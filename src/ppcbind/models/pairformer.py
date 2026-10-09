"""Small Pairformer-inspired joint-grid trunk, not an AF3 reproduction.

Only monomer sequence separation and precomputed intrachain contacts enter
the pair initializer. Experimental interchain labels are never inputs.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TriangleMultiplication(nn.Module):
    def __init__(self, channels: int, hidden: int, incoming: bool, chunk: int):
        super().__init__()
        self.incoming, self.chunk = incoming, chunk
        self.norm = nn.LayerNorm(channels)
        self.left = nn.Linear(channels, hidden)
        self.right = nn.Linear(channels, hidden)
        self.left_gate = nn.Linear(channels, hidden)
        self.right_gate = nn.Linear(channels, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, channels)
        self.output_gate = nn.Linear(channels, channels)

    def forward(self, z):
        n = z.shape[0]
        normalized = self.norm(z)
        a = self.left(normalized) * self.left_gate(normalized).sigmoid()
        b = self.right(normalized) * self.right_gate(normalized).sigmoid()
        a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)
        if self.incoming:
            a, b = a.transpose(1, 2), b.transpose(1, 2)
        rows = []
        for start in range(0, n, self.chunk):
            end = min(n, start + self.chunk)
            product = torch.bmm(a[:, start:end], b.transpose(1, 2)) / math.sqrt(n)
            update = self.output(self.output_norm(product.permute(1, 2, 0)))
            rows.append(update * self.output_gate(normalized[start:end]).sigmoid())
        return torch.cat(rows, dim=0)


class PairformerBlock(nn.Module):
    def __init__(self, single: int, pair: int, triangle: int, use_triangle: bool, chunk: int, pooling: str = "mean"):
        super().__init__()
        self.chunk = chunk
        if pooling not in {"mean", "softmax"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pool_score = nn.Linear(pair, 1) if pooling == "softmax" else None
        self.outgoing = TriangleMultiplication(pair, triangle, False, chunk) if use_triangle else None
        self.incoming = TriangleMultiplication(pair, triangle, True, chunk) if use_triangle else None
        self.transition = nn.Sequential(nn.LayerNorm(pair), nn.Linear(pair, 2 * pair), nn.GELU(), nn.Linear(2 * pair, pair))
        self.single_update = nn.Sequential(nn.LayerNorm(single + 2 * pair), nn.Linear(single + 2 * pair, single), nn.GELU(), nn.Linear(single, single))
        self.single_norm = nn.LayerNorm(single)
        self.to_row, self.to_col = nn.Linear(single, pair), nn.Linear(single, pair)

    def forward(self, s, z, split):
        normalized = self.single_norm(s)
        z = z + self.to_row(normalized)[:, None] + self.to_col(normalized)[None, :]
        if self.outgoing is not None:
            z = z + self.outgoing(z)
            z = z + self.incoming(z)
        z = torch.cat([row + self.transition(row) for row in z.split(self.chunk)], dim=0)
        # Cross-chain row/column contexts, not real contact probabilities.
        def pool(grid, dim):
            if self.pool_score is None:
                return grid.mean(dim)
            weights = self.pool_score(grid).squeeze(-1).float().softmax(dim).to(grid.dtype)
            return (grid * weights.unsqueeze(-1)).sum(dim)
        a = torch.cat([pool(z[:split, split:], 1), pool(z[split:, :split], 0)], -1)
        b = torch.cat([pool(z[split:, :split], 1), pool(z[:split, split:], 0)], -1)
        s = s + self.single_update(torch.cat([s, torch.cat([a, b])], -1))
        return s, z


class JointPairformerAdapter(nn.Module):
    def __init__(self, d_model: int, config: dict):
        super().__init__()
        channels = int(config.get("pair_hidden", 32))
        triangle = int(config.get("triangle_hidden", 16))
        chunk = int(config.get("pair_chunk", 128))
        depth = int(config.get("pairformer_blocks", 2))
        if min(channels, triangle, chunk, depth) < 1:
            raise ValueError("Pairformer dimensions/depth/chunk must be positive")
        self.checkpoint_blocks = bool(config.get("checkpoint_blocks", True))
        self.single_norm = nn.LayerNorm(d_model)
        self.to_row, self.to_col = nn.Linear(d_model, channels), nn.Linear(d_model, channels)
        # 0..64: signed clipped intrachain offset; 65: unknown interchain offset.
        self.relative = nn.Embedding(66, channels)
        self.contact = nn.Linear(2, channels, bias=False)
        self.outer_rank = int(config.get("outer_rank", 0))
        if self.outer_rank < 0:
            raise ValueError("outer_rank must be nonnegative")
        self.outer_chunk = chunk
        if self.outer_rank:
            self.outer_residue = nn.Linear(d_model, self.outer_rank)
            self.outer_pair = nn.Linear(self.outer_rank**2, channels)
        self.blocks = nn.ModuleList([PairformerBlock(d_model, channels, triangle, bool(config.get("triangle_updates", True)), chunk, str(config.get("pooling", "mean"))) for _ in range(depth)])
        self.readout = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, left, right, left_contact, right_contact):
        split, n = left.shape[0], left.shape[0] + right.shape[0]
        s = torch.cat([left, right])
        norm = self.single_norm(s)
        positions = torch.cat([torch.arange(split, device=s.device), torch.arange(n-split, device=s.device)])
        chain = torch.arange(n, device=s.device) >= split
        same = chain[:, None] == chain[None, :]
        relative = (positions[:, None] - positions[None, :]).clamp(-32, 32) + 32
        relative = torch.where(same, relative, 65)
        row, col = self.to_row(norm), self.to_col(norm)
        z = row[:, None] + col[None, :] + self.relative(relative).to(row.dtype)
        contacts = torch.zeros((n, n, 2), device=s.device, dtype=z.dtype)
        contacts[:split, :split] = left_contact.to(z.dtype)
        contacts[split:, split:] = right_contact.to(z.dtype)
        z = z + self.contact(contacts)
        if self.outer_rank:
            low = self.outer_residue(norm)
            def cross_outer(a, b):
                rows = []
                for row in a.split(self.outer_chunk):
                    product = torch.einsum("ir,js->ijrs", row, b)
                    product = 0.5 * (product + product.transpose(-1, -2))
                    rows.append(self.outer_pair(product.flatten(-2)))
                return torch.cat(rows)
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                cross = checkpoint(cross_outer, low[:split], low[split:], use_reentrant=False)
            else:
                cross = cross_outer(low[:split], low[split:])
            # Only AB/BA receive the symmetric outer-product initializer.
            upper = torch.cat([z[:split, :split], z[:split, split:] + cross], 1)
            lower = torch.cat([z[split:, :split] + cross.transpose(0, 1), z[split:, split:]], 1)
            z = torch.cat([upper, lower], 0)
        for block in self.blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                s, z = checkpoint(block, s, z, split, use_reentrant=False)
            else:
                s, z = block(s, z, split)
        logits = self.readout(s)
        return logits[:split], logits[split:]


def monomer_contact_features(inputs: dict, index: int, length: int, dtype: torch.dtype):
    """Use the batched ESM2 graph; absent edges are unknown, not negatives."""
    result = torch.zeros((length, length, 2), device=inputs["protein_mask"].device, dtype=dtype)
    edges = inputs.get("contact_edge_index")
    if edges is None or not edges.numel():
        return result
    width = inputs["protein_mask"].shape[1]
    local = edges - index * width
    keep = ((local >= 0) & (local < length)).all(0)
    src, dst = local[:, keep]
    scores = inputs.get("contact_edge_scores")
    values = torch.ones(src.shape[0], device=result.device, dtype=dtype) if scores is None else scores[keep].to(dtype)
    values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    flat = result[..., 0].contiguous().view(-1)
    flat.scatter_reduce_(0, src * length + dst, values, reduce="amax", include_self=True)
    result[..., 0] = flat.view(length, length)
    result[src, dst, 1] = 1
    return result
