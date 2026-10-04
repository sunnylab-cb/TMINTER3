"""Symmetric partner-conditioned extension of the single-chain M18 model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ppcbind.models.esm_site import ESMSiteClassifier


def _segment_pool(h: torch.Tensor, mask: torch.Tensor, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool each partner into contiguous sequence segments with bounded token count."""
    pooled_rows: list[torch.Tensor] = []
    lengths: list[int] = []
    for sample, sample_mask in zip(h, mask):
        length = int(sample_mask.sum().item())
        n_tokens = min(max(1, int(max_tokens)), max(1, length))
        boundaries = torch.linspace(0, length, n_tokens + 1, device=h.device).round().long()
        tokens: list[torch.Tensor] = []
        for idx in range(n_tokens):
            start = int(boundaries[idx].item())
            stop = max(start + 1, int(boundaries[idx + 1].item()))
            stop = min(stop, length)
            tokens.append(sample[start:stop].mean(dim=0))
        pooled_rows.append(torch.stack(tokens))
        lengths.append(n_tokens)
    width = max(lengths)
    output = h.new_zeros((h.shape[0], width, h.shape[-1]))
    output_mask = torch.zeros((h.shape[0], width), dtype=torch.bool, device=h.device)
    for idx, row in enumerate(pooled_rows):
        output[idx, : row.shape[0]] = row
        output_mask[idx, : row.shape[0]] = True
    return output, output_mask


class SymmetricPartnerAdapter(nn.Module):
    """Shared bidirectional cross-attention with a zero-initialized residual."""

    def __init__(self, d_model: int, n_heads: int, bottleneck: int, dropout: float, max_partner_tokens: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.max_partner_tokens = max(1, int(max_partner_tokens))
        self.query_norm = nn.LayerNorm(d_model)
        self.partner_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        partner: torch.Tensor,
        partner_mask: torch.Tensor,
    ) -> torch.Tensor:
        partner_tokens, token_mask = _segment_pool(partner, partner_mask, self.max_partner_tokens)
        cross, _ = self.cross_attention(
            self.query_norm(query),
            self.partner_norm(partner_tokens),
            self.partner_norm(partner_tokens),
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        features = torch.cat([query, cross, query - cross, query * cross], dim=-1)
        delta = self.residual(features)
        return (query + delta).masked_fill(~query_mask.unsqueeze(-1), 0.0)


class SymmetricOuterProductAdapter(nn.Module):
    """Residue-pair outer products with a symmetric, zero-init site residual."""

    def __init__(
        self,
        d_model: int,
        pair_rank: int,
        pair_hidden: int,
        bottleneck: int,
        dropout: float,
        conv_layers: int,
        pooling: str,
        pooling_topk: int,
        pooling_temperature: float,
        endpoint_features: bool = False,
    ) -> None:
        super().__init__()
        self.pair_rank = max(2, int(pair_rank))
        self.pair_hidden = max(4, int(pair_hidden))
        self.pooling = str(pooling).lower()
        if self.pooling not in {"softmax", "topk", "null_softmax"}:
            raise ValueError(f"Unsupported pair pooling={self.pooling!r}")
        self.pooling_topk = max(1, int(pooling_topk))
        self.pooling_temperature = max(1e-3, float(pooling_temperature))
        self.endpoint_features = bool(endpoint_features)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.pair_rank),
        )
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(self.pair_rank * self.pair_rank + (3 * self.pair_rank if self.endpoint_features else 0)),
            nn.Linear(self.pair_rank * self.pair_rank + (3 * self.pair_rank if self.endpoint_features else 0), self.pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        conv_blocks: list[nn.Module] = []
        for _ in range(max(0, int(conv_layers))):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.pair_hidden,
                        self.pair_hidden,
                        kernel_size=3,
                        padding=1,
                        groups=self.pair_hidden,
                    ),
                    nn.GELU(),
                    nn.Conv2d(self.pair_hidden, self.pair_hidden, kernel_size=1),
                    nn.GELU(),
                )
            )
        self.pair_convs = nn.ModuleList(conv_blocks)
        self.contact_head = nn.Linear(self.pair_hidden, 1)
        if self.pooling == "null_softmax":
            self.null_logit = nn.Parameter(torch.zeros(()))
        self.site_residual = nn.Sequential(
            nn.LayerNorm(d_model + self.pair_hidden + (2 if self.pooling == "null_softmax" else 1)),
            nn.Linear(d_model + self.pair_hidden + (2 if self.pooling == "null_softmax" else 1), bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2),
        )
        nn.init.zeros_(self.site_residual[-1].weight)
        nn.init.zeros_(self.site_residual[-1].bias)

    def _pair_features(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_low = self.residue_projection(left)
        right_low = self.residue_projection(right)
        outer = torch.einsum("ir,js->ijrs", left_low, right_low)
        # Sharing the residue projection and symmetrizing the two latent axes
        # guarantees z(A,B)[i,j] == z(B,A)[j,i].
        outer = 0.5 * (outer + outer.transpose(-1, -2))
        features = outer.flatten(start_dim=-2)
        if self.endpoint_features:
            a, b = left_low[:, None, :], right_low[None, :, :]
            features = torch.cat([features, a + b, torch.abs(a - b), a * b], dim=-1)
        pair = self.pair_projection(features)
        if self.pair_convs:
            pair_2d = pair.permute(2, 0, 1).unsqueeze(0)
            for block in self.pair_convs:
                # Strided FP16 depthwise-convolution inputs can take a very slow
                # backend path; materialize both orientations in NCHW order.
                forward = block(pair_2d.contiguous())
                reverse = block(pair_2d.transpose(-1, -2).contiguous()).transpose(-1, -2)
                pair_2d = pair_2d + 0.5 * (forward + reverse)
            pair = pair_2d.squeeze(0).permute(1, 2, 0)
        return pair

    def _pool_one_axis(
        self,
        pair: torch.Tensor,
        contact_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling == "topk":
            count = min(self.pooling_topk, pair.shape[1])
            values, indices = torch.topk(contact_logits, k=count, dim=1)
            selected = torch.gather(
                pair,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, pair.shape[-1]),
            )
            return selected.mean(dim=1), values.mean(dim=1, keepdim=True)
        if self.pooling == "null_softmax":
            # Equal total prior mass for real candidates and the null choice,
            # independent of partner length. The null carries zero context.
            scaled = contact_logits / self.pooling_temperature
            scaled = scaled - scaled.new_tensor(float(pair.shape[1])).log()
            null = self.null_logit.to(scaled.dtype).expand(scaled.shape[0], 1)
            weights = torch.softmax(torch.cat([scaled, null], dim=1), dim=1)[:, :-1]
            context = torch.sum(weights.unsqueeze(-1) * pair, dim=1)
            evidence = torch.sum(weights * contact_logits, dim=1, keepdim=True)
            return context, torch.cat([evidence, weights.sum(dim=1, keepdim=True)], dim=1)
        weights = torch.softmax(contact_logits / self.pooling_temperature, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * pair, dim=1)
        evidence = torch.sum(weights * contact_logits, dim=1, keepdim=True)
        return context, evidence

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self._pair_features(left, right)
        contact_logits = self.contact_head(pair).squeeze(-1)
        left_context, left_evidence = self._pool_one_axis(pair, contact_logits)
        right_context, right_evidence = self._pool_one_axis(
            pair.transpose(0, 1), contact_logits.transpose(0, 1)
        )
        left_delta = self.site_residual(torch.cat([left, left_context, left_evidence], dim=-1))
        right_delta = self.site_residual(torch.cat([right, right_context, right_evidence], dim=-1))
        return left_delta, right_delta, contact_logits


class ESMPairSiteClassifier(nn.Module):
    """Apply one shared M18 encoder to both chains and condition them symmetrically."""

    def __init__(self, base_config: dict[str, Any], pair_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        pair_config = pair_config or {}
        self.base = ESMSiteClassifier(base_config)
        self.mode = str(pair_config.get("mode", "cross_attention")).lower()
        if self.mode not in {"independent", "cross_attention", "outer_product"}:
            raise ValueError(f"Unsupported pair mode={self.mode!r}")
        d_model = int(self.base.config.d_model)
        if self.mode == "outer_product":
            self.adapter = SymmetricOuterProductAdapter(
                d_model=d_model,
                pair_rank=int(pair_config.get("pair_rank", 8)),
                pair_hidden=int(pair_config.get("pair_hidden", 32)),
                bottleneck=int(pair_config.get("bottleneck", 64)),
                dropout=float(pair_config.get("dropout", 0.10)),
                conv_layers=int(pair_config.get("conv_layers", 0)),
                pooling=str(pair_config.get("pooling", "softmax")),
                pooling_topk=int(pair_config.get("pooling_topk", 4)),
                pooling_temperature=float(pair_config.get("pooling_temperature", 0.5)),
                endpoint_features=bool(pair_config.get("endpoint_features", False)),
            )
        else:
            self.adapter = SymmetricPartnerAdapter(
                d_model=d_model,
                n_heads=int(pair_config.get("n_heads", 8)),
                bottleneck=int(pair_config.get("bottleneck", 64)),
                dropout=float(pair_config.get("dropout", 0.10)),
                max_partner_tokens=int(pair_config.get("max_partner_tokens", 32)),
            )

    def load_base_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
        if not isinstance(state, dict):
            raise ValueError("Base checkpoint does not contain a state dict")
        self.base.load_state_dict(state, strict=True)

    def forward(
        self,
        left_inputs: dict[str, torch.Tensor],
        right_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        left_base = self.base(**left_inputs)
        right_base = self.base(**right_inputs)
        if self.mode == "independent":
            return {
                "left_logits": left_base["logits"],
                "right_logits": right_base["logits"],
                "left_h": left_base["h_residue"],
                "right_h": right_base["h_residue"],
                "pair_contact_logits": [],
            }
        if self.mode == "outer_product":
            left_delta = torch.zeros_like(left_base["logits"])
            right_delta = torch.zeros_like(right_base["logits"])
            pair_contact_logits: list[torch.Tensor] = []
            for idx in range(left_base["h_residue"].shape[0]):
                left_length = int(left_inputs["protein_mask"][idx].sum().item())
                right_length = int(right_inputs["protein_mask"][idx].sum().item())
                delta_left, delta_right, contact_logits = self.adapter(
                    left_base["h_residue"][idx, :left_length],
                    right_base["h_residue"][idx, :right_length],
                )
                left_delta[idx, :left_length] = delta_left
                right_delta[idx, :right_length] = delta_right
                pair_contact_logits.append(contact_logits)
            left_logits = left_base["logits"] + left_delta
            right_logits = right_base["logits"] + right_delta
            return {
                "left_logits": left_logits,
                "right_logits": right_logits,
                "left_h": left_base["h_residue"],
                "right_h": right_base["h_residue"],
                "pair_contact_logits": pair_contact_logits,
            }
        left_h = self.adapter(
            left_base["h_residue"],
            left_inputs["protein_mask"].bool(),
            right_base["h_residue"],
            right_inputs["protein_mask"].bool(),
        )
        right_h = self.adapter(
            right_base["h_residue"],
            right_inputs["protein_mask"].bool(),
            left_base["h_residue"],
            left_inputs["protein_mask"].bool(),
        )
        left_logits = self.base.classifier(left_h).masked_fill(
            ~left_inputs["protein_mask"].bool().unsqueeze(-1), 0.0
        )
        right_logits = self.base.classifier(right_h).masked_fill(
            ~right_inputs["protein_mask"].bool().unsqueeze(-1), 0.0
        )
        return {
            "left_logits": left_logits,
            "right_logits": right_logits,
            "left_h": left_h,
            "right_h": right_h,
            "pair_contact_logits": [],
        }
