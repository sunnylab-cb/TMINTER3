"""Joint-grid pair module: permutation, batching, gradients and chunk checks."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
from ppcbind.models.pairformer import JointPairformerAdapter, TriangleMultiplication, monomer_contact_features
from ppcbind.models.esm_pair import ESMPairSiteClassifier

torch.set_num_threads(2)
torch.manual_seed(49)
for incoming in (False, True):
    m = TriangleMultiplication(8, 4, incoming, 2)
    z = torch.randn(7, 7, 8, requires_grad=True)
    n = m.norm(z)
    a, b = m.left(n)*m.left_gate(n).sigmoid(), m.right(n)*m.right_gate(n).sigmoid()
    reference = torch.einsum("kic,kjc->ijc" if incoming else "ikc,jkc->ijc", a, b) / 7**0.5
    reference = m.output(m.output_norm(reference)) * m.output_gate(n).sigmoid()
    torch.testing.assert_close(m(z), reference, atol=1e-6, rtol=1e-5)

for triangles, pooling, rank in ((False, "mean", 0), (True, "mean", 0), (True, "softmax", 0), (True, "softmax", 4), (False, "softmax", 4)):
    config = dict(pair_hidden=8, triangle_hidden=4, pair_chunk=3, pairformer_blocks=2, triangle_updates=triangles, pooling=pooling, outer_rank=rank)
    m = JointPairformerAdapter(16, config).eval()
    a, b = torch.randn(5, 16), torch.randn(9, 16)
    ca, cb = torch.rand(5, 5, 2), torch.rand(9, 9, 2)
    assert m(a, b, ca, cb)[0].count_nonzero() == 0
    torch.nn.init.normal_(m.readout[-1].weight, std=0.1)
    x, y = m(a, b, ca, cb)
    sy, sx = m(b, a, cb, ca)
    torch.testing.assert_close(x, sx, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(y, sy, atol=2e-6, rtol=1e-5)
    m.train()
    other = deepcopy(m)
    other.checkpoint_blocks = False
    out = m(a, b, ca, cb)
    out2 = other(a, b, ca, cb)
    sum(t.square().sum() for t in out).backward()
    sum(t.square().sum() for t in out2).backward()
    for p, q in zip(m.parameters(), other.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-5)
    print("PASS adapter", triangles, pooling, rank)

inputs = {"protein_mask": torch.ones(2, 6, dtype=torch.bool),
          "contact_edge_index": torch.tensor([[0, 6, 6], [1, 7, 7]]),
          "contact_edge_scores": torch.tensor([0.1, 0.3, 0.8])}
contacts = monomer_contact_features(inputs, 1, 3, torch.float32)
assert contacts[0, 1, 0] == 0.8 and contacts[..., 1].sum() == 1

base = dict(d_esm=12, d_seq=3, d_model=16, d_hidden=32, dropout=0.0, classifier_dropout=0.0,
            n_transformer_layers=0, use_tcn_context=False, use_contact_graph=False)
m = ESMPairSiteClassifier(base, dict(mode="pairformer", pair_hidden=8, triangle_hidden=4,
                                    pair_chunk=3, pairformer_blocks=2)).eval()
def side(lengths):
    width = max(lengths)
    return {"esm_embeddings": torch.randn(len(lengths), width, 12),
            "seq_features": torch.randn(len(lengths), width, 3),
            "protein_mask": torch.arange(width)[None] < torch.tensor(lengths)[:, None]}
left, right = side([5, 3]), side([9, 4])
torch.nn.init.normal_(m.adapter.readout[-1].weight, std=0.1)
with torch.no_grad():
    output = m(left, right)
    swapped = m(right, left)
    torch.testing.assert_close(output["left_logits"], swapped["right_logits"], atol=2e-6, rtol=1e-5)
    assert output["left_logits"][1, 3:].count_nonzero() == 0
    for data in (left, right):
        for key in ("esm_embeddings", "seq_features"):
            data[key][~data["protein_mask"]] = 10000
    changed = m(left, right)
    torch.testing.assert_close(output["left_logits"], changed["left_logits"], atol=2e-6, rtol=1e-5)
print("PASS full model padding and swap; graph isolation; chunked triangle reference")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/training"))
from train_esm_pair import PairTokenBatchSampler
lengths = [(5, 9), (2300, 2300), (20, 30), (100, 1)]
sampler = PairTokenBatchSampler(lengths, 64, 65536, 4000000, 25000000, False, 49, joint_grid=True)
assert sorted(i for batch in sampler for i in batch) == list(range(len(lengths)))
assert any(batch == [1] for batch in sampler)
try:
    PairTokenBatchSampler([(3000, 3000)], 64, 65536, 4000000, 25000000, False, 49, joint_grid=True)
except ValueError:
    pass
else:
    raise AssertionError("Oversized joint grid was not rejected")
print("PASS joint-grid sampler coverage and explicit oversize rejection")
