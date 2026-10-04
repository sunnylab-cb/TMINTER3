"""Small numerical tests for auxiliary losses without loading feature assets."""
import ast
from pathlib import Path
from typing import Any

import torch


source = Path(__file__).resolve().parents[1] / "scripts/training/train_esm_pair.py"
tree = ast.parse(source.read_text())
names = {"_pair_contact_loss", "_site_contact_consistency"}
module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
namespace = {"torch": torch, "Any": Any}
exec(compile(module, str(source), "exec"), namespace)
contact_loss = namespace["_pair_contact_loss"]
consistency = namespace["_site_contact_consistency"]
scores = torch.tensor([[0.2, -0.5, 99.0], [0.8, -1.0, -99.0]], requires_grad=True)
contacts = torch.tensor([[0, 1], [0, 1]])
left_mask, right_mask = torch.tensor([True, True]), torch.tensor([True, True, False])
loss = contact_loss([scores], [contacts], [left_mask], [right_mask])
expected = 0.5 * (torch.nn.functional.softplus(-torch.stack([scores[0, 0], scores[1, 1]])).mean()
                  + torch.nn.functional.softplus(torch.stack([scores[0, 1], scores[1, 0]])).mean())
torch.testing.assert_close(loss, expected)
loss.backward()
assert torch.equal(scores.grad[:, 2], torch.zeros(2))
scores.grad = None
hard = contact_loss([scores], [contacts], [left_mask], [right_mask], hard_negative_fraction=1.0,
                    negatives_per_positive=1, max_sampled_negatives=1)
hard.backward()
assert scores.grad[1, 0] > 0 and scores.grad[0, 1] == 0
assert torch.equal(scores.grad[:, 2], torch.zeros(2))
out = {"pair_contact_logits": [scores], "left_logits": torch.randn(1, 2, 2, requires_grad=True),
       "right_logits": torch.randn(1, 3, 2, requires_grad=True)}
batch = {"left_observed_mask": [left_mask], "right_observed_mask": [right_mask]}
c = consistency(out, batch)
rev = {"pair_contact_logits": [scores.T], "left_logits": out["right_logits"], "right_logits": out["left_logits"]}
torch.testing.assert_close(c, consistency(rev, {"left_observed_mask": [right_mask], "right_observed_mask": [left_mask]}))
c.backward()
assert torch.equal(out["right_logits"].grad[0, 2], torch.zeros(2))
assert torch.isfinite(c)
print("PASS: canonical loss, hard-negative selection, masked gradients, consistency swap symmetry")
