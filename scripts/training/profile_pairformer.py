"""Measure a real longest train pair, with no truncation or test access."""
import argparse
import json
import time
from pathlib import Path
import torch
import yaml
from train_esm_pair import _build_loader, _move_side, PairResidueLoss
from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
from ppcbind.models.esm_pair import ESMPairSiteClassifier

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
torch.set_num_threads(4)
torch.manual_seed(49)
dataset, loader = _build_loader(config, "train", False)
index = max(range(len(dataset)), key=lambda i: sum(dataset.pair_lengths[i]))
batch = collate_esm_pair_features([dataset[index]])
device = torch.device("cuda:0")
a, ay = _move_side(batch["left"], device)
b, by = _move_side(batch["right"], device)
model = ESMPairSiteClassifier(config["model"], config["pair_model"]).to(device).train()
pos, neg = dataset.class_counts
criterion = PairResidueLoss(min(20, neg / pos), 0.05, 0.25).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
print(json.dumps({"event": "stress_start", "pair": batch["pair_id"], "lengths": dataset.pair_lengths[index],
                  "n_train_pairs": len(dataset), "n_batches": len(loader), "device": torch.cuda.get_device_name()}), flush=True)
with torch.amp.autocast("cuda"):
    out = model(a, b)
    loss = criterion(out["left_logits"], ay, out["right_logits"], by)
assert torch.isfinite(loss)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
scaler.step(optimizer)
scaler.update()
torch.cuda.synchronize()
print(json.dumps({"event": "stress_pass", "seconds": time.monotonic()-started,
                  "peak_GiB": torch.cuda.max_memory_allocated()/2**30, "loss": float(loss)}), flush=True)
