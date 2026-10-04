"""Validation-only, length-matched partner replacement; not negative labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train_esm_pair import _build_loader, _move_side, _summarize_predictions
from ppcbind.models.esm_pair import ESMPairSiteClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["config"]
    dataset, loader = _build_loader(config, "val", False)
    model = ESMPairSiteClassifier(config["model"], config["pair_model"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {row["pair_id"]: row for row in dataset.rows}
    cache, queries, info = {}, {}, {}
    amp = bool(config["training"].get("amp", True))
    device = torch.device("cuda:0")
    equivalence_error = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            for side, opposite in (("left", "right"), ("right", "left")):
                inputs, labels = _move_side(batch[side], device)
                with torch.amp.autocast("cuda", enabled=amp):
                    result = model.base(**inputs)
                for i, pair_id in enumerate(batch["pair_id"]):
                    row = metadata[pair_id]
                    key = row[f"{side}_seq_id"]
                    length = int(inputs["protein_mask"][i].sum())
                    if key not in cache:
                        cache[key] = (result["h_residue"][i, :length].cpu(), result["logits"][i, :length].cpu())
                        info[key] = {"pdb_id": row["pdb_id"], "sha1": row[f"{side}_sequence_sha1"], "length": length}
                    queries[f"{pair_id}/{side}"] = {
                        "query": key, "partner": row[f"{opposite}_seq_id"],
                        "pdb_id": row["pdb_id"], "labels": labels[i, :length].cpu().numpy(),
                        "homomer": row["left_sequence_sha1"] == row["right_sequence_sha1"],
                    }
            if batch_index == 0:
                left, _ = _move_side(batch["left"], device)
                right, _ = _move_side(batch["right"], device)
                with torch.amp.autocast("cuda", enabled=amp):
                    reference = model(left, right)
                    for i, pair_id in enumerate(batch["pair_id"]):
                        row = metadata[pair_id]
                        a, logits = cache[row["left_seq_id"]]
                        b, _ = cache[row["right_seq_id"]]
                        delta = model.adapter(a.cuda(), b.cuda())[0] if model.mode == "outer_product" else 0.0
                        reconstructed = logits.cuda() + delta
                        error = float((reference["left_logits"][i, :a.shape[0]] - reconstructed).abs().max())
                        equivalence_error = max(equivalence_error, error)
                if equivalence_error > 0.02:
                    raise RuntimeError(f"Cached adapter reconstruction differs: {equivalence_error}")
            if batch_index % 20 == 0:
                print(json.dumps({"stage": "encode_val", "batch": batch_index, "cached_chains": len(cache)}), flush=True)

        keys = sorted(cache)
        max_cells = int(config["training"]["max_single_pair_cells"])
        candidates = {}
        for partner in sorted({q["partner"] for q in queries.values()}):
            candidates[partner] = sorted(
                (k for k in keys if info[k]["pdb_id"] != info[partner]["pdb_id"] and info[k]["sha1"] != info[partner]["sha1"]),
                key=lambda k: (abs(info[k]["length"] - info[partner]["length"]), k),
            )

        def predict(query, partner):
            h, logits = cache[query]
            with torch.amp.autocast("cuda", enabled=amp):
                if model.mode == "independent":
                    result = logits.cuda()
                elif model.mode == "outer_product":
                    delta, _, _ = model.adapter(h.cuda(), cache[partner][0].cuda())
                    result = logits.cuda() + delta
                else:
                    raise ValueError("Diagnostic supports independent/outer_product only")
            return result.float().softmax(-1)[:, 1].cpu().numpy()

        records, summaries = [], {}
        skipped = []
        eligible = {}
        for identifier, q in queries.items():
            pool = [k for k in candidates[q["partner"]] if info[k]["sha1"] != info[q["query"]]["sha1"] and info[k]["length"] * info[q["query"]]["length"] <= max_cells][:64]
            if pool:
                eligible[identifier] = pool
            else:
                skipped.append(identifier)
        correct = {}
        if not eligible:
            raise RuntimeError("No eligible replacement partners; diagnostic cohort is empty")
        for index, identifier in enumerate(eligible):
            q = queries[identifier]
            correct[identifier] = predict(q["query"], q["partner"])
            if index % 500 == 0:
                print(json.dumps({"stage": "correct_partner", "directional_chains": index}), flush=True)
        for seed in args.seeds:
            reference_rows, replaced_rows = [], []
            start = len(records)
            for index, (identifier, pool) in enumerate(eligible.items()):
                q = queries[identifier]
                local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:8], "little")
                replacement = pool[int(np.random.default_rng(local_seed).integers(len(pool)))]
                valid = q["labels"] != -100
                y = q["labels"][valid]
                p = correct[identifier][valid]
                counterfactual = predict(q["query"], replacement)[valid]
                if model.mode == "independent" and not np.array_equal(p, counterfactual):
                    raise RuntimeError("Independent-model negative control changed after partner replacement")
                ref_ap = float(average_precision_score(y, p)) if y.any() else 0.0
                alt_ap = float(average_precision_score(y, counterfactual)) if y.any() else 0.0
                records.append({"id": identifier, "pdb_id": q["pdb_id"], "homomer": q["homomer"], "seed": seed,
                                "original_partner": q["partner"], "replacement": replacement,
                                "original_partner_length": info[q["partner"]]["length"],
                                "replacement_length": info[replacement]["length"],
                                "correct_ap": ref_ap, "replaced_ap": alt_ap, "ap_drop": ref_ap - alt_ap,
                                "mean_absolute_probability_change": float(np.abs(p-counterfactual).mean())})
                reference_rows.append((identifier, p, y))
                replaced_rows.append((identifier, counterfactual, y))
                if index % 500 == 0:
                    print(json.dumps({"stage": "replacement", "seed": seed, "directional_chains": index}), flush=True)
            current = records[start:]
            groups = {}
            for group, selected in (("all", current), ("homomer", [r for r in current if r["homomer"]]), ("heteromer", [r for r in current if not r["homomer"]])):
                pdbs = {}
                for row in selected:
                    pdbs.setdefault(row["pdb_id"], []).append(row["ap_drop"])
                groups[group] = {"n": len(selected), "n_pdb": len(pdbs),
                                 "mean_ap_drop": float(np.mean([r["ap_drop"] for r in selected])) if selected else None,
                                 "pdb_macro_ap_drop": float(np.mean([np.mean(v) for v in pdbs.values()])) if pdbs else None}
            summaries[str(seed)] = {"correct": _summarize_predictions(reference_rows, [], False),
                                    "replaced": _summarize_predictions(replaced_rows, [], False), "groups": groups}
            with (args.output / "directional_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            (args.output / "summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint),
                "split": "val", "test_read": False, "completed_seeds": list(summaries),
                "scope": "counterfactual sensitivity; replacement partners are not biological negative labels",
                "replacement_policy": "sample among 64 nearest-length eligible validation partners, with replacement",
                "equivalence_max_logit_error": equivalence_error, "skipped": skipped, "results": summaries}, indent=2))
    print(json.dumps({"stage": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
