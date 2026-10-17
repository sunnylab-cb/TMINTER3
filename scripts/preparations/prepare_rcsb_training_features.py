"""Gate RCSB pair rows on local features and fit train-only raw-89 statistics."""

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FEATURE_COUNT = 89
FEATURE_KEY = "spatial_scalar_raw_features"
FEATURE_SOURCE = "protenix_mini_esm_v0.5.0"
SPLITS = ("train", "val", "test")
UID_PATTERN = re.compile(r"^useq_[0-9a-f]{64}$")
REQUIRED_FIELDS = {
    "pair_id", "split", "left_seq_id", "right_seq_id", "left_length",
    "right_length", "label_path", "label_offset",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        require(fields is not None, "Pair manifest has no CSV header")
        require(len(fields) == len(set(fields)), "Pair manifest has duplicate columns")
        require(REQUIRED_FIELDS <= set(fields),
                "Pair manifest is missing required columns: " + ",".join(sorted(REQUIRED_FIELDS - set(fields))))
        return fields


def pair_rows(path, expected_fields):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == expected_fields, "Pair manifest header changed while processing")
        for line_number, row in enumerate(reader, start=2):
            require(None not in row and all(value is not None for value in row.values()),
                    f"Malformed pair row at line {line_number}")
            yield row


def parse_endpoint(row, side, line_number):
    uid = row[f"{side}_seq_id"]
    require(UID_PATTERN.fullmatch(uid) is not None,
            f"Invalid {side} sequence ID at line {line_number}: {uid!r}")
    try:
        length = int(row[f"{side}_length"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {side} length at line {line_number}") from exc
    require(length > 0, f"Non-positive {side} length at line {line_number}")
    return uid, length


def index_manifest(path, fields):
    uid_splits, uid_lengths = {}, {}
    source_pairs = Counter({split: 0 for split in SPLITS})
    for line_number, row in enumerate(pair_rows(path, fields), start=2):
        split = row["split"]
        require(split in SPLITS, f"Unexpected split at line {line_number}: {split!r}")
        require(row["pair_id"], f"Missing pair_id at line {line_number}")
        source_pairs[split] += 1
        endpoints = (parse_endpoint(row, "left", line_number),
                     parse_endpoint(row, "right", line_number))
        for uid, length in endpoints:
            previous_split = uid_splits.setdefault(uid, split)
            require(previous_split == split,
                    f"Sequence ID crosses splits: {uid} ({previous_split}, {split})")
            previous_length = uid_lengths.setdefault(uid, length)
            require(previous_length == length,
                    f"Conflicting lengths for {uid}: {previous_length} and {length}")
    return uid_splits, uid_lengths, source_pairs


def feature_paths(feature_root, uid):
    return (
        Path(feature_root) / "TMINTER3-ESMC/features/unique" / uid[5:7] / f"{uid}.pt",
        Path(feature_root) / "protenix/features/complete_raw_v1/pt" / uid[5:7] / f"{uid}.pt",
    )


def file_state(path):
    try:
        if not path.is_file():
            return "missing"
        if path.stat().st_size == 0:
            return "empty"
        return "present"
    except OSError:
        return "unreadable"


def inspect_uid(task):
    uid, root = task
    esmc_path, protenix_path = feature_paths(root, uid)
    return uid, file_state(esmc_path), file_state(protenix_path)


def inspect_features(uid_splits, feature_root, workers):
    statuses = {}
    uids = sorted(uid_splits)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Keep submitted work bounded while allowing workers to overlap filesystem checks.
        for start in range(0, len(uids), workers * 4):
            batch = [(uid, feature_root) for uid in uids[start:start + workers * 4]]
            for uid, esmc, protenix in executor.map(inspect_uid, batch):
                statuses[uid] = (esmc, protenix)
    return statuses


def uid_reason(status):
    names = ("esmc", "protenix")
    return ";".join(f"{state}_{name}" for name, state in zip(names, status) if state != "present")


def pair_reasons(row, statuses):
    reasons = {}
    for side in ("left", "right"):
        reason = uid_reason(statuses[row[f"{side}_seq_id"]])
        if reason:
            reasons[side] = reason
    return reasons


def reverse_rank(pair_id, ordinal):
    return bytes(255 - value for value in pair_id.encode("utf-8")), -ordinal


def offer_smoke(heap, row, ordinal, limit):
    rank = reverse_rank(row["pair_id"], ordinal)
    item = (rank, ordinal, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank > heap[0][0]:
        heapq.heapreplace(heap, item)


def selected_smoke_rows(heap):
    return [row for _, _, row in sorted(heap, key=lambda item: (item[2]["pair_id"], item[1]))]


def load_validated_raw89(task):
    uid, expected_length, path = task
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        require(isinstance(data, dict), f"{uid}: Protenix payload is not a dictionary")
        sequence = data.get("sequence")
        require(isinstance(sequence, str), f"{uid}: missing sequence")
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        require(data.get("source") == FEATURE_SOURCE, f"{uid}: unexpected Protenix source")
        require(data.get("sequence_sha256") == digest == uid[5:], f"{uid}: sequence hash mismatch")
        require(len(sequence) == expected_length, f"{uid}: sequence length mismatch")
        features = data.get(FEATURE_KEY)
        require(torch.is_tensor(features), f"{uid}: {FEATURE_KEY} is not a tensor")
        require(features.ndim == 2 and tuple(features.shape) == (expected_length, FEATURE_COUNT),
                f"{uid}: {FEATURE_KEY} has wrong shape")
        require(features.is_floating_point(), f"{uid}: {FEATURE_KEY} is not floating point")
        require(bool(torch.isfinite(features).all().item()), f"{uid}: {FEATURE_KEY} contains nonfinite values")
        return features.detach().to(device="cpu")
    except Exception as exc:
        raise ValueError(f"Invalid Protenix feature for {uid}: {exc}") from exc


def load_train_batch_stats(task):
    uid, expected_length, _ = task
    import torch

    features = load_validated_raw89(task).to(dtype=torch.float64)
    try:
        mean = features.mean(dim=0)
        centered = features - mean
        m2 = (centered * centered).sum(dim=0)
        return uid, expected_length, mean.tolist(), m2.tolist()
    except Exception as exc:
        raise ValueError(f"Unable to compute training moments for {uid}: {exc}") from exc


def validate_heldout_feature(task):
    uid, expected_length, _ = task
    load_validated_raw89(task)
    return uid, expected_length


def validate_heldout_features(ready_uids, uid_lengths, feature_root, workers):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split in ("val", "test"):
            uids = sorted(ready_uids[split])
            residues = 0
            for start in range(0, len(uids), workers * 4):
                tasks = []
                for uid in uids[start:start + workers * 4]:
                    _, protenix_path = feature_paths(feature_root, uid)
                    tasks.append((uid, uid_lengths[uid], protenix_path))
                for _, length in executor.map(validate_heldout_feature, tasks):
                    residues += length
            results[split] = {"unique_ids": len(uids), "residues": residues}
    return results


def fit_train_stats(train_uids, uid_lengths, feature_root, workers):
    require(bool(train_uids), "No ready training endpoints; cannot fit normalization statistics")
    import torch

    torch.set_num_threads(1)
    train_ids = sorted(train_uids)
    residue_total = sum(uid_lengths[uid] for uid in train_uids)
    total = 0
    processed = 0
    mean = [0.0] * FEATURE_COUNT
    m2 = [0.0] * FEATURE_COUNT
    batch_size = workers * 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(train_ids), batch_size):
            tasks = []
            for uid in train_ids[start:start + batch_size]:
                _, protenix_path = feature_paths(feature_root, uid)
                tasks.append((uid, uid_lengths[uid], protenix_path))
            for _, count, batch_mean, batch_m2 in executor.map(
                    load_train_batch_stats, tasks):
                combined = total + count
                if total == 0:
                    mean = batch_mean
                    m2 = batch_m2
                else:
                    delta = [batch_mean[i] - mean[i] for i in range(FEATURE_COUNT)]
                    mean = [mean[i] + delta[i] * count / combined for i in range(FEATURE_COUNT)]
                    m2 = [m2[i] + batch_m2[i] + delta[i] * delta[i] * total * count / combined
                          for i in range(FEATURE_COUNT)]
                total = combined
                processed += 1
                if processed % 1000 == 0 or processed == len(train_ids):
                    print(json.dumps({
                        "stage": "normalization_fit",
                        "train_unique_ids_read": processed,
                        "train_unique_ids_total": len(train_ids),
                        "train_residues_read": total,
                        "train_residues_total": residue_total,
                    }), flush=True)

    require(total == residue_total, "Training residue count mismatch")
    std = [max((value / total) ** 0.5, 1e-6) for value in m2]
    train_ids_text = "".join(uid + "\n" for uid in sorted(train_uids))
    train_uid_checksum = hashlib.sha256(train_ids_text.encode("utf-8")).hexdigest()
    return {
        "mean": mean,
        "std": std,
        "feature_count": FEATURE_COUNT,
        "fit_split": "train",
        "train_unique_ids_sha256": train_uid_checksum,
        "train_uid_checksum": train_uid_checksum,
        "train_unique_count": len(train_uids),
        "train_residue_count": total,
        "counts": {"unique": len(train_uids), "residues": total},
        "provenance": {
            "source": FEATURE_SOURCE,
            "feature_key": FEATURE_KEY,
            "aggregation": "float64 parallel-variance merge over each unique training sequence once",
            "variance": "population; residues weighted equally",
            "std_minimum": 1e-6,
        },
    }


def collect_ready_metadata(path, fields, statuses, uid_lengths):
    ready_pairs = Counter({split: 0 for split in SPLITS})
    excluded_pairs = Counter({split: 0 for split in SPLITS})
    ready_uids = {split: set() for split in SPLITS}
    ready_residues = Counter({split: 0 for split in SPLITS})
    train_uids = set()
    smoke_heaps = {"train": [], "val": []}
    smoke_candidates = Counter({split: 0 for split in SPLITS})

    for ordinal, row in enumerate(pair_rows(path, fields)):
        split = row["split"]
        reasons = pair_reasons(row, statuses)
        if reasons:
            excluded_pairs[split] += 1
            continue
        ready_pairs[split] += 1
        left_uid, right_uid = row["left_seq_id"], row["right_seq_id"]
        ready_uids[split].update((left_uid, right_uid))
        ready_residues[split] += uid_lengths[left_uid] + uid_lengths[right_uid]
        if split == "train":
            train_uids.update((left_uid, right_uid))
        if split in ("train", "val") and int(row["left_length"]) <= 512 and int(row["right_length"]) <= 512:
            smoke_candidates[split] += 1
            limit = 256 if split == "train" else 64
            offer_smoke(smoke_heaps[split], row, ordinal, limit)

    return (ready_pairs, excluded_pairs, ready_uids, ready_residues, train_uids,
            smoke_heaps, smoke_candidates)


def write_outputs(args, fields, manifest_sha256, uid_splits, statuses, heldout_validation,
                  source_pairs, ready_pairs, excluded_pairs, ready_uids, ready_residues,
                  train_uids, smoke_heaps, smoke_candidates, stats):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    excluded_fields = fields + ["left_reason", "right_reason", "reason"]
    with (output / "ready_pair_manifest.csv").open("x", newline="", encoding="utf-8") as ready_file, \
            (output / "excluded_pairs.csv").open("x", newline="", encoding="utf-8") as excluded_file:
        ready_writer = csv.DictWriter(ready_file, fieldnames=fields)
        excluded_writer = csv.DictWriter(excluded_file, fieldnames=excluded_fields)
        ready_writer.writeheader()
        excluded_writer.writeheader()
        for row in pair_rows(args.pair_manifest, fields):
            reasons = pair_reasons(row, statuses)
            if reasons:
                excluded_writer.writerow({
                    **row,
                    "left_reason": reasons.get("left", ""),
                    "right_reason": reasons.get("right", ""),
                    "reason": ";".join(f"{side}:{reason}" for side, reason in reasons.items()),
                })
            else:
                ready_writer.writerow(row)

    with (output / "missing_unique.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("unique_id", "split", "reason", "esmc_path", "protenix_path"))
        for uid in sorted(uid_splits):
            reason = uid_reason(statuses[uid])
            if reason:
                esmc_path, protenix_path = feature_paths(args.feature_root, uid)
                writer.writerow((uid, uid_splits[uid], reason, str(esmc_path), str(protenix_path)))

    for split, limit in (("train", 256), ("val", 64)):
        rows = selected_smoke_rows(smoke_heaps[split])
        with (output / f"smoke_{split}_pair_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:limit])

    train_ids_text = "".join(uid + "\n" for uid in sorted(train_uids))
    (output / "train_unique_ids.txt").write_text(train_ids_text, encoding="utf-8")
    stats["input_manifest_sha256"] = manifest_sha256
    stats["provenance"].update({
        "pair_manifest": str(args.pair_manifest),
        "feature_root": str(args.feature_root),
        "esmc_gate": "file presence and nonzero size only; tensors are not loaded",
        "normalization_fit_ids_file": "train_unique_ids.txt",
    })
    (output / "normalization_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": "complete",
        "input_manifest_sha256": manifest_sha256,
        "source_pairs_by_split": dict(source_pairs),
        "ready_pairs_by_split": dict(ready_pairs),
        "excluded_pairs_by_split": dict(excluded_pairs),
        "source_unique_endpoints_by_split": {
            split: sum(value == split for value in uid_splits.values()) for split in SPLITS
        },
        "ready_unique_endpoints_by_split": {split: len(ready_uids[split]) for split in SPLITS},
        "ready_pair_weighted_residues_by_split": dict(ready_residues),
        "heldout_protenix_validated_unique_endpoints_by_split": {
            split: heldout_validation[split]["unique_ids"] for split in ("val", "test")
        },
        "heldout_protenix_validated_residues_by_split": {
            split: heldout_validation[split]["residues"] for split in ("val", "test")
        },
        "feature_validation_scope": {
            "esmc": "file presence and nonzero size only; tensor content not validated",
            "protenix_train": "content validated and used for train-only normalization",
            "protenix_heldout": "content validated; no contribution to normalization statistics",
        },
        "smoke": {
            "train_eligible_pairs": smoke_candidates["train"],
            "train_selected_pairs": len(smoke_heaps["train"]),
            "train_limit": 256,
            "val_eligible_pairs": smoke_candidates["val"],
            "val_selected_pairs": len(smoke_heaps["val"]),
            "val_limit": 64,
            "test_pairs_excluded": ready_pairs["test"],
        },
        "normalization_train_unique_count": stats["train_unique_count"],
        "normalization_train_residue_count": stats["train_residue_count"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("/data1/wholeset/RCSBprot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    require(args.workers > 0, "--workers must be positive")
    require(args.pair_manifest.is_file(), f"Pair manifest not found: {args.pair_manifest}")
    require(not args.output.exists(), f"Output path already exists: {args.output}")

    fields = read_header(args.pair_manifest)
    manifest_sha256 = file_sha256(args.pair_manifest)
    uid_splits, uid_lengths, source_pairs = index_manifest(args.pair_manifest, fields)
    statuses = inspect_features(uid_splits, args.feature_root, args.workers)
    (ready_pairs, excluded_pairs, ready_uids, ready_residues, train_uids,
     smoke_heaps, smoke_candidates) = collect_ready_metadata(
        args.pair_manifest, fields, statuses, uid_lengths)
    stats = fit_train_stats(train_uids, uid_lengths, args.feature_root, args.workers)
    heldout_validation = validate_heldout_features(
        ready_uids, uid_lengths, args.feature_root, args.workers)
    write_outputs(args, fields, manifest_sha256, uid_splits, statuses, heldout_validation,
                  source_pairs, ready_pairs, excluded_pairs, ready_uids, ready_residues,
                  train_uids, smoke_heaps, smoke_candidates, stats)


if __name__ == "__main__":
    main()
