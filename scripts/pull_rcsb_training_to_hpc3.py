"""HPC3-side, read-only-from-star, restartable training-asset transfer (Python 3.6+)."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


SOURCE = "star:/data1/wholeset/RCSBprot/"
STAGES = [
    "sequences",
    "training/ppis_ge50_v1/split",
    "training/ppis_ge50_v1/pairs",
    "training/ppis_ge50_v1/ready_v1",
    "TMINTER3-ESMC/manifests",
    "training/ppis_ge50_v1/launch_records",
    "protenix/features/complete_raw_v1",
    "TMINTER3-ESMC/features",
]
RETRY_CODES = {10, 12, 30, 35, 255}
EXPECTED_PAIR_COUNTS = {"train": 1150993, "val": 123863, "test": 131829}
EXPECTED_SOURCE_DIGEST = "cfecd7c24e0e5a6b93ebd4576ba5efd58a812960267e7f72016ac47919ca7834"


def write_json(path, obj):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


def command(root, stage, bandwidth):
    return ["rsync", "-rtL", "--partial", "--partial-dir=.rsync-partial",
            "--timeout=180", "--info=progress2", "--stats",
            "--bwlimit={}".format(bandwidth), "-e",
            "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "
            "-o ServerAliveInterval=30 -o ServerAliveCountMax=3",
            SOURCE + stage + "/", str(root / stage) + "/"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_snapshot(root):
    ready = root / "training/ppis_ge50_v1/ready_v1"
    summary = json.loads((ready / "summary.json").read_text())
    if summary["ready_pairs_by_split"] != EXPECTED_PAIR_COUNTS:
        raise ValueError("Ready counts differ from frozen v1 snapshot")
    parent = root / "training/ppis_ge50_v1/pairs/pair_manifest.csv"
    if summary["input_manifest_sha256"] != EXPECTED_SOURCE_DIGEST or digest(parent) != EXPECTED_SOURCE_DIGEST:
        raise ValueError("Parent manifest differs from frozen v1 SHA-256")
    return {"parent_sha256": EXPECTED_SOURCE_DIGEST,
            "ready_sha256": digest(ready / "ready_pair_manifest.csv"),
            "ready_pairs_by_split": EXPECTED_PAIR_COUNTS}


def audit_ready(root, require_esmc=True):
    snapshot = check_snapshot(root)
    receipt = root / "_transfer/metadata_snapshot.json"
    if receipt.exists() and json.loads(receipt.read_text()) != snapshot:
        raise ValueError("Metadata changed during feature transfer")
    ready = root / "training/ppis_ge50_v1/ready_v1"
    counts, uid_splits, label_paths = Counter(), {}, set()
    with (ready / "ready_pair_manifest.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            counts[split] += 1
            label_path = Path(row["label_path"])
            if label_path.is_absolute() or ".." in label_path.parts:
                raise ValueError("Nonportable label path: {}".format(label_path))
            label_paths.add(label_path)
            for side in ("left", "right"):
                uid = row[side + "_seq_id"]
                if len(uid) != 69 or not uid.startswith("useq_") or any(c not in "0123456789abcdef" for c in uid[5:]):
                    raise ValueError("Invalid sequence ID: {}".format(uid))
                previous = uid_splits.setdefault(uid, split)
                if previous != split:
                    raise ValueError("Cross-split exact sequence: {}".format(uid))
    if dict(counts) != EXPECTED_PAIR_COUNTS:
        raise ValueError("Transferred pair counts do not match summary")
    for path in label_paths:
        full = root / "training/ppis_ge50_v1/pairs" / path
        if not full.is_file() or not full.stat().st_size:
            raise ValueError("Missing label shard: {}".format(full))
    missing = []
    directories = ["protenix/features/complete_raw_v1/pt"]
    if require_esmc:
        directories.append("TMINTER3-ESMC/features/unique")
    for uid in uid_splits:
        for directory in directories:
            path = root / directory / uid[5:7] / (uid + ".pt")
            if not path.is_file() or not path.stat().st_size:
                missing.append(str(path))
    result = {"ready_pairs_by_split": dict(counts), "unique_endpoints": len(uid_splits),
              "label_shards": len(label_paths), "missing_features": len(missing),
              "missing_examples": missing[:20], "tensor_content_validation": False,
              "esmc_checked": require_esmc, "esmc_deferred": not require_esmc}
    audit_name = "coverage_audit.json" if require_esmc else "coverage_audit_structure.json"
    write_json(root / "_transfer" / audit_name, result)
    if missing:
        raise ValueError("{} missing/empty feature files".format(len(missing)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bwlimit-kib", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--feature-scope", choices=("all", "structure-only"), default="all")
    args = parser.parse_args()
    if args.bwlimit_kib < 0 or args.attempts < 1:
        parser.error("Bandwidth must be nonnegative (0 = unlimited); attempts must be positive")
    root = args.root.resolve()
    control = root / "_transfer"
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "pull.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A rerun rechecks all stages with rsync rather than trusting stale markers.
    require_esmc = args.feature_scope == "all"
    complete = control / ("COMPLETED.json" if require_esmc else "STRUCTURE_COMPLETED.json")
    if complete.exists():
        os.replace(str(complete), str(control / ("previous_" + complete.name)))
    started = time.time()
    state = {"pid": os.getpid(), "source": SOURCE, "destination": str(root),
             "started_epoch": started, "bwlimit_kib": args.bwlimit_kib,
             "completed_stages": [], "status": "running", "feature_scope": args.feature_scope}
    def transfer_stage(index, stage):
        (root / stage).mkdir(parents=True, exist_ok=True)
        log_path = control / "{:02d}_{}.log".format(index + 1, stage.replace("/", "_"))
        stage_status = log_path.with_suffix(".status.json")
        stage_state = {"stage": stage, "log": str(log_path), "status": "running"}
        try:
            for attempt in range(1, args.attempts + 1):
                stage_state.update(attempt=attempt, updated_epoch=time.time())
                write_json(stage_status, stage_state)
                print(json.dumps(stage_state), flush=True)
                with log_path.open("a") as log:
                    log.write("\nATTEMPT {} epoch {}\n".format(attempt, time.time()))
                    log.flush()
                    rc = subprocess.call(command(root, stage, args.bwlimit_kib), stdout=log, stderr=subprocess.STDOUT)
                if rc == 0:
                    stage_state.update(status="complete", completed_epoch=time.time())
                    write_json(stage_status, stage_state)
                    return stage
                if rc not in RETRY_CODES or attempt == args.attempts:
                    raise RuntimeError("rsync stage={} exit={}; see {}".format(stage, rc, log_path))
                time.sleep(min(30 * attempt, 300))
        except Exception as exc:
            stage_state.update(status="failed", error=str(exc), updated_epoch=time.time())
            write_json(stage_status, stage_state)
            raise

    try:
        for index, stage in enumerate(STAGES[:-2]):
            state.update(stage=stage, updated_epoch=time.time())
            write_json(control / "status.json", state)
            state["completed_stages"].append(transfer_stage(index, stage))
        write_json(control / "metadata_snapshot.json", check_snapshot(root))
        # Structure-only intentionally excludes the ESMC bank and has its own receipt.
        indices = list(range(len(STAGES) - 2, len(STAGES))) if require_esmc else [len(STAGES) - 2]
        state.update(stage="parallel_feature_banks" if require_esmc else "structure_features",
                     active_stages=[STAGES[i] for i in indices], updated_epoch=time.time())
        write_json(control / "status.json", state)
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(transfer_stage, i, STAGES[i]): STAGES[i]
                       for i in indices}
            for future in as_completed(futures):
                stage = futures[future]
                state["active_stages"].remove(stage)
                try:
                    state["completed_stages"].append(future.result())
                except Exception as exc:
                    failures.append(str(exc))
                    state.update(status="running_with_failures", failures=failures)
                state["updated_epoch"] = time.time()
                write_json(control / "status.json", state)
        if failures:
            raise RuntimeError("; ".join(failures))
        state.update(status="auditing", updated_epoch=time.time())
        write_json(control / "status.json", state)
        state["audit"] = audit_ready(root, require_esmc=require_esmc)
        state.update(status="complete" if require_esmc else "structure_complete_esmc_deferred", completed_epoch=time.time())
        state["elapsed_seconds"] = time.time() - started
        write_json(complete, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc), updated_epoch=time.time())
        write_json(control / "status.json", state)
        raise
    write_json(control / "status.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
