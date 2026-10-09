"""Read-only raw-mmCIF census. Outputs are exclusively created outside raw."""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw, output = args.raw_root.resolve(), args.output.resolve()
    if not raw.is_dir():
        raise ValueError("raw-root must exist")
    if raw == output or raw in output.parents:
        raise ValueError("Output must not be inside raw-root")
    output.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    ids = collections.Counter()
    sizes = 0
    samples = []
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    def walk_error(error):
        raise error
    with (output / "files.tsv").open("x") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relative_path", "entry_filename", "bytes", "mtime_ns"])
        for parent, dirs, files in os.walk(str(raw), onerror=walk_error, followlinks=False):
            linked = [name for name in dirs if (Path(parent) / name).is_symlink()]
            counts["symlink_directories_skipped"] += len(linked)
            dirs[:] = [name for name in dirs if name not in linked]
            dirs.sort()
            for name in sorted(files):
                path = Path(parent) / name
                if path.is_symlink():
                    counts["symlink_skipped"] += 1
                    continue
                if not (name.endswith(".cif.gz") or name.endswith(".cif")):
                    counts["non_cif"] += 1
                    continue
                stat = path.stat()
                entry = name[:-7].lower() if name.endswith(".cif.gz") else name[:-4].lower()
                writer.writerow([str(path.relative_to(raw)), entry, stat.st_size, stat.st_mtime_ns])
                ids[entry] += 1
                sizes += stat.st_size
                counts["cif_files"] += 1
                if stat.st_size == 0:
                    counts["zero_bytes"] += 1
                if len(samples) < 16:
                    samples.append(path)
                if counts["cif_files"] % 25000 == 0:
                    print(json.dumps(dict(counts)), flush=True)
    checks = []
    for path in samples:
        try:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(str(path), "rb") as handle:
                prefix = handle.read(65536)
            checks.append({"path": str(path.relative_to(raw)), "prefix_has_data_block": b"data_" in prefix,
                           "prefix_sha256": hashlib.sha256(prefix).hexdigest()})
        except Exception as error:
            checks.append({"path": str(path.relative_to(raw)), "error": str(error)})
    with (output / "summary.json").open("x") as handle:
        json.dump({"raw_root": str(raw), "started_utc": start,
                   "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "counts": dict(counts), "bytes": sizes, "unique_filename_ids": len(ids),
                   "duplicate_filename_ids": {k: v for k, v in ids.items() if v > 1},
                   "sample_prefix_checks": checks,
                   "scope": "File inventory only; not full gzip integrity, biological assembly or trainability validation"},
                  handle, indent=2)
    print("Inventory complete: " + str(output), flush=True)


if __name__ == "__main__":
    main()
