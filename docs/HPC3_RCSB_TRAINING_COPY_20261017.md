# HPC3 RCSB training-data replica

## Current policy (user revision)

Prioritize structure features. Resume metadata, then copy only the roughly
119 GB PS bank without a bandwidth cap. Defer the roughly 1.04 TB ESMC bank;
HPC3 re-extraction is now the selected route, not yet started. Do not transfer
the ESMC feature bank from star. Use
`--feature-scope structure-only`. Success produces `STRUCTURE_COMPLETED.json`
with `structure_complete_esmc_deferred`, not `COMPLETED.json`. The dataset is
not ready for model training until ESMC coverage/content is also validated.
The all-bank parallel plan below is retained as the optional later mode.

Destination:
`/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1`

Source (read only): `star:/data1/wholeset/RCSBprot/`.
The controller and rsync client run on HPC3, not star. Star serves read-only
rsync requests; no installation, source-file changes, source deletions or
training-job changes are part of this transfer.

Copy the following, preserving the relative layout:

1. `sequences/`: identity and chain/sequence mappings.
2. `training/ppis_ge50_v1/{split,pairs,ready_v1}/`: split, labels, ready/excluded
   pair manifests and frozen train-only normalization.
3. `TMINTER3-ESMC/manifests/`: feature extraction provenance.
4. `training/ppis_ge50_v1/launch_records/`: frozen source/config provenance.
5. `protenix/features/complete_raw_v1/`: structure descriptors, about 119 GB.
6. `TMINTER3-ESMC/features/`: ESMC bank, about 1.04 TB.

Decimal units; observed feature-bank disk use was 1,158,262,439,936 bytes in
total. These banks include sequences outside the current feature-ready pair
subset. Plan at least 1.3 TB plus operational headroom. User explicitly
confirmed storage is not a constraint and requested unlimited transfer speed.

Do not duplicate raw CIF: HPC3 already has
`/data/user/shuang886/Folding/Dataset/raw/pdb_mmcif/`.
Do not copy live training outputs, checkpoints, repeated folding runs or
transport archives. This is a train-ready feature replica, not a mirror of
every RCSB project artifact. Raw predicted CIF can be added separately if
re-extraction is required; current training consumes the descriptors.

The original source configs deliberately retain their star paths for
provenance. Before training on HPC3 create separate local configs replacing
the source data-root prefix with the destination above. Do not mutate split,
labels, normalization or training contract. No HPC3 training is launched by
this transfer script.

Run on HPC3 with system Python 3.6+ and rsync 3.1+ (HPC3 verified: 3.1.3):

```bash
python3 -u /data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_transfer/pull_rcsb_training_to_hpc3.py \
  --root /data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1 \
  --bwlimit-kib 0 --feature-scope structure-only
```

Metadata first, then two concurrent feature-bank streams with no bandwidth cap,
no compression, private partial directories,
bounded connection-error retries, an exclusive lock preventing duplicate
controllers. Rerun the same command to resume. No `--delete`, `--inplace`, or
source-file removal. Space/quota and missing-source errors fail explicitly.

Progress: `_transfer/status.json`, per-stage rsync logs and `controller.log`.
Before bulk features, require frozen v1 pair counts and source pair-manifest
SHA-256 `cfecd7c24e0e5a6b93ebd4576ba5efd58a812960267e7f72016ac47919ca7834`;
record ready-manifest SHA-256 and recheck it at completion.
After all stages, audit ready pair counts, exact-sequence split separation,
relative label-shard paths and nonempty required feature files. Only then
write `_transfer/COMPLETED.json`. This is a coverage/transfer check, not full
tensor-content validation; the training adapter retains its payload checks.
Rsync validates transferred file reconstruction with its transfer checksum;
this is not an independent whole-bank SHA-256 audit of existing target files.

Frozen current ready counts: train 1,150,993, val 123,863, test 131,829 pairs.
Test labels may be copied but are not evaluated, used for calibration or
used for fitting normalization. This operation does not alter star job 2349
or array 2355.

## Initial launch receipt (superseded by structure-first mode)

Launched on HPC3 in tmux session `rcsb_ppis_pull_20261017`; initial controller
PID 3485284. `status.json` confirms `bwlimit_kib: 0`. The first sequence stage
has transferred actual data (over 203 MB observed); end-to-end rsync progress
options are working. Initial metadata-file throughput was 6-9 MB/s, not a
reliable feature-bank ETA. After metadata, ESMC and PS run concurrently.
No artificial bandwidth cap remains. Script tests: 7 passed; HPC3 Python3.6
syntax checked. No Python/Torch packages were installed on either machine.

The initial controller was interrupted on HPC3 while receiving ready metadata
to prevent ESMC starting under the old policy. Completed sequence/split/pair
copies and partial files remain reusable. Structure-first resume rechecks them
without deleting destination files. Tests for this mode explicitly assert no
ESMC feature-bank command and no full-completion receipt (8 pull tests pass).
