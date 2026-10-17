#!/bin/bash
#SBATCH --job-name=rcsb_esmc
#SBATCH --partition=acd_u
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_esmc/logs/esmc_%A_%a.log
set -euo pipefail
ROOT=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1
PY=/data/user/shuang886/.conda/envs/tminter3_esmc/bin/python
test -x "$PY"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset PYTHONPATH
MODE="${1:-full}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$INDEX" =~ ^[0-3]$ ]] || exit 64
if [[ "$MODE" == smoke ]]; then
    test "$INDEX" == 0
    MANIFEST="$ROOT/_esmc/manifests/ready_v1/smoke.tsv"
    OUT="$ROOT/_esmc/smoke/features"
elif [[ "$MODE" == full ]]; then
    "$PY" - "$ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
receipt = json.loads((root / "_esmc/SMOKE_APPROVED.json").read_text())
for name in ("_esmc/manifests/ready_v1/summary.json",
             "_esmc/manifests/ready_v1/smoke.tsv",
             "_esmc/scripts/extract_esmc_unique.py",
             "_esmc/scripts/hpc3_esmc_ready.slurm"):
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if receipt["file_sha256"][name] != actual:
        raise SystemExit("Smoke receipt mismatch: " + name)
summary = json.loads((root / "_esmc/manifests/ready_v1/summary.json").read_text())
if [s["file"] for s in summary["shards"]] != ["shard_%02d.tsv" % i for i in range(4)]:
    raise SystemExit("Expected exactly four ordered shards")
for shard in summary["shards"]:
    path = root / "_esmc/manifests/ready_v1" / shard["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
        raise SystemExit("Shard checksum mismatch: " + str(path))
environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
if receipt["environment_sha256"] != hashlib.sha256(environment).hexdigest():
    raise SystemExit("Environment changed since smoke validation")
if receipt["passed"] is not True:
    raise SystemExit("Smoke not approved")
PY
    printf -v SHARD 'shard_%02d.tsv' "$INDEX"
    MANIFEST="$ROOT/_esmc/manifests/ready_v1/$SHARD"
    OUT="$ROOT/TMINTER3-ESMC/features/unique"
else
    exit 64
fi
exec 9>"$ROOT/_esmc/logs/${MODE}_${INDEX}.lock"
flock -n 9
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$PY" -u "$ROOT/_esmc/scripts/extract_esmc_unique.py" \
    --manifest "$MANIFEST" --output-root "$OUT" \
    --cache-dir "$ROOT/_esmc/hf_cache" \
    --repo-id biohub/esmc-600m-2024-12 \
    --revision e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c \
    --device cuda:0 --dtype bfloat16 --output-dtype float32 \
    --layers 12,24,36 --hidden-normalization final_layernorm \
    --max-tokens-per-batch 4096 --max-batch-size 64 \
    --local-files-only --no-use-flash-attn --tag "hpc3_${MODE}_${INDEX}"
