#!/bin/bash
#SBATCH --job-name=rcsb_model_matrix
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data1/wholeset/RCSBprot/training/ppis_ge50_v1/logs/matrix_%A_%a.log
set -euo pipefail
# ROOT is a dedicated, frozen source snapshot, never the live baseline tree.
ROOT="${1:?Pass the absolute frozen code snapshot directory}"
PY=/home/husrcf/anaconda3/envs/AIAA/bin/python
CONFIG_DIR="$ROOT/configs/experiments/20261017_rcsb_matrix"
META=$("$PY" - "$CONFIG_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import yaml
root = Path(sys.argv[1])
row = json.loads((root / "manifest.json").read_text())["variants"][int(sys.argv[2])]
path = root / row["config"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
config = yaml.safe_load(path.read_text())
assert config["training"]["output_dir"] == row["output_dir"]
assert not config["training"]["evaluate_test"]
print(path)
print(row["output_dir"])
PY
)
mapfile -t fields <<< "$META"
CONFIG="${fields[0]}"
OUT="${fields[1]}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
if (( ${SLURM_RESTART_COUNT:-0} > 32 )); then
    printf 'Restart limit exceeded; inspect checkpoints\n' >&2
    exit 1
fi
if [[ -f "$OUT/final_metrics.json" ]]; then
    printf 'Training already completed: %s\n' "$OUT"
    exit 0
fi
args=(--config "$CONFIG")
if [[ -f "$OUT/progress.pt" ]]; then
    args+=(--resume)
fi
printf 'Starting job=%s array=%s task=%s restart=%s config=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "${SLURM_RESTART_COUNT:-0}" "$CONFIG"
set +e
srun --ntasks=1 "$PY" -u scripts/training/train_esm_pair.py "${args[@]}"
rc=$?
set -e
if [[ "$rc" == 75 ]]; then
    test -s "$OUT/progress.pt"
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$rc"
