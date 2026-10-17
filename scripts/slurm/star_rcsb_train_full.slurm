#!/bin/bash
#SBATCH --job-name=rcsb_ppis_full49
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data1/wholeset/RCSBprot/training/ppis_ge50_v1/logs/full_%j.log
set -euo pipefail
ROOT=/home/husrcf/Code/TMINTER3
CONFIG="$ROOT/configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml"
OUT=/data1/wholeset/RCSBprot/training/ppis_ge50_v1/runs/rcsb_s5_noESM2contact_full_seed49_v1
PY=/home/husrcf/anaconda3/envs/AIAA/bin/python
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
if (( ${SLURM_RESTART_COUNT:-0} > 32 )); then
    printf 'Restart limit exceeded; inspect checkpoints before continuing\n' >&2
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
printf 'Starting job=%s restart=%s config=%s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$CONFIG"
set +e
srun --ntasks=1 "$PY" -u scripts/training/train_esm_pair.py "${args[@]}"
rc=$?
set -e
if [[ "$rc" == 75 ]]; then
    test -s "$OUT/progress.pt"
    printf 'Graceful checkpoint stop; requeueing job %s\n' "$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rc"
