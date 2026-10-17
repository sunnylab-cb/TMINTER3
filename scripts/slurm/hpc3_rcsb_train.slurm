#!/bin/bash
#SBATCH --job-name=rcsb_ppis49
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_hpc3_training/20261017_v1/logs/train_%j.log
set -euo pipefail
DEPLOY=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1/_hpc3_training/20261017_v1
PY=/data/user/shuang886/.conda/envs/tminter3_esmc/bin/python
MODE="${1:-smoke}"
[[ "$MODE" == smoke || "$MODE" == full ]] || exit 64
export PYTHONPATH="$DEPLOY/code/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec 9>"$DEPLOY/logs/$MODE.lock"
flock -n 9
"$PY" -u "$DEPLOY/check_hpc3_training.py" "$DEPLOY" "$MODE"
cd "$DEPLOY/code"
OUT="$DEPLOY/runs/$MODE"
if (( ${SLURM_RESTART_COUNT:-0} > 32 )); then exit 1; fi
if [[ -f "$OUT/final_metrics.json" ]]; then exit 0; fi
args=(--config "$DEPLOY/configs/$MODE.yaml")
if [[ -f "$OUT/progress.pt" ]]; then args+=(--resume); fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf 'Starting mode=%s job=%s restart=%s\n' "$MODE" "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}"
set +e
srun --ntasks=1 "$PY" -u scripts/training/train_esm_pair.py "${args[@]}"
rc=$?
set -e
if [[ "$rc" == 75 ]]; then
    test -s "$OUT/progress.pt"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rc"
