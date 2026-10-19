#!/bin/bash
#SBATCH --job-name=ppis_objective_audit
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1
DEPLOY="$ROOT/_hpc3_training/20261017_v1"
AUDIT="$DEPLOY/diagnostics/objective_audit_v2_20261019"
PY=/data/user/shuang886/.conda/envs/tminter3_esmc/bin/python
MANIFEST="$ROOT/training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv"

for SPLIT in train val; do
    "$PY" -u "$AUDIT/code/audit_pair_objective.py" \
        --manifest "$MANIFEST" \
        --trainer "$DEPLOY/code/scripts/training/train_esm_pair.py" \
        --config full "$DEPLOY/runs/full/config.json" \
        --config batch4m "$DEPLOY/runs/large_batch4m/config.json" \
        --config batch8m "$DEPLOY/runs/large_batch8m/config.json" \
        --split "$SPLIT" --output "$AUDIT"/"$SPLIT"_objective.json
    EXTRA=()
    if [[ "$SPLIT" == val ]]; then EXTRA+=(--context-detail); fi
    "$PY" -u "$AUDIT/code/audit_repeated_pair_labels.py" \
        --manifest "$MANIFEST" --label-root "$ROOT/training/ppis_ge50_v1/pairs" \
        --split "$SPLIT" --output "$AUDIT"/"$SPLIT"_label_diversity "${EXTRA[@]}"
done
