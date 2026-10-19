#!/bin/bash
#SBATCH --job-name=ppis_input_gap_audit
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
ROOT=/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1
DEPLOY="$ROOT/_hpc3_training/20261017_v1"
AUDIT="$DEPLOY/diagnostics/input_gap_audit_20261019"
PY=/data/user/shuang886/.conda/envs/tminter3_esmc/bin/python

cd "$DEPLOY/code"
"$PY" -u "$AUDIT/code/audit_pair_input_contract.py" \
    --frozen-code "$DEPLOY/code" \
    --checkpoint "$DEPLOY/diagnostics/frozen_20261018/rcsb_epoch3.pt" \
    --device cuda:0 --output "$AUDIT/input_contract.json"
"$PY" -u "$AUDIT/code/audit_observed_sequence_gaps.py" \
    --manifest "$ROOT/training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv" \
    --source-mapping /data/user/shuang886/tminter3/artifacts/rcsb_sequences_full_success_v1/chain_to_unique.tsv \
    --expected-mapping-sha256 acc34bac9f7abf68f2423cb9b529ef118c836a68975d3c149a985e4e6a955909 \
    --workers 4 --output "$AUDIT/validation_gaps"
