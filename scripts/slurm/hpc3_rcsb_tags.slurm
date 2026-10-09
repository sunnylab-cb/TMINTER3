#!/bin/bash
#SBATCH --job-name=rcsb_tags
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/shuang886/tminter3/logs/%x_%A_%a.out
#SBATCH --error=/data/user/shuang886/tminter3/logs/%x_%A_%a.err
set -euo pipefail
cd /data/user/shuang886/tminter3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/user/shuang886/.conda/envs/fold/bin/python
if [[ "${MODE:-full}" == smoke ]]; then
  "$PY" -u scripts/preparations/build_rcsb_pair_tags.py --inventory artifacts/rcsb_inventory_704380/files.tsv \
    --raw-root /data/user/shuang886/Folding/Dataset/raw/pdb_mmcif \
    --output artifacts/rcsb_tags_smoke_v1 --workers 4 --shards 256 --shard 0 --limit 32
  "$PY" -u scripts/preparations/rcsb_tag_manifest.py --tags artifacts/rcsb_tags_smoke_v1 \
    --output artifacts/rcsb_sequences_smoke_v1
else
  "$PY" -u scripts/preparations/build_rcsb_pair_tags.py --inventory artifacts/rcsb_inventory_704380/files.tsv \
    --raw-root /data/user/shuang886/Folding/Dataset/raw/pdb_mmcif \
    --output artifacts/rcsb_tags_v1 --workers 12 --shards "${SHARDS:-32}" --shard "${SLURM_ARRAY_TASK_ID:?}"
fi
