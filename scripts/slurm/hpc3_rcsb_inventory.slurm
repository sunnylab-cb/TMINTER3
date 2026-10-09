#!/bin/bash
#SBATCH --job-name=tminter3_inventory
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/shuang886/tminter3/logs/%x_%j.out
#SBATCH --error=/data/user/shuang886/tminter3/logs/%x_%j.err
set -euo pipefail
cd /data/user/shuang886/tminter3
/bin/python -u scripts/preparations/inventory_rcsb_raw.py \
  --raw-root /data/user/shuang886/Folding/Dataset/raw/pdb_mmcif \
  --output "artifacts/rcsb_inventory_${SLURM_JOB_ID}"
