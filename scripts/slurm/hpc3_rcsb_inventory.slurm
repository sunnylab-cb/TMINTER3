#!/bin/bash
#SBATCH --job-name=tminter3_inventory
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/shuang886/tminter3/logs/%x_%j.out
#SBATCH --error=/data/user/shuang886/tminter3/logs/%x_%j.err
set -euo pipefail
cd /data/user/shuang886/tminter3
# ACD requires a GPU allocation even for this short CPU inventory.
/data/user/shuang886/.conda/envs/fold/bin/python -u -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)'
/bin/python -u scripts/preparations/inventory_rcsb_raw.py \
  --raw-root /data/user/shuang886/Folding/Dataset/raw/pdb_mmcif \
  --output "artifacts/rcsb_inventory_${SLURM_JOB_ID}"
