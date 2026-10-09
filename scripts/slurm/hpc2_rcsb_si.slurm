#!/bin/bash
#SBATCH --job-name=rcsb_si
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%A_%a.err
set -euo pipefail
cd /hpc2hdd/home/shuang886/jhspoolers/PPC/experiments/tminter3_rcsb_20261009
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u scripts/preparations/rcsb_pairwise_si.py \
 --manifest "${MANIFEST:?}" --output "${OUTPUT:?}/shard_${SLURM_ARRAY_TASK_ID:-0}" --workers 120 \
 --shards "${SHARDS:-1}" --shard "${SLURM_ARRAY_TASK_ID:-0}"
