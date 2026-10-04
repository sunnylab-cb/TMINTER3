#!/bin/bash
#SBATCH --job-name=ppc_pair_labels
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
PYTHON_BIN="${PYTHON_BIN:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
cd "${PROJECT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
"${PYTHON_BIN}" scripts/preparations/generate_pairwise_contact_labels_from_features.py \
  --chain-manifest features/contact_labels_5p5A/splits_pairwise_global_si30_deeptminter_tmk_no_len_limit_20260708/pypropel_chain_filtered_no_len_limit/chain_manifest.csv \
  --output-root features/contact_labels_pairwise_5p5A_20261001 \
  --cutoff 5.5 \
  --workers 24 \
  --smoke-train-pairs 128 \
  --smoke-eval-pairs 32
