#!/bin/bash
#SBATCH --job-name=tminter3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${PROJECT_ROOT:?Set PROJECT_ROOT to the TMINTER3 checkout}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN:?Set PYTHON_BIN to an existing compatible environment}" \
  scripts/training/train_esm_pair.py --config "${CONFIG:?Set CONFIG to a YAML path}"
