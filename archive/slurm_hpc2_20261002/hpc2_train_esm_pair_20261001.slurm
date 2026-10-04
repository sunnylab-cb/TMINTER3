#!/bin/bash
#SBATCH --job-name=ppc_pair_m18
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PPC}"
PYTHON_BIN="${PYTHON_BIN:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CONFIG="${CONFIG:?CONFIG is required}"
cd "${PROJECT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} config=${CONFIG} python=${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"${PYTHON_BIN}" scripts/training/train_esm_pair.py --config "${CONFIG}"
