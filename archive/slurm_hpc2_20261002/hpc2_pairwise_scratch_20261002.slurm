#!/bin/bash
#SBATCH --job-name=ppc_scratch
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
cd /hpc2hdd/home/shuang886/jhspoolers/PPC
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON_BIN=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
CONFIG="${CONFIG:?CONFIG must identify the scratch variant}"
GPU_UUID=$("$PYTHON_BIN" -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith(("GPU-", "MIG-")) else "GPU-" + u)')
nvidia-smi -i "$GPU_UUID" --query-gpu=uuid,name --format=csv
nvidia-smi -i "$GPU_UUID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw --format=csv --loop=5 > "logs/ppc_scratch_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$MONITOR_PID"
"$PYTHON_BIN" scripts/training/train_esm_pair.py --config "$CONFIG"
