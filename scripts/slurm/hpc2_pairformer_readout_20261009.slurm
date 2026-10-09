#!/bin/bash
#SBATCH --job-name=pf_readout
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PPC/logs/%x_%j.err
set -euo pipefail
cd /hpc2hdd/home/shuang886/jhspoolers/PPC
CODE="$PWD/experiments/tminter3_pairformer_readout_20261009"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
GPU_UUID=$("$PY" -u -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith(("GPU-", "MIG-")) else "GPU-" + u)')
nvidia-smi -i "$GPU_UUID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw --format=csv --loop=5 > "logs/pairformer_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$MONITOR_PID"
if [[ "${MODE:-train}" == "smoke" ]]; then
  "$PY" -u "$CODE/tests/test_pairformer.py"
  for name in A_selective B_outer_triangle C_outer_no_triangle; do
    "$PY" -u "$CODE/scripts/training/train_esm_pair.py" --config "$CODE/configs/experiments/20261009_pairformer_readout/${name}_smoke.yaml"
    "$PY" -u "$CODE/scripts/training/profile_pairformer.py" --config "$CODE/configs/experiments/20261009_pairformer_readout/${name}.yaml"
  done
else
  "$PY" -u "$CODE/scripts/training/train_esm_pair.py" --config "$CODE/configs/experiments/20261009_pairformer_readout/${VARIANT:?set VARIANT}.yaml"
fi
