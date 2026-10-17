#!/bin/bash
#SBATCH --job-name=rcsb_ppis_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/data1/wholeset/RCSBprot/training/ppis_ge50_v1/logs/train_%j.log
set -euo pipefail
ROOT=/home/husrcf/Code/TMINTER3
CONFIG=${1:?Provide the reviewed training configuration}
PY=/home/husrcf/anaconda3/envs/AIAA/bin/python
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
if [[ ${PREPARE_RCSB_ASSETS:-0} == 1 ]]; then
    # Star requires a GPU allocation even for the CPU-only preparation phase.
    bash "$ROOT/scripts/slurm/star_rcsb_prepare.slurm"
fi
"$PY" -u -c 'import torch; print(torch.__version__, flush=True); assert torch.cuda.is_available(), "Allocated CUDA GPU is unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
exec "$PY" -u scripts/training/train_esm_pair.py --config "$CONFIG"
