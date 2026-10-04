#!/bin/bash
#SBATCH --job-name=ppc_followup
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
SNAPSHOT="$PWD/experiments/pairwise_followup_20261002"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
GPU_UUID=$("$PY" -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith(("GPU-", "MIG-")) else "GPU-" + u)')
nvidia-smi -i "$GPU_UUID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw --format=csv --loop=5 > "logs/ppc_followup_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$MONITOR_PID"
case "${MODE:-train}" in
  smoke)
    "$PY" "$SNAPSHOT/test_followup_adapter.py"
    for name in S5_site_only S6_no_ratio S7_no_smoothing S8_endpoints S9_null_pool; do
      "$PY" "$SNAPSHOT/train_esm_pair.py" --config "configs/experiments/20261002_pairwise_followup/${name}_smoke.yaml"
    done
    for name in S0_independent S1_opm; do
      "$PY" "$SNAPSHOT/diagnose_partner_use.py" --checkpoint "runs/pairwise_scratch_${name}_seed49_20261002_hpc2_smoke/best.pt" --output "runs/pairwise_followup_partner_diagnostic_smoke_${name}_${SLURM_JOB_ID}" --seeds 11
    done
    "$PY" "$SNAPSHOT/diagnose_partner_use.py" --checkpoint runs/pairwise_followup_S8_endpoints_seed49_20261002_hpc2_smoke/best.pt --output "runs/pairwise_followup_partner_diagnostic_smoke_S8_${SLURM_JOB_ID}" --seeds 11
    ;;
  train)
    "$PY" "$SNAPSHOT/train_esm_pair.py" --config "${CONFIG:?CONFIG required}"
    ;;
  diagnostic)
    for name in S0_independent S1_opm; do
      "$PY" "$SNAPSHOT/diagnose_partner_use.py" --checkpoint "runs/pairwise_scratch_${name}_seed49_20261002_hpc2/best.pt" --output "runs/pairwise_followup_partner_diagnostic_${name}_${SLURM_JOB_ID}"
    done
    ;;
  *) exit 2 ;;
esac
