#!/bin/bash
#SBATCH --job-name=rcsb_ppis_prepare
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data1/wholeset/RCSBprot/training/ppis_ge50_v1/logs/prepare_%j.log
set -euo pipefail
ROOT=/data1/wholeset/RCSBprot/training/ppis_ge50_v1
CODE=/home/husrcf/Code/TMINTER3
SOURCE=/data/user/shuang886/tminter3/artifacts/ppis_only_ge50_training_v1
PY=/home/husrcf/anaconda3/envs/AIAA/bin/python
SOURCE_JOB=${SOURCE_JOB:?Set the HPC3 materialization job ID}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RSYNC_RSH='ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3'

deadline=$((SECONDS + 3600))
until ssh -o BatchMode=yes -o ConnectTimeout=20 hpc3 "test -f '$SOURCE/pairs/COMPLETED'"; do
    state=$(ssh -o BatchMode=yes -o ConnectTimeout=20 hpc3 \
        "sacct -j '$SOURCE_JOB' -X --format=State -n -P" | head -n 1)
    printf 'Waiting for HPC3 job %s, state=%s\n' "$SOURCE_JOB" "$state"
    case "$state" in
        FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|COMPLETED*)
            printf 'Source job ended without a complete pair artifact\n' >&2
            exit 1 ;;
    esac
    if (( SECONDS >= deadline )); then
        printf 'Source wait timed out; no source job was stopped\n' >&2
        exit 1
    fi
    sleep 30
done
mkdir -p "$ROOT/pairs" "$ROOT/split"
rsync -a --partial --partial-dir=.rsync-partial --info=stats2 \
    --exclude=COMPLETED --exclude=parts/ "hpc3:$SOURCE/pairs/" "$ROOT/pairs/"
rsync -a "hpc3:$SOURCE/pairs/COMPLETED" "$ROOT/pairs/"
rsync -a --partial "hpc3:$SOURCE/split/" "$ROOT/split/"
"$PY" -u "$CODE/scripts/preparations/prepare_rcsb_training_features.py" \
    --pair-manifest "$ROOT/pairs/pair_manifest.csv" \
    --feature-root /data1/wholeset/RCSBprot \
    --output "$ROOT/ready_v1" --workers "${SLURM_CPUS_PER_TASK:-16}"
