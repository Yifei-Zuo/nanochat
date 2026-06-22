#!/bin/bash
#SBATCH --job-name=plx-sft
#SBATCH --partition=main
#SBATCH --qos=guest-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/home/guests/yifei/nanochat/output/logs/sbatch_sft_%j.out

# Batch submission of the SFT-only pipeline (skips pretraining, uses the existing base
# checkpoint). One 8xH100 node; runs/sft_parallax.sh launches torchrun itself.
# No --time set (account default 3-day limit is ample; the job exits when SFT finishes).
# Submit:  sbatch runs/sft_parallax.sbatch

export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under the batch shell
cd /home/guests/yifei/nanochat || exit 1
echo "[sbatch] host=$(hostname) job=${SLURM_JOB_ID} gpus=${SLURM_GPUS_ON_NODE:-?} start=$(date)"
bash runs/sft_parallax.sh
rc=$?
echo "[sbatch] sft exited rc=${rc} at $(date)"
exit $rc
