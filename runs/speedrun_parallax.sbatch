#!/bin/bash
#SBATCH --job-name=plx-speedrun
#SBATCH --partition=main
#SBATCH --qos=guest-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/home/guests/yifei/nanochat/output/logs/sbatch_speedrun_%j.out

# Full Parallax speedrun as a batch job. The run is long (~7-8h): a 6000-step d24
# pretrain with several max500 CORE evals, then base_eval + SFT + chat_eval. Runs on a
# single 8xH100 node; runs/speedrun_parallax.sh launches torchrun --standalone
# --nproc_per_node=8 itself (no srun needed). No --time set: the account's default
# limit (3 days) is ample, and the job exits as soon as the speedrun finishes.
#
# Submit:  sbatch runs/speedrun_parallax.sbatch
# (wandb logs to project "parallax-nanochat", run "parallax-d24"; needs `wandb login`.)

export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH under the batch shell
cd /home/guests/yifei/nanochat || exit 1
echo "[sbatch] host=$(hostname) job=${SLURM_JOB_ID} gpus=${SLURM_GPUS_ON_NODE:-?} start=$(date)"
bash runs/speedrun_parallax.sh
rc=$?
echo "[sbatch] speedrun exited rc=${rc} at $(date)"
exit $rc
