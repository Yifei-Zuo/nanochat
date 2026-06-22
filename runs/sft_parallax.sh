#!/bin/bash

# Parallax SFT-only: SKIP pretraining and use an existing base checkpoint
# (output/base_checkpoints/parallax-d<depth>) to run the SFT stage + chat eval only.
# Use this when base_train already produced a checkpoint and you want to (re)run SFT,
# e.g. after a crash in the SFT phase of runs/speedrun_parallax.sh.
#
# Mirrors the env of runs/speedrun_parallax.sh (./output, model tag, wandb project,
# node-local Triton cache + autotune persistence). Same model config / attn_impl is
# inherited from the base checkpoint's meta.json.
#
# Launch:  bash runs/sft_parallax.sh   (or: sbatch runs/sft_parallax.sbatch)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
set -eo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$REPO_DIR/output"
DEPTH=24
MODEL_TAG="parallax-d${DEPTH}"
WANDB_PROJECT="parallax-nanochat"
WANDB_RUN=$MODEL_TAG
mkdir -p "$NANOCHAT_BASE_DIR"

# Tee console output to a timestamped log under the output dir.
mkdir -p "$NANOCHAT_BASE_DIR/logs"
LOG_FILE="$NANOCHAT_BASE_DIR/logs/sft_parallax_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[sft-parallax] console log -> $LOG_FILE"

# Node-local Triton cache + persist autotune results across this job's torchrun stages.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-$USER}"
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-1}"

# -----------------------------------------------------------------------------
# Python venv setup with uv (includes the parallax deps)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --group parallax
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Require an existing base checkpoint (we deliberately skip pretraining).
if ! ls "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"/model_*.pt >/dev/null 2>&1; then
    echo "[sft-parallax] ERROR: no base checkpoint at $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
    echo "[sft-parallax] Run pretraining (runs/speedrun_parallax.sh) first, or fix the model tag."
    exit 1
fi
echo "[sft-parallax] using base checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

# Identity conversations for SFT (download only if missing, fail on HTTP error so we never
# leave a partial file).
IDENTITY="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
[ -s "$IDENTITY" ] || curl -fL -o "$IDENTITY" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# SFT (inherits attn_impl=parallax + model config from the base checkpoint) + chat eval.
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" --device-batch-size=16 \
    --wandb-project="$WANDB_PROJECT" --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -g "$MODEL_TAG"

# -----------------------------------------------------------------------------
python -m nanochat.report generate
echo "[sft-parallax] done."
