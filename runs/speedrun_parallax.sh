#!/bin/bash

# Parallax speedrun: same recipe and model configuration as runs/speedrun.sh
# (GPT-2 grade LLM, pretraining + SFT, designed for an 8XH100 node, ~3 hours), with the
# ONLY change being the attention implementation switched to Parallax (--attn-impl=parallax).
#
# Differences from runs/speedrun.sh:
#   - attention: Parallax instead of FA3/SDPA softmax attention.
#   - artifacts go to ./output (NANOCHAT_BASE_DIR), not ~/.cache, under model tag
#     "parallax-d<depth>" (mirrors runs/parallax_smoke.sh).
#   - wandb logs to project "parallax-nanochat".
#   - env is set up with the parallax dependency group (uv sync --extra gpu --group parallax).
#
# Launch:
#   bash runs/speedrun_parallax.sh
# With wandb logging (otherwise disabled):
#   wandb login
#   WANDB_RUN=parallax-d24 bash runs/speedrun_parallax.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export OMP_NUM_THREADS=1
# Output dir + model tag + wandb project (mirrors runs/parallax_smoke.sh; not ~/.cache).
export NANOCHAT_BASE_DIR="$REPO_DIR/output"
DEPTH=24
MODEL_TAG="parallax-d${DEPTH}"
WANDB_PROJECT="parallax-nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
# NOTE: do not redirect TORCHINDUCTOR_CACHE_DIR to the NFS output dir here — under
# multi-rank torchrun the ranks race on the shared cache files (FileNotFoundError). The
# default node-local /tmp cache (as in runs/speedrun.sh) is multi-rank safe.

# -----------------------------------------------------------------------------
# Python venv setup with uv (includes the parallax deps: triton + nvidia-cutlass-dsl)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --group parallax
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup. To log, run `wandb login` first and set WANDB_RUN to a run name.
# Default "dummy" disables wandb logging.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report (writes to $NANOCHAT_BASE_DIR/report)
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer: download ~2B chars (8 shards), kick off the rest in the background, then train.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) — Parallax attention
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24, slightly undertrained to beat GPT-2 (data:params ratio 8). Same as the baseline
# speedrun, plus --attn-impl=parallax and the parallax output tag / wandb project.
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --attn-impl=parallax --model-tag="$MODEL_TAG" \
    --wandb-project="$WANDB_PROJECT" --run=$WANDB_RUN
# Evaluate the base model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# chat_sft inherits the model config (incl. attn_impl=parallax) from the base checkpoint.
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" --device-batch-size=16 \
    --wandb-project="$WANDB_PROJECT" --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -g "$MODEL_TAG"

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
