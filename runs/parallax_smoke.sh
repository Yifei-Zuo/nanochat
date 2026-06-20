#!/bin/bash
# Parallax smoke test.
#   1. Pretrains a tiny Parallax model for 10 steps.
#   2. Runs a 10-step evaluation (BPB + generation) on the saved checkpoint.
#   3. Removes the checkpoints afterwards.
#
# All artifacts are written under ./output (NOT ~/.cache/nanochat), in the save
# folder "parallax-smoke" (i.e. output/base_checkpoints/parallax-smoke).
#
# Requires an environment with nanochat + the parallax deps and a GPU:
#     uv sync --extra gpu --group parallax
# Run it (override the interpreter with PYTHON=... if no venv is activated):
#     bash runs/parallax_smoke.sh
# On a Slurm cluster, e.g.:
#     srun -p main --gres=gpu:1 bash runs/parallax_smoke.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- Output location: ./output instead of ~/.cache; save folder "parallax-smoke" ---
export NANOCHAT_BASE_DIR="$REPO_DIR/output"
MODEL_TAG="parallax-smoke"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON="${PYTHON:-python}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Isolate the torch.compile (Inductor) cache under the output dir so the smoke test is
# reproducible and never replays a stale graph from the shared /tmp/torchinductor cache.
# Override via the env var if you want to share a cache. (The Triton autotune cache is
# separate and stays in its default location.)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NANOCHAT_BASE_DIR/.inductor_cache}"

# Always remove this run's checkpoints on exit (success or failure).
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
cleanup() { rm -rf "$CKPT_DIR"; echo "[parallax-smoke] removed checkpoints: $CKPT_DIR"; }
trap cleanup EXIT

# --- Data prep (idempotent): reuse a shared ~/.cache/nanochat copy if present, else fetch ---
CACHE_DIR="$HOME/.cache/nanochat"
if [ ! -e "$NANOCHAT_BASE_DIR/base_data_climbmix" ]; then
    if [ -d "$CACHE_DIR/base_data_climbmix" ]; then
        ln -s "$CACHE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_data_climbmix"
    else
        $PYTHON -m nanochat.dataset -n 8
    fi
fi
if [ ! -e "$NANOCHAT_BASE_DIR/tokenizer" ]; then
    if [ -d "$CACHE_DIR/tokenizer" ]; then
        ln -s "$CACHE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
    else
        $PYTHON -m scripts.tok_train --max-chars=100000000
    fi
fi

# --- 1. Pretrain the Parallax model for 10 steps (saves a checkpoint at step 10) ---
$PYTHON -m scripts.base_train \
    --attn-impl=parallax \
    --model-tag="$MODEL_TAG" \
    --depth=4 --max-seq-len=512 --window-pattern=L \
    --device-batch-size=1 --total-batch-size=512 \
    --num-iterations=10 --save-every=10 \
    --eval-every=-1 --sample-every=-1 --core-metric-every=-1

# --- 2. Evaluate the saved checkpoint: 10-step BPB + generation samples ---
#   split-tokens=5120 == 10 steps at device-batch-size=1 x max-seq-len=512.
#   (add 'core' to --eval for the CORE metric; it downloads eval_bundle.zip once.)
$PYTHON -m scripts.base_eval \
    --model-tag="$MODEL_TAG" \
    --eval=bpb,sample \
    --device-batch-size=1 --split-tokens=5120

echo "[parallax-smoke] done."
