#!/usr/bin/env bash
# =============================================================================
# Token-Position Probing: full pipeline for Colab (or any clone of the repo).
# Run from repo root, or from llm_eval_mutual_funds, or from this script's dir.
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"   # or set HF_TOKEN below
#   bash token_position_probing/run_token_position_experiment.sh
# =============================================================================
set -e

# -----------------------------------------------------------------------------
# CONFIGURABLE VARIABLES — edit these as needed
# -----------------------------------------------------------------------------

# Hugging Face token (required for downloading gated models). Prefer setting
# via env: export HF_TOKEN="hf_..."
: "${HF_TOKEN:=}"

# Model and condition (used for extraction, probing, and plotting)
MODEL="${MODEL:-qwen3-4b}"
CONDITION="${CONDITION:-2_fewshot_cot_temp0}"

# Extraction (Phase 1)
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
POSITION_STEP="${POSITION_STEP:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
DEVICE="${DEVICE:-cuda}"

# Probing (Phase 2) — one feature per run
FEATURE="${FEATURE:-beta_f1_lower}"
N_WORKERS="${N_WORKERS:-4}"

# Plotting (Phase 3) — optional; leave empty to skip annotations
FEATURE_POSITION="${FEATURE_POSITION:-}"       # e.g. -89 from find_token_position.py
LAST_TOKEN_ACCURACY="${LAST_TOKEN_ACCURACY:-}" # e.g. 0.746 from last-token probe
LAYERS="${LAYERS:-}"                           # e.g. "10 19 23" for selected-layers plot

# Whether to run each phase (set to "no" to skip)
RUN_DOWNLOAD="${RUN_DOWNLOAD:-yes}"
RUN_EXTRACTION="${RUN_EXTRACTION:-yes}"
RUN_PROBING="${RUN_PROBING:-yes}"
RUN_PLOTTING="${RUN_PLOTTING:-yes}"

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# llm_eval_mutual_funds/ (parent of token_position_probing)
LLM_EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Repo root (parent of llm_eval_mutual_funds)
REPO_ROOT="$(cd "$LLM_EVAL_ROOT/.." && pwd)"

# -----------------------------------------------------------------------------
# EXPORT HUGGING FACE TOKEN
# -----------------------------------------------------------------------------
if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
  echo "Using HF_TOKEN from environment."
else
  echo "Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=\"hf_...\""
  echo "Download may fail for gated models (Llama, etc.)."
fi

# -----------------------------------------------------------------------------
# ENSURE WE RUN FROM llm_eval_mutual_funds
# -----------------------------------------------------------------------------
cd "$LLM_EVAL_ROOT"
echo "Working directory: $LLM_EVAL_ROOT"

# -----------------------------------------------------------------------------
# PHASE 0: DOWNLOAD MODELS (Llama + Qwen)
# -----------------------------------------------------------------------------
if [ "$RUN_DOWNLOAD" = "yes" ]; then
  echo ""
  echo "========== Phase 0: Downloading models (Llama + Qwen) =========="
  python download_model.py
else
  echo "Skipping download (RUN_DOWNLOAD != yes)."
fi

# -----------------------------------------------------------------------------
# PHASE 1: EXTRACTION
# -----------------------------------------------------------------------------
if [ "$RUN_EXTRACTION" = "yes" ]; then
  echo ""
  echo "========== Phase 1: Extracting activations at all grid positions =========="
  python token_position_probing/extract_all_positions.py \
    --model "$MODEL" \
    --condition "$CONDITION" \
    --sample-size "$SAMPLE_SIZE" \
    --position-step "$POSITION_STEP" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
else
  echo "Skipping extraction (RUN_EXTRACTION != yes)."
fi

# -----------------------------------------------------------------------------
# PHASE 2: PROBING
# -----------------------------------------------------------------------------
if [ "$RUN_PROBING" = "yes" ]; then
  echo ""
  echo "========== Phase 2: Probing feature across positions =========="
  python token_position_probing/probe_positions.py \
    --model "$MODEL" \
    --condition "$CONDITION" \
    --feature "$FEATURE" \
    --n-workers "$N_WORKERS"
else
  echo "Skipping probing (RUN_PROBING != yes)."
fi

# -----------------------------------------------------------------------------
# PHASE 3: VISUALIZATION
# -----------------------------------------------------------------------------
if [ "$RUN_PLOTTING" = "yes" ]; then
  echo ""
  echo "========== Phase 3: Plotting results =========="
  RESULTS_CSV="data/probe_results/token_position/tp_probe_${FEATURE}_${MODEL}_${CONDITION}.csv"
  if [ ! -f "$RESULTS_CSV" ]; then
    echo "Error: Results CSV not found: $RESULTS_CSV"
    echo "Run probing first (RUN_PROBING=yes) or set RUN_PLOTTING=no."
    exit 1
  fi

  PLOT_ARGS=(
    --results-csv "$RESULTS_CSV"
    --feature "$FEATURE"
  )
  [ -n "$FEATURE_POSITION" ]    && PLOT_ARGS+=(--feature-position "$FEATURE_POSITION")
  [ -n "$LAST_TOKEN_ACCURACY" ] && PLOT_ARGS+=(--last-token-accuracy "$LAST_TOKEN_ACCURACY")
  [ -n "$LAYERS" ]              && PLOT_ARGS+=(--layers $LAYERS)

  python token_position_probing/plot_position_results.py "${PLOT_ARGS[@]}"
else
  echo "Skipping plotting (RUN_PLOTTING != yes)."
fi

echo ""
echo "========== Token-position experiment finished =========="
echo "Results: data/probe_results/token_position/"
