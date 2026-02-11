#!/bin/bash
#SBATCH --job-name=linear_probe_mf
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

# ============================================================================
# SLURM Job Script for Linear Probing (Mutual Funds)
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Linear Probing for Mutual Fund Mechanistic Interpretability"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $PROJECT_DIR"
echo "============================================================"

cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

mkdir -p logs data/probe_results data/probe_results/logs

CONDITION="2_fewshot_cot_temp0"

LLAMA_ACT="data/activations/llama-3.2-3b_${CONDITION}_activations.npz"
QWEN_ACT="data/activations/qwen3-4b_${CONDITION}_activations.npz"

echo ""
echo "Running pre-flight checks..."

LLAMA_READY=false
QWEN_READY=false

if [ -f "$LLAMA_ACT" ]; then
    echo "  Llama activations found"
    LLAMA_READY=true
else
    echo "  WARNING: Llama activations not found at $LLAMA_ACT"
fi

if [ -f "$QWEN_ACT" ]; then
    echo "  Qwen activations found"
    QWEN_READY=true
else
    echo "  WARNING: Qwen activations not found at $QWEN_ACT"
fi

echo ""
echo "Pre-flight checks passed!"
echo ""

cd "$PROJECT_DIR/linear_probing" || exit 1

LLAMA_EXIT=0
QWEN_EXIT=0

if [ "$LLAMA_READY" = true ]; then
    echo "============================================================"
    echo "Probing Llama-3.2-3B-Instruct activations"
    echo "Condition: $CONDITION"
    echo "============================================================"

    python probe.py \
        --model llama-3.2-3b \
        --condition $CONDITION

    LLAMA_EXIT=$?
fi

if [ "$QWEN_READY" = true ]; then
    echo ""
    echo "============================================================"
    echo "Probing Qwen3-4B-Instruct activations"
    echo "Condition: $CONDITION"
    echo "============================================================"

    python probe.py \
        --model qwen3-4b \
        --condition $CONDITION

    QWEN_EXIT=$?
fi

echo ""
echo "============================================================"
echo "Linear Probing Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo "============================================================"

if [ "$LLAMA_READY" = false ] && [ "$QWEN_READY" = false ]; then
    exit 1
elif [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    exit 0
fi
exit 1
