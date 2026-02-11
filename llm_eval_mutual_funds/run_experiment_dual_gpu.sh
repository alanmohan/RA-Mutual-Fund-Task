#!/bin/bash
#SBATCH --job-name=mutual_funds_dual_gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# SLURM Job Script for Dual-GPU LLM Mutual Fund Comparison Experiment
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="$PROJECT_DIR/../input_csvs/mutual_funds_pairs_no_date.csv"

echo "============================================================"
echo "DUAL-GPU LLM MUTUAL FUND COMPARISON EXPERIMENT"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $PROJECT_DIR"
echo "GPU Mode: Dual-GPU (parallel batches)"
echo "============================================================"

cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

mkdir -p logs data/results data/checkpoints

echo ""
echo "Running pre-flight checks..."

if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  ERROR: Dual-GPU mode requires 2 GPUs, found $GPU_COUNT"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "  Checking data file..."
if [ ! -f "$DATA_FILE" ]; then
    echo "  ERROR: $DATA_FILE not found"
    exit 1
fi
echo "  Data file found"

echo ""
echo "Pre-flight checks passed!"
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting dual-GPU experiment..."
echo ""

python main.py --dual-gpu

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds"
echo "============================================================"

exit $EXIT_CODE
