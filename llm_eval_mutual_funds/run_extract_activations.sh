#!/bin/bash
#SBATCH --job-name=extract_activations_mf
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

# ============================================================================
# SLURM Job Script for Activation Extraction (Mutual Funds)
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="$PROJECT_DIR/../input_csvs/mutual_funds_pairs_no_date.csv"

echo "============================================================"
echo "Activation Extraction for Mutual Fund Linear Probing"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $PROJECT_DIR"
echo "============================================================"

cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

mkdir -p logs data/activations data/checkpoints

echo ""
echo "Running pre-flight checks..."

if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "  Checking data file..."
if [ ! -f "$DATA_FILE" ]; then
    echo "  ERROR: $DATA_FILE not found"
    exit 1
fi
NUM_SAMPLES=$(tail -n +2 "$DATA_FILE" | wc -l)
echo "  Found $NUM_SAMPLES samples"

echo "  Checking models..."
if [ ! -d "models/Llama-3.2-3B-Instruct" ]; then
    echo "  ERROR: models/Llama-3.2-3B-Instruct not found"
    exit 1
fi
if [ ! -d "models/Qwen3-4B-Instruct-2507" ]; then
    echo "  ERROR: models/Qwen3-4B-Instruct-2507 not found"
    exit 1
fi
echo "  Both models found"

echo ""
echo "Pre-flight checks passed!"
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd "$PROJECT_DIR/linear_probing" || exit 1

BATCH_SIZE=50
CONDITION="2_fewshot_cot_temp0"

echo "============================================================"
echo "Extracting Llama-3.2-3B-Instruct activations"
echo "Condition: $CONDITION"
echo "============================================================"

python extract_activations.py \
    --model llama-3.2-3b \
    --condition $CONDITION \
    --batch-size $BATCH_SIZE \
    --device cuda \
    --no-resume

LLAMA_EXIT=$?

echo ""
echo "============================================================"
echo "Extracting Qwen3-4B-Instruct activations"
echo "Condition: $CONDITION"
echo "============================================================"

python extract_activations.py \
    --model qwen3-4b \
    --condition $CONDITION \
    --batch-size $BATCH_SIZE \
    --device cuda \
    --no-resume

QWEN_EXIT=$?

echo ""
echo "============================================================"
echo "Extraction Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo "============================================================"

if [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    exit 0
fi
exit 1
