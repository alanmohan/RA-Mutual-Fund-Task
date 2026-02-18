### RENAMED: lp_config.py
# -*- coding: utf-8 -*-
"""
Configuration for Linear Probing Pipeline (Mutual Funds)
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATH = PROJECT_ROOT.parent / "input_csvs" / "mutual_funds_pairs_no_date.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"

# Activation storage
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Probe results
PROBE_RESULTS_DIR = PROJECT_ROOT / "data" / "probe_results"
PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Model configurations
# use_transformerlens: True = use TransformerLens, False = use manual HuggingFace hooks
MODELS = {
    "llama-3.2-3b": {
        "hf_name": "meta-llama/Llama-3.2-3B-Instruct",
        "local_path": PROJECT_ROOT / "models" / "Llama-3.2-3B-Instruct",
        "n_layers": 28,
        "d_model": 3072,
        "use_transformerlens": True,  # Supported by TransformerLens
    },
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B-Instruct-2507",
        "local_path": PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507",
        "n_layers": 36,
        "d_model": 2560,
        "use_transformerlens": False,  # NOT supported - use manual HF extraction
    },
}

# ============================================================================
# EXTRACTION SETTINGS
# ============================================================================

# Which layers to extract (None = all layers)
LAYERS_TO_EXTRACT = None  # Will extract all layers

# Hook pattern for residual stream post-attention
HOOK_PATTERN = "hook_resid_post"

# Token position to extract (-1 = last token)
TOKEN_POSITION = -1

# Batch size for extraction (adjust based on GPU memory)
EXTRACTION_BATCH_SIZE = 50  # Optimized for A6000 (48GB)

# Checkpoint frequency
EXTRACTION_CHECKPOINT_INTERVAL = 500

# Sample size for activation extraction (None = use all data)
EXTRACTION_SAMPLE_SIZE = 5000  # Default: extract activations from 5000 samples
EXTRACTION_RANDOM_STATE = 42  # Random seed for sampling

# ============================================================================
# PROBING SETTINGS
# ============================================================================

# Features to probe for (binary labels derived from data)
PROBE_FEATURES = [
    "expense_ratio_f1_lower",
    "sharpe_f1_higher",
    "stdev_f1_lower",
    "return_3yr_f1_higher",
    "beta_f1_lower",
    "tenure_f1_longer",
    "inception_f1_older",
    "assets_f1_higher",
    "turnover_f1_lower",
    "load_f1_no",
    "ntf_f1_yes",
    "medalist_f1_higher",  # Ground truth target
]

# Data split ratios
TRAIN_RATIO = 0.70  # 70% for training (with CV for hyperparameter selection)
VAL_RATIO = 0.10    # 10% for validation
TEST_RATIO = 0.20   # 20% for final unbiased evaluation

# Cross-validation settings (used within training set only)
CV_FOLDS = 5
PROBE_MAX_ITER = 3000  # Base iterations (will be multiplied for final model) - increased to reduce convergence warnings
PROBE_RANDOM_STATE = 42

# Regularization values to try
PROBE_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Statistical settings
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_THRESHOLD = 0.05

# ============================================================================
# PROMPT STRATEGIES (will be updated based on experiment results)
# ============================================================================

# This will be set based on winning prompt from experiments
WINNING_PROMPT_STRATEGY = None  # e.g., "2_fewshot_cot_temp0"

# Import prompt functions from parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
