# -*- coding: utf-8 -*-
"""
Configuration for LLM Mutual Fund Comparison Experiment
Edit values here to change experiment settings
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = PROJECT_ROOT.parent
DATA_PATH = REPO_ROOT / "input_csvs" / "mutual_funds_pairs_no_date.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Checkpoint settings
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples

# Sampling
SAMPLE_SIZE = 10  # Change this to run on different sample sizes (default: 10 for testing)
RANDOM_STATE = 42

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Generation settings
MAX_NEW_TOKENS = 512
BATCH_SIZE = 30  # Adjust based on GPU memory

# Models to test (if using inference.py)
MODELS = {
    # Default to Llama only (LM Studio endpoint). Add Qwen here if you have it locally.
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}

# Local model cache directory
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"

# ============================================================================
# SYSTEM MESSAGE
# ============================================================================

SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "system_prompt.txt"
SYSTEM_MSG = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

# ============================================================================
# OPENAI-COMPATIBLE SETTINGS (LM Studio)
# ============================================================================

# Default LM Studio OpenAI-compatible base URL
OPENAI_BASE_URL = "http://127.0.0.1:1234"
# LM Studio does not require a real API key, but OpenAI client needs a value
OPENAI_API_KEY = "lm-studio"
# Default model name served by LM Studio
OPENAI_MODEL = "llama-3.2-3b-instruct"
# Use OpenAI-compatible endpoint for Llama model by default
USE_OPENAI_FOR_LLAMA = True

# ============================================================================
# GROUND TRUTH SETTINGS
# ============================================================================

# How to handle ties in Medalist rating:
# - "exclude": ties are excluded from accuracy calculations
# - "count_as_correct": ties are counted as correct regardless of prediction
TIE_HANDLING = "exclude"

# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

# Each condition defines:
# - prompt_fn: which prompt builder to use (imported from prompts.py)
# - do_sample: whether to use sampling
# - temperature, top_p: sampling parameters (if do_sample=True)

CONDITIONS = {
    "1_zeroshot_cot_temp0": {
        "prompt_fn": "build_prompt_zero_shot_cot",
        "temperature": 0.0,
        "do_sample": False,
    },
    "2_fewshot_cot_temp0": {
        "prompt_fn": "build_prompt_few_shot_cot",
        "temperature": 0.0,
        "do_sample": True,
    }
}
