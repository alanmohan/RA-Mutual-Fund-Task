# -*- coding: utf-8 -*-
"""
Configuration for Token Position Probing Experiment.

Extracts activations at a grid of token positions (from end) across all layers,
then sweeps logistic regression probes to map how feature encoding varies
across the prompt.
"""
import sys
import importlib.util
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

_THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _THIS_DIR.parent.resolve()          # llm_eval_mutual_funds/
_LP_DIR = PROJECT_ROOT / "linear_probing"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_lp_config = _load_module("_tp_lp_config", _LP_DIR / "lp_config.py")
_lp_utils = _load_module("_tp_lp_utils", _LP_DIR / "lp_utils.py")

# Re-export from lp_config
MODELS = _lp_config.MODELS
DATA_PATH = _lp_config.DATA_PATH
PROBE_FEATURES = _lp_config.PROBE_FEATURES
PROBE_CS = _lp_config.PROBE_CS
PROBE_MAX_ITER = _lp_config.PROBE_MAX_ITER
PROBE_RANDOM_STATE = _lp_config.PROBE_RANDOM_STATE
EXTRACTION_RANDOM_STATE = _lp_config.EXTRACTION_RANDOM_STATE
BOOTSTRAP_ITERATIONS = _lp_config.BOOTSTRAP_ITERATIONS
CONFIDENCE_LEVEL = _lp_config.CONFIDENCE_LEVEL
SIGNIFICANCE_THRESHOLD = _lp_config.SIGNIFICANCE_THRESHOLD
TRAIN_RATIO = _lp_config.TRAIN_RATIO
VAL_RATIO = _lp_config.VAL_RATIO
TEST_RATIO = _lp_config.TEST_RATIO

# Re-export from lp_utils
create_feature_labels = _lp_utils.create_feature_labels
print_banner = _lp_utils.print_banner

# ============================================================================
# TOKEN-POSITION EXTRACTION SETTINGS
# ============================================================================

# Position grid: extract every POSITION_STEP-th token from the end.
# -1, -1-step, -1-2*step, ...  until we exceed the shortest sequence.
POSITION_STEP = 5

# Number of samples to extract (fewer than full probing to keep storage sane).
TP_SAMPLE_SIZE = 1000

# Batch size for extraction — sized for a T4 (16 GB VRAM).
TP_EXTRACTION_BATCH_SIZE = 5

# ============================================================================
# PROBING SETTINGS
# ============================================================================

# Number of parallel workers for layer-level probing within each position.
N_PROBE_WORKERS = 4

# ============================================================================
# OUTPUT PATHS
# ============================================================================

TP_ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations" / "token_position"
TP_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

TP_RESULTS_DIR = PROJECT_ROOT / "data" / "probe_results" / "token_position"
TP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_tp_hdf5_path(model: str, condition: str) -> Path:
    return TP_ACTIVATIONS_DIR / f"{model}_{condition}_all_positions.h5"


def get_tp_results_csv(feature: str, model: str, condition: str) -> Path:
    return TP_RESULTS_DIR / f"tp_probe_{feature}_{model}_{condition}.csv"
