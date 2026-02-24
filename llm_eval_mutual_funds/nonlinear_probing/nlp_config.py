# -*- coding: utf-8 -*-
"""
Configuration for Nonlinear (MLP) Probing Pipeline (Mutual Funds).
Layer selection applies only to nonlinear probes; linear probes always run on all layers.
"""
from pathlib import Path

# ============================================================================
# NONLINEAR PROBE SETTINGS
# ============================================================================
# If True, run_linear_and_nonlinear.py skips linear probing and runs only nonlinear + control.
# Default False: run both linear and nonlinear (and control).
SKIP_LINEAR_PROBING = True

# Features to run nonlinear (and linear, when --features not passed) probes on.
# None = use all features (from linear_probing lp_config.PROBE_FEATURES).
# A list of feature names = probe only those features, e.g. ["beta_f1_lower", "volatility_f1_lower"].
# CLI --features overrides this when provided.
# NONLINEAR_PROBE_FEATURES = ['beta_f1_lower', 'stdev_f1_lower', 'medalist_f1_higher', 'turnover_f1_lower', 'inception_f1_older', 'assets_f1_higher', 'tenure_f1_longer']  # Set to e.g. ["beta_f1_lower", "volatility_f1_lower"] to probe a subset
NONLINEAR_PROBE_FEATURES = ['stdev_f1_lower']  # Set to e.g. ["beta_f1_lower", "volatility_f1_lower"] to probe a subset

# Layers to run nonlinear (MLP) probes on. Saves time vs running on all layers.
# None = all layers; or a list of layer indices, e.g. [0, 5, 10, 15, 20, 25]
# This does NOT affect linear probes (they always run on every layer).
NONLINEAR_PROBE_LAYERS = [21]  # Set to e.g. [0, 5, 10, 15, 20, 25] to probe a subset

# Loss for binary classifier: "bce" (BCEWithLogitsLoss, labels 0/1) or "hinge" (hinge loss, labels -1/1)
NONLINEAR_PROBE_LOSS = "bce"

# MLP architecture defaults (used when tuning is off)
NONLINEAR_PROBE_HIDDEN = (32,)
NONLINEAR_PROBE_DROPOUT = 0.3
NONLINEAR_PROBE_LR = 5e-4
NONLINEAR_PROBE_MAX_EPOCHS = 300
NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE = 30
NONLINEAR_PROBE_RANDOM_STATE = 42
NONLINEAR_PROBE_INPUT_NOISE_STD = 0.1

# Class weighting: True = balanced pos_weight in BCEWithLogitsLoss (matches linear probe class_weight="balanced")
NONLINEAR_PROBE_USE_CLASS_WEIGHT = True

# LR scheduler: True = warmup + cosine annealing; False = flat LR
NONLINEAR_PROBE_USE_SCHEDULER = True
NONLINEAR_PROBE_WARMUP_EPOCHS = 10

# Hyperparameter tuning: grid search per (feature, layer) selecting by val accuracy.
# Enable with --tune CLI flag or set NONLINEAR_PROBE_TUNE = True.
NONLINEAR_PROBE_TUNE = False
NONLINEAR_PROBE_TUNE_GRID = {
    "hidden_sizes": [(16,), (32,), (64,), (128,), (32, 16), (64, 32)],
    "lr": [1e-3, 5e-4, 1e-4],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "weight_decay": [1e-4, 1e-3, 1e-2, 5e-2],
    "input_noise_std": [0.0, 0.1, 0.3],
}

# Control task: shuffled labels (same activations, random labels). Used to check selectivity.
# If control accuracy is high (comparable to real probe), the MLP may be overfitting.
CONTROL_TASK_SEED = 42
