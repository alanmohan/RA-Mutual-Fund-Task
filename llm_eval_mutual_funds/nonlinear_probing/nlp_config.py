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
SKIP_LINEAR_PROBING = False

# Layers to run nonlinear (MLP) probes on. Saves time vs running on all layers.
# None = all layers; or a list of layer indices, e.g. [0, 5, 10, 15, 20, 25]
# This does NOT affect linear probes (they always run on every layer).
NONLINEAR_PROBE_LAYERS = None  # Set to e.g. [0, 5, 10, 15, 20, 25] to probe a subset

# MLP architecture: hidden layer sizes (input is d_model, output is 1)
NONLINEAR_PROBE_HIDDEN = (128, 64)
NONLINEAR_PROBE_DROPOUT = 0.2
NONLINEAR_PROBE_MAX_EPOCHS = 200
NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE = 15
NONLINEAR_PROBE_RANDOM_STATE = 42

# Control task: shuffled labels (same activations, random labels). Used to check selectivity.
# If control accuracy is high (comparable to real probe), the MLP may be overfitting.
CONTROL_TASK_SEED = 42
