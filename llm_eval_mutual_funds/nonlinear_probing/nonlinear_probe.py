# -*- coding: utf-8 -*-
"""
Nonlinear (MLP) probing module for Mutual Funds.
Runs on the same activations and splits as linear probes; layer set is configurable.
Lives in nonlinear_probing/ and imports from linear_probing/.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_LINEAR_PROBING_DIR = _THIS_DIR.parent / "linear_probing"

lp_config = importlib.util.spec_from_file_location(
    "lp_config", str(_LINEAR_PROBING_DIR / "lp_config.py")
)
lp_config_mod = importlib.util.module_from_spec(lp_config)
lp_config.loader.exec_module(lp_config_mod)

lp_utils = importlib.util.spec_from_file_location(
    "lp_utils", str(_LINEAR_PROBING_DIR / "lp_utils.py")
)
lp_utils_mod = importlib.util.module_from_spec(lp_utils)
lp_utils.loader.exec_module(lp_utils_mod)

# Import from probe.py (linear) for shared types and helpers.
# Reuse sys.modules["probe"] if already loaded (e.g. by run_linear_and_nonlinear) so pickle works.
if "probe" in sys.modules:
    probe_mod = sys.modules["probe"]
else:
    probe_module_path = _LINEAR_PROBING_DIR / "probe.py"
    spec = importlib.util.spec_from_file_location("probe", str(probe_module_path))
    probe_mod = importlib.util.module_from_spec(spec)
    sys.modules["probe"] = probe_mod
    spec.loader.exec_module(probe_mod)

ProbeResult = probe_mod.ProbeResult
ProbeExperiment = probe_mod.ProbeExperiment
create_stratified_splits = probe_mod.create_stratified_splits
compute_binomial_ci = probe_mod.compute_binomial_ci
compute_p_value_vs_chance = probe_mod.compute_p_value_vs_chance
bootstrap_accuracy = probe_mod.bootstrap_accuracy
setup_logging = probe_mod.setup_logging
print_banner = probe_mod.print_banner
load_activations = lp_utils_mod.load_activations

PROBE_FEATURES = lp_config_mod.PROBE_FEATURES
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
TRAIN_RATIO = lp_config_mod.TRAIN_RATIO
VAL_RATIO = lp_config_mod.VAL_RATIO
TEST_RATIO = lp_config_mod.TEST_RATIO
BOOTSTRAP_ITERATIONS = lp_config_mod.BOOTSTRAP_ITERATIONS
CONFIDENCE_LEVEL = lp_config_mod.CONFIDENCE_LEVEL
SIGNIFICANCE_THRESHOLD = lp_config_mod.SIGNIFICANCE_THRESHOLD
PROBE_RANDOM_STATE = lp_config_mod.PROBE_RANDOM_STATE

# Nonlinear probe config (from this folder only; does not affect linear probes)
nlp_config = importlib.util.spec_from_file_location(
    "nlp_config", str(_THIS_DIR / "nlp_config.py")
)
nlp_config_mod = importlib.util.module_from_spec(nlp_config)
nlp_config.loader.exec_module(nlp_config_mod)

NONLINEAR_PROBE_LAYERS = nlp_config_mod.NONLINEAR_PROBE_LAYERS
NONLINEAR_PROBE_HIDDEN = nlp_config_mod.NONLINEAR_PROBE_HIDDEN
NONLINEAR_PROBE_DROPOUT = nlp_config_mod.NONLINEAR_PROBE_DROPOUT
NONLINEAR_PROBE_MAX_EPOCHS = nlp_config_mod.NONLINEAR_PROBE_MAX_EPOCHS
NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE = nlp_config_mod.NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE
NONLINEAR_PROBE_RANDOM_STATE = nlp_config_mod.NONLINEAR_PROBE_RANDOM_STATE
CONTROL_TASK_SEED = getattr(nlp_config_mod, "CONTROL_TASK_SEED", 42)


def _scrambled_hierarchy_control_labels(
    value_1: np.ndarray,
    value_2: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Scrambled Hierarchy control: assign random hidden ranks to unique values, then
    Control_Label = True iff Rank(Fund1_value) < Rank(Fund2_value). Preserves transitivity
    but removes the real mathematical ordering. Reproducible via seed."""
    n = len(value_1)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = (~np.isnan(value_1)) & (~np.isnan(value_2))
    if not np.any(valid):
        return out
    unique_vals = np.unique(np.concatenate([value_1[valid], value_2[valid]]))
    rng = np.random.RandomState(seed)
    hidden_rank = np.arange(1, len(unique_vals) + 1, dtype=np.float64)
    rng.shuffle(hidden_rank)
    value_to_rank = {v: r for v, r in zip(unique_vals, hidden_rank)}
    for i in range(n):
        if not valid[i]:
            continue
        r1 = value_to_rank[value_1[i]]
        r2 = value_to_rank[value_2[i]]
        out[i] = 1.0 if r1 < r2 else 0.0
    return out


def _shuffle_labels_for_control(labels: np.ndarray, feature: str, seed: int) -> np.ndarray:
    """Fallback control: shuffled labels (used when raw values not available)."""
    out = np.full_like(labels, np.nan, dtype=labels.dtype)
    valid = ~np.isnan(labels)
    if not np.any(valid):
        return out
    valid_idx = np.where(valid)[0]
    values = np.array(labels[valid_idx], copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(values)
    out[valid_idx] = values
    return out


def train_and_evaluate_nonlinear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_sizes: tuple = NONLINEAR_PROBE_HIDDEN,
    dropout: float = NONLINEAR_PROBE_DROPOUT,
    max_epochs: int = NONLINEAR_PROBE_MAX_EPOCHS,
    patience: int = NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE,
    random_state: int = NONLINEAR_PROBE_RANDOM_STATE,
) -> Dict[str, Any]:
    """Train an MLP probe and return metrics (same structure as linear probe for compatibility)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=1e-4,
            max_iter=max_epochs,
            shuffle=True,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            n_iter_no_change=patience,
            tol=1e-4,
        )
        if dropout > 0:
            # sklearn doesn't have dropout; we use alpha (L2) for regularization
            clf.alpha = 1e-3
        clf.fit(X_train_scaled, y_train)

    y_train_pred = clf.predict(X_train_scaled)
    y_val_pred = clf.predict(X_val_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    y_train_prob = clf.predict_proba(X_train_scaled)[:, 1]
    y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
    y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    def safe_auc(y_true, y_prob):
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_prob)

    train_auc = safe_auc(y_train, y_train_prob)
    val_auc = safe_auc(y_val, y_val_prob)
    test_auc = safe_auc(y_test, y_test_prob)

    n_correct = (y_test_pred == y_test).sum()
    n_test = len(y_test)
    ci_lower, ci_upper = compute_binomial_ci(n_correct, n_test)
    p_value = compute_p_value_vs_chance(n_correct, n_test)
    _, bootstrap_ci_lower, bootstrap_ci_upper = bootstrap_accuracy(y_test, y_test_pred)

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "cv_mean": val_acc,
        "cv_std": 0.0,
        "best_C": 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_ci_lower": bootstrap_ci_lower,
        "bootstrap_ci_upper": bootstrap_ci_upper,
        "p_value": p_value,
        "is_significant": p_value < SIGNIFICANCE_THRESHOLD,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": n_test,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }


def probe_layer_nonlinear(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    split_indices: Dict[str, np.ndarray],
    feature_name: str = "target",
    logger: Optional[logging.Logger] = None,
) -> ProbeResult:
    """Run nonlinear (MLP) probe for one layer and one feature."""
    X = activations[:, layer, :]
    y = labels.astype(float)

    X_train = X[split_indices["train"]]
    y_train = y[split_indices["train"]]
    X_val = X[split_indices["val"]]
    y_val = y[split_indices["val"]]
    X_test = X[split_indices["test"]]
    y_test = y[split_indices["test"]]

    def remove_nans(X, y):
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[valid_mask], y[valid_mask]

    X_train, y_train = remove_nans(X_train, y_train)
    X_val, y_val = remove_nans(X_val, y_val)
    X_test, y_test = remove_nans(X_test, y_test)

    min_samples = 20
    if len(X_train) < min_samples or len(X_test) < min_samples:
        if logger:
            logger.warning(f"Layer {layer}, {feature_name}: Insufficient samples")
        return ProbeResult(
            layer=layer,
            feature=feature_name,
            test_accuracy=0.5,
            test_auc=0.5,
            test_n_samples=len(X_test),
            val_accuracy=0.5,
            val_auc=0.5,
            train_accuracy=0.5,
            cv_mean=0.5,
            cv_std=0.0,
            accuracy_ci_lower=0.5,
            accuracy_ci_upper=0.5,
            p_value=1.0,
            is_significant=False,
            best_C=0.0,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        )

    metrics = train_and_evaluate_nonlinear_probe(
        X_train,
        y_train.astype(int),
        X_val,
        y_val.astype(int),
        X_test,
        y_test.astype(int),
    )

    return ProbeResult(
        layer=layer,
        feature=feature_name,
        test_accuracy=metrics["test_accuracy"],
        test_auc=metrics["test_auc"],
        test_n_samples=metrics["n_test"],
        val_accuracy=metrics["val_accuracy"],
        val_auc=metrics["val_auc"],
        train_accuracy=metrics["train_accuracy"],
        cv_mean=metrics["cv_mean"],
        cv_std=metrics["cv_std"],
        accuracy_ci_lower=metrics["ci_lower"],
        accuracy_ci_upper=metrics["ci_upper"],
        p_value=metrics["p_value"],
        is_significant=metrics["is_significant"],
        best_C=metrics["best_C"],
        n_train=metrics["n_train"],
        n_val=metrics["n_val"],
        n_test=metrics["n_test"],
    )


def _layers_to_probe(n_layers: int) -> List[int]:
    """Return list of layer indices to run nonlinear probes on."""
    if NONLINEAR_PROBE_LAYERS is None:
        return list(range(n_layers))
    return [int(l) for l in NONLINEAR_PROBE_LAYERS if 0 <= l < n_layers]


def run_nonlinear_probing_experiment(
    activations: np.ndarray,
    feature_labels: pd.DataFrame,
    ground_truth_labels: np.ndarray,
    model_name: str,
    condition: str,
    features_to_probe: List[str] = None,
    output_dir: Path = None,
    logger: logging.Logger = None,
    feature_raw_values: Optional[Dict[str, tuple]] = None,
) -> tuple:
    """Run nonlinear (MLP) probing on configured layers only, plus control.
    Control uses Scrambled Hierarchy when feature_raw_values is provided (recommended);
    otherwise falls back to shuffled labels. Returns (experiment, control_experiment)."""
    n_samples, n_layers, d_model = activations.shape

    if features_to_probe is None:
        features_to_probe = PROBE_FEATURES
    if output_dir is None:
        output_dir = PROBE_RESULTS_DIR / "nonlinear"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = setup_logging(output_dir, model_name, condition)

    layers_to_run = _layers_to_probe(n_layers)
    print_banner(f"Nonlinear Probing: {model_name} / {condition}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Condition: {condition}")
    logger.info(f"Layers to probe: {layers_to_run} (config: NONLINEAR_PROBE_LAYERS)")
    logger.info(f"Features: {features_to_probe}")
    if feature_raw_values:
        logger.info("Control task: Scrambled Hierarchy (random hidden ranks, preserves transitivity)")
    else:
        logger.info("Control task: shuffled labels (fallback; re-extract activations with raw values for Scrambled Hierarchy)")

    gt_for_split = np.where(np.isnan(ground_truth_labels), 0, ground_truth_labels).astype(int)
    split_indices = create_stratified_splits(n_samples, gt_for_split)

    results = []
    control_results = []
    features_valid = []
    for feature in features_to_probe:
        if feature in feature_labels.columns:
            labels = feature_labels[feature].values
        elif feature == "medalist_f1_higher":
            labels = ground_truth_labels.copy()
        else:
            logger.warning(f"Feature '{feature}' not found, skipping")
            continue
        features_valid.append(feature)
    total_probes = 2 * len(layers_to_run) * len(features_valid)
    pbar = tqdm(total=total_probes, desc="Nonlinear + control")

    for feature in features_to_probe:
        if feature in feature_labels.columns:
            labels = feature_labels[feature].values
        elif feature == "medalist_f1_higher":
            labels = ground_truth_labels.copy()
        else:
            continue
        seed = CONTROL_TASK_SEED + hash(feature) % (2**32)
        if feature_raw_values and feature in feature_raw_values:
            v1, v2 = feature_raw_values[feature]
            control_labels = _scrambled_hierarchy_control_labels(
                np.asarray(v1), np.asarray(v2), seed=seed
            )
        else:
            control_labels = _shuffle_labels_for_control(labels, feature, seed=seed)
        for layer in layers_to_run:
            result = probe_layer_nonlinear(
                activations=activations,
                labels=labels,
                layer=layer,
                split_indices=split_indices,
                feature_name=feature,
                logger=logger,
            )
            results.append(result)
            control_result = probe_layer_nonlinear(
                activations=activations,
                labels=control_labels,
                layer=layer,
                split_indices=split_indices,
                feature_name=feature,
                logger=logger,
            )
            control_results.append(control_result)
            if result.is_significant and result.test_accuracy > 0.55:
                logger.info(
                    f"  {feature} L{layer}: acc={result.test_accuracy:.3f} "
                    f"control={control_result.test_accuracy:.3f} "
                    f"[{result.accuracy_ci_lower:.3f}, {result.accuracy_ci_upper:.3f}] p={result.p_value:.4f} *"
                )
            pbar.update(2)
    pbar.close()

    common_config = {
        "probe_type": "nonlinear",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "d_model": d_model,
        "nonlinear_probe_layers": layers_to_run,
        "hidden_sizes": list(NONLINEAR_PROBE_HIDDEN),
        "max_epochs": NONLINEAR_PROBE_MAX_EPOCHS,
    }
    experiment = ProbeExperiment(
        model_name=model_name,
        condition=condition,
        results=results,
        split_indices=split_indices,
        config=common_config,
    )
    control_task_name = "control_scrambled_hierarchy" if feature_raw_values else "control_shuffled_labels"
    control_experiment = ProbeExperiment(
        model_name=model_name,
        condition=condition,
        results=control_results,
        split_indices=split_indices,
        config={**common_config, "task": control_task_name, "control_seed": CONTROL_TASK_SEED},
    )
    return experiment, control_experiment


def export_nonlinear_results(
    experiment: ProbeExperiment,
    output_dir: Path,
    control_experiment: Optional[ProbeExperiment] = None,
):
    """Export nonlinear probe results to CSV; optionally export control (Scrambled Hierarchy or shuffled-labels) results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = experiment.to_dataframe()
    csv_path = output_dir / f"probe_nonlinear_results_{experiment.model_name}_{experiment.condition}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved nonlinear results to {csv_path}")

    matrix_acc = experiment.get_layer_feature_matrix("test_accuracy")
    matrix_path = output_dir / f"probe_nonlinear_matrix_accuracy_{experiment.model_name}_{experiment.condition}.csv"
    matrix_acc.to_csv(matrix_path)
    print(f"Saved nonlinear accuracy matrix to {matrix_path}")

    config_path = output_dir / f"probe_nonlinear_config_{experiment.model_name}_{experiment.condition}.json"
    config = {
        "model_name": experiment.model_name,
        "condition": experiment.condition,
        "timestamp": experiment.timestamp,
        "probe_type": "nonlinear",
        **experiment.config,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved nonlinear config to {config_path}")

    if control_experiment is not None:
        cdf = control_experiment.to_dataframe()
        c_csv = output_dir / f"probe_nonlinear_control_results_{control_experiment.model_name}_{control_experiment.condition}.csv"
        cdf.to_csv(c_csv, index=False)
        print(f"Saved control results to {c_csv}")
        c_matrix = control_experiment.get_layer_feature_matrix("test_accuracy")
        c_matrix_path = output_dir / f"probe_nonlinear_control_matrix_accuracy_{control_experiment.model_name}_{control_experiment.condition}.csv"
        c_matrix.to_csv(c_matrix_path)
        print(f"Saved control accuracy matrix to {c_matrix_path}")

    return csv_path
