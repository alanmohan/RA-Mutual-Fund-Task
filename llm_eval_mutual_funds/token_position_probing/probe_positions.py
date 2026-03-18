# -*- coding: utf-8 -*-
"""
Probe a single feature across all (position, layer) pairs from the
token-position HDF5 file produced by extract_all_positions.py.

Layers within each position are probed in parallel via joblib.

Usage (from llm_eval_mutual_funds/):
    python token_position_probing/probe_positions.py \
        --model qwen3-4b --condition 2_fewshot_cot_temp0 \
        --feature beta_f1_lower --n-workers 4
"""
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()
_LP_DIR = _PROJECT_ROOT / "linear_probing"

for _p in (_PROJECT_ROOT, _PROJECT_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp_config = _load_mod("_pp_tp_config", _THIS_DIR / "tp_config.py")

MODELS = tp_config.MODELS
PROBE_FEATURES = tp_config.PROBE_FEATURES
PROBE_CS = tp_config.PROBE_CS
PROBE_MAX_ITER = tp_config.PROBE_MAX_ITER
PROBE_RANDOM_STATE = tp_config.PROBE_RANDOM_STATE
BOOTSTRAP_ITERATIONS = tp_config.BOOTSTRAP_ITERATIONS
CONFIDENCE_LEVEL = tp_config.CONFIDENCE_LEVEL
SIGNIFICANCE_THRESHOLD = tp_config.SIGNIFICANCE_THRESHOLD
TRAIN_RATIO = tp_config.TRAIN_RATIO
VAL_RATIO = tp_config.VAL_RATIO
TEST_RATIO = tp_config.TEST_RATIO
print_banner = tp_config.print_banner


# ============================================================================
# SPLIT + PROBE HELPERS (self-contained to avoid pickling issues with joblib)
# ============================================================================


def create_stratified_splits(
    n_samples: int,
    labels: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = PROBE_RANDOM_STATE,
):
    indices = np.arange(n_samples)
    train_val_ratio = train_ratio + val_ratio
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=random_state,
    )
    val_ratio_adj = val_ratio / train_val_ratio
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adj,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def compute_binomial_ci(n_correct, n_total, confidence=CONFIDENCE_LEVEL):
    if n_total == 0:
        return 0.5, 0.5
    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n_total
    center = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total) / denom
    return max(0, center - margin), min(1, center + margin)


def compute_p_value_vs_chance(n_correct, n_total, chance=0.5):
    if n_total == 0:
        return 1.0
    return stats.binomtest(n_correct, n_total, chance, alternative="greater").pvalue


def bootstrap_accuracy(y_true, y_pred, n_iterations=BOOTSTRAP_ITERATIONS,
                        confidence=CONFIDENCE_LEVEL):
    n = len(y_true)
    rng = np.random.RandomState(PROBE_RANDOM_STATE)
    accs = [accuracy_score(y_true[rng.choice(n, n, replace=True)],
                           y_pred[rng.choice(n, n, replace=True)])
            for _ in range(n_iterations)]
    accs = np.array(accs)
    alpha = 1 - confidence
    return np.mean(accs), np.percentile(accs, 100 * alpha / 2), np.percentile(accs, 100 * (1 - alpha / 2))


def probe_single_layer(
    X_all: np.ndarray,
    y: np.ndarray,
    layer: int,
    split_indices: dict,
    position: int,
    use_gpu: bool = True,
):
    """Train and evaluate a logistic-regression probe for one (position, layer).

    Uses cuML on GPU when available and use_gpu=True, otherwise sklearn.
    Returns a dict with all metrics (matches the CSV schema).
    """
    use_cuml = use_gpu and CUML_AVAILABLE

    X = X_all[:, layer, :]
    y_float = y.astype(float)

    def _split(X, y, idx):
        mask = ~(np.isnan(X[idx]).any(axis=1) | np.isnan(y[idx]))
        return X[idx][mask], y[idx][mask]

    X_tr, y_tr = _split(X, y_float, split_indices["train"])
    X_va, y_va = _split(X, y_float, split_indices["val"])
    X_te, y_te = _split(X, y_float, split_indices["test"])

    min_samples = 20
    if len(X_tr) < min_samples or len(X_te) < min_samples:
        return _chance_result(position, layer, len(X_te))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    best_C, best_val = None, 0.0

    if use_cuml:
        cuml_logger = logging.getLogger("cuml")
        orig_level = cuml_logger.level
        cuml_logger.setLevel(logging.ERROR)

        X_tr_gpu = cp.asarray(X_tr_s.astype(np.float32))
        y_tr_gpu = cp.asarray(y_tr.astype(np.float32))
        X_va_gpu = cp.asarray(X_va_s.astype(np.float32))
        X_te_gpu = cp.asarray(X_te_s.astype(np.float32))

        for C in PROBE_CS:
            clf = cuLogisticRegression(C=C, max_iter=PROBE_MAX_ITER, tol=1e-3, solver="qn")
            clf.fit(X_tr_gpu, y_tr_gpu)
            val_acc = accuracy_score(y_va, cp.asnumpy(clf.predict(X_va_gpu)))
            if val_acc > best_val:
                best_val = val_acc
                best_C = C

        final = cuLogisticRegression(C=best_C, max_iter=PROBE_MAX_ITER * 2, tol=1e-3, solver="qn")
        final.fit(X_tr_gpu, y_tr_gpu)

        y_te_pred = cp.asnumpy(final.predict(X_te_gpu))
        y_te_prob = cp.asnumpy(final.predict_proba(X_te_gpu))[:, 1]

        cuml_logger.setLevel(orig_level)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for C in PROBE_CS:
                clf = LogisticRegression(
                    C=C, max_iter=PROBE_MAX_ITER, random_state=PROBE_RANDOM_STATE,
                    class_weight="balanced", solver="saga", tol=1e-3, n_jobs=-1,
                )
                clf.fit(X_tr_s, y_tr)
                val_acc = accuracy_score(y_va, clf.predict(X_va_s))
                if val_acc > best_val:
                    best_val = val_acc
                    best_C = C

            final = LogisticRegression(
                C=best_C, max_iter=PROBE_MAX_ITER * 2, random_state=PROBE_RANDOM_STATE,
                class_weight="balanced", solver="saga", tol=1e-4, n_jobs=-1,
            )
            final.fit(X_tr_s, y_tr)

        y_te_pred = final.predict(X_te_s)
        y_te_prob = final.predict_proba(X_te_s)[:, 1]

    test_acc = accuracy_score(y_te, y_te_pred)
    test_auc = _safe_auc(y_te, y_te_prob)

    n_correct = int((y_te_pred == y_te).sum())
    n_test = len(y_te)
    ci_lo, ci_hi = compute_binomial_ci(n_correct, n_test)
    p_val = compute_p_value_vs_chance(n_correct, n_test)
    _, bs_lo, bs_hi = bootstrap_accuracy(y_te, y_te_pred)

    return {
        "position": position,
        "layer": layer,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "best_C": best_C,
        "val_accuracy": best_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "bootstrap_ci_lower": bs_lo,
        "bootstrap_ci_upper": bs_hi,
        "p_value": p_val,
        "is_significant": p_val < SIGNIFICANCE_THRESHOLD,
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "n_test": n_test,
    }


def _chance_result(position, layer, n_test):
    return {
        "position": position,
        "layer": layer,
        "test_accuracy": 0.5,
        "test_auc": 0.5,
        "best_C": None,
        "val_accuracy": 0.5,
        "ci_lower": 0.5,
        "ci_upper": 0.5,
        "bootstrap_ci_lower": 0.5,
        "bootstrap_ci_upper": 0.5,
        "p_value": 1.0,
        "is_significant": False,
        "n_train": 0,
        "n_val": 0,
        "n_test": n_test,
    }


# ============================================================================
# MAIN SWEEP
# ============================================================================


def probe_feature_across_positions(
    model_key: str,
    condition: str,
    feature: str,
    n_workers: int = 4,
    use_gpu: bool = True,
    position_min: int | None = None,
    position_max: int | None = None,
    output_path: Path | None = None,
):
    hdf5_path = tp_config.get_tp_hdf5_path(model_key, condition)
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"HDF5 not found: {hdf5_path}\nRun extract_all_positions.py first."
        )

    use_cuml = use_gpu and CUML_AVAILABLE

    print_banner(f"Token-Position Probing: {feature}")
    print(f"Model: {model_key}  Condition: {condition}")
    print(f"Backend: {'cuML (GPU)' if use_cuml else 'sklearn (CPU)'}")
    if not use_cuml:
        print(f"Workers: {n_workers}")

    # ---- Load metadata and labels from HDF5 -------------------------------
    with h5py.File(hdf5_path, "r") as f:
        position_grid = [int(x) for x in list(f["positions"][:])]
        n_layers = int(f.attrs["n_layers"])

        feature_columns = [c.decode() if isinstance(c, bytes) else c for c in f["feature_columns"][:]]
        feature_values = f["feature_values"][:]
        labels = f["labels"][:]

    feature_labels_df = pd.DataFrame(feature_values, columns=feature_columns)

    if feature not in feature_labels_df.columns:
        raise ValueError(f"Feature '{feature}' not in HDF5. Available: {feature_columns}")

    y = feature_labels_df[feature].values.astype(float)

    # ---- Restrict positions (optional) ------------------------------------
    # Example: position_min=-200, position_max=-1 probes only the last 200 tokens from end.
    if position_min is not None or position_max is not None:
        lo = position_min if position_min is not None else min(position_grid)
        hi = position_max if position_max is not None else max(position_grid)
        if lo > hi:
            raise ValueError(f"Invalid position range: min={lo} > max={hi}")
        before = len(position_grid)
        position_grid = [p for p in position_grid if (lo <= p <= hi)]
        if len(position_grid) == 0:
            raise ValueError(
                f"No positions remain after filtering to [{lo}, {hi}]. "
                f"Available range: [{min(position_grid)}, {max(position_grid)}]"
            )
        print(f"Position filter: [{lo}, {hi}]  (kept {len(position_grid)}/{before})")

    # ---- Splits (stratified on ground-truth medalist label) ---------------
    valid_mask = ~np.isnan(y)
    strat_labels = labels[valid_mask]
    splits_all = create_stratified_splits(int(valid_mask.sum()), strat_labels)

    valid_indices = np.where(valid_mask)[0]
    split_indices = {k: valid_indices[v] for k, v in splits_all.items()}

    n_positions = len(position_grid)
    total_probes = n_positions * n_layers
    print(f"Positions: {n_positions}  Layers: {n_layers}  Total probes: {total_probes}")

    if output_path is None:
        output_path = tp_config.get_tp_results_csv(feature, model_key, condition)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Sweep: outer=positions, inner=layers -----------------------------
    # With cuML: run layers sequentially (GPU contention makes joblib counterproductive).
    # With sklearn: parallelize layers via joblib.
    all_results = []
    pbar = tqdm(position_grid, desc="Positions")
    for pos in pbar:
        with h5py.File(hdf5_path, "r") as f:
            X_pos = f[f"activations/pos_{pos}"][:].astype(np.float32)

        if use_cuml:
            layer_results = [
                probe_single_layer(X_pos, y, layer, split_indices, pos, use_gpu=True)
                for layer in range(n_layers)
            ]
        else:
            layer_results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(probe_single_layer)(X_pos, y, layer, split_indices, pos, use_gpu=False)
                for layer in range(n_layers)
            )
        all_results.extend(layer_results)

        best_in_pos = max(layer_results, key=lambda r: r["test_accuracy"])
        pbar.set_postfix({
            "pos": pos,
            "best_layer": best_in_pos["layer"],
            "best_acc": f"{best_in_pos['test_accuracy']:.3f}",
        })

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results: {output_path}")
    print(f"Total probes: {len(df)}")

    best = df.loc[df["test_accuracy"].idxmax()]
    print(f"Best overall: position={int(best['position'])}, layer={int(best['layer'])}, "
          f"accuracy={best['test_accuracy']:.4f}")
    return df


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Probe a feature across all token positions and layers."
    )
    parser.add_argument("--model", "-m", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--condition", "-c", default="2_fewshot_cot_temp0")
    parser.add_argument(
        "--feature", "-f", required=True, choices=PROBE_FEATURES,
        help="Feature to probe (run once per feature).",
    )
    parser.add_argument(
        "--n-workers", "-w", type=int, default=tp_config.N_PROBE_WORKERS,
        help=f"Parallel workers per position (default {tp_config.N_PROBE_WORKERS})",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Override output CSV path")
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force sklearn (CPU) even when cuML is available.",
    )
    parser.add_argument(
        "--position-min",
        type=int,
        default=None,
        help="Minimum (most negative) token position to probe (from end). Example: -200.",
    )
    parser.add_argument(
        "--position-max",
        type=int,
        default=None,
        help="Maximum (closest to end) token position to probe (from end). Example: -1.",
    )
    args = parser.parse_args()

    probe_feature_across_positions(
        model_key=args.model,
        condition=args.condition,
        feature=args.feature,
        n_workers=args.n_workers,
        use_gpu=not args.no_gpu,
        position_min=args.position_min,
        position_max=args.position_max,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
