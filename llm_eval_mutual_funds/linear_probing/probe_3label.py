# -*- coding: utf-8 -*-
"""
3-Label Linear Probing: fund1-better (1) / fund2-better (0) / too-close-to-tell (2)

Instead of forcing a binary label when two fund values are nearly identical,
this script introduces a third class (label=2) for pairs whose absolute
difference falls within a narrow band around zero:

    |value_1 - value_2| < THRESHOLD_FACTOR * std(all non-NaN differences)

where THRESHOLD_FACTOR defaults to 0.1 (configurable via --threshold).

For categorical features (load, ntf) the third label captures same-value
pairs (both Y/Y or both N/N). For medalist, ties remain NaN (excluded).

Everything else (activation loading, splits, probe training, CSV export)
reuses the existing pipeline in probe.py and lp_utils.py without modifying
those files.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import json
import logging
from tqdm import tqdm
import importlib.util

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import warnings

# Optional cuML acceleration (multiclass logistic regression)
try:
    import cuml  # noqa: F401
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.preprocessing import StandardScaler as cuStandardScaler

    CUML_AVAILABLE = True
except Exception:
    CUML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import existing modules via importlib (same pattern as probe.py)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()

lp_config_spec = importlib.util.spec_from_file_location("lp_config", str(_THIS_DIR / "lp_config.py"))
lp_config_mod = importlib.util.module_from_spec(lp_config_spec)
lp_config_spec.loader.exec_module(lp_config_mod)

lp_utils_spec = importlib.util.spec_from_file_location("lp_utils", str(_THIS_DIR / "lp_utils.py"))
lp_utils_mod = importlib.util.module_from_spec(lp_utils_spec)
lp_utils_spec.loader.exec_module(lp_utils_mod)

probe_spec = importlib.util.spec_from_file_location("probe", str(_THIS_DIR / "probe.py"))
probe_mod = importlib.util.module_from_spec(probe_spec)
# Ensure a single canonical module identity for pickling.
# If `probe` is already imported elsewhere (or in interactive environments),
# we overwrite it with this loaded module so `pickle` sees the same object.
sys.modules["probe"] = probe_mod
probe_spec.loader.exec_module(probe_mod)

# Re-use configs and helpers
MODELS = lp_config_mod.MODELS
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
PROBE_FEATURES = lp_config_mod.PROBE_FEATURES
PROBE_MAX_ITER = lp_config_mod.PROBE_MAX_ITER
PROBE_RANDOM_STATE = lp_config_mod.PROBE_RANDOM_STATE
PROBE_CS = lp_config_mod.PROBE_CS
DATA_PATH = lp_config_mod.DATA_PATH
TRAIN_RATIO = lp_config_mod.TRAIN_RATIO
VAL_RATIO = lp_config_mod.VAL_RATIO
TEST_RATIO = lp_config_mod.TEST_RATIO

load_activations = lp_utils_mod.load_activations
get_activation_path = lp_utils_mod.get_activation_path
print_banner = lp_utils_mod.print_banner
MEDALIST_HIERARCHY = lp_utils_mod.MEDALIST_HIERARCHY
get_medalist_value = lp_utils_mod.get_medalist_value

create_stratified_splits = probe_mod.create_stratified_splits
compute_binomial_ci = probe_mod.compute_binomial_ci
compute_p_value_vs_chance = probe_mod.compute_p_value_vs_chance
bootstrap_accuracy = probe_mod.bootstrap_accuracy
setup_logging = probe_mod.setup_logging
ProbeResult = probe_mod.ProbeResult
ProbeExperiment = probe_mod.ProbeExperiment

BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_THRESHOLD = 0.05

DEFAULT_THRESHOLD_FACTOR = 0.1

# Features where "lower is better for fund 1" (strict < means label 1)
_LOWER_IS_BETTER = {
    "expense_ratio_f1_lower",
    "stdev_f1_lower",
    "beta_f1_lower",
    "turnover_f1_lower",
}
# Features where "higher is better for fund 1"
_HIGHER_IS_BETTER = {
    "sharpe_f1_higher",
    "return_3yr_f1_higher",
    "tenure_f1_longer",
    "assets_f1_higher",
}
# Date feature: earlier inception = "older"
_DATE_FEATURES = {"inception_f1_older"}
# Categorical features
_CATEGORICAL_FEATURES = {"load_f1_no", "ntf_f1_yes"}


# ============================================================================
# 3-label creation
# ============================================================================

def create_3label_feature_labels(
    df: pd.DataFrame,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
) -> pd.DataFrame:
    """
    Like create_feature_labels but with three classes:
      0 = fund 2 is better (or fund 1 does NOT satisfy the criterion)
      1 = fund 1 is better
      2 = too close to tell (|diff| < threshold_factor * sd)

    For categorical features (load, ntf): label 2 when both funds have the
    same value.  Medalist keeps the existing NaN-for-ties convention.
    """
    labels = pd.DataFrame(index=df.index)

    def safe_float(val):
        try:
            if pd.isna(val):
                return np.nan
            return float(str(val).replace(",", "").strip())
        except Exception:
            return np.nan

    def safe_years(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(val).lower().replace("years", "").replace("year", "").strip()
            return float(s)
        except Exception:
            return np.nan

    def safe_date(val):
        try:
            if pd.isna(val):
                return pd.NaT
            return pd.to_datetime(val, errors="coerce")
        except Exception:
            return pd.NaT

    def safe_yes_no(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s == "Y":
            return 1
        if s == "N":
            return 0
        return np.nan

    def _numeric_3label(v1_arr, v2_arr, higher_is_better: bool) -> list:
        diff = v1_arr - v2_arr
        valid = ~np.isnan(diff)
        sd = np.nanstd(diff) if valid.sum() > 1 else 0.0
        threshold = threshold_factor * sd
        out = []
        for a, b, d in zip(v1_arr, v2_arr, diff):
            if np.isnan(a) or np.isnan(b):
                out.append(np.nan)
            elif abs(d) < threshold:
                out.append(2)
            elif higher_is_better:
                out.append(1 if d > 0 else 0)
            else:
                out.append(1 if d < 0 else 0)
        return out

    # --- Numerical features ---
    er_1 = df["Expense Ratio - Net_1"].apply(safe_float).values
    er_2 = df["Expense Ratio - Net_2"].apply(safe_float).values
    labels["expense_ratio_f1_lower"] = _numeric_3label(er_1, er_2, higher_is_better=False)

    sh_1 = df["3 Year Sharpe Ratio_1"].apply(safe_float).values
    sh_2 = df["3 Year Sharpe Ratio_2"].apply(safe_float).values
    labels["sharpe_f1_higher"] = _numeric_3label(sh_1, sh_2, higher_is_better=True)

    sd_1 = df["Standard Deviation_1"].apply(safe_float).values
    sd_2 = df["Standard Deviation_2"].apply(safe_float).values
    labels["stdev_f1_lower"] = _numeric_3label(sd_1, sd_2, higher_is_better=False)

    r3_1 = df["3 Yr_1"].apply(safe_float).values
    r3_2 = df["3 Yr_2"].apply(safe_float).values
    labels["return_3yr_f1_higher"] = _numeric_3label(r3_1, r3_2, higher_is_better=True)

    b_1 = df["Beta_1"].apply(safe_float).values
    b_2 = df["Beta_2"].apply(safe_float).values
    labels["beta_f1_lower"] = _numeric_3label(b_1, b_2, higher_is_better=False)

    t_1 = df["Manager Tenure_1"].apply(safe_years).values
    t_2 = df["Manager Tenure_2"].apply(safe_years).values
    labels["tenure_f1_longer"] = _numeric_3label(t_1, t_2, higher_is_better=True)

    # Inception date: convert to ordinal for numeric comparison
    d_1_raw = df["Inception Date_1"].apply(safe_date)
    d_2_raw = df["Inception Date_2"].apply(safe_date)
    d_1 = np.array([d.toordinal() if not pd.isna(d) else np.nan for d in d_1_raw], dtype=float)
    d_2 = np.array([d.toordinal() if not pd.isna(d) else np.nan for d in d_2_raw], dtype=float)
    labels["inception_f1_older"] = _numeric_3label(d_1, d_2, higher_is_better=False)

    a_1 = df["Assets (Millions)_1"].apply(safe_float).values
    a_2 = df["Assets (Millions)_2"].apply(safe_float).values
    labels["assets_f1_higher"] = _numeric_3label(a_1, a_2, higher_is_better=True)

    tr_1 = df["Turnover Rates_1"].apply(safe_float).values
    tr_2 = df["Turnover Rates_2"].apply(safe_float).values
    labels["turnover_f1_lower"] = _numeric_3label(tr_1, tr_2, higher_is_better=False)

    # --- Categorical: load ---
    l_1 = df["Load (Y/N)_1"].apply(safe_yes_no)
    l_2 = df["Load (Y/N)_2"].apply(safe_yes_no)
    load_labels = []
    for a, b in zip(l_1, l_2):
        if pd.isna(a) or pd.isna(b):
            load_labels.append(np.nan)
        elif a == b:
            load_labels.append(2)
        else:
            load_labels.append(1 if (a == 0 and b == 1) else 0)
    labels["load_f1_no"] = load_labels

    # --- Categorical: ntf ---
    n_1 = df["NTF_1"].apply(safe_yes_no)
    n_2 = df["NTF_2"].apply(safe_yes_no)
    ntf_labels = []
    for a, b in zip(n_1, n_2):
        if pd.isna(a) or pd.isna(b):
            ntf_labels.append(np.nan)
        elif a == b:
            ntf_labels.append(2)
        else:
            ntf_labels.append(1 if (a == 1 and b == 0) else 0)
    labels["ntf_f1_yes"] = ntf_labels

    # --- Medalist: keep existing NaN-for-ties ---
    medalist_labels = []
    for m1, m2 in zip(df["Medalist_1"], df["Medalist_2"]):
        v1 = get_medalist_value(m1)
        v2 = get_medalist_value(m2)
        if v1 < 0 or v2 < 0 or v1 == v2:
            medalist_labels.append(np.nan)
        else:
            medalist_labels.append(1 if v1 > v2 else 0)
    labels["medalist_f1_higher"] = medalist_labels

    return labels


def _label_distribution_summary(labels_col: np.ndarray, feature: str) -> str:
    valid = labels_col[~np.isnan(labels_col)]
    n = len(valid)
    if n == 0:
        return f"  {feature}: no valid samples"
    n0 = (valid == 0).sum()
    n1 = (valid == 1).sum()
    n2 = (valid == 2).sum()
    return (
        f"  {feature}: n={n}  "
        f"label0={n0} ({100*n0/n:.1f}%)  "
        f"label1={n1} ({100*n1/n:.1f}%)  "
        f"label2(close)={n2} ({100*n2/n:.1f}%)"
    )


# ============================================================================
# 3-class probe training (multinomial logistic regression)
# ============================================================================

def train_and_evaluate_probe_3class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = PROBE_MAX_ITER,
    Cs: List[float] = None,
    random_state: int = PROBE_RANDOM_STATE,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    if Cs is None:
        Cs = PROBE_CS

    use_cuml = use_gpu and CUML_AVAILABLE
    best_C = None
    best_val_score = 0

    # -----------------------------------------------------------------------
    # GPU path (cuML) with fallback to sklearn if anything fails.
    # -----------------------------------------------------------------------
    if use_cuml:
        try:
            import cupy as cp

            scaler = cuStandardScaler()
            X_train_scaled = scaler.fit_transform(cp.asarray(X_train.astype(np.float32)))
            X_val_scaled = scaler.transform(cp.asarray(X_val.astype(np.float32)))
            X_test_scaled = scaler.transform(cp.asarray(X_test.astype(np.float32)))

            y_train_gpu = cp.asarray(y_train.astype(int))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*failed to converge.*")

                for C in Cs:
                    probe = cuLogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        tol=1e-3,
                        solver="qn",
                    )
                    probe.fit(X_train_scaled, y_train_gpu)
                    val_pred_gpu = probe.predict(X_val_scaled)
                    val_pred = cp.asnumpy(val_pred_gpu).astype(int)
                    val_acc = accuracy_score(y_val, val_pred)

                    if val_acc > best_val_score:
                        best_val_score = val_acc
                        best_C = C

                final_probe = cuLogisticRegression(
                    C=best_C,
                    max_iter=max_iter * 2,
                    tol=1e-4,
                    solver="qn",
                )
                final_probe.fit(X_train_scaled, y_train_gpu)

            y_train_pred = cp.asnumpy(final_probe.predict(X_train_scaled)).astype(int)
            y_val_pred = cp.asnumpy(final_probe.predict(X_val_scaled)).astype(int)
            y_test_pred = cp.asnumpy(final_probe.predict(X_test_scaled)).astype(int)
            y_test_prob = cp.asnumpy(final_probe.predict_proba(X_test_scaled))

        except Exception as e:
            # If cuML multiclass isn't supported in the current environment,
            # fall back to sklearn.
            warnings.warn(f"cuML 3-class probe failed ({e}); falling back to sklearn.")
            use_cuml = False

    # -----------------------------------------------------------------------
    # CPU path (sklearn)
    # -----------------------------------------------------------------------
    if not use_cuml:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*failed to converge.*")

            for C in Cs:
                probe = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight="balanced",
                    solver="saga",
                    tol=1e-3,
                    n_jobs=-1,
                )
                probe.fit(X_train_scaled, y_train)
                val_pred = probe.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)

                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_C = C

            final_probe = LogisticRegression(
                C=best_C,
                max_iter=max_iter * 2,
                random_state=random_state,
                class_weight="balanced",
                solver="saga",
                tol=1e-4,
                n_jobs=-1,
            )
            final_probe.fit(X_train_scaled, y_train)

        y_train_pred = final_probe.predict(X_train_scaled)
        y_val_pred = final_probe.predict(X_val_scaled)
        y_test_pred = final_probe.predict(X_test_scaled)
        y_test_prob = final_probe.predict_proba(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    def safe_auc(y_true, y_prob):
        classes_present = np.unique(y_true)
        if len(classes_present) < 2:
            return 0.5
        try:
            return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except ValueError:
            return 0.5

    test_auc = safe_auc(y_test, y_test_prob)

    n_correct = (y_test_pred == y_test).sum()
    n_test = len(y_test)
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    chance = 1.0 / n_classes

    ci_lower, ci_upper = compute_binomial_ci(n_correct, n_test)
    p_value = compute_p_value_vs_chance(n_correct, n_test, chance=chance)

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "cv_mean": best_val_score,
        "best_C": best_C,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": p_value < SIGNIFICANCE_THRESHOLD,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": n_test,
        "n_classes": n_classes,
        "chance_level": chance,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }


def probe_layer_3class(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    split_indices: Dict[str, np.ndarray],
    feature_name: str = "target",
    logger: logging.Logger = None,
    use_gpu: bool = True,
) -> ProbeResult:
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
            layer=layer, feature=feature_name,
            test_accuracy=0.5, test_auc=0.5, test_n_samples=len(X_test),
            val_accuracy=0.5, val_auc=0.5, train_accuracy=0.5,
            cv_mean=0.5, cv_std=0.0,
            accuracy_ci_lower=0.5, accuracy_ci_upper=0.5,
            p_value=1.0, is_significant=False, best_C=1.0,
            n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
        )

    # Ensure each split has at least 2 classes; if not, fall back
    unique_train = np.unique(y_train.astype(int))
    unique_test = np.unique(y_test.astype(int))
    if len(unique_train) < 2:
        if logger:
            logger.warning(f"Layer {layer}, {feature_name}: Only 1 class in train")
        return ProbeResult(
            layer=layer, feature=feature_name,
            test_accuracy=0.5, test_auc=0.5, test_n_samples=len(X_test),
            val_accuracy=0.5, val_auc=0.5, train_accuracy=0.5,
            cv_mean=0.5, cv_std=0.0,
            accuracy_ci_lower=0.5, accuracy_ci_upper=0.5,
            p_value=1.0, is_significant=False, best_C=1.0,
            n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
        )

    metrics = train_and_evaluate_probe_3class(
        X_train, y_train.astype(int),
        X_val, y_val.astype(int),
        X_test, y_test.astype(int),
        use_gpu=use_gpu,
    )

    return ProbeResult(
        layer=layer,
        feature=feature_name,
        test_accuracy=metrics["test_accuracy"],
        test_auc=metrics["test_auc"],
        test_n_samples=metrics["n_test"],
        val_accuracy=metrics["val_accuracy"],
        val_auc=0.5,
        train_accuracy=metrics["train_accuracy"],
        cv_mean=metrics["cv_mean"],
        cv_std=0.0,
        accuracy_ci_lower=metrics["ci_lower"],
        accuracy_ci_upper=metrics["ci_upper"],
        p_value=metrics["p_value"],
        is_significant=metrics["is_significant"],
        best_C=metrics["best_C"],
        n_train=metrics["n_train"],
        n_val=metrics["n_val"],
        n_test=metrics["n_test"],
    )


# ============================================================================
# Experiment runner
# ============================================================================

def run_3label_probing(
    activations: np.ndarray,
    feature_labels_3: pd.DataFrame,
    ground_truth_labels: np.ndarray,
    model_name: str,
    condition: str,
    threshold_factor: float,
    features_to_probe: List[str] = None,
    output_dir: Path = None,
    logger: logging.Logger = None,
    use_gpu: bool = True,
) -> ProbeExperiment:
    n_samples, n_layers, d_model = activations.shape

    if features_to_probe is None:
        features_to_probe = PROBE_FEATURES
    if output_dir is None:
        output_dir = PROBE_RESULTS_DIR
    if logger is None:
        logger = setup_logging(output_dir, model_name, condition)

    print_banner(f"3-Label Probing: {model_name} / {condition}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Condition: {condition}")
    logger.info(f"Threshold factor: {threshold_factor}")
    logger.info(f"Samples: {n_samples}, Layers: {n_layers}, D_model: {d_model}")
    logger.info(f"Features: {features_to_probe}")
    logger.info(
        f"Split: {TRAIN_RATIO*100:.0f}/{VAL_RATIO*100:.0f}/{TEST_RATIO*100:.0f}"
    )

    # Label distribution
    logger.info("Label distribution (3-label):")
    for feat in features_to_probe:
        if feat in feature_labels_3.columns:
            logger.info(_label_distribution_summary(feature_labels_3[feat].values, feat))

    gt_for_split = np.where(np.isnan(ground_truth_labels), 0, ground_truth_labels).astype(int)
    split_indices = create_stratified_splits(n_samples, gt_for_split)

    logger.info(f"  Train: {len(split_indices['train'])} samples")
    logger.info(f"  Val:   {len(split_indices['val'])} samples")
    logger.info(f"  Test:  {len(split_indices['test'])} samples")

    results = []
    total_probes = n_layers * len(features_to_probe)
    pbar = tqdm(total=total_probes, desc="3-Label Probing")

    for feature in features_to_probe:
        if feature in feature_labels_3.columns:
            labels = feature_labels_3[feature].values
        elif feature == "medalist_f1_higher":
            labels = ground_truth_labels
        else:
            logger.warning(f"Feature '{feature}' not found, skipping")
            continue

        logger.info(f"Probing feature: {feature}")

        for layer in range(n_layers):
            result = probe_layer_3class(
                activations=activations,
                labels=labels,
                layer=layer,
                split_indices=split_indices,
                feature_name=feature,
                logger=logger,
                use_gpu=use_gpu,
            )
            results.append(result)

            if result.is_significant and result.test_accuracy > 0.40:
                logger.info(
                    f"  Layer {layer}: acc={result.test_accuracy:.3f} "
                    f"[{result.accuracy_ci_lower:.3f}, {result.accuracy_ci_upper:.3f}] "
                    f"p={result.p_value:.4f} *"
                )
            pbar.update(1)

    pbar.close()

    experiment = ProbeExperiment(
        model_name=model_name,
        condition=condition,
        results=results,
        split_indices=split_indices,
        config={
            "probe_type": "3-label",
            "threshold_factor": threshold_factor,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "d_model": d_model,
        },
    )
    return experiment


def export_3label_results(experiment: ProbeExperiment, output_dir: Path, threshold_factor: float):
    tag = f"3label_t{threshold_factor}"
    df = experiment.to_dataframe()

    csv_path = output_dir / f"probe_results_{tag}_{experiment.model_name}_{experiment.condition}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    matrix_acc = experiment.get_layer_feature_matrix("test_accuracy")
    matrix_path = output_dir / f"probe_matrix_accuracy_{tag}_{experiment.model_name}_{experiment.condition}.csv"
    matrix_acc.to_csv(matrix_path)
    print(f"Saved accuracy matrix to {matrix_path}")

    best_layers = []
    for feature in df["feature"].unique():
        feat_df = df[df["feature"] == feature]
        best_row = feat_df.loc[feat_df["test_accuracy"].idxmax()]
        best_layers.append({
            "feature": feature,
            "best_layer": int(best_row["layer"]),
            "test_accuracy": best_row["test_accuracy"],
            "test_auc": best_row["test_auc"],
            "ci_lower": best_row["accuracy_ci_lower"],
            "ci_upper": best_row["accuracy_ci_upper"],
            "p_value": best_row["p_value"],
            "is_significant": best_row["is_significant"],
        })

    best_df = pd.DataFrame(best_layers)
    best_path = output_dir / f"probe_best_layers_{tag}_{experiment.model_name}_{experiment.condition}.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Saved best layers to {best_path}")

    config_path = output_dir / f"probe_config_{tag}_{experiment.model_name}_{experiment.condition}.json"
    config = {
        "model_name": experiment.model_name,
        "condition": experiment.condition,
        "timestamp": experiment.timestamp,
        "probe_type": "3-label",
        "threshold_factor": threshold_factor,
        **{k: v for k, v in experiment.config.items() if k not in ("probe_type", "threshold_factor")},
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    return {"detailed": csv_path, "matrix_accuracy": matrix_path, "best_layers": best_path, "config": config_path}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3-Label linear probing (fund1-better / fund2-better / too-close)"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        choices=list(MODELS.keys()), help="Model to probe",
    )
    parser.add_argument(
        "--condition", "-c", type=str, required=True,
        help="Experimental condition (e.g. 2_fewshot_cot_temp0)",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD_FACTOR,
        help=f"Fraction of std(diff) used as closeness band (default: {DEFAULT_THRESHOLD_FACTOR})",
    )
    parser.add_argument(
        "--features", type=str, nargs="+", default=None,
        help="Features to probe (default: all)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory (default: data/probe_results/3label)",
    )
    parser.add_argument(
        "--token-position", type=int, default=None,
        help="Token position used for extraction (default: -1, last token)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable cuML GPU acceleration (force sklearn CPU).",
    )

    args = parser.parse_args()

    default_out = PROBE_RESULTS_DIR / "3label"
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.model, args.condition)

    use_gpu = (not args.no_gpu) and CUML_AVAILABLE
    if not CUML_AVAILABLE:
        logger.info("cuML not available: using sklearn CPU.")
    elif args.no_gpu:
        logger.info("GPU disabled via --no-gpu: using sklearn CPU.")
    else:
        logger.info("Using cuML GPU acceleration for 3-class probes.")

    token_position = args.token_position if args.token_position is not None else -1
    activation_path = get_activation_path(ACTIVATIONS_DIR, args.model, args.condition, token_position=token_position)

    if not activation_path.exists():
        logger.error(f"Activations not found at {activation_path}")
        logger.error("Run extract_activations.py first.")
        return

    logger.info(f"Loading activations from {activation_path}")
    data = load_activations(activation_path)

    activations = data["activations"]
    original_feature_labels = data["feature_labels"]
    gt_labels = data["labels"]

    logger.info(f"Loaded activations: {activations.shape}")

    # Build 3-label feature labels from the original CSV
    logger.info(f"Loading dataset from {DATA_PATH} to create 3-label features (threshold={args.threshold})...")
    raw_df = pd.read_csv(DATA_PATH)

    # The activations were extracted from a (possibly sampled) subset.
    # We need to recreate the same subset to compute 3-label features.
    # The sample indices are stored in the activations file.
    sample_indices = data["sample_indices"]
    n_act = len(activations)

    if n_act < len(raw_df):
        _EXTRACTION_RANDOM_STATE = getattr(lp_config_mod, "EXTRACTION_RANDOM_STATE", 42)
        _EXTRACTION_SAMPLE_SIZE = getattr(lp_config_mod, "EXTRACTION_SAMPLE_SIZE", 5000)
        logger.info(f"Activations have {n_act} samples (dataset has {len(raw_df)}). "
                     f"Re-sampling with EXTRACTION_RANDOM_STATE={_EXTRACTION_RANDOM_STATE}.")
        raw_df_sampled = raw_df.sample(
            n=min(_EXTRACTION_SAMPLE_SIZE, len(raw_df)),
            random_state=_EXTRACTION_RANDOM_STATE,
        ).reset_index(drop=True)
    else:
        raw_df_sampled = raw_df.copy()

    if len(raw_df_sampled) != n_act:
        logger.warning(
            f"Sample count mismatch: activations={n_act}, re-sampled CSV={len(raw_df_sampled)}. "
            f"Using first {n_act} rows."
        )
        raw_df_sampled = raw_df_sampled.iloc[:n_act]

    feature_labels_3 = create_3label_feature_labels(raw_df_sampled, threshold_factor=args.threshold)

    logger.info("3-label distribution:")
    for feat in (args.features or PROBE_FEATURES):
        if feat in feature_labels_3.columns:
            logger.info(_label_distribution_summary(feature_labels_3[feat].values, feat))

    experiment = run_3label_probing(
        activations=activations,
        feature_labels_3=feature_labels_3,
        ground_truth_labels=gt_labels,
        model_name=args.model,
        condition=args.condition,
        threshold_factor=args.threshold,
        features_to_probe=args.features,
        output_dir=output_dir,
        logger=logger,
        use_gpu=use_gpu,
    )

    tag = f"3label_t{args.threshold}"
    pickle_path = output_dir / f"probe_{tag}_{args.model}_{args.condition}.pkl"
    experiment.save(pickle_path)

    export_3label_results(experiment, output_dir, args.threshold)

    print_banner("3-Label Probing Complete!")

    df = experiment.to_dataframe()
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (3-LABEL)")
    logger.info("=" * 60)
    logger.info(f"Threshold factor: {args.threshold}")

    logger.info("\nBest Layer per Feature (Test Set):")
    for feature in df["feature"].unique():
        feat_df = df[df["feature"] == feature]
        best_row = feat_df.loc[feat_df["test_accuracy"].idxmax()]
        sig = "*" if best_row["is_significant"] else ""
        logger.info(
            f"  {feature}: Layer {int(best_row['layer'])}: "
            f"acc={best_row['test_accuracy']:.3f} "
            f"[{best_row['accuracy_ci_lower']:.3f}, {best_row['accuracy_ci_upper']:.3f}] "
            f"AUC={best_row['test_auc']:.3f} "
            f"p={best_row['p_value']:.4f} {sig}"
        )

    n_sig = df["is_significant"].sum()
    total = len(df)
    logger.info(f"\nSignificant probes: {n_sig}/{total} ({100*n_sig/total:.1f}%)")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
