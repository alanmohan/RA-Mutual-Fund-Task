# -*- coding: utf-8 -*-
"""
Linear Probing Module (Mutual Funds)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import argparse
import json
import logging
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import warnings

try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.preprocessing import StandardScaler as cuStandardScaler

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
lp_config = importlib.util.spec_from_file_location("lp_config", str(_THIS_DIR / "lp_config.py"))
lp_config_mod = importlib.util.module_from_spec(lp_config)
lp_config.loader.exec_module(lp_config_mod)
lp_utils = importlib.util.spec_from_file_location("lp_utils", str(_THIS_DIR / "lp_utils.py"))
lp_utils_mod = importlib.util.module_from_spec(lp_utils)
lp_utils.loader.exec_module(lp_utils_mod)

MODELS = lp_config_mod.MODELS
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
PROBE_FEATURES = lp_config_mod.PROBE_FEATURES
CV_FOLDS = lp_config_mod.CV_FOLDS
PROBE_MAX_ITER = lp_config_mod.PROBE_MAX_ITER
PROBE_RANDOM_STATE = lp_config_mod.PROBE_RANDOM_STATE
PROBE_CS = lp_config_mod.PROBE_CS
load_activations = lp_utils_mod.load_activations
print_banner = lp_utils_mod.print_banner

# Split settings
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Statistical settings
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_THRESHOLD = 0.05


def setup_logging(output_dir: Path, model_name: str, condition: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"probe_{model_name}_{condition}_{timestamp}.log"

    logger = logging.getLogger(f"probe_{model_name}_{condition}")
    logger.setLevel(logging.INFO)

    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")

    return logger


@dataclass
class ProbeResult:
    """Results from a single probe (one layer, one feature)."""

    layer: int
    feature: str
    test_accuracy: float
    test_auc: float
    test_n_samples: int
    val_accuracy: float
    val_auc: float
    train_accuracy: float
    cv_mean: float
    cv_std: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    p_value: float
    is_significant: bool
    best_C: float
    n_train: int
    n_val: int
    n_test: int

    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "feature": self.feature,
            "test_accuracy": self.test_accuracy,
            "test_auc": self.test_auc,
            "test_n_samples": self.test_n_samples,
            "val_accuracy": self.val_accuracy,
            "val_auc": self.val_auc,
            "train_accuracy": self.train_accuracy,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "accuracy_ci_lower": self.accuracy_ci_lower,
            "accuracy_ci_upper": self.accuracy_ci_upper,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "best_C": self.best_C,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


@dataclass
class ProbeExperiment:
    """Full probing experiment results."""

    model_name: str
    condition: str
    results: List[ProbeResult]
    split_indices: Dict[str, np.ndarray] = None
    timestamp: str = None
    config: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dataframe(self) -> pd.DataFrame:
        records = [r.to_dict() for r in self.results]
        return pd.DataFrame(records)

    def get_layer_feature_matrix(self, metric: str = "test_accuracy") -> pd.DataFrame:
        df = self.to_dataframe()
        return df.pivot(index="layer", columns="feature", values=metric)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved probe results to {path}")

    @classmethod
    def load(cls, path: Path) -> "ProbeExperiment":
        with open(path, "rb") as f:
            return pickle.load(f)


def compute_binomial_ci(n_correct: int, n_total: int, confidence: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    if n_total == 0:
        return 0.5, 0.5

    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denominator

    return max(0, center - margin), min(1, center + margin)


def compute_p_value_vs_chance(n_correct: int, n_total: int, chance: float = 0.5) -> float:
    if n_total == 0:
        return 1.0
    result = stats.binomtest(n_correct, n_total, chance, alternative="greater")
    return result.pvalue


def bootstrap_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       n_iterations: int = BOOTSTRAP_ITERATIONS,
                       confidence: float = CONFIDENCE_LEVEL) -> Tuple[float, float, float]:
    n_samples = len(y_true)
    accuracies = []
    rng = np.random.RandomState(PROBE_RANDOM_STATE)

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        acc = accuracy_score(y_true[indices], y_pred[indices])
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    alpha = 1 - confidence
    ci_lower = np.percentile(accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))

    return mean_acc, ci_lower, ci_upper


def create_stratified_splits(
    n_samples: int,
    labels: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = PROBE_RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.arange(n_samples)

    train_val_ratio = train_ratio + val_ratio
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=random_state
    )

    val_ratio_adjusted = val_ratio / train_val_ratio
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def train_and_evaluate_probe(
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    best_C = None
    best_val_score = 0
    val_scores_all = {}

    if use_cuml:
        # Suppress cuML convergence warnings (they're informational, not errors)
        cuml_logger = logging.getLogger('cuml')
        original_level = cuml_logger.level
        cuml_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        
        import cupy as cp
        X_train_gpu = cp.asarray(X_train_scaled.astype(np.float32))
        y_train_gpu = cp.asarray(y_train.astype(np.float32))
        X_val_gpu = cp.asarray(X_val_scaled.astype(np.float32))
        y_val_gpu = cp.asarray(y_val.astype(np.float32))
        X_test_gpu = cp.asarray(X_test_scaled.astype(np.float32))

        for C in Cs:
            # Relaxed tolerance from 1e-4 to 1e-3 for faster convergence
            probe = cuLogisticRegression(C=C, max_iter=max_iter, tol=1e-3, solver="qn")
            probe.fit(X_train_gpu, y_train_gpu)
            val_pred = cp.asnumpy(probe.predict(X_val_gpu))
            val_acc = accuracy_score(y_val, val_pred)
            val_scores_all[C] = val_acc

            if val_acc > best_val_score:
                best_val_score = val_acc
                best_C = C

        # Relaxed tolerance from 1e-5 to 1e-3 for final model
        final_probe = cuLogisticRegression(
            C=best_C, max_iter=max_iter * 2, tol=1e-3, solver="qn"
        )
        final_probe.fit(X_train_gpu, y_train_gpu)
        
        # Restore original logging level
        cuml_logger.setLevel(original_level)

    else:
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
                val_scores_all[C] = val_acc

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

    if use_cuml:
        import cupy as cp

        y_train_pred = cp.asnumpy(final_probe.predict(X_train_gpu))
        y_val_pred = cp.asnumpy(final_probe.predict(X_val_gpu))
        y_test_pred = cp.asnumpy(final_probe.predict(X_test_gpu))

        y_train_prob = cp.asnumpy(final_probe.predict_proba(X_train_gpu))[:, 1]
        y_val_prob = cp.asnumpy(final_probe.predict_proba(X_val_gpu))[:, 1]
        y_test_prob = cp.asnumpy(final_probe.predict_proba(X_test_gpu))[:, 1]
    else:
        y_train_pred = final_probe.predict(X_train_scaled)
        y_val_pred = final_probe.predict(X_val_scaled)
        y_test_pred = final_probe.predict(X_test_scaled)

        y_train_prob = final_probe.predict_proba(X_train_scaled)[:, 1]
        y_val_prob = final_probe.predict_proba(X_val_scaled)[:, 1]
        y_test_prob = final_probe.predict_proba(X_test_scaled)[:, 1]

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

    val_scores_list = list(val_scores_all.values())
    val_scores_std = np.std(val_scores_list) if len(val_scores_list) > 1 else 0.0

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "cv_mean": best_val_score,
        "cv_std": val_scores_std,
        "best_C": best_C,
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


def probe_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    split_indices: Dict[str, np.ndarray],
    feature_name: str = "target",
    logger: logging.Logger = None,
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
            best_C=1.0,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        )

    metrics = train_and_evaluate_probe(
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


def run_probing_experiment(
    activations: np.ndarray,
    feature_labels: pd.DataFrame,
    ground_truth_labels: np.ndarray,
    model_name: str,
    condition: str,
    features_to_probe: List[str] = None,
    output_dir: Path = None,
    logger: logging.Logger = None,
) -> ProbeExperiment:
    n_samples, n_layers, d_model = activations.shape

    if features_to_probe is None:
        features_to_probe = PROBE_FEATURES

    if output_dir is None:
        output_dir = PROBE_RESULTS_DIR

    if logger is None:
        logger = setup_logging(output_dir, model_name, condition)

    print_banner(f"Probing Experiment: {model_name} / {condition}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Condition: {condition}")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Layers: {n_layers}")
    logger.info(f"D_model: {d_model}")
    logger.info(f"Features: {features_to_probe}")
    logger.info(f"Split: {TRAIN_RATIO*100:.0f}/{VAL_RATIO*100:.0f}/{TEST_RATIO*100:.0f} (train/val/test)")
    logger.info("C selection: Validation set (fast mode)")
    logger.info("GPU acceleration: ENABLED (cuML)") if CUML_AVAILABLE else logger.info(
        "GPU acceleration: DISABLED (CPU sklearn)"
    )

    # Create stratified splits using ground truth labels
    logger.info("Creating stratified train/val/test splits...")
    gt_clean = ground_truth_labels.copy()
    # For stratification, drop NaNs by temporary fill (will be filtered later anyway)
    gt_for_split = np.where(np.isnan(gt_clean), 0, gt_clean).astype(int)
    split_indices = create_stratified_splits(n_samples, gt_for_split)

    logger.info(f"  Train: {len(split_indices['train'])} samples")
    logger.info(f"  Val: {len(split_indices['val'])} samples")
    logger.info(f"  Test: {len(split_indices['test'])} samples")

    results = []
    total_probes = n_layers * len(features_to_probe)
    pbar = tqdm(total=total_probes, desc="Probing")

    for feature in features_to_probe:
        if feature in feature_labels.columns:
            labels = feature_labels[feature].values
        elif feature == "medalist_f1_higher":
            labels = ground_truth_labels
        else:
            logger.warning(f"Feature '{feature}' not found, skipping")
            continue

        logger.info(f"Probing feature: {feature}")

        for layer in range(n_layers):
            result = probe_layer(
                activations=activations,
                labels=labels,
                layer=layer,
                split_indices=split_indices,
                feature_name=feature,
                logger=logger,
            )
            results.append(result)

            if result.is_significant and result.test_accuracy > 0.55:
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
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "cv_folds": CV_FOLDS,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": CONFIDENCE_LEVEL,
            "significance_threshold": SIGNIFICANCE_THRESHOLD,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "d_model": d_model,
        },
    )

    return experiment


def export_results_to_csv(experiment: ProbeExperiment, output_dir: Path):
    df = experiment.to_dataframe()

    csv_path = output_dir / f"probe_results_{experiment.model_name}_{experiment.condition}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    matrix_acc = experiment.get_layer_feature_matrix("test_accuracy")
    matrix_path = output_dir / f"probe_matrix_accuracy_{experiment.model_name}_{experiment.condition}.csv"
    matrix_acc.to_csv(matrix_path)
    print(f"Saved accuracy matrix to {matrix_path}")

    matrix_auc = experiment.get_layer_feature_matrix("test_auc")
    matrix_auc_path = output_dir / f"probe_matrix_auc_{experiment.model_name}_{experiment.condition}.csv"
    matrix_auc.to_csv(matrix_auc_path)
    print(f"Saved AUC matrix to {matrix_auc_path}")

    summary = df.groupby("feature").agg(
        {
            "test_accuracy": ["mean", "std", "max"],
            "test_auc": ["mean", "std", "max"],
            "is_significant": "sum",
            "p_value": "min",
        }
    ).round(4)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary_path = output_dir / f"probe_summary_{experiment.model_name}_{experiment.condition}.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")

    best_layers = []
    for feature in df["feature"].unique():
        feat_df = df[df["feature"] == feature]
        best_row = feat_df.loc[feat_df["test_accuracy"].idxmax()]
        best_layers.append(
            {
                "feature": feature,
                "best_layer": int(best_row["layer"]),
                "test_accuracy": best_row["test_accuracy"],
                "test_auc": best_row["test_auc"],
                "ci_lower": best_row["accuracy_ci_lower"],
                "ci_upper": best_row["accuracy_ci_upper"],
                "p_value": best_row["p_value"],
                "is_significant": best_row["is_significant"],
            }
        )

    best_df = pd.DataFrame(best_layers)
    best_path = output_dir / f"probe_best_layers_{experiment.model_name}_{experiment.condition}.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Saved best layers to {best_path}")

    return {
        "detailed": csv_path,
        "matrix_accuracy": matrix_path,
        "matrix_auc": matrix_auc_path,
        "summary": summary_path,
        "best_layers": best_path,
    }


def export_config(experiment: ProbeExperiment, output_dir: Path):
    config_path = output_dir / f"probe_config_{experiment.model_name}_{experiment.condition}.json"

    config = {
        "model_name": experiment.model_name,
        "condition": experiment.condition,
        "timestamp": experiment.timestamp,
        **experiment.config,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved config to {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Run linear probing on activations")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to probe",
    )
    parser.add_argument(
        "--condition", "-c", type=str, required=True, help="Experimental condition"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="Features to probe (default: all)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None, help="Output directory for results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROBE_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.model, args.condition)

    activation_path = ACTIVATIONS_DIR / f"{args.model}_{args.condition}_activations.npz"

    if not activation_path.exists():
        logger.error(f"Activations not found at {activation_path}")
        logger.error("Run extract_activations.py first.")
        return

    logger.info(f"Loading activations from {activation_path}")
    data = load_activations(activation_path)

    activations = data["activations"]
    feature_labels = data["feature_labels"]
    labels = data["labels"]

    logger.info(f"Loaded activations: {activations.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Feature labels columns: {feature_labels.columns.tolist()}")

    experiment = run_probing_experiment(
        activations=activations,
        feature_labels=feature_labels,
        ground_truth_labels=labels,
        model_name=args.model,
        condition=args.condition,
        features_to_probe=args.features,
        output_dir=output_dir,
        logger=logger,
    )

    pickle_path = output_dir / f"probe_{args.model}_{args.condition}.pkl"
    experiment.save(pickle_path)

    logger.info("Exporting results to CSV...")
    export_results_to_csv(experiment, output_dir)

    export_config(experiment, output_dir)

    print_banner("Probing Complete!")

    df = experiment.to_dataframe()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nBest Layer per Feature (Test Set):")
    for feature in df["feature"].unique():
        feat_df = df[df["feature"] == feature]
        best_row = feat_df.loc[feat_df["test_accuracy"].idxmax()]
        sig_marker = "*" if best_row["is_significant"] else ""
        logger.info(
            f"  {feature}:"
            f" Layer {int(best_row['layer'])}: "
            f"acc={best_row['test_accuracy']:.3f} "
            f"[{best_row['accuracy_ci_lower']:.3f}, {best_row['accuracy_ci_upper']:.3f}] "
            f"AUC={best_row['test_auc']:.3f} "
            f"p={best_row['p_value']:.4f} {sig_marker}"
        )

    n_significant = df["is_significant"].sum()
    total_probes = len(df)
    logger.info(
        f"\nSignificant probes: {n_significant}/{total_probes} "
        f"({100*n_significant/total_probes:.1f}%)"
    )

    logger.info("\nMean Test Accuracy by Layer (top 5):")
    layer_means = df.groupby("layer")["test_accuracy"].mean().sort_values(ascending=False)
    for layer, acc in layer_means.head().items():
        logger.info(f"  Layer {int(layer)}: {acc:.3f}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
