# -*- coding: utf-8 -*-
"""
Plot linear vs nonlinear probe comparison and nonlinear-only views.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

FEATURE_LABELS = {
    "expense_ratio_f1_lower": "Expense Ratio (Lower)",
    "sharpe_f1_higher": "Sharpe (Higher)",
    "stdev_f1_lower": "Stdev (Lower)",
    "return_3yr_f1_higher": "3Y Return (Higher)",
    "beta_f1_lower": "Beta (Lower)",
    "tenure_f1_longer": "Tenure (Longer)",
    "inception_f1_older": "Inception (Older)",
    "assets_f1_higher": "Assets (Higher)",
    "turnover_f1_lower": "Turnover (Lower)",
    "load_f1_no": "Load (No)",
    "ntf_f1_yes": "NTF (Yes)",
    "medalist_f1_higher": "Medalist (Target)",
}

FEATURE_ORDER = [
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
    "medalist_f1_higher",
]


def _get_exp(obj):
    """Get experiment; accept object with .to_dataframe() or ProbeExperiment."""
    if hasattr(obj, "to_dataframe"):
        return obj.to_dataframe()
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError("Need ProbeExperiment or DataFrame")


def plot_linear_vs_nonlinear_by_feature(
    linear_df: Optional[pd.DataFrame],
    nonlinear_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    condition: str,
    control_df: Optional[pd.DataFrame] = None,
):
    """Bar chart: for each feature, best-layer linear vs nonlinear vs control (shuffled labels)."""
    if linear_df is not None:
        features = [f for f in FEATURE_ORDER if f in linear_df["feature"].values and f in nonlinear_df["feature"].values]
    else:
        features = [f for f in FEATURE_ORDER if f in nonlinear_df["feature"].values]
    if not features:
        return

    linear_best = []
    nonlinear_best = []
    control_best = []
    for f in features:
        if linear_df is not None:
            linear_best.append(linear_df[linear_df["feature"] == f]["test_accuracy"].max())
        nonlinear_best.append(nonlinear_df[nonlinear_df["feature"] == f]["test_accuracy"].max())
        if control_df is not None and f in control_df["feature"].values:
            control_best.append(control_df[control_df["feature"] == f]["test_accuracy"].max())
        else:
            control_best.append(np.nan)

    has_linear = linear_df is not None
    has_control = control_df is not None and len(control_best) == len(features) and not np.all(np.isnan(control_best))
    n_series = sum([has_linear, True, has_control])  # nonlinear always
    width = 0.8 / n_series if n_series else 0.35
    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(12, 5))
    offset = -width * (n_series - 1) / 2
    if has_linear:
        ax.bar(x + offset, linear_best, width, label="Linear (best layer)", color="#3498DB")
        offset += width
    ax.bar(x + offset, nonlinear_best, width, label="Nonlinear (best layer)", color="#E74C3C")
    offset += width
    if has_control:
        ax.bar(x + offset, control_best, width, label="Control (shuffled labels)", color="#95A5A6")
    ax.set_ylabel("Test accuracy")
    title = "Linear vs " if has_linear else ""
    ax.set_title(f"{title}Nonlinear vs Control: Best-Layer Accuracy — {model_name} / {condition}")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in features], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_linear_vs_nonlinear_by_layer(
    linear_df: Optional[pd.DataFrame],
    nonlinear_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    condition: str,
    control_df: Optional[pd.DataFrame] = None,
):
    """Line plot: for each layer, mean accuracy (over features) for linear vs nonlinear vs control."""
    layers = sorted(nonlinear_df["layer"].unique())
    if not layers:
        return

    linear_means = []
    nonlinear_means = []
    control_means = []
    for layer in layers:
        if linear_df is not None:
            linear_means.append(linear_df[linear_df["layer"] == layer]["test_accuracy"].mean())
        nonlinear_means.append(nonlinear_df[nonlinear_df["layer"] == layer]["test_accuracy"].mean())
        if control_df is not None and layer in control_df["layer"].values:
            control_means.append(control_df[control_df["layer"] == layer]["test_accuracy"].mean())
        else:
            control_means.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    if linear_df is not None:
        ax.plot(layers, linear_means, "o-", label="Linear", color="#3498DB", linewidth=2)
    ax.plot(layers, nonlinear_means, "s-", label="Nonlinear", color="#E74C3C", linewidth=2)
    if control_df is not None and not np.all(np.isnan(control_means)):
        ax.plot(layers, control_means, "^-", label="Control (shuffled labels)", color="#95A5A6", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean test accuracy (over features)")
    title = "Linear vs " if linear_df is not None else ""
    ax.set_title(f"{title}Nonlinear vs Control by Layer — {model_name} / {condition}")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_nonlinear_only_best(
    nonlinear_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    condition: str,
    control_df: Optional[pd.DataFrame] = None,
):
    """Bar chart: nonlinear vs control best-layer accuracy per feature (selectivity check)."""
    features = [f for f in FEATURE_ORDER if f in nonlinear_df["feature"].values]
    if not features:
        return

    nonlinear_best = [nonlinear_df[nonlinear_df["feature"] == f]["test_accuracy"].max() for f in features]
    control_best = []
    for f in features:
        if control_df is not None and f in control_df["feature"].values:
            control_best.append(control_df[control_df["feature"] == f]["test_accuracy"].max())
        else:
            control_best.append(np.nan)
    has_control = control_df is not None and not np.all(np.isnan(control_best))

    x = np.arange(len(features))
    width = 0.35 if has_control else 0.6
    fig, ax = plt.subplots(figsize=(12, 5))
    if has_control:
        ax.bar(x - width / 2, nonlinear_best, width, label="Nonlinear (best layer)", color="#E74C3C", alpha=0.8)
        ax.bar(x + width / 2, control_best, width, label="Control (shuffled labels)", color="#95A5A6", alpha=0.8)
    else:
        ax.bar(x, nonlinear_best, width, color="#E74C3C", alpha=0.8)
    ax.set_ylabel("Test accuracy (best layer)")
    ax.set_title(f"Nonlinear vs Control: Best-Layer Accuracy — {model_name} / {condition}")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in features], rotation=45, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    if has_control:
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_comparison_plots(
    linear_experiment,
    nonlinear_experiment,
    output_dir: Path,
    model_name: str,
    condition: str,
    control_experiment=None,
):
    """Generate all comparison plots (linear if provided, nonlinear, and control if provided)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_df = _get_exp(linear_experiment) if linear_experiment is not None else None
    nonlinear_df = _get_exp(nonlinear_experiment)
    control_df = _get_exp(control_experiment) if control_experiment is not None else None

    safe_name = model_name.replace(".", "-")
    plot_linear_vs_nonlinear_by_feature(
        linear_df,
        nonlinear_df,
        output_dir / f"linear_vs_nonlinear_by_feature_{safe_name}.png",
        model_name,
        condition,
        control_df=control_df,
    )
    plot_linear_vs_nonlinear_by_layer(
        linear_df,
        nonlinear_df,
        output_dir / f"linear_vs_nonlinear_by_layer_{safe_name}.png",
        model_name,
        condition,
        control_df=control_df,
    )
    plot_nonlinear_only_best(
        nonlinear_df,
        output_dir / f"nonlinear_best_by_feature_{safe_name}.png",
        model_name,
        condition,
        control_df=control_df,
    )
