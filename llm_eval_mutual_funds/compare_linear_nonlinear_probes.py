#!/usr/bin/env python3
"""
Compare linear vs non-linear probe test accuracies per feature (beta_f1_lower, stdev_f1_lower).
Reads linear results from probe_results_*.csv and non-linear from nonlinear_torch_beta/ and
nonlinear_torch_stddev/, then plots layer vs test accuracy for each feature.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Paths relative to this script (script lives in llm_eval_mutual_funds/)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "probe_results"

# Model and condition
MODEL_CONDITION = "qwen3-4b_2_fewshot_cot_temp0"
LINEAR_CSV = DATA_DIR / f"probe_results_{MODEL_CONDITION}.csv"
NONLINEAR_BETA_DIR = DATA_DIR / "nonlinear_torch_beta"
NONLINEAR_STDDEV_DIR = DATA_DIR / "nonlinear_torch_stddev"

FEATURE_CONFIG = [
    {
        "feature": "beta_f1_lower",
        "nonlinear_csv": NONLINEAR_BETA_DIR / f"probe_nonlinear_results_{MODEL_CONDITION}.csv",
        "title": "beta_f1_lower",
    },
    {
        "feature": "stdev_f1_lower",
        "nonlinear_csv": NONLINEAR_STDDEV_DIR / f"probe_nonlinear_results_{MODEL_CONDITION}.csv",
        "title": "stdev_f1_lower",
    },
]


def load_linear_by_feature(csv_path: Path, feature: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["feature"] == feature][["layer", "test_accuracy"]].sort_values("layer").reset_index(drop=True)


def load_nonlinear(csv_path: Path, feature: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["feature"] == feature][["layer", "test_accuracy"]].sort_values("layer").reset_index(drop=True)


def main():
    if not LINEAR_CSV.exists():
        raise FileNotFoundError(f"Linear results not found: {LINEAR_CSV}")

    n_features = len(FEATURE_CONFIG)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    for ax, cfg in zip(axes, FEATURE_CONFIG):
        feature = cfg["feature"]
        nl_path = cfg["nonlinear_csv"]

        linear = load_linear_by_feature(LINEAR_CSV, feature)
        if linear.empty:
            print(f"Warning: no linear rows for feature={feature}")
            ax.set_title(cfg["title"])
            continue

        if not nl_path.exists():
            print(f"Warning: non-linear file not found: {nl_path}")
            ax.plot(linear["layer"], linear["test_accuracy"], "o-", label="Linear", color="C0")
            ax.set_title(cfg["title"])
            ax.legend()
            ax.set_xlabel("Layer")
            ax.set_ylabel("Test accuracy")
            ax.grid(True, alpha=0.3)
            continue

        nonlinear = load_nonlinear(nl_path, feature)
        if nonlinear.empty:
            print(f"Warning: no non-linear rows for feature={feature} in {nl_path}")

        ax.plot(linear["layer"], linear["test_accuracy"], "o-", label="Linear", color="C0", markersize=4)
        if not nonlinear.empty:
            ax.plot(nonlinear["layer"], nonlinear["test_accuracy"], "s-", label="Non-linear", color="C1", markersize=4)
        ax.set_title(cfg["title"])
        ax.legend()
        ax.set_xlabel("Layer")
        ax.set_ylabel("Test accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

    fig.suptitle(f"Linear vs non-linear probe test accuracy by layer ({MODEL_CONDITION})", fontsize=12)
    plt.tight_layout()

    out_path = DATA_DIR / f"compare_linear_nonlinear_{MODEL_CONDITION}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
