#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and Visualize Linear Probing Results (Mutual Funds)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

FEATURE_LABELS = {
    "expense_ratio_f1_lower": "Expense Ratio (Lower)",
    "sharpe_f1_higher": "Sharpe (Higher)",
    "stdev_f1_lower": "Stdev (Lower)",
    "return_3yr_f1_higher": "3Y Return (Higher)",
    "beta_f1_closer_to_1": "Beta (Closer to 1)",
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
    "beta_f1_closer_to_1",
    "tenure_f1_longer",
    "inception_f1_older",
    "assets_f1_higher",
    "turnover_f1_lower",
    "load_f1_no",
    "ntf_f1_yes",
    "medalist_f1_higher",
]

MODEL_COLORS = {
    "llama-3.2-3b": "#E74C3C",
    "qwen3-4b": "#3498DB",
}

MODEL_LABELS = {
    "llama-3.2-3b": "Llama-3.2-3B",
    "qwen3-4b": "Qwen3-4B",
}


def load_results(results_dir: Path, model: str, condition: str):
    results = {}

    results_csv = results_dir / f"probe_results_{model}_{condition}.csv"
    if results_csv.exists():
        results["detailed"] = pd.read_csv(results_csv)

    acc_matrix = results_dir / f"probe_matrix_accuracy_{model}_{condition}.csv"
    if acc_matrix.exists():
        results["accuracy_matrix"] = pd.read_csv(acc_matrix, index_col=0)

    auc_matrix = results_dir / f"probe_matrix_auc_{model}_{condition}.csv"
    if auc_matrix.exists():
        results["auc_matrix"] = pd.read_csv(auc_matrix, index_col=0)

    best_layers = results_dir / f"probe_best_layers_{model}_{condition}.csv"
    if best_layers.exists():
        results["best_layers"] = pd.read_csv(best_layers)

    config_file = results_dir / f"probe_config_{model}_{condition}.json"
    if config_file.exists():
        with open(config_file) as f:
            results["config"] = json.load(f)

    return results


def plot_accuracy_heatmap(df: pd.DataFrame, model: str, output_dir: Path, figsize=(16, 9)):
    pivot = df.pivot(index="layer", columns="feature", values="test_accuracy")
    cols_ordered = [f for f in FEATURE_ORDER if f in pivot.columns]
    pivot = pivot[cols_ordered]
    pivot.columns = [FEATURE_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=False,
        cmap="RdYlGn",
        center=0.7,
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Test Accuracy"},
    )

    ax.set_title(f"Linear Probe Accuracy by Layer\n{MODEL_LABELS.get(model, model)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / f"heatmap_accuracy_{model}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_accuracy_curves(df: pd.DataFrame, model: str, output_dir: Path, figsize=(14, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(FEATURE_ORDER)))

    for i, feature in enumerate(FEATURE_ORDER):
        feature_df = df[df["feature"] == feature].sort_values("layer")
        if len(feature_df) == 0:
            continue

        label = FEATURE_LABELS.get(feature, feature)
        is_target = feature == "medalist_f1_higher"

        ax.plot(
            feature_df["layer"],
            feature_df["test_accuracy"],
            marker="o" if is_target else ".",
            markersize=8 if is_target else 4,
            linewidth=3 if is_target else 1.5,
            label=label,
            color=colors[i],
            linestyle="--" if is_target else "-",
            alpha=1.0 if is_target else 0.8,
        )

        if "accuracy_ci_lower" in feature_df.columns and "accuracy_ci_upper" in feature_df.columns:
            ax.fill_between(
                feature_df["layer"],
                feature_df["accuracy_ci_lower"],
                feature_df["accuracy_ci_upper"],
                alpha=0.1,
                color=colors[i],
            )

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Probe Accuracy Across Layers\n{MODEL_LABELS.get(model, model)}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"accuracy_curves_{model}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_best_layer_comparison(results, output_dir: Path, figsize=(13, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    models = list(results.keys())
    features = [f for f in FEATURE_ORDER if f != "medalist_f1_higher"]

    ax1 = axes[0]
    x = np.arange(len(features))
    width = 0.35

    for i, model in enumerate(models):
        if "best_layers" not in results[model]:
            continue
        best_df = results[model]["best_layers"]
        accuracies = []
        for f in features:
            row = best_df[best_df["feature"] == f]
            acc = row["test_accuracy"].values[0] if len(row) > 0 else 0
            accuracies.append(acc)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset,
            accuracies,
            width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, f"C{i}"),
        )

        for bar, acc in zip(bars, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Best Test Accuracy", fontsize=11)
    ax1.set_title("Input Feature Encoding", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([FEATURE_LABELS[f] for f in features], rotation=45, ha="right")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    ax2 = axes[1]
    price_accs = []
    input_means = []
    for model in models:
        if "best_layers" not in results[model]:
            continue
        best_df = results[model]["best_layers"]
        row = best_df[best_df["feature"] == "medalist_f1_higher"]
        acc = row["test_accuracy"].values[0] if len(row) > 0 else 0
        price_accs.append((model, acc))
        input_df = best_df[best_df["feature"] != "medalist_f1_higher"]
        mean_acc = input_df["test_accuracy"].mean()
        input_means.append((model, mean_acc))

    x2 = np.arange(len(models))
    width2 = 0.35

    bars1 = ax2.bar(x2 - width2 / 2, [m[1] for m in input_means], width2, label="Inputs (avg)", color="#2ECC71")
    bars2 = ax2.bar(x2 + width2 / 2, [p[1] for p in price_accs], width2, label="Medalist (Target)", color="#E74C3C")

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Best Test Accuracy", fontsize=11)
    ax2.set_title("The 'Last Mile' Problem", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax2.legend(loc="upper right")
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    for i, (model, input_acc) in enumerate(input_means):
        medalist_acc = price_accs[i][1]
        gap = input_acc - medalist_acc
        ax2.annotate(f"Gap: {gap:.1%}", xy=(i, (input_acc + medalist_acc) / 2), fontsize=10, fontweight="bold", color="#8E44AD")

    plt.tight_layout()
    output_path = output_dir / "best_layer_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_significance_summary(results, output_dir: Path, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, (model, res) in enumerate(results.items()):
        if "detailed" not in res:
            continue
        df = res["detailed"]
        ax = axes[idx]

        pivot = df.pivot(index="layer", columns="feature", values="p_value")
        cols_ordered = [f for f in FEATURE_ORDER if f in pivot.columns]
        pivot = pivot[cols_ordered]

        log_p = -np.log10(pivot.clip(lower=1e-300))
        log_p = log_p.clip(upper=50)
        log_p.columns = [FEATURE_LABELS.get(c, c) for c in log_p.columns]

        sns.heatmap(log_p, cmap="YlOrRd", ax=ax, cbar_kws={"label": "-log₁₀(p-value)"})
        ax.set_title(f"{MODEL_LABELS.get(model, model)}\nStatistical Significance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Layer")

    plt.tight_layout()
    output_path = output_dir / "significance_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze linear probing results")
    parser.add_argument("--results-dir", type=str, default="data/probe_results", help="Directory containing probe results")
    parser.add_argument("--output-dir", type=str, default="data/probe_results/plots", help="Directory to save plots")
    parser.add_argument("--condition", type=str, default="2_fewshot_cot_temp0", help="Experiment condition")
    parser.add_argument("--models", type=str, nargs="+", default=["llama-3.2-3b", "qwen3-4b"], help="Models to analyze")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Linear Probing Results Analysis (Mutual Funds)")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Condition: {args.condition}")
    print(f"Models: {args.models}")
    print("=" * 60)

    all_results = {}
    for model in args.models:
        print(f"\nLoading {model}...")
        res = load_results(results_dir, model, args.condition)
        if res:
            all_results[model] = res
            print(f"  Loaded {len(res)} result files")
        else:
            print("  No results found")

    if not all_results:
        print("No results to analyze!")
        return

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    for model, res in all_results.items():
        if "detailed" in res:
            print(f"\n{model}:")
            plot_accuracy_heatmap(res["detailed"], model, output_dir)
            plot_accuracy_curves(res["detailed"], model, output_dir)

    if len(all_results) >= 1:
        print("\nCross-model analysis:")
        plot_best_layer_comparison(all_results, output_dir)
        plot_significance_summary(all_results, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
