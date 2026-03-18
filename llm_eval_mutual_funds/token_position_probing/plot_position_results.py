# -*- coding: utf-8 -*-
"""
Visualize token-position probing results.

Generates:
  1. Heatmap  – layers (y) vs positions (x), coloured by test accuracy.
  2. Best-layer line plot – best accuracy across layers for each position.
  3. Selected-layers line plot – accuracy vs position for user-chosen layers.

Usage (from llm_eval_mutual_funds/):
    python token_position_probing/plot_position_results.py \
        --results-csv data/probe_results/token_position/tp_probe_beta_f1_lower_qwen3-4b_2_fewshot_cot_temp0.csv \
        --feature beta_f1_lower \
        --feature-position -89
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()

for _p in (_PROJECT_ROOT, _PROJECT_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp_config = _load_mod("_plot_tp_config", _THIS_DIR / "tp_config.py")
TP_RESULTS_DIR = tp_config.TP_RESULTS_DIR


def _stem(feature, model, condition):
    return f"{feature}_{model}_{condition}"


def _snap_to_grid(pos: int, positions: list[int]) -> int:
    """Return the position in *positions* closest to *pos*."""
    return min(positions, key=lambda p: abs(p - pos))


def auto_find_feature_position(feature: str, model: str, condition: str) -> int | None:
    """Try to auto-detect the feature's token position using find_feature_position.

    Returns position_from_end (negative int) or None on failure.
    """
    try:
        from token_position_probing.find_feature_position import (
            find_feature_token_position,
            FEATURE_TO_LINE_PREFIX,
        )
        if feature not in FEATURE_TO_LINE_PREFIX:
            return None
        result = find_feature_token_position(feature=feature, model_key=model, condition=condition)
        pos = result["position_from_end"]
        print(f"Auto-detected feature position for {feature}: {pos} "
              f"(value={result['value_snippet']!r}, token={result['decoded_token']!r})")
        return pos
    except Exception as e:
        print(f"Could not auto-detect feature position: {e}")
        return None


# ============================================================================
# PLOT 1: HEATMAP
# ============================================================================

def plot_heatmap(
    df: pd.DataFrame,
    feature: str,
    model: str,
    condition: str,
    feature_position: int | None = None,
    output_dir: Path | None = None,
):
    pivot = df.pivot(index="layer", columns="position", values="test_accuracy")
    positions = sorted(df["position"].unique())
    layers = sorted(df["layer"].unique())

    fig, ax = plt.subplots(figsize=(max(12, len(positions) * 0.18), max(6, len(layers) * 0.25)))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.45,
        vmax=min(1.0, pivot.values.max() + 0.02),
        origin="lower",
    )
    ax.set_xlabel("Token Position (from end)")
    ax.set_ylabel("Layer")
    ax.set_title(f"Probe Accuracy — {feature}\n{model} / {condition}")

    # Tick labels (show a subset to avoid clutter)
    n_xticks = min(30, len(positions))
    step = max(1, len(positions) // n_xticks)
    ax.set_xticks(range(0, len(positions), step))
    ax.set_xticklabels([positions[i] for i in range(0, len(positions), step)], rotation=45, fontsize=7)
    n_yticks = min(36, len(layers))
    ystep = max(1, len(layers) // n_yticks)
    ax.set_yticks(range(0, len(layers), ystep))
    ax.set_yticklabels([layers[i] for i in range(0, len(layers), ystep)], fontsize=7)

    if feature_position is not None and feature_position in positions:
        x_idx = positions.index(feature_position)
        ax.axvline(x_idx, color="blue", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Feature value ({feature_position})")
        ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(im, ax=ax, label="Test Accuracy", shrink=0.8)
    fig.tight_layout()

    out = (output_dir or TP_RESULTS_DIR) / f"tp_heatmap_{_stem(feature, model, condition)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ============================================================================
# PLOT 2: BEST-LAYER ACCURACY VS POSITION
# ============================================================================

def plot_best_layer(
    df: pd.DataFrame,
    feature: str,
    model: str,
    condition: str,
    feature_position: int | None = None,
    last_token_accuracy: float | None = None,
    output_dir: Path | None = None,
):
    best = df.loc[df.groupby("position")["test_accuracy"].idxmax()]
    best = best.sort_values("position")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(best["position"], best["test_accuracy"], "o-", markersize=3, linewidth=1.2, color="tab:blue", label="Best-layer accuracy")
    ax1.set_xlabel("Token Position (from end)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Best-Layer Probe Accuracy vs Token Position — {feature}\n{model} / {condition}")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")

    if last_token_accuracy is not None:
        ax1.axhline(last_token_accuracy, color="tab:red", linestyle="--", alpha=0.7, label=f"Last-token accuracy ({last_token_accuracy:.3f})")

    if feature_position is not None:
        ax1.axvline(feature_position, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Feature value pos ({feature_position})")

    ax2 = ax1.twinx()
    ax2.plot(best["position"], best["layer"], "x", markersize=4, color="tab:orange", alpha=0.5, label="Best layer (right axis)")
    ax2.set_ylabel("Best Layer", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    fig.tight_layout()
    out = (output_dir or TP_RESULTS_DIR) / f"tp_best_layer_{_stem(feature, model, condition)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ============================================================================
# PLOT 3: SELECTED LAYERS VS POSITION
# ============================================================================

def plot_selected_layers(
    df: pd.DataFrame,
    feature: str,
    model: str,
    condition: str,
    layers: list[int] | None = None,
    feature_position: int | None = None,
    output_dir: Path | None = None,
):
    if layers is None:
        best_row = df.loc[df["test_accuracy"].idxmax()]
        best_layer = int(best_row["layer"])
        all_layers = sorted(df["layer"].unique())
        layers = sorted(set([all_layers[0], best_layer, all_layers[-1]]))

    fig, ax = plt.subplots(figsize=(12, 5))
    for layer in layers:
        sub = df[df["layer"] == layer].sort_values("position")
        ax.plot(sub["position"], sub["test_accuracy"], "o-", markersize=3, linewidth=1, label=f"Layer {layer}")

    ax.set_xlabel("Token Position (from end)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Probe Accuracy by Layer vs Position — {feature}\n{model} / {condition}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    if feature_position is not None:
        ax.axvline(feature_position, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Feature value pos ({feature_position})")

    ax.legend(fontsize=8)
    fig.tight_layout()

    out = (output_dir or TP_RESULTS_DIR) / f"tp_layer_lines_{_stem(feature, model, condition)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot token-position probing results.")
    parser.add_argument("--results-csv", "-r", required=True, help="Path to tp_probe_*.csv")
    parser.add_argument("--feature", "-f", required=True, help="Feature name (for titles)")
    parser.add_argument("--model", "-m", default=None, help="Model key (auto-detected from filename if omitted)")
    parser.add_argument("--condition", "-c", default=None, help="Condition (auto-detected from filename if omitted)")
    parser.add_argument(
        "--feature-position", type=int, default=None,
        help="Token position where the feature value appears (from find_feature_position.py). Drawn as vertical line.",
    )
    parser.add_argument(
        "--auto-feature-position", action="store_true",
        help="Auto-detect feature position using find_feature_position.py (requires tokenizer).",
    )
    parser.add_argument(
        "--last-token-accuracy", type=float, default=None,
        help="Accuracy from the standard last-token probe (drawn as horizontal reference).",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layers for the selected-layers plot (default: auto-pick).",
    )
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)

    model = args.model
    condition = args.condition
    if model is None or condition is None:
        stem = csv_path.stem  # tp_probe_{feature}_{model}_{condition}
        parts = stem.replace("tp_probe_", "").split("_")
        feature_parts = args.feature.split("_")
        n_feat = len(feature_parts)
        remaining = parts[n_feat:]
        if model is None:
            model = remaining[0] if remaining else "unknown"
        if condition is None:
            condition = "_".join(remaining[1:]) if len(remaining) > 1 else "unknown"

    feature_position = args.feature_position
    if feature_position is None and args.auto_feature_position:
        raw_pos = auto_find_feature_position(args.feature, model, condition)
        if raw_pos is not None:
            positions = sorted(df["position"].unique())
            feature_position = _snap_to_grid(raw_pos, positions)
            print(f"Snapped to nearest grid position: {feature_position}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    plot_heatmap(df, args.feature, model, condition, feature_position, output_dir)
    plot_best_layer(df, args.feature, model, condition, feature_position, args.last_token_accuracy, output_dir)
    plot_selected_layers(df, args.feature, model, condition, args.layers, feature_position, output_dir)


if __name__ == "__main__":
    main()
