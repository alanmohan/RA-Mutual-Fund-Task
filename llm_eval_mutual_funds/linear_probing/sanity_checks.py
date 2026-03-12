# -*- coding: utf-8 -*-
"""
Sanity checks for linear probing on mutual fund comparison.

1. Class imbalance: Check balance of binary labels (0/1) per feature before probing.
2. Margin of error: Histograms of |value_1 - value_2| for wrong vs right predictions (same bins & scale to compare).
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()

# Load config and utils
lp_config = importlib.util.spec_from_file_location("lp_config", str(_THIS_DIR / "lp_config.py"))
lp_config_mod = importlib.util.module_from_spec(lp_config)
lp_config.loader.exec_module(lp_config_mod)

lp_utils_spec = importlib.util.spec_from_file_location("lp_utils", str(_THIS_DIR / "lp_utils.py"))
lp_utils_mod = importlib.util.module_from_spec(lp_utils_spec)
lp_utils_spec.loader.exec_module(lp_utils_mod)

DATA_PATH = lp_config_mod.DATA_PATH
ACTIVATIONS_DIR = lp_config_mod.ACTIVATIONS_DIR
PROBE_RESULTS_DIR = lp_config_mod.PROBE_RESULTS_DIR
PROBE_FEATURES = lp_config_mod.PROBE_FEATURES
PROBE_RANDOM_STATE = lp_config_mod.PROBE_RANDOM_STATE
TRAIN_RATIO = getattr(lp_config_mod, "TRAIN_RATIO", 0.70)
VAL_RATIO = getattr(lp_config_mod, "VAL_RATIO", 0.10)
TEST_RATIO = getattr(lp_config_mod, "TEST_RATIO", 0.20)

create_feature_labels = lp_utils_mod.create_feature_labels
load_activations = lp_utils_mod.load_activations


def _shared_hist_bins(wrong_arr: np.ndarray, right_arr: np.ndarray, n_bins: int = 30):
    """
    Return shared bin edges for wrong and right so both histograms are directly comparable.
    Uses union of both data ranges with a small padding. Goal: compare if wrong predictions
    concentrate at small |diff| (values close together) vs right at larger |diff|.
    """
    combined = np.concatenate([np.asarray(wrong_arr).ravel(), np.asarray(right_arr).ravel()])
    combined = combined[~(np.isnan(combined) | np.isinf(combined))]
    if len(combined) == 0:
        return np.linspace(0, 1, n_bins + 1)
    x_min, x_max = np.nanmin(combined), np.nanmax(combined)
    padding = (x_max - x_min) * 0.02 if x_max > x_min else 0.1
    x_lo = max(0, x_min - padding)
    x_hi = x_max + padding
    return np.linspace(x_lo, x_hi, n_bins + 1)


def _plot_wrong_right_histograms(
    ax_wrong, ax_right, abs_diff_wrong, abs_diff_right, n_wrong, n_right, x_max_zoom: Optional[float] = None
):
    """
    Plot Wrong and Right as histograms with shared bins and shared x/y limits.
    If x_max_zoom is set, x-axis is limited to [x_lo, x_max_zoom] to zoom into small |diff|.
    """
    bins = _shared_hist_bins(abs_diff_wrong, abs_diff_right)
    x_lo, x_hi = bins[0], bins[-1]
    if x_max_zoom is not None:
        x_hi = min(x_hi, max(x_lo + 1e-9, x_max_zoom))

    if n_wrong > 0:
        ax_wrong.hist(abs_diff_wrong, bins=bins, density=True, color="coral", alpha=0.7, edgecolor="white", label=f"Wrong (n={n_wrong})")
    else:
        ax_wrong.text(0.5, 0.5, "No wrong predictions", ha="center", va="center", transform=ax_wrong.transAxes)
    ax_wrong.set_xlim(x_lo, x_hi)
    ax_wrong.set_xlabel("|value_1 − value_2|")
    ax_wrong.set_ylabel("Density")
    ax_wrong.set_title("Wrong predictions")
    ax_wrong.legend(loc="upper right")

    if n_right > 0:
        ax_right.hist(abs_diff_right, bins=bins, density=True, color="steelblue", alpha=0.7, edgecolor="white", label=f"Right (n={n_right})")
    else:
        ax_right.text(0.5, 0.5, "No right predictions", ha="center", va="center", transform=ax_right.transAxes)
    ax_right.set_xlim(x_lo, x_hi)
    ax_right.set_xlabel("|value_1 − value_2|")
    ax_right.set_ylabel("Density")
    ax_right.set_title("Right predictions")
    ax_right.legend(loc="upper right")

    # Shared y-axis scale so bar heights are comparable
    y_max_w = ax_wrong.get_ylim()[1] if n_wrong > 0 else 0
    y_max_r = ax_right.get_ylim()[1] if n_right > 0 else 0
    y_max = max(y_max_w, y_max_r, 1e-9)
    ax_wrong.set_ylim(0, y_max)
    ax_right.set_ylim(0, y_max)


def _mann_whitney_wrong_less_than_right(abs_diff_wrong: np.ndarray, abs_diff_right: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Mann-Whitney U, one-sided: test if wrong has smaller |diff| than right.
    Returns (U_statistic, pvalue, rank_biserial_r).
    rank_biserial_r = 1 - 2*U/(n_w*n_r) in [-1,1]; positive => wrong tends smaller than right (effect size).
    """
    if len(abs_diff_wrong) < 3 or len(abs_diff_right) < 3:
        return None, None, None
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(abs_diff_wrong, abs_diff_right, alternative="less")
        U = res.statistic
        n_w, n_r = len(abs_diff_wrong), len(abs_diff_right)
        r_rb = 1.0 - (2.0 * U) / (n_w * n_r)  # rank-biserial correlation
        return U, res.pvalue, r_rb
    except Exception:
        return None, None, None


def _plot_margin_results(
    results: Dict[str, Any],
    model_name: str,
    condition: str,
    features: List[str],
    output_dir: Path,
    plt,  # matplotlib.pyplot or None
):
    """
    Plot margin-of-error histograms (full + zoomed) and write summary CSV.
    Uses same bins and scales; adds Mann-Whitney p-value (wrong < right) and zoom into small |diff|.
    """
    if plt is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zoom threshold: show small-|diff| region (30th percentile of combined data)
    ZOOM_PERCENTILE = 30

    for feat in features:
        r = results.get(feat, {"wrong": np.array([]), "right": np.array([])})
        aw = r["wrong"] if isinstance(r, dict) else np.array([])
        ar = r["right"] if isinstance(r, dict) else np.array([])
        n_wrong, n_right = len(aw), len(ar)
        if n_wrong == 0 and n_right == 0:
            continue

        combined = np.concatenate([np.ravel(aw), np.ravel(ar)])
        combined = combined[~(np.isnan(combined) | np.isinf(combined))]
        zoom_x_max = float(np.percentile(combined, ZOOM_PERCENTILE)) if len(combined) > 0 else None

        med_w = np.nanmedian(aw) if n_wrong > 0 else np.nan
        med_r = np.nanmedian(ar) if n_right > 0 else np.nan
        _, mw_p, r_rb = _mann_whitney_wrong_less_than_right(aw, ar)
        stats_text = f"Median |diff|: Wrong={med_w:.4g}  Right={med_r:.4g}"
        if mw_p is not None:
            stats_text += f"\nMann-Whitney (wrong<right) p={mw_p:.4g}"
        if r_rb is not None:
            stats_text += f"  r_rb={r_rb:.3f}"
        else:
            stats_text += "\nMann-Whitney: n/a"

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_w_full, ax_r_full = axes[0, 0], axes[0, 1]
        ax_w_zoom, ax_r_zoom = axes[1, 0], axes[1, 1]

        _plot_wrong_right_histograms(ax_w_full, ax_r_full, aw, ar, n_wrong, n_right, x_max_zoom=None)
        ax_w_full.set_title(f"{feat} — Wrong (full)")
        ax_r_full.set_title(f"{feat} — Right (full)")

        _plot_wrong_right_histograms(ax_w_zoom, ax_r_zoom, aw, ar, n_wrong, n_right, x_max_zoom=zoom_x_max)
        ax_w_zoom.set_title(f"{feat} — Wrong (zoomed to small |diff|)")
        ax_r_zoom.set_title(f"{feat} — Right (zoomed)")

        fig.text(0.5, 0.02, stats_text, ha="center", fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        fig.suptitle(
            f"Margin of error: {feat} — {model_name} {condition}\n(small |diff| = values close → harder to distinguish)",
            y=1.02,
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        safe_name = feat.replace("/", "_")
        fig.savefig(out_dir / f"margin_of_error_{safe_name}_{model_name}_{condition}.png", dpi=150)
        plt.close(fig)
        print(f"      -> saved margin_of_error_{safe_name}_{model_name}_{condition}.png")

    # Combined grid (full view only, one row per feature)
    feats_with_data = [f for f in features if f in results and (len(results[f]["wrong"]) > 0 or len(results[f]["right"]) > 0)]
    if feats_with_data:
        n_rows = len(feats_with_data)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        for idx, feat in enumerate(feats_with_data):
            r = results[feat]
            aw, ar = r["wrong"], r["right"]
            ax_w, ax_r = axes[idx, 0], axes[idx, 1]
            _plot_wrong_right_histograms(ax_w, ax_r, aw, ar, len(aw), len(ar), x_max_zoom=None)
            ax_w.set_title(f"{feat} — Wrong")
            ax_r.set_title(f"{feat} — Right")
        fig.suptitle(
            f"Margin of error: |value_1 − value_2| — {model_name} {condition}\n(small |diff| = values close → harder to distinguish)",
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"margin_of_error_all_{model_name}_{condition}.png", dpi=150)
        plt.close(fig)
        print(f"Saved combined plot: margin_of_error_all_{model_name}_{condition}.png")

    # Summary CSV with Mann-Whitney
    rows = []
    for feat in features:
        r = results.get(feat, {"wrong": np.array([]), "right": np.array([])})
        arr_w = r["wrong"] if isinstance(r, dict) else np.array([])
        arr_r = r["right"] if isinstance(r, dict) else np.array([])
        n_wrong = len(arr_w)
        n_right = len(arr_r) if isinstance(arr_r, np.ndarray) else 0
        _, mw_p, r_rb = _mann_whitney_wrong_less_than_right(arr_w, arr_r)
        rows.append({
            "feature": feat,
            "n_wrong_predictions": n_wrong,
            "n_right_predictions": n_right,
            "median_abs_diff_wrong": np.nanmedian(arr_w) if n_wrong > 0 else np.nan,
            "median_abs_diff_right": np.nanmedian(arr_r) if n_right > 0 else np.nan,
            "median_ratio_right_over_wrong": (np.nanmedian(arr_r) / np.nanmedian(arr_w)) if (n_right > 0 and n_wrong > 0 and np.nanmedian(arr_w) > 0) else np.nan,
            "mann_whitney_p_wrong_less_than_right": mw_p if mw_p is not None else np.nan,
            "rank_biserial_r": r_rb if r_rb is not None else np.nan,
            "mean_abs_diff_wrong": np.nanmean(arr_w) if n_wrong > 0 else np.nan,
            "mean_abs_diff_right": np.nanmean(arr_r) if n_right > 0 else np.nan,
        })
    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"margin_of_error_summary_{model_name}_{condition}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Margin-of-error summary saved to {summary_path}")


def _load_probe_module():
    """Lazy load probe module (heavy deps: sklearn/cuml) only when needed."""
    probe_spec = importlib.util.spec_from_file_location("probe", str(_THIS_DIR / "probe.py"))
    probe_mod = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe_mod)
    return probe_mod


def check_class_imbalance(
    data_path: Path = DATA_PATH,
    features: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute class balance for each binary feature label (0 vs 1) and NaN counts.
    Uses the full dataset from CSV (labels as formed before probing).
    """
    if features is None:
        features = PROBE_FEATURES

    df = pd.read_csv(data_path)
    labels_df = create_feature_labels(df)

    rows = []
    for feat in features:
        if feat not in labels_df.columns:
            rows.append({"feature": feat, "n_0": np.nan, "n_1": np.nan, "n_nan": np.nan, "pct_0": np.nan, "pct_1": np.nan, "imbalance_ratio": np.nan})
            continue
        col = labels_df[feat]
        n_0 = (col == 0).sum()
        n_1 = (col == 1).sum()
        n_nan = col.isna().sum()
        n_valid = n_0 + n_1
        if n_valid > 0:
            pct_0 = 100 * n_0 / n_valid
            pct_1 = 100 * n_1 / n_valid
            imbalance_ratio = min(n_0, n_1) / max(n_0, n_1) if max(n_0, n_1) > 0 else 0
        else:
            pct_0 = pct_1 = imbalance_ratio = np.nan
        rows.append({
            "feature": feat,
            "n_0": int(n_0),
            "n_1": int(n_1),
            "n_nan": int(n_nan),
            "n_valid": int(n_valid),
            "pct_0": round(pct_0, 2) if not np.isnan(pct_0) else np.nan,
            "pct_1": round(pct_1, 2) if not np.isnan(pct_1) else np.nan,
            "imbalance_ratio": round(imbalance_ratio, 4) if not np.isnan(imbalance_ratio) else np.nan,
        })

    out = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "sanity_class_imbalance.csv"
        out.to_csv(csv_path, index=False)
        print(f"Class imbalance summary saved to {csv_path}")

    return out


def _margin_of_error_data_path(output_dir: Path, model_name: str, condition: str) -> Path:
    """Path for saved margin-of-error results (wrong/right arrays per feature)."""
    return output_dir / f"margin_of_error_data_{model_name}_{condition}.pkl"


def margin_of_error_density(
    model_name: str,
    condition: str,
    features: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    activations_dir: Path = ACTIVATIONS_DIR,
    probe_results_dir: Path = PROBE_RESULTS_DIR,
    load_from_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    For each feature, get test-set samples the probe got wrong and right, compute
    |value_1 - value_2| for each, and plot histograms (wrong vs right, same bins & scale)
    plus a zoomed view and Mann-Whitney test. Optionally load from saved data (--plot-only).
    Returns dict feature -> {"wrong": array, "right": array} of absolute differences.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if features is None:
        features = PROBE_FEATURES

    # Load from saved results (--plot-only) to re-plot without re-running probes
    if load_from_path is not None and Path(load_from_path).exists():
        print(f"Loading margin-of-error data from {load_from_path} (plot-only mode)...")
        with open(load_from_path, "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        model_name = saved["model_name"]
        condition = saved["condition"]
        features = saved.get("features", list(results.keys()))
        out_dir = Path(output_dir) if output_dir else Path(load_from_path).parent
        _plot_margin_results(
            results=results,
            model_name=model_name,
            condition=condition,
            features=features,
            output_dir=out_dir,
            plt=plt,
        )
        print("Margin-of-error check done (plot-only).")
        return results

    activation_path = activations_dir / f"{model_name}_{condition}_activations.npz"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activations not found: {activation_path}. Run extract_activations.py first.")

    print(f"Loading activations from {activation_path} ...")
    data = load_activations(activation_path)
    activations = data["activations"]
    feature_labels = data["feature_labels"]
    ground_truth = data["labels"]
    raw_values = data.get("feature_raw_values")
    print(f"  Activations shape: {activations.shape}")

    if raw_values is None:
        raise ValueError("Activations file has no feature_raw_values (raw_val1/raw_val2). Re-run extract_activations to save raw values.")
    print(f"  Raw values present for {len(raw_values)} features.")

    print("Loading probe module (train/eval) ...")
    probe_mod = _load_probe_module()
    create_stratified_splits = probe_mod.create_stratified_splits
    get_probe_test_predictions = probe_mod.get_probe_test_predictions

    n_samples = len(activations)
    gt_for_split = np.where(np.isnan(ground_truth), 0, ground_truth).astype(int)
    split_indices = create_stratified_splits(n_samples, gt_for_split)
    n_test = len(split_indices["test"])
    print(f"  Splits: train={len(split_indices['train'])}, val={len(split_indices['val'])}, test={n_test}")

    best_path = probe_results_dir / f"probe_best_layers_{model_name}_{condition}.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Best layers CSV not found: {best_path}. Run probe.py first.")
    best_df = pd.read_csv(best_path)
    print(f"Loaded best layers for {len(best_df)} features from {best_path}")

    # Map feature name -> column index in raw_val1/raw_val2
    raw_feature_names = list(raw_values.keys())
    feature_to_col = {name: i for i, name in enumerate(raw_feature_names)}

    results = {}
    n_features = sum(1 for f in features if f in feature_labels.columns and f in feature_to_col and not best_df[best_df["feature"] == f].empty)
    done = 0
    print(f"Computing margin-of-error for {n_features} features ...")
    for feat in features:
        if feat not in feature_labels.columns or feat not in feature_to_col:
            continue
        best_row = best_df[best_df["feature"] == feat]
        if best_row.empty:
            continue
        done += 1
        best_layer = int(best_row["best_layer"].iloc[0])
        labels = feature_labels[feat].values

        y_true, y_pred, test_indices = get_probe_test_predictions(
            activations, labels, best_layer, split_indices, feat
        )
        if len(y_true) == 0:
            results[feat] = {"wrong": np.array([]), "right": np.array([])}
            print(f"  [{done}/{n_features}] {feat}: no valid test samples, skip")
            continue

        wrong_mask = (y_pred != y_true)
        right_mask = (y_pred == y_true)
        wrong_indices = test_indices[wrong_mask]
        right_indices = test_indices[right_mask]

        v1 = raw_values[feat][0]  # (n_samples,)
        v2 = raw_values[feat][1]

        abs_diff_wrong = np.abs(v1[wrong_indices].astype(float) - v2[wrong_indices].astype(float))
        valid_w = ~(np.isnan(v1[wrong_indices]) | np.isnan(v2[wrong_indices]))
        abs_diff_wrong = abs_diff_wrong[valid_w]

        abs_diff_right = np.abs(v1[right_indices].astype(float) - v2[right_indices].astype(float))
        valid_r = ~(np.isnan(v1[right_indices]) | np.isnan(v2[right_indices]))
        abs_diff_right = abs_diff_right[valid_r]

        results[feat] = {"wrong": abs_diff_wrong, "right": abs_diff_right}

        n_wrong = len(abs_diff_wrong)
        n_right = len(abs_diff_right)
        mean_diff_w = np.nanmean(abs_diff_wrong) if n_wrong > 0 else np.nan
        print(f"  [{done}/{n_features}] {feat}: layer={best_layer}, n_wrong={n_wrong}, n_right={n_right}, mean|diff|_wrong={mean_diff_w:.4g}")

    # Save results so we can re-plot without re-running probes (--plot-only)
    if output_dir is not None and results:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = _margin_of_error_data_path(out_dir, model_name, condition)
        with open(data_path, "wb") as f:
            pickle.dump({"results": results, "model_name": model_name, "condition": condition, "features": features}, f)
        print(f"Saved margin-of-error data to {data_path} (use --plot-only to re-plot without re-running)")

    if output_dir is not None and plt is not None and results:
        _plot_margin_results(
            results=results,
            model_name=model_name,
            condition=condition,
            features=features,
            output_dir=Path(output_dir),
            plt=plt,
        )
        print(f"Margin-of-error plots saved to {output_dir}")

    print("Margin-of-error check done.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for linear probing (class imbalance, margin of error)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to mutual_funds_pairs_no_date.csv")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model key (e.g. qwen3-4b) for margin-of-error")
    parser.add_argument("--condition", "-c", type=str, default=None, help="Condition (e.g. 2_fewshot_cot_temp0) for margin-of-error")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: data/probe_results/sanity)")
    parser.add_argument("--class-imbalance-only", action="store_true", help="Only run class imbalance check")
    parser.add_argument("--margin-only", action="store_true", help="Only run margin-of-error (requires --model and --condition)")
    parser.add_argument("--plot-only", action="store_true", help="Load saved margin-of-error data and re-plot only (no probe re-run); requires -m and -c")

    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else DATA_PATH
    output_dir = Path(args.output_dir) if args.output_dir else (PROBE_RESULTS_DIR / "sanity")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.margin_only:
        print("Running class imbalance check...")
        imb = check_class_imbalance(data_path=data_path, output_dir=output_dir)
        print(imb.to_string(index=False))

    if not args.class_imbalance_only:
        if not args.model or not args.condition:
            print("For margin-of-error, provide --model and --condition (e.g. -m qwen3-4b -c 2_fewshot_cot_temp0)")
        else:
            load_path = _margin_of_error_data_path(output_dir, args.model, args.condition) if args.plot_only else None
            if args.plot_only and not load_path.exists():
                print(f"Saved data not found at {load_path}. Run without --plot-only first to generate it.")
            else:
                if load_path is None:
                    print(f"Running margin-of-error (wrong vs right |diff|) for {args.model} / {args.condition}...")
                margin_of_error_density(
                    model_name=args.model,
                    condition=args.condition,
                    output_dir=output_dir,
                    load_from_path=load_path if (args.plot_only and load_path.exists()) else None,
                )
    print("Done.")


if __name__ == "__main__":
    main()
