# -*- coding: utf-8 -*-
"""
Results processing and saving for LLM Mutual Fund Comparison Experiment
"""

import os
import numpy as np
import pandas as pd
from config import RESULTS_DIR, TIE_HANDLING
from utils import compare_medalist


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _is_tie(medalist_1, medalist_2) -> bool:
    """Return True if both Medalist ratings are known and equal."""
    return (
        pd.notna(medalist_1)
        and pd.notna(medalist_2)
        and str(medalist_1).strip() == str(medalist_2).strip()
    )


def calculate_accuracy(results_df):
    """
    Calculate accuracy of predictions vs Medalist ground truth.

    Args:
        results_df: DataFrame with pred_choice, Medalist_1, Medalist_2 columns

    Returns:
        dict with accuracy, n_valid, n_total, n_ties
    """
    # Compute true choice from Medalist ratings
    results_df["true_choice"] = results_df.apply(
        lambda r: compare_medalist(r["Medalist_1"], r["Medalist_2"]), axis=1
    )

    # Identify ties for reporting
    results_df["is_tie"] = results_df.apply(
        lambda r: _is_tie(r["Medalist_1"], r["Medalist_2"]), axis=1
    )

    # Check correctness
    results_df["correct"] = results_df["pred_choice"] == results_df["true_choice"]

    if TIE_HANDLING == "count_as_correct":
        # Count ties as correct if a prediction is present
        tie_mask = results_df["is_tie"] & results_df["pred_choice"].notna()
        results_df.loc[tie_mask, "correct"] = True

        valid = results_df.dropna(subset=["pred_choice"])
    else:
        # Default: exclude ties and missing ground truth
        valid = results_df.dropna(subset=["pred_choice", "true_choice"])

    accuracy = valid["correct"].mean() if len(valid) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "n_valid": len(valid),
        "n_total": len(results_df),
        "n_ties": int(results_df["is_tie"].sum()),
    }


def save_results(results_df, model_name, condition_name):
    """
    Save results DataFrame to CSV.

    Args:
        results_df: DataFrame with experiment results
        model_name: Name of the model
        condition_name: Name of the condition
    """
    ensure_results_dir()
    filename = f"results_{model_name}_{condition_name}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    results_df.to_csv(filepath, index=False)
    print(f"  Saved: {filename}")


def save_experiment_summary(summary_rows, model_name):
    """
    Save experiment summary CSV.

    Args:
        summary_rows: List of dicts with condition results
        model_name: Name of the model
    """
    ensure_results_dir()
    summary_df = pd.DataFrame(summary_rows)
    filename = f"experiment_summary_{model_name}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    summary_df.to_csv(filepath, index=False)
    print(f"\nSaved: {filename}")


def print_experiment_summary(model_results, model_name, sample_size):
    """
    Print formatted experiment summary table.

    Args:
        model_results: Dict of condition_name -> result dict
        model_name: Name of the model
        sample_size: Number of samples per condition
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {model_name.upper()} (n={sample_size} pairs per condition)")
    print("=" * 70)

    print("-" * 70)
    print(f"{'Condition':<35} {'Accuracy':>10} {'Valid':>10} {'Ties':>8} {'Time(s)':>10}")
    print("-" * 70)

    for cond_name, result in model_results.items():
        acc = result["accuracy"]
        n_valid = result["n_valid"]
        n_ties = result.get("n_ties", 0)
        time_s = result["time_sec"]
        print(f"{cond_name:<35} {acc:>10.3f} {n_valid:>10} {n_ties:>8} {time_s:>10.1f}")

    print("=" * 70)
