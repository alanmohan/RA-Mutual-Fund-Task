from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


# Medalist rating hierarchy (higher is better)
MEDALIST_HIERARCHY = {
    "Negative": 0,
    "Neutral": 1,
    "Bronze": 2,
    "Silver": 3,
    "Gold": 4,
}


def get_medalist_value(medalist: str) -> int:
    """Get numeric value for medalist rating."""
    return MEDALIST_HIERARCHY.get(medalist, -1)


def determine_ground_truth(medalist_1: str, medalist_2: str) -> str:
    """Determine ground truth comparison: fund1>fund2, fund1<fund2, or fund1=fund2."""
    val_1 = get_medalist_value(medalist_1)
    val_2 = get_medalist_value(medalist_2)
    
    if val_1 > val_2:
        return "fund1>fund2"
    elif val_1 < val_2:
        return "fund1<fund2"
    else:
        return "fund1=fund2"


def parse_llm_prediction(prediction: str) -> Optional[str]:
    """Parse LLM prediction to fund1>fund2 or fund1<fund2."""
    prediction_lower = prediction.lower().strip()
    
    if "mutual fund 1" in prediction_lower or "fund 1" in prediction_lower:
        return "fund1>fund2"
    elif "mutual fund 2" in prediction_lower or "fund 2" in prediction_lower:
        return "fund1<fund2"
    else:
        return None


def load_ground_truth(csv_path: Path) -> Dict[Tuple[str, str], Dict]:
    """Load ground truth data and create lookup by (Name_1, Name_2)."""
    df = pd.read_csv(csv_path)
    
    lookup = {}
    for _, row in df.iterrows():
        name_1 = str(row.get("Name_1", "")).strip()
        name_2 = str(row.get("Name_2", "")).strip()
        medalist_1 = str(row.get("Medalist_1", "")).strip()
        medalist_2 = str(row.get("Medalist_2", "")).strip()
        
        if name_1 and name_2:
            # Create lookup key (normalize names)
            key = (name_1, name_2)
            lookup[key] = {
                "medalist_1": medalist_1,
                "medalist_2": medalist_2,
                "ground_truth": determine_ground_truth(medalist_1, medalist_2),
            }
    
    return lookup


def load_llm_predictions(csv_path: Path, prediction_column: str = "better_fund_response") -> list[Dict]:
    """Load LLM predictions from CSV."""
    df = pd.read_csv(csv_path)
    
    predictions = []
    for _, row in df.iterrows():
        name_1 = str(row.get("name_1", row.get("Name_1", ""))).strip()
        name_2 = str(row.get("name_2", row.get("Name_2", ""))).strip()
        prediction = str(row.get(prediction_column, "")).strip()
        
        if name_1 and name_2 and prediction:
            predictions.append({
                "name_1": name_1,
                "name_2": name_2,
                "prediction": prediction,
                "parsed_prediction": parse_llm_prediction(prediction),
            })
    
    return predictions


def create_confusion_matrix(
    predictions: list[Dict],
    ground_truth_lookup: Dict[Tuple[str, str], Dict]
) -> Tuple[np.ndarray, Dict]:
    """Create 2x3 confusion matrix and return statistics."""
    # Initialize confusion matrix
    # Rows: LLM predictions (fund1>fund2, fund1<fund2)
    # Columns: Ground truth (fund1>fund2, fund1<fund2, fund1=fund2)
    confusion_matrix = np.zeros((2, 3), dtype=int)
    
    # Labels
    pred_labels = ["fund1>fund2", "fund1<fund2"]
    truth_labels = ["fund1>fund2", "fund1<fund2", "fund1=fund2"]
    
    matched = 0
    unmatched = 0
    invalid_predictions = 0
    
    for pred in predictions:
        name_1 = pred["name_1"]
        name_2 = pred["name_2"]
        parsed_pred = pred["parsed_prediction"]
        
        # Lookup ground truth
        key = (name_1, name_2)
        if key not in ground_truth_lookup:
            # Try reverse order
            key_reverse = (name_2, name_1)
            if key_reverse in ground_truth_lookup:
                # Swap the ground truth
                gt_data = ground_truth_lookup[key_reverse]
                gt = gt_data["ground_truth"]
                # Swap ground truth
                if gt == "fund1>fund2":
                    gt = "fund1<fund2"
                elif gt == "fund1<fund2":
                    gt = "fund1>fund2"
                # fund1=fund2 stays the same
            else:
                unmatched += 1
                continue
        else:
            gt_data = ground_truth_lookup[key]
            gt = gt_data["ground_truth"]
        
        matched += 1
        
        # Handle invalid predictions
        if parsed_pred is None:
            invalid_predictions += 1
            continue
        
        # Map to indices
        pred_idx = pred_labels.index(parsed_pred)
        truth_idx = truth_labels.index(gt)
        
        confusion_matrix[pred_idx, truth_idx] += 1
    
    stats = {
        "matched": matched,
        "unmatched": unmatched,
        "invalid_predictions": invalid_predictions,
        "pred_labels": pred_labels,
        "truth_labels": truth_labels,
    }
    
    return confusion_matrix, stats


def print_confusion_matrix(matrix: np.ndarray, stats: Dict) -> None:
    """Print confusion matrix in a readable format."""
    print("=" * 80)
    print("CONFUSION MATRIX: LLM Predictions vs Medalist Ground Truth")
    print("=" * 80)
    print()
    print("Rows: LLM Predictions")
    print("Columns: Ground Truth (based on Medalist ratings)")
    print()
    print(f"Medalist Hierarchy: Gold > Silver > Bronze > Neutral > Negative")
    print()
    
    # Header
    print(f"{'':<20}", end="")
    for label in stats["truth_labels"]:
        print(f"{label:>20}", end="")
    print(f"{'Total':>20}")
    print("-" * 80)
    
    # Rows
    total_correct = 0
    total_all = 0
    for i, pred_label in enumerate(stats["pred_labels"]):
        print(f"{pred_label:<20}", end="")
        row_total = 0
        for j in range(len(stats["truth_labels"])):
            count = matrix[i, j]
            row_total += count
            total_all += count
            # Count correct predictions (diagonal for fund1>fund2 and fund1<fund2)
            # For ties (fund1=fund2), we count both predictions as acceptable
            if (i == 0 and j == 0) or (i == 1 and j == 1):
                total_correct += count
            print(f"{count:>20}", end="")
        print(f"{row_total:>20}")
    
    print("-" * 80)
    print(f"{'Total':<20}", end="")
    col_totals = matrix.sum(axis=0)
    for total in col_totals:
        print(f"{total:>20}", end="")
    print(f"{total_all:>20}")
    print()
    
    # Statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total matched pairs: {stats['matched']}")
    print(f"Unmatched pairs: {stats['unmatched']}")
    print(f"Invalid predictions: {stats['invalid_predictions']}")
    print()
    
    # Accuracy for cases where ground truth is not "equal"
    non_equal_mask = col_totals[0] + col_totals[1] > 0
    if non_equal_mask:
        non_equal_total = col_totals[0] + col_totals[1]
        non_equal_correct = matrix[0, 0] + matrix[1, 1]
        accuracy_non_equal = (non_equal_correct / non_equal_total * 100) if non_equal_total > 0 else 0
        print(f"Accuracy (excluding ties): {non_equal_correct}/{non_equal_total} = {accuracy_non_equal:.2f}%")
    
    # Overall accuracy (treating ties as correct if predicted either way)
    if total_all > 0:
        # For ties, count as correct if predicted either way (since LLM can only predict one of two options)
        tie_correct = matrix[0, 2] + matrix[1, 2]
        overall_correct = total_correct + tie_correct
        overall_accuracy = (overall_correct / total_all * 100) if total_all > 0 else 0
        print(f"Overall accuracy (ties counted as correct): {overall_correct}/{total_all} = {overall_accuracy:.2f}%")
        if col_totals[2] > 0:
            print(f"  Note: For {col_totals[2]} tie cases (fund1=fund2), both predictions are acceptable")
    
    # Per-class breakdown
    print()
    print("Per-class breakdown:")
    for i, pred_label in enumerate(stats["pred_labels"]):
        for j, truth_label in enumerate(stats["truth_labels"]):
            count = matrix[i, j]
            if count > 0:
                pct = (count / col_totals[j] * 100) if col_totals[j] > 0 else 0
                print(f"  {pred_label} when truth is {truth_label}: {count} ({pct:.1f}% of {truth_label} cases)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM predictions against Medalist ground truth."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to CSV file with LLM predictions (e.g., direct_output.csv)",
    )
    parser.add_argument(
        "--ground-truth",
        default="mutual_funds_pairs_no_date.csv",
        help="Path to CSV file with ground truth (mutual_funds_pairs_no_date.csv)",
    )
    parser.add_argument(
        "--prediction-column",
        default="better_fund_response",
        help="Column name in predictions CSV containing LLM predictions",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: Path to save confusion matrix as CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load data
    print("Loading ground truth data...")
    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        return
    
    ground_truth_lookup = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth_lookup)} pairs from ground truth file.")
    
    print("\nLoading LLM predictions...")
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        return
    
    predictions = load_llm_predictions(predictions_path, args.prediction_column)
    print(f"Loaded {len(predictions)} predictions from predictions file.")
    
    # Create confusion matrix
    print("\nCreating confusion matrix...")
    confusion_matrix, stats = create_confusion_matrix(predictions, ground_truth_lookup)
    
    # Print results
    print_confusion_matrix(confusion_matrix, stats)
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(
            confusion_matrix,
            index=stats["pred_labels"],
            columns=stats["truth_labels"]
        )
        df.to_csv(output_path)
        print(f"\nConfusion matrix saved to: {output_path}")


if __name__ == "__main__":
    main()

