from __future__ import annotations

import pandas as pd
from pathlib import Path


def calculate_metrics(matrix: pd.DataFrame) -> dict:
    """Calculate accuracy metrics from confusion matrix."""
    # Extract values
    fund1_gt_fund1 = matrix.loc["fund1>fund2", "fund1>fund2"]
    fund1_gt_fund2 = matrix.loc["fund1>fund2", "fund1<fund2"]
    fund1_gt_equal = matrix.loc["fund1>fund2", "fund1=fund2"]
    
    fund2_gt_fund1 = matrix.loc["fund1<fund2", "fund1>fund2"]
    fund2_gt_fund2 = matrix.loc["fund1<fund2", "fund1<fund2"]
    fund2_gt_equal = matrix.loc["fund1<fund2", "fund1=fund2"]
    
    # Totals
    total_fund1_gt = matrix["fund1>fund2"].sum()
    total_fund2_gt = matrix["fund1<fund2"].sum()
    total_equal_gt = matrix["fund1=fund2"].sum()
    total = matrix.sum().sum()
    
    # Correct predictions (excluding ties)
    correct_non_tie = fund1_gt_fund1 + fund2_gt_fund2
    total_non_tie = total_fund1_gt + total_fund2_gt
    
    # Overall accuracy (ties counted as correct)
    correct_with_ties = fund1_gt_fund1 + fund2_gt_fund2 + fund1_gt_equal + fund2_gt_equal
    overall_accuracy = (correct_with_ties / total * 100) if total > 0 else 0
    
    # Accuracy excluding ties
    accuracy_excl_ties = (correct_non_tie / total_non_tie * 100) if total_non_tie > 0 else 0
    
    return {
        "total": int(total),
        "total_fund1_gt": int(total_fund1_gt),
        "total_fund2_gt": int(total_fund2_gt),
        "total_equal_gt": int(total_equal_gt),
        "correct_non_tie": int(correct_non_tie),
        "total_non_tie": int(total_non_tie),
        "accuracy_excl_ties": accuracy_excl_ties,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": {
            "fund1_pred_fund1_gt": int(fund1_gt_fund1),
            "fund1_pred_fund2_gt": int(fund1_gt_fund2),
            "fund1_pred_equal_gt": int(fund1_gt_equal),
            "fund2_pred_fund1_gt": int(fund2_gt_fund1),
            "fund2_pred_fund2_gt": int(fund2_gt_fund2),
            "fund2_pred_equal_gt": int(fund2_gt_equal),
        }
    }


def main() -> None:
    confusion_matrices_dir = Path("batch_experiments/confusion_matrices")
    
    # Find all confusion matrix CSV files
    matrix_files = sorted(confusion_matrices_dir.glob("*_confusion_matrix.csv"))
    
    if not matrix_files:
        print("No confusion matrix files found.")
        return
    
    results = []
    
    for matrix_file in matrix_files:
        experiment_name = matrix_file.stem.replace("_confusion_matrix", "")
        
        # Read confusion matrix
        matrix = pd.read_csv(matrix_file, index_col=0)
        
        # Calculate metrics
        metrics = calculate_metrics(matrix)
        
        results.append({
            "experiment": experiment_name,
            **metrics
        })
    
    # Create markdown table
    markdown_content = """# Confusion Matrix Results Summary

## Results Summary

| Experiment | Total Pairs | fund1>fund2 (GT) | fund1<fund2 (GT) | fund1=fund2 (GT) | Correct (non-tie) | Accuracy (excl. ties) | Overall Accuracy |
|-----------|-------------|------------------|------------------|------------------|-------------------|----------------------|------------------|
"""
    
    for result in results:
        markdown_content += (
            f"| {result['experiment']} | "
            f"{result['total']} | "
            f"{result['total_fund1_gt']} | "
            f"{result['total_fund2_gt']} | "
            f"{result['total_equal_gt']} | "
            f"{result['correct_non_tie']}/{result['total_non_tie']} | "
            f"{result['accuracy_excl_ties']:.2f}% | "
            f"{result['overall_accuracy']:.2f}% |\n"
        )
    
    markdown_content += "\n## Detailed Confusion Matrices\n\n"
    
    for result in results:
        exp_name = result['experiment']
        cm = result['confusion_matrix']
        
        markdown_content += f"### {exp_name}\n\n"
        markdown_content += "| | fund1>fund2 (GT) | fund1<fund2 (GT) | fund1=fund2 (GT) |\n"
        markdown_content += "|--|------------------|------------------|------------------|\n"
        markdown_content += (
            f"| **fund1>fund2 (Pred)** | {cm['fund1_pred_fund1_gt']} | "
            f"{cm['fund1_pred_fund2_gt']} | {cm['fund1_pred_equal_gt']} |\n"
        )
        markdown_content += (
            f"| **fund1<fund2 (Pred)** | {cm['fund2_pred_fund1_gt']} | "
            f"{cm['fund2_pred_fund2_gt']} | {cm['fund2_pred_equal_gt']} |\n"
        )
        markdown_content += "\n"
    
    markdown_content += """
## Notes

- **Accuracy (excl. ties)**: Percentage of correct predictions when ground truth is not a tie (fund1>fund2 or fund1<fund2)
- **Overall Accuracy**: Percentage of correct predictions including ties (ties counted as correct since LLM can only predict one of two options)
- **Ground Truth**: Based on Medalist ratings (Gold > Silver > Bronze > Neutral > Negative)
- **GT**: Ground Truth
- **Pred**: Prediction
"""
    
    # Save markdown file
    output_file = confusion_matrices_dir / "results_summary.md"
    output_file.write_text(markdown_content)
    
    print(f"Results summary saved to: {output_file}")
    print(f"\nProcessed {len(results)} experiments:")
    for result in results:
        print(f"  - {result['experiment']}: {result['accuracy_excl_ties']:.2f}% (excl. ties), {result['overall_accuracy']:.2f}% (overall)")


if __name__ == "__main__":
    main()

