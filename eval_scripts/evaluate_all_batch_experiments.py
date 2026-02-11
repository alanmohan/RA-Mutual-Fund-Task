from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    batch_experiments_dir = Path("batch_experiments")
    output_dir = batch_experiments_dir / "confusion_matrices"
    output_dir.mkdir(exist_ok=True)
    
    ground_truth_file = "mutual_funds_pairs_no_date.csv"
    
    # Find all CSV files in batch_experiments (excluding eval_summary.csv)
    csv_files = [
        f for f in batch_experiments_dir.glob("*.csv")
        if f.name != "eval_summary.csv"
    ]
    
    if not csv_files:
        print("No CSV files found in batch_experiments folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files to evaluate:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Ground truth file: {ground_truth_file}\n")
    print("=" * 80)
    
    results = []
    
    for csv_file in sorted(csv_files):
        experiment_name = csv_file.stem
        output_file = output_dir / f"{experiment_name}_confusion_matrix.csv"
        
        print(f"\nEvaluating: {csv_file.name}")
        print("-" * 80)
        
        try:
            # Run the evaluation script
            cmd = [
                sys.executable,
                "evaluate_medalist.py",
                "--predictions", str(csv_file),
                "--ground-truth", ground_truth_file,
                "--prediction-column", "better_fund",
                "--output", str(output_file),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract key statistics from output
            output_lines = result.stdout.split("\n")
            matched = None
            accuracy = None
            
            for line in output_lines:
                if "Total matched pairs:" in line:
                    matched = line.split(":")[-1].strip()
                if "Accuracy (excluding ties):" in line:
                    accuracy = line.split("=")[-1].strip()
            
            results.append({
                "experiment": experiment_name,
                "status": "success",
                "matched": matched,
                "accuracy": accuracy,
                "output_file": output_file.name,
            })
            
            print(f"✓ Successfully evaluated {csv_file.name}")
            print(f"  Output saved to: {output_file.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error evaluating {csv_file.name}:")
            print(f"  {e.stderr}")
            results.append({
                "experiment": experiment_name,
                "status": "error",
                "error": str(e),
            })
        except Exception as e:
            print(f"✗ Unexpected error evaluating {csv_file.name}:")
            print(f"  {e}")
            results.append({
                "experiment": experiment_name,
                "status": "error",
                "error": str(e),
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"\nSuccessfully evaluated: {len(successful)}/{len(results)}")
    
    if successful:
        print("\nResults:")
        for result in successful:
            print(f"  {result['experiment']}:")
            print(f"    Matched pairs: {result['matched']}")
            print(f"    Accuracy: {result['accuracy']}")
            print(f"    Output: {result['output_file']}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for result in failed:
            print(f"  {result['experiment']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nAll confusion matrices saved to: {output_dir}")


if __name__ == "__main__":
    main()

