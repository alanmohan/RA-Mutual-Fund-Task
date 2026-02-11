from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def analyze_results(csv_path: Path) -> Dict:
    """Analyze the guess results CSV and return statistics."""
    results = []
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    total = len(results)
    matches = 0
    match_details = []
    
    for result in results:
        actual = str(result.get("actual_name", "")).lower().strip()
        guessed = str(result.get("guessed_name", "")).lower().strip()
        
        # Skip empty or error entries
        if not guessed or guessed.startswith("[") or not actual:
            continue
        
        # Check if guessed_name is contained in actual_name
        if guessed in actual:
            matches += 1
            match_details.append({
                "actual_name": result.get("actual_name", ""),
                "guessed_name": result.get("guessed_name", ""),
                "confidence": result.get("confidence", ""),
            })
    
    return {
        "total": total,
        "matches": matches,
        "match_rate": matches / total * 100 if total > 0 else 0,
        "match_details": match_details,
    }


def print_analysis(stats: Dict) -> None:
    """Print the analysis results."""
    print("=" * 60)
    print("GUESS FUND NAME ANALYSIS")
    print("=" * 60)
    print(f"\nTotal funds analyzed: {stats['total']}")
    print(f"Matches (guessed_name contained in actual_name): {stats['matches']}")
    print(f"Match rate: {stats['matches']}/{stats['total']} = {stats['match_rate']:.1f}%")
    
    if stats['match_details']:
        print(f"\n{'=' * 60}")
        print("MATCH DETAILS:")
        print(f"{'=' * 60}")
        for i, detail in enumerate(stats['match_details'], 1):
            print(f"\n{i}. Actual: {detail['actual_name']}")
            print(f"   Guessed: {detail['guessed_name']}")
            print(f"   Confidence: {detail['confidence']}")
    else:
        print("\nNo matches found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze results from guess_fund_name.py output CSV."
    )
    parser.add_argument(
        "--csv",
        default="guess_fund_name_output.csv",
        help="Path to the output CSV file from guess_fund_name.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: Path to save detailed match results (CSV format)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    stats = analyze_results(csv_path)
    print_analysis(stats)
    
    # Optionally save match details to a separate file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", newline="", encoding="utf-8") as f:
            if stats['match_details']:
                fieldnames = ["actual_name", "guessed_name", "confidence"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for detail in stats['match_details']:
                    writer.writerow(detail)
            else:
                f.write("actual_name,guessed_name,confidence\n")
        
        print(f"\nMatch details saved to: {output_path}")


if __name__ == "__main__":
    main()

