from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize(value: str) -> str:
    return (value or "").strip().lower()


def evaluate_csv(path: Path) -> dict:
    total = 0
    better_match = 0
    next_match = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            lowest = normalize(row.get("lowest_rank", ""))
            better = normalize(row.get("better_fund", ""))
            next_month = normalize(row.get("next_month", ""))

            if lowest and better == lowest:
                better_match += 1
            if lowest and next_month == lowest:
                next_match += 1

    better_pct = (better_match / total * 100.0) if total else 0.0
    next_pct = (next_match / total * 100.0) if total else 0.0

    return {
        "file": path.name,
        "rows": total,
        "better_match_pct": better_pct,
        "next_month_match_pct": next_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate batch experiment CSV outputs."
    )
    parser.add_argument(
        "--dir",
        default="batch_experiments",
        help="Directory containing experiment CSVs.",
    )
    parser.add_argument(
        "--output",
        default="batch_experiments/eval_summary.csv",
        help="Path to write the evaluation summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.dir)

    csv_files = sorted(p for p in target_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {target_dir}")
        return

    summary = []
    for csv_path in csv_files:
        result = evaluate_csv(csv_path)
        summary.append(result)

    # Print summary to stdout
    print("file,rows,better_match_pct,next_month_match_pct")
    for row in summary:
        print(
            f"{row['file']},"
            f"{row['rows']},"
            f"{row['better_match_pct']:.2f},"
            f"{row['next_month_match_pct']:.2f}"
        )

    # Write summary to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "rows",
                "better_match_pct",
                "next_month_match_pct",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()

