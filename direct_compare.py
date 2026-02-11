from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pairs(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def first_n_pairs(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"Requested n={n} exceeds dataset size {len(df)}.")
    return df.head(n).reset_index(drop=True)


def build_user_prompt(row: pd.Series, template: str) -> str:
    return template.format(
        name_1=row.get("Name_1", ""),
        name_2=row.get("Name_2", ""),
        expense_ratio_net_1=row["Expense Ratio - Net_1"],
        sharpe_3y_1=row["3 Year Sharpe Ratio_1"],
        std_dev_1=row["Standard Deviation_1"],
        return_3y_1=row["3 Yr_1"],
        beta_1=row["Beta_1"],
        manager_tenure_1=row["Manager Tenure_1"],
        inception_date_1=row["Inception Date_1"],
        assets_millions_1=row["Assets (Millions)_1"],
        turnover_rates_1=row["Turnover Rates_1"],
        load_yn_1=row["Load (Y/N)_1"],
        ntf_1=row["NTF_1"],
        expense_ratio_net_2=row["Expense Ratio - Net_2"],
        sharpe_3y_2=row["3 Year Sharpe Ratio_2"],
        std_dev_2=row["Standard Deviation_2"],
        return_3y_2=row["3 Yr_2"],
        beta_2=row["Beta_2"],
        manager_tenure_2=row["Manager Tenure_2"],
        inception_date_2=row["Inception Date_2"],
        assets_millions_2=row["Assets (Millions)_2"],
        turnover_rates_2=row["Turnover Rates_2"],
        load_yn_2=row["Load (Y/N)_2"],
        ntf_2=row["NTF_2"],
    )


def call_model(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_output_tokens=128,
    )
    return response.output_text or ""


def write_results(
    rows: Iterable[dict], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "name_1",
        "name_2",
        "better_fund_response",
        "next_month_response",
        "model",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct API comparisons for mutual fund pairs."
    )
    parser.add_argument(
        "--csv",
        default="mutual_funds_pairs_no_date.csv",
        help="Path to mutual_funds_pairs.csv",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of pairs to process (first n rows).",
    )
    parser.add_argument(
        "--system-prompt",
        default="system_prompt.txt",
        help="Path to system prompt file.",
    )
    parser.add_argument(
        "--user-prompt",
        default="zero_shot_prompt_template.txt",
        help="Path to user prompt template file.",
    )
    parser.add_argument(
        "--user-prompt-next-month",
        default="zero_shot_next_month_prompt_template.txt",
        help="Path to next-month user prompt template file.",
    )
    parser.add_argument(
        "--output-csv",
        default="direct_output.csv",
        help="Where to save the CSV results.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model name for requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    system_prompt = load_prompt(Path(args.system_prompt))
    user_template = load_prompt(Path(args.user_prompt))
    user_template_next = load_prompt(Path(args.user_prompt_next_month))

    df = read_pairs(Path(args.csv))
    pairs = first_n_pairs(df, args.n)

    client = OpenAI()
    results = []
    for idx, row in pairs.iterrows():
        print(f"Processing row {idx}")
        user_prompt = build_user_prompt(row, user_template)
        response_text = call_model(
            client, args.model, system_prompt, user_prompt
        )
        user_prompt_next = build_user_prompt(row, user_template_next)
        response_text_next = call_model(
            client, args.model, system_prompt, user_prompt_next
        )
        results.append(
            {
                "pair_index": idx,
                "name_1": row.get("Name_1", ""),
                "name_2": row.get("Name_2", ""),
                "better_fund_response": response_text.strip(),
                "next_month_response": response_text_next.strip(),
                "model": args.model,
            }
        )

    write_results(results, Path(args.output_csv))
    print(f"Wrote results to {args.output_csv}")


if __name__ == "__main__":
    main()

