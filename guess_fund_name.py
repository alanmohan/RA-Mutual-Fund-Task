from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class FundNameGuess(BaseModel):
    """Structured output for fund name guessing."""
    guessed_name: str
    confidence: int  # 0-100


def read_pairs(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_guess_prompt(row: pd.Series, fund_num: int) -> str:
    """Build a prompt asking the LLM to guess the fund name based on features."""
    if fund_num == 1:
        name = row.get("Name_1", "")
        expense_ratio = row["Expense Ratio - Net_1"]
        sharpe = row["3 Year Sharpe Ratio_1"]
        std_dev = row["Standard Deviation_1"]
        return_3y = row["3 Yr_1"]
        beta = row["Beta_1"]
        manager_tenure = row["Manager Tenure_1"]
        inception_date = row["Inception Date_1"]
        assets = row["Assets (Millions)_1"]
        turnover = row["Turnover Rates_1"]
        load = row["Load (Y/N)_1"]
        ntf = row["NTF_1"]
        rank = row.get("3-year Rank_1", "")
        category = row.get("Morningstar Category", "")
    else:
        name = row.get("Name_2", "")
        expense_ratio = row["Expense Ratio - Net_2"]
        sharpe = row["3 Year Sharpe Ratio_2"]
        std_dev = row["Standard Deviation_2"]
        return_3y = row["3 Yr_2"]
        beta = row["Beta_2"]
        manager_tenure = row["Manager Tenure_2"]
        inception_date = row["Inception Date_2"]
        assets = row["Assets (Millions)_2"]
        turnover = row["Turnover Rates_2"]
        load = row["Load (Y/N)_2"]
        ntf = row["NTF_2"]
        rank = row.get("3-year Rank_2", "")
        category = row.get("Morningstar Category", "")
    
    prompt = f"""Task: Based on the following mutual fund features, guess the name of the mutual fund. Provide your best guess and a confidence score (0-100).

Mutual Fund Features:
3-year Rank: {rank}
Expense Ratio - Net: {expense_ratio}
3 Year Sharpe Ratio: {sharpe}
Standard Deviation: {std_dev}
3 Yr: {return_3y}%
Beta: {beta}
Manager Tenure: {manager_tenure}
Inception Date: {inception_date}
Assets (Millions): {assets}
Turnover Rates: {turnover}%
Load (Y/N): {load}
NTF: {ntf}
"""
    
    return prompt


def call_model(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str
) -> FundNameGuess:
    """Call the OpenAI API using structured outputs and return parsed response."""
    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_output_tokens=200,
            text_format=FundNameGuess,
        )
        
        # Get the parsed response - structured outputs ensures it matches the schema
        parsed = response.output_parsed
        if parsed is None:
            # Fallback if parsing failed (shouldn't happen with structured outputs)
            return FundNameGuess(
                guessed_name="[Parse Error]",
                confidence=0
            )
        return parsed
        
    except Exception as e:
        # Handle any errors (network, API, etc.)
        print(f"Error calling API: {e}")
        return FundNameGuess(
            guessed_name=f"[Error: {str(e)[:50]}]",
            confidence=0
        )


def write_results(
    results: Iterable[dict], output_path: Path
) -> None:
    """Write results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "actual_name",
        "guessed_name",
        "confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guess mutual fund names based on features using LLM."
    )
    parser.add_argument(
        "--csv",
        default="mutual_funds_pairs_no_date.csv",
        help="Path to mutual_funds_pairs_no_date.csv",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of rows to sample randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--system-prompt",
        default="system_prompt.txt",
        help="Path to system prompt file.",
    )
    parser.add_argument(
        "--output-csv",
        default="guess_fund_name_output.csv",
        help="Where to save the CSV results.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="Model name for requests (must support structured outputs: gpt-4o-2024-08-06, gpt-4o-mini-2024-07-18, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    # Use a system prompt appropriate for name guessing task
    system_prompt = """You are a financial analyst assistant specializing in mutual fund identification. 
Your task is to analyze mutual fund features (expense ratios, performance metrics, risk measures, fund characteristics) 
and make an educated guess about the fund's name based on these characteristics. 

Provide your best guess for the fund name and a confidence score between 0 and 100."""

    df = read_pairs(Path(args.csv))
    
    # Validate n
    if args.n > len(df):
        print(f"Warning: Requested {args.n} rows but only {len(df)} available. Processing all rows.")
        args.n = len(df)
    
    # Sample n random rows, preserving original index
    if args.seed is not None:
        sampled_rows = df.sample(n=args.n, random_state=args.seed)
    else:
        sampled_rows = df.sample(n=args.n, random_state=None)

    client = OpenAI()
    results = []
    processed_fund_names = set()  # Track unique fund names
    
    print(f"Sampling {args.n} random rows and processing unique mutual funds...")
    
    for original_idx, row in sampled_rows.iterrows():
        
        # Process fund 1 if not already processed
        fund_name_1 = row.get("Name_1", "")
        if fund_name_1 and fund_name_1 not in processed_fund_names:
            print(f"Processing fund 1: {fund_name_1}")
            user_prompt_1 = build_guess_prompt(row, 1)
            response_1 = call_model(client, args.model, system_prompt, user_prompt_1)
            results.append({
                "row_index": original_idx,
                "actual_name": fund_name_1,
                "guessed_name": response_1.guessed_name,
                "confidence": response_1.confidence,
            })
            processed_fund_names.add(fund_name_1)
        elif fund_name_1:
            print(f"Skipping duplicate fund 1: {fund_name_1}")
        
        # Process fund 2 if not already processed
        fund_name_2 = row.get("Name_2", "")
        if fund_name_2 and fund_name_2 not in processed_fund_names:
            print(f"Processing fund 2: {fund_name_2}")
            user_prompt_2 = build_guess_prompt(row, 2)
            response_2 = call_model(client, args.model, system_prompt, user_prompt_2)
            results.append({
                "row_index": original_idx,
                "actual_name": fund_name_2,
                "guessed_name": response_2.guessed_name,
                "confidence": response_2.confidence,
            })
            processed_fund_names.add(fund_name_2)
        elif fund_name_2:
            print(f"Skipping duplicate fund 2: {fund_name_2}")

    write_results(results, Path(args.output_csv))
    
    # Check for matches (guessed_name contained in actual_name)
    matches = 0
    for result in results:
        actual = str(result["actual_name"]).lower()
        guessed = str(result["guessed_name"]).lower()
        if guessed and guessed in actual:
            matches += 1
    
    print(f"\nCompleted! Wrote results to {args.output_csv}")
    print(f"Processed {len(results)} unique mutual funds from {args.n} sampled rows.")
    print(f"Total unique funds found: {len(processed_fund_names)}")
    print(f"Matches (guessed_name contained in actual_name): {matches} out of {len(results)} ({matches/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()

