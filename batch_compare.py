from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI, APIError, APITimeoutError

load_dotenv()

@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    better_template: str
    next_month_template: str

def load_prompts(
    system_path: Path, better_path: Path, next_month_path: Path
) -> PromptBundle:
    return PromptBundle(
        system_prompt=system_path.read_text(encoding="utf-8"),
        better_template=better_path.read_text(encoding="utf-8"),
        next_month_template=next_month_path.read_text(encoding="utf-8"),
    )

def read_pairs(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def sample_pairs(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"Requested n={n} exceeds dataset size {len(df)}.")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def get_indices_from_existing_csvs(output_dir: Path) -> set[int]:
    """Read indices from existing CSV files in output directory."""
    csv_files = sorted(output_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {output_dir} to resume from.")
    
    all_indices = set()
    for csv_file in csv_files:
        try:
            df_csv = pd.read_csv(csv_file)
            if "index" in df_csv.columns:
                indices = df_csv["index"].dropna().astype(int).unique()
                all_indices.update(indices)
        except Exception as e:
            print(f"Warning: Could not read indices from {csv_file}: {e}")
            continue
    
    if not all_indices:
        raise ValueError(f"No valid indices found in CSV files in {output_dir}.")
    
    return all_indices

def filter_df_by_indices(df: pd.DataFrame, indices: set[int]) -> pd.DataFrame:
    """Filter dataframe to only include rows with specified indices."""
    filtered = df[df.index.isin(indices)].copy()
    if len(filtered) == 0:
        raise ValueError(f"No rows found with indices {indices} in dataset.")
    return filtered.reset_index(drop=True)

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

def batch_requests(
    df: pd.DataFrame,
    prompts: PromptBundle,
    model: str,
    exp_id: str,
    hyperparams: dict,
) -> Iterable[dict]:
    for _, row in df.iterrows():
        for prompt_type, template in (
            ("better", prompts.better_template),
            ("next_month", prompts.next_month_template),
        ):
            user_prompt = build_user_prompt(row, template)
            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": prompts.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_output_tokens": 16,
            }
            body.update(hyperparams)
            yield {
                "custom_id": f"{exp_id}|{prompt_type}|{row['index']}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }

def write_jsonl(requests: Iterable[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

def create_batch(
    client: OpenAI, input_jsonl: Path, completion_window: str = "24h"
) -> str:
    file_obj = client.files.create(
        file=input_jsonl.open("rb"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return batch.id


def wait_for_batch_capacity(
    client: OpenAI,
    min_idle: int = 1,
    poll_seconds: int = 60,
) -> None:
    while True:
        batches = client.batches.list(limit=50)
        active = [
            b
            for b in batches.data
            if b.status in {"validating", "in_progress", "finalizing"}
        ]
        if len(active) < min_idle:
            return
        print(
            f"Waiting for batch capacity (active={len(active)}). "
            f"Sleeping {poll_seconds}s..."
        )
        time.sleep(poll_seconds)

def poll_batch(
    client: OpenAI,
    batch_id: str,
    poll_seconds: int = 10,
    label: str = "",
) -> dict:
    last_report = 0.0
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        now = time.time()
        if now - last_report >= 60:
            counts = batch.request_counts
            completed = getattr(counts, "completed", 0) if counts else 0
            total = getattr(counts, "total", 0) if counts else 0
            failed = getattr(counts, "failed", 0) if counts else 0
            elapsed_s = int(now - start)
            prefix = f"[{label}] " if label else ""
            print(
                f"{prefix}Batch status: {batch.status} | "
                f"completed {completed}/{total} | failed {failed} | "
                f"elapsed {elapsed_s}s"
            )
            last_report = now
        if batch.status in {"completed", "failed", "cancelled", "expired"}:
            return batch.model_dump()
        time.sleep(poll_seconds)

def download_batch_output(client: OpenAI, output_file_id: str, dest: Path) -> None:
    content = client.files.content(output_file_id)
    dest.write_bytes(content.read())

def parse_rank_value(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lowest_rank_label(rank_1: str | None, rank_2: str | None) -> str:
    r1 = parse_rank_value(rank_1)
    r2 = parse_rank_value(rank_2)
    if r1 is not None and (r2 is None or r1 <= r2):
        return "mutual fund 1"
    if r2 is not None:
        return "mutual fund 2"
    return ""


def extract_response_text(body: dict) -> str:
    output = body.get("output") or []
    if output:
        content = output[0].get("content") or []
        if content:
            return content[0].get("text", "").strip()
    return ""


def write_experiment_csv(
    output_jsonls: list[Path],
    index_lookup: dict,
    output_csv: Path,
) -> None:
    import csv as _csv

    combined = {}
    for output_jsonl in output_jsonls:
        with output_jsonl.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                custom_id = item.get("custom_id")
                response_body = item.get("response", {}).get("body", {})
                text = extract_response_text(response_body)
                if not custom_id:
                    continue
                try:
                    _, prompt_type, index_str = custom_id.split("|", 2)
                    index_val = int(index_str)
                except ValueError:
                    continue
                row = index_lookup.get(index_val)
                if row is None:
                    continue
                entry = combined.setdefault(
                    index_val,
                    {
                        "index": index_val,
                        "name_1": row.get("Name_1", ""),
                        "name_2": row.get("Name_2", ""),
                        "better_fund": "",
                        "next_month": "",
                        "lowest_rank": lowest_rank_label(
                            row.get("3-year Rank_1"),
                            row.get("3-year Rank_2"),
                        ),
                    },
                )
                if prompt_type == "better":
                    entry["better_fund"] = text
                elif prompt_type == "next_month":
                    entry["next_month"] = text

    rows = [combined[k] for k in sorted(combined)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "name_1",
                "name_2",
                "better_fund",
                "next_month",
                "lowest_rank",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch compare mutual fund pairs with OpenAI."
    )
    parser.add_argument(
        "--csv",
        default="mutual_funds_pairs_no_date.csv",
        help="Path to mutual_funds_pairs.csv",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--system-prompt",
        default="system_prompt.txt",
        help="Path to system prompt file.",
    )
    parser.add_argument(
        "--user-prompt-better-zero",
        default="zero_shot_prompt_template.txt",
        help="Zero-shot template for better fund.",
    )
    parser.add_argument(
        "--user-prompt-next-month-zero",
        default="zero_shot_next_month_prompt_template.txt",
        help="Zero-shot template for next-month return.",
    )
    parser.add_argument(
        "--user-prompt-better-single",
        default="single_shot_prompt_template.txt",
        help="Single-shot template for better fund.",
    )
    parser.add_argument(
        "--user-prompt-next-month-single",
        default="single_shot_next_month_prompt_template.txt",
        help="Single-shot template for next-month return.",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_experiments",
        help="Directory to write batch inputs and outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per batch chunk. If not set, calculates to produce exactly 2 chunks per experiment.",
    )
    parser.add_argument(
        "--wait-capacity",
        action="store_true",
        help="If set, wait for batch capacity before each chunk.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="If set, resume from existing CSV indices instead of random sampling.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model name for batch requests.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    df = read_pairs(csv_path)
    df = df.copy()
    df["index"] = df.index

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(
        timeout=120.0,
        max_retries=5,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample once and reuse across all experiments (to keep experiments comparable)
    if args.skip:
        print(f"[--skip] Reading indices from existing CSVs in {output_dir}")
        indices = get_indices_from_existing_csvs(output_dir)
        print(f"[--skip] Found {len(indices)} unique indices")
        sampled_master = filter_df_by_indices(df, indices)
        print(f"[--skip] Filtered dataset to {len(sampled_master)} rows")
    else:
        sampled_master = sample_pairs(df, args.n, args.seed)

    experiments = [
        {
            "id": "zero_shot_temp0",
            "prompt_type": "zero",
            "hyperparams": {"temperature": 0.0},
        },
        {
            "id": "zero_shot_top_p_0_1",
            "prompt_type": "zero",
            "hyperparams": {"top_p": 0.1},
        },
        {
            "id": "single_shot_temp0",
            "prompt_type": "single",
            "hyperparams": {"temperature": 0.0},
        },
        {
            "id": "single_shot_top_p_0_1",
            "prompt_type": "single",
            "hyperparams": {"top_p": 0.1},
        },
    ]

    for exp_idx, exp in enumerate(experiments, start=1):
        sampled = sampled_master
        if exp["prompt_type"] == "zero":
            prompts = load_prompts(
                Path(args.system_prompt),
                Path(args.user_prompt_better_zero),
                Path(args.user_prompt_next_month_zero),
            )
        else:
            prompts = load_prompts(
                Path(args.system_prompt),
                Path(args.user_prompt_better_single),
                Path(args.user_prompt_next_month_single),
            )

        index_lookup = {
            int(row["index"]): row for _, row in sampled.iterrows()
        }

        output_jsonls: list[Path] = []

        # Calculate chunk_size to ensure exactly 2 chunks per experiment
        if args.chunk_size is None:
            chunk_size = (len(sampled) + 1) // 2  # ceil(n/2) to get 2 chunks
        else:
            chunk_size = args.chunk_size

        total_chunks = (len(sampled) + chunk_size - 1) // chunk_size
        for chunk_idx in range(0, len(sampled), chunk_size):
            chunk = sampled.iloc[chunk_idx : chunk_idx + chunk_size]
            chunk_num = chunk_idx // chunk_size + 1
            chunk_id = f"{exp['id']}_chunk_{chunk_num}"

            input_jsonl = output_dir / f"{chunk_id}_input.jsonl"
            write_jsonl(
                batch_requests(
                    chunk, prompts, args.model, exp["id"], exp["hyperparams"]
                ),
                input_jsonl,
            )

            if args.wait_capacity:
                wait_for_batch_capacity(client)

            try:
                batch_id = create_batch(client, input_jsonl)
            except (APITimeoutError, APIError) as exc:
                print(
                    f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                    f"Failed to create batch: {exc}"
                )
                continue

            print(
                f"[{chunk_id}] ({chunk_num}/{total_chunks}) Created batch: {batch_id}"
            )

            batch_info = poll_batch(
                client,
                batch_id,
                label=f"{chunk_id} {chunk_num}/{total_chunks}",
            )
            print(
                f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                f"Final status: {batch_info['status']}"
            )

            output_jsonl = output_dir / f"{chunk_id}_output.jsonl"
            if batch_info.get("output_file_id"):
                download_batch_output(
                    client, batch_info["output_file_id"], output_jsonl
                )
                output_jsonls.append(output_jsonl)
                print(
                    f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                    f"Downloaded output to {output_jsonl}"
                )
            elif batch_info.get("error_file_id"):
                error_path = output_dir / f"{chunk_id}_errors.jsonl"
                download_batch_output(
                    client, batch_info["error_file_id"], error_path
                )
                print(
                    f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                    f"Downloaded error report to {error_path}"
                )
                continue
            else:
                counts = batch_info.get("request_counts", {})
                error_path = output_dir / f"{chunk_id}_batch_info.json"
                error_path.write_text(
                    json.dumps(batch_info, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                    f"No output_file_id or error_file_id. "
                    f"Status: {batch_info.get('status')} | counts: {counts}. "
                    f"Wrote batch details to {error_path}"
                )
                continue

        if not output_jsonls:
            print(f"[{exp['id']}] No outputs to write; skipping CSV.")
            continue

        output_csv = output_dir / f"{exp['id']}.csv"
        write_experiment_csv(output_jsonls, index_lookup, output_csv)
        print(f"[{exp['id']}] Wrote CSV to {output_csv}")


if __name__ == "__main__":
    main()

