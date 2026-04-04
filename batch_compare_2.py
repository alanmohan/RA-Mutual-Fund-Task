from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI, APIError, APITimeoutError

import sys

from batch_compare import (
    PromptBundle,
    batch_requests,
    create_batch,
    download_batch_output,
    load_prompts,
    poll_batch,
    read_pairs,
    wait_for_batch_capacity,
    write_experiment_csv,
    write_jsonl,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def load_indices_from_result_csv(indices_csv: Path) -> list[int]:
    ref = pd.read_csv(indices_csv)
    if "index" not in ref.columns:
        raise ValueError(f"{indices_csv} must have an 'index' column.")
    return ref["index"].dropna().astype(int).tolist()


def build_sampled_frame(df_full: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Rows in order of ``indices``; skips must be applied before calling."""
    missing = [i for i in indices if i not in df_full.index]
    if missing:
        raise ValueError(
            f"Indices not found in pairs CSV (first few): {missing[:10]}"
        )
    sampled = df_full.reindex(indices).copy()
    sampled["index"] = sampled.index
    return sampled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-run the single-shot batch compare on the same pair indices as "
            "single_shot_temp0.csv, using single_shot_prompt_template_2.txt for "
            "the primary (better) user prompt."
        )
    )
    p.add_argument(
        "--pairs-csv",
        default="input_csvs/mutual_funds_pairs_no_date.csv",
        help="Full mutual fund pairs dataset (same schema as batch_compare.py).",
    )
    p.add_argument(
        "--indices-csv",
        default="batch_experiments/single_shot_temp0.csv",
        help="CSV with an 'index' column listing original row indices to run.",
    )
    p.add_argument(
        "--system-prompt",
        default="prompts/system_prompt.txt",
        help="System prompt path.",
    )
    p.add_argument(
        "--user-prompt-better",
        default="prompts/single_shot_prompt_template_2.txt",
        help="User template for which fund is better / higher return (v2).",
    )
    p.add_argument(
        "--user-prompt-next-month",
        default="prompts/single_shot_next_month_prompt_template.txt",
        help="User template for next-month prediction (same as single_shot run).",
    )
    p.add_argument(
        "--output-dir",
        default="batch_experiments",
        help="Directory for batch JSONL I/O and output CSV.",
    )
    p.add_argument(
        "--exp-id",
        default="single_shot_temp0_template2",
        help="Experiment id (output: {exp_id}.csv and batch chunk names).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per batch chunk; default ceil(n/2) for two chunks.",
    )
    p.add_argument(
        "--wait-capacity",
        action="store_true",
        help="Wait for batch capacity before each chunk.",
    )
    p.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model name for batch requests.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0 to match single_shot_temp0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("batch_compare_2: starting", flush=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    pairs_path = Path(args.pairs_csv)
    indices_path = Path(args.indices_csv)
    df_full = read_pairs(pairs_path)
    indices = load_indices_from_result_csv(indices_path)
    valid_idx = [i for i in indices if i in df_full.index]
    skipped = [i for i in indices if i not in df_full.index]
    if skipped:
        print(
            f"Warning: {len(skipped)} indices from {indices_path} are not in "
            f"{pairs_path} (max row index {df_full.index.max()}). "
            f"Skipping them. First few: {skipped[:8]}"
        )
    if not valid_idx:
        raise ValueError(
            "No indices overlap the pairs CSV; cannot build a sample."
        )
    sampled = build_sampled_frame(df_full, valid_idx)

    prompts: PromptBundle = load_prompts(
        Path(args.system_prompt),
        Path(args.user_prompt_better),
        Path(args.user_prompt_next_month),
    )

    hyperparams = {"temperature": args.temperature}
    exp_id = args.exp_id

    index_lookup = {int(row["index"]): row for _, row in sampled.iterrows()}

    client = OpenAI(timeout=120.0, max_retries=5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_size is None:
        chunk_size = (len(sampled) + 1) // 2
    else:
        chunk_size = args.chunk_size

    total_chunks = (len(sampled) + chunk_size - 1) // chunk_size
    output_jsonls: list[Path] = []

    for chunk_idx in range(0, len(sampled), chunk_size):
        chunk = sampled.iloc[chunk_idx : chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        chunk_id = f"{exp_id}_chunk_{chunk_num}"

        input_jsonl = output_dir / f"{chunk_id}_input.jsonl"
        write_jsonl(
            batch_requests(chunk, prompts, args.model, exp_id, hyperparams),
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
            download_batch_output(client, batch_info["output_file_id"], output_jsonl)
            output_jsonls.append(output_jsonl)
            print(
                f"[{chunk_id}] ({chunk_num}/{total_chunks}) "
                f"Downloaded output to {output_jsonl}"
            )
        elif batch_info.get("error_file_id"):
            error_path = output_dir / f"{chunk_id}_errors.jsonl"
            download_batch_output(client, batch_info["error_file_id"], error_path)
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
        print(f"[{exp_id}] No outputs to write; skipping CSV.")
        return

    output_csv = output_dir / f"{exp_id}.csv"
    write_experiment_csv(output_jsonls, index_lookup, output_csv)
    print(f"[{exp_id}] Wrote CSV to {output_csv}")


if __name__ == "__main__":
    main()
