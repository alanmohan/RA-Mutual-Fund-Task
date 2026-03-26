#!/usr/bin/env python3
"""
Feature ablation experiment for mutual fund comparison.

Measures baseline accuracy on 200 non-tie pairs using the zero-shot prompt,
then removes Beta, Standard Deviation, and Turnover Rates one at a time
(from both fund blocks) and re-measures accuracy to quantify each feature's
contribution to the model's decision quality.

Usage (Colab or local):
    python feature_ablation.py                           # from HF Hub
    python feature_ablation.py --model-path ./models/Llama-3.2-3B-Instruct
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_PATH = REPO_ROOT / "input_csvs" / "mutual_funds_pairs_no_date.csv"
PROMPT_TEMPLATE_PATH = REPO_ROOT / "prompts" / "zero_shot_prompt_template.txt"
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "system_prompt.txt"
OUTPUT_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
N_SAMPLES = 200
RANDOM_STATE = 42
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

FEATURES_TO_ABLATE = ["Beta", "Standard Deviation", "Turnover Rates"]

FEATURE_TEMPLATE_LINES = {
    "Beta": {
        "fund1": "Beta: {beta_1}\n",
        "fund2": "Beta: {beta_2}\n",
    },
    "Standard Deviation": {
        "fund1": "Standard Deviation: {std_dev_1}\n",
        "fund2": "Standard Deviation: {std_dev_2}\n",
    },
    "Turnover Rates": {
        "fund1": "Turnover Rates: {turnover_rates_1}\n",
        "fund2": "Turnover Rates: {turnover_rates_2}\n",
    },
}

# ---------------------------------------------------------------------------
# Medalist ground truth
# ---------------------------------------------------------------------------
MEDALIST_HIERARCHY = {"Negative": 0, "Neutral": 1, "Bronze": 2, "Silver": 3, "Gold": 4}


def compare_medalist(m1, m2):
    v1 = MEDALIST_HIERARCHY.get(str(m1).strip(), -1)
    v2 = MEDALIST_HIERARCHY.get(str(m2).strip(), -1)
    if v1 < 0 or v2 < 0:
        return np.nan
    if v1 > v2:
        return 1
    if v2 > v1:
        return 2
    return np.nan


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
MUTUAL_FUND_RE = re.compile(r"mutual\s+fund\s+([12])", re.IGNORECASE)
FUND_NUM_RE = re.compile(r"\bfund\s+([12])\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"ANSWER:\s*([12])", re.IGNORECASE)


def parse_choice(text: str):
    if not text or not isinstance(text, str):
        return np.nan
    text = text.strip()
    if not text:
        return np.nan

    tail = text[-500:] if len(text) > 500 else text
    matches = list(MUTUAL_FUND_RE.finditer(tail))
    if matches:
        return int(matches[-1].group(1))

    matches = list(MUTUAL_FUND_RE.finditer(text))
    if matches:
        return int(matches[-1].group(1))

    small_tail = text[-200:] if len(text) > 200 else text
    matches = list(FUND_NUM_RE.finditer(small_tail))
    if matches:
        return int(matches[-1].group(1))

    matches = list(ANSWER_RE.finditer(tail))
    if matches:
        return int(matches[-1].group(1))

    last_100 = text[-100:]
    if re.search(r"\b1\b", last_100) and not re.search(r"\b2\b", last_100):
        return 1
    if re.search(r"\b2\b", last_100) and not re.search(r"\b1\b", last_100):
        return 2

    return np.nan


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def clean_str(x):
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    return s if s else "NA"


TEMPLATE_VALUES_MAP = {
    "expense_ratio_net_1": "Expense Ratio - Net_1",
    "sharpe_3y_1": "3 Year Sharpe Ratio_1",
    "std_dev_1": "Standard Deviation_1",
    "return_3y_1": "3 Yr_1",
    "beta_1": "Beta_1",
    "manager_tenure_1": "Manager Tenure_1",
    "inception_date_1": "Inception Date_1",
    "assets_millions_1": "Assets (Millions)_1",
    "turnover_rates_1": "Turnover Rates_1",
    "load_yn_1": "Load (Y/N)_1",
    "ntf_1": "NTF_1",
    "expense_ratio_net_2": "Expense Ratio - Net_2",
    "sharpe_3y_2": "3 Year Sharpe Ratio_2",
    "std_dev_2": "Standard Deviation_2",
    "return_3y_2": "3 Yr_2",
    "beta_2": "Beta_2",
    "manager_tenure_2": "Manager Tenure_2",
    "inception_date_2": "Inception Date_2",
    "assets_millions_2": "Assets (Millions)_2",
    "turnover_rates_2": "Turnover Rates_2",
    "load_yn_2": "Load (Y/N)_2",
    "ntf_2": "NTF_2",
}


def _build_template_values(row):
    return {k: clean_str(row[col]) for k, col in TEMPLATE_VALUES_MAP.items()}


def build_prompt(template: str, row):
    return template.format(**_build_template_values(row))


def remove_feature_from_template(template: str, feature_name: str) -> str:
    """Remove the lines for a feature from both fund blocks in the template."""
    lines_to_remove = FEATURE_TEMPLATE_LINES[feature_name]
    result = template
    for line_template in lines_to_remove.values():
        pattern = re.escape(line_template.split(":")[0]) + r":.*\n"
        result = re.sub(pattern, "", result)
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        print(f"  Loaded on GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        model.to("mps")
        print("  Loaded on Apple MPS")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu"
        )
        print("  Loaded on CPU")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, tokenizer, prompts, system_msg, batch_size=BATCH_SIZE):
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch_prompts = prompts[i : i + batch_size]

        batch_texts = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        for output in outputs:
            new_tokens = output[inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

        if (i // batch_size + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(responses, ground_truths):
    predictions = [parse_choice(r) for r in responses]
    n_total = len(predictions)
    n_valid = sum(1 for p in predictions if not pd.isna(p))
    n_correct = sum(
        1 for p, gt in zip(predictions, ground_truths)
        if not pd.isna(p) and not pd.isna(gt) and int(p) == int(gt)
    )
    n_parseable = sum(
        1 for p, gt in zip(predictions, ground_truths)
        if not pd.isna(p) and not pd.isna(gt)
    )
    accuracy = n_correct / n_parseable if n_parseable > 0 else 0.0
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parseable": n_parseable,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature ablation experiment")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path or HF Hub model name (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_SAMPLES, help="Number of non-tie pairs to sample"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Inference batch size"
    )
    args = parser.parse_args()

    model_path = args.model_path or MODEL_NAME
    n_samples = args.n_samples
    batch_size = args.batch_size

    # Load data
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total pairs: {len(df)}")

    valid_mask = df.apply(
        lambda row: not pd.isna(compare_medalist(row["Medalist_1"], row["Medalist_2"])),
        axis=1,
    )
    df_valid = df[valid_mask].copy()
    print(f"Non-tie pairs: {len(df_valid)}")

    n_pairs = min(n_samples, len(df_valid))
    df_sample = df_valid.sample(n=n_pairs, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled {n_pairs} pairs (random_state={RANDOM_STATE})")

    ground_truths = [
        compare_medalist(row["Medalist_1"], row["Medalist_2"])
        for _, row in df_sample.iterrows()
    ]

    # Load templates
    template_full = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    system_msg = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    # Load model
    model, tokenizer = load_model(model_path)

    # Results collector
    all_results = {}

    # -----------------------------------------------------------------------
    # 1. Baseline (all features)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONDITION: Baseline (all features)")
    print("=" * 60)

    prompts_baseline = [build_prompt(template_full, row) for _, row in df_sample.iterrows()]
    print(f"Prompt preview (sample 0):\n{prompts_baseline[0][:300]}...\n")

    t0 = time.time()
    responses_baseline = run_inference(model, tokenizer, prompts_baseline, system_msg, batch_size)
    elapsed = time.time() - t0

    metrics = evaluate(responses_baseline, ground_truths)
    metrics["time_sec"] = elapsed
    all_results["baseline"] = metrics

    print(
        f"  Accuracy: {metrics['accuracy']:.4f} "
        f"({metrics['n_correct']}/{metrics['n_parseable']} parseable, "
        f"{metrics['n_valid']}/{metrics['n_total']} valid)"
    )
    print(f"  Time: {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Ablations (remove one feature at a time)
    # -----------------------------------------------------------------------
    for feature_name in FEATURES_TO_ABLATE:
        print("\n" + "=" * 60)
        print(f"CONDITION: Remove {feature_name}")
        print("=" * 60)

        template_ablated = remove_feature_from_template(template_full, feature_name)

        prompts_ablated = [
            build_prompt(template_ablated, row) for _, row in df_sample.iterrows()
        ]
        print(f"Prompt preview (sample 0):\n{prompts_ablated[0][:300]}...\n")

        t0 = time.time()
        responses_ablated = run_inference(
            model, tokenizer, prompts_ablated, system_msg, batch_size
        )
        elapsed = time.time() - t0

        metrics = evaluate(responses_ablated, ground_truths)
        metrics["time_sec"] = elapsed
        condition_key = f"remove_{feature_name.lower().replace(' ', '_')}"
        all_results[condition_key] = metrics

        baseline_acc = all_results["baseline"]["accuracy"]
        delta = metrics["accuracy"] - baseline_acc
        print(
            f"  Accuracy: {metrics['accuracy']:.4f} "
            f"({metrics['n_correct']}/{metrics['n_parseable']} parseable, "
            f"{metrics['n_valid']}/{metrics['n_total']} valid)"
        )
        print(f"  Delta from baseline: {delta:+.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 3. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline_acc = all_results["baseline"]["accuracy"]
    print(f"\n{'Condition':<30} {'Accuracy':>10} {'Delta':>10} {'Correct/Parse':>15}")
    print("-" * 70)
    for cond_name, m in all_results.items():
        delta = m["accuracy"] - baseline_acc
        delta_str = f"{delta:+.4f}" if cond_name != "baseline" else "—"
        print(
            f"{cond_name:<30} {m['accuracy']:>10.4f} {delta_str:>10} "
            f"{m['n_correct']:>6}/{m['n_parseable']:<6}"
        )

    # -----------------------------------------------------------------------
    # 4. Save results
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "feature_ablation_results.json"

    save_data = {}
    for cond_name, m in all_results.items():
        save_data[cond_name] = {
            "accuracy": m["accuracy"],
            "n_total": m["n_total"],
            "n_valid": m["n_valid"],
            "n_parseable": m["n_parseable"],
            "n_correct": m["n_correct"],
            "time_sec": m["time_sec"],
            "delta_from_baseline": m["accuracy"] - baseline_acc if cond_name != "baseline" else 0.0,
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save per-sample predictions for detailed analysis
    detail_rows = []
    for i, row in df_sample.iterrows():
        entry = {
            "sample_idx": i,
            "ground_truth": int(ground_truths[i]) if not pd.isna(ground_truths[i]) else None,
        }
        for cond_name, m in all_results.items():
            pred = m["predictions"][i]
            entry[f"pred_{cond_name}"] = int(pred) if not pd.isna(pred) else None
        detail_rows.append(entry)

    detail_df = pd.DataFrame(detail_rows)
    detail_path = OUTPUT_DIR / "feature_ablation_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Per-sample predictions saved to {detail_path}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    main()
