# -*- coding: utf-8 -*-
"""
Find the token position where a feature's value appears in the prompt (e.g. "Beta: 1.20"
for mutual fund 2). Use the output to set lp_config.TOKEN_POSITION or pass
--token-position to extract_activations.py for token-position probing experiments.

The position is given as "from end" (negative index): -1 = last token, -2 = second-to-last, etc.
Set TOKEN_POSITION to that value so extraction uses the representation at the value token.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()   # llm_eval_mutual_funds
_REPO_ROOT = _PROJECT_ROOT.parent.resolve()  # repo root
for _p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Feature (probe name) -> (line prefix in prompt, placeholder key for fund 2)
# We find the last occurrence of this line in the prompt (evaluation pair, fund 2)
FEATURE_TO_LINE_PREFIX = {
    "expense_ratio_f1_lower": "Expense Ratio - Net: ",
    "sharpe_f1_higher": "3 Year Sharpe Ratio: ",
    "stdev_f1_lower": "Standard Deviation: ",
    "return_3yr_f1_higher": "3 Yr: ",
    "beta_f1_lower": "Beta: ",
    "tenure_f1_longer": "Manager Tenure: ",
    "inception_f1_older": "Inception Date: ",
    "assets_f1_higher": "Assets (Millions): ",
    "turnover_f1_lower": "Turnover Rates: ",
    "load_f1_no": "Load (Y/N): ",
    "ntf_f1_yes": "NTF: ",
}


def get_prompt_for_row(row, prompt_builder, tokenizer, system_msg: str):
    """Build full prompt (with chat template) for the given row, same as extract_activations."""
    user_content = prompt_builder(row)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{system_msg}\n<|user|>\n{user_content}\n<|assistant|>\n"


def find_value_span_in_prompt(full_prompt: str, line_prefix: str):
    """
    Find the character span of the value (the part after the prefix until newline)
    for the *last* occurrence of the line in the prompt (evaluation pair, fund 2).
    Returns (char_start, char_end) or (None, None).
    """
    idx = full_prompt.rfind(line_prefix)
    if idx < 0:
        return None, None
    value_start = idx + len(line_prefix)
    end = full_prompt.find("\n", value_start)
    if end < 0:
        end = len(full_prompt)
    return value_start, end


def find_token_index_for_span(tokenizer, full_prompt: str, char_start: int, char_end: int, use_last_token_of_span: bool = True):
    """
    Tokenize prompt with return_offsets_mapping and return the token index (0-based)
    that covers [char_start, char_end]. If use_last_token_of_span, return the last token
    that overlaps the span (so the representation has "seen" the full value).
    """
    enc = tokenizer(
        full_prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    offset_mapping = enc["offset_mapping"]
    ids = enc["input_ids"]
    n = len(ids)
    overlapping = []
    for i, (s, e) in enumerate(offset_mapping):
        if s is None or e is None:
            continue
        if not (e <= char_start or s >= char_end):
            overlapping.append(i)
    if not overlapping:
        return None, n
    idx = overlapping[-1] if use_last_token_of_span else overlapping[0]
    return idx, n


def main():
    parser = argparse.ArgumentParser(
        description="Find token position where a feature value appears (e.g. Beta for fund 2). "
        "Set TOKEN_POSITION in lp_config or use --token-position in extract_activations.py."
    )
    parser.add_argument("--model", "-m", type=str, required=True, choices=["llama-3.2-3b", "qwen3-4b"], help="Model key for tokenizer")
    parser.add_argument("--condition", "-c", type=str, default="2_fewshot_cot_temp0", help="Condition (prompt builder)")
    parser.add_argument("--feature", "-f", type=str, required=True, choices=list(FEATURE_TO_LINE_PREFIX.keys()), help="Probe feature (value is for fund 2)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to mutual_funds_pairs CSV")
    parser.add_argument("--row", "-r", type=int, default=0, help="Row index in CSV to build sample prompt (default 0)")
    args = parser.parse_args()

    # Paths
    import importlib.util
    spec = importlib.util.spec_from_file_location("lp_config", _THIS_DIR / "lp_config.py")
    lp_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lp_config)
    if args.data_path is None:
        data_path = Path(lp_config.DATA_PATH)
    else:
        data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return 1

    # Load tokenizer
    from transformers import AutoTokenizer
    model_key = args.model
    model_info = lp_config.MODELS[model_key]
    local_path = model_info["local_path"]
    if Path(local_path).resolve().exists() and (Path(local_path).resolve() / "config.json").exists():
        load_path = str(Path(local_path).resolve())
    else:
        load_path = model_info["hf_name"]
    print(f"Loading tokenizer from {load_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

    # Prompt builder and system msg (same as extract_activations)
    if "fewshot" in args.condition or "few_shot" in args.condition:
        from prompts import build_prompt_few_shot_cot
        prompt_builder = build_prompt_few_shot_cot
    elif "zeroshot" in args.condition:
        from prompts import build_prompt_zero_shot_cot
        prompt_builder = build_prompt_zero_shot_cot
    else:
        from prompts import build_prompt_baseline
        prompt_builder = build_prompt_baseline
    from config import SYSTEM_MSG
    system_msg = SYSTEM_MSG

    # Load row and build prompt
    df = pd.read_csv(data_path)
    row = df.iloc[args.row]
    full_prompt = get_prompt_for_row(row, prompt_builder, tokenizer, system_msg)

    line_prefix = FEATURE_TO_LINE_PREFIX[args.feature]
    char_start, char_end = find_value_span_in_prompt(full_prompt, line_prefix)
    if char_start is None:
        print(f"Line prefix {line_prefix!r} not found in prompt.")
        return 1

    value_snippet = full_prompt[char_start:char_end]
    token_idx, seq_len = find_token_index_for_span(tokenizer, full_prompt, char_start, char_end, use_last_token_of_span=True)
    if token_idx is None:
        print("No token overlaps the value span.")
        return 1

    # Position from end (for TOKEN_POSITION: negative index)
    pos_from_end = token_idx - seq_len  # -1 = last, -2 = second-to-last, ...

    enc = tokenizer(full_prompt, add_special_tokens=True)
    token_ids = enc["input_ids"]
    the_token = tokenizer.decode([token_ids[token_idx]], skip_special_tokens=False)

    print()
    print("=" * 60)
    print(f"Feature: {args.feature}  (fund 2 value in evaluation pair)")
    print(f"Sample row: {args.row}")
    print(f"Line prefix: {line_prefix!r}")
    print(f"Value (snippet): {value_snippet!r}")
    print(f"Sequence length (tokens): {seq_len}")
    print(f"Token index (0-based): {token_idx}")
    print(f"Token (decoded): {the_token!r}")
    print(f"Position from end: {pos_from_end}  (-1 = last token)")
    print()
    print("To extract activations at this position, set in lp_config.py:")
    print(f"  TOKEN_POSITION = {pos_from_end}")
    print()
    print("Or run extract_activations.py with:")
    print(f"  --token-position {pos_from_end}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
