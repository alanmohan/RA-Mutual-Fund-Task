# -*- coding: utf-8 -*-
"""
Find the approximate token position (from end) where a feature's value
appears in the prompt for mutual fund 2.

Works on the raw prompt template — no model weights needed, only a tokenizer.
The position is approximate because the actual values have variable length,
but for typical numeric values the error is at most 1-2 tokens.

Can be called as a standalone CLI or imported by plot_position_results.py
to auto-annotate plots.

Usage (from llm_eval_mutual_funds/):
    python token_position_probing/find_feature_position.py \
        --feature beta_f1_lower --model qwen3-4b

    python token_position_probing/find_feature_position.py \
        --feature sharpe_f1_higher --model qwen3-4b \
        --template prompts/single_shot_prompt_template.txt
"""
import argparse
import sys
from pathlib import Path

import importlib.util

_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.resolve()   # llm_eval_mutual_funds
_REPO_ROOT = _PROJECT_ROOT.parent.resolve()   # repo root

for _p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tp_config = _load_mod("_ffp_tp_config", _THIS_DIR / "tp_config.py")

MODELS = tp_config.MODELS

# Feature name -> line prefix in the prompt (fund 2, evaluation pair).
# The value follows immediately after the prefix until the next newline.
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

# Default prompt template (single-shot, fund 2 evaluation pair)
DEFAULT_TEMPLATE = _REPO_ROOT / "prompts" / "single_shot_prompt_template.txt"


def _find_last_occurrence_span(text: str, prefix: str):
    """Return (char_start, char_end) of the value after the last occurrence
    of *prefix* in *text* (up to the next newline)."""
    idx = text.rfind(prefix)
    if idx < 0:
        return None, None
    val_start = idx + len(prefix)
    nl = text.find("\n", val_start)
    val_end = nl if nl >= 0 else len(text)
    return val_start, val_end


def find_feature_token_position(
    feature: str,
    model_key: str,
    template_path: Path | None = None,
    condition: str = "2_fewshot_cot_temp0",
    row_idx: int = 0,
) -> dict:
    """Find the token position (from end) where *feature*'s fund-2 value appears.

    Uses the first CSV row to fill the template, tokenizes the full prompt
    (with chat template), and locates the value tokens.

    Returns a dict with keys:
        feature, line_prefix, value_snippet, token_index, seq_len,
        position_from_end, decoded_token
    """
    import pandas as pd
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Failed to import HuggingFace transformers tokenizer. "
            "This is usually due to an incompatible 'safetensors'/'torch' setup in your environment. "
            "Try using the repo's recommended environment / reinstalling deps."
        ) from e

    if feature not in FEATURE_TO_LINE_PREFIX:
        raise ValueError(f"Unknown feature: {feature}. Available: {list(FEATURE_TO_LINE_PREFIX)}")

    line_prefix = FEATURE_TO_LINE_PREFIX[feature]

    # Load tokenizer
    model_info = MODELS[model_key]
    local_path = Path(model_info["local_path"]).resolve()
    load_path = str(local_path) if (local_path / "config.json").exists() else model_info["hf_name"]
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

    # Build a real prompt from the first data row
    data = pd.read_csv(tp_config.DATA_PATH)
    row = data.iloc[row_idx]

    if "fewshot" in condition or "few_shot" in condition:
        from prompts import build_prompt_few_shot_cot as builder
    elif "zeroshot" in condition:
        from prompts import build_prompt_zero_shot_cot as builder
    else:
        from prompts import build_prompt_baseline as builder

    from config import SYSTEM_MSG
    user_content = builder(row)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{user_content}\n<|assistant|>\n"

    # Find the value span (last occurrence = fund 2 in evaluation pair)
    char_start, char_end = _find_last_occurrence_span(full_prompt, line_prefix)
    if char_start is None:
        raise ValueError(f"Line prefix {line_prefix!r} not found in prompt.")

    value_snippet = full_prompt[char_start:char_end]

    # Tokenize with offset mapping to find token indices
    enc = tokenizer(
        full_prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    offset_mapping = enc["offset_mapping"]
    token_ids = enc["input_ids"]
    seq_len = len(token_ids)

    overlapping = []
    for i, (s, e) in enumerate(offset_mapping):
        if s is None or e is None:
            continue
        if not (e <= char_start or s >= char_end):
            overlapping.append(i)

    if not overlapping:
        raise ValueError("No token overlaps the value span.")

    # Last overlapping token = the one that has "seen" the full value
    token_idx = overlapping[-1]
    pos_from_end = token_idx - seq_len  # negative: -1 = last token
    decoded_token = tokenizer.decode([token_ids[token_idx]], skip_special_tokens=False)

    return {
        "feature": feature,
        "line_prefix": line_prefix,
        "value_snippet": value_snippet,
        "token_index": token_idx,
        "seq_len": seq_len,
        "position_from_end": pos_from_end,
        "decoded_token": decoded_token,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find the token position (from end) where a feature's value "
        "appears in the prompt for fund 2."
    )
    parser.add_argument(
        "--feature", "-f", required=False,
        choices=list(FEATURE_TO_LINE_PREFIX.keys()),
        help="Probe feature name.",
    )
    parser.add_argument(
        "--model", "-m", required=True,
        choices=list(MODELS.keys()),
        help="Model key (for tokenizer).",
    )
    parser.add_argument(
        "--condition", "-c", default="2_fewshot_cot_temp0",
        help="Condition / prompt builder.",
    )
    parser.add_argument(
        "--template", "-t", type=str, default=None,
        help=f"Prompt template path (default: {DEFAULT_TEMPLATE}).",
    )
    parser.add_argument(
        "--row", "-r", type=int, default=0,
        help="CSV row index to build the sample prompt (default: 0).",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Print positions for ALL features (ignores --feature).",
    )
    args = parser.parse_args()

    if args.all:
        features = list(FEATURE_TO_LINE_PREFIX.keys())
    else:
        if args.feature is None:
            parser.error("Missing --feature/-f (or pass --all/-a).")
        features = [args.feature]

    for feat in features:
        result = find_feature_token_position(
            feature=feat,
            model_key=args.model,
            template_path=Path(args.template) if args.template else None,
            condition=args.condition,
            row_idx=args.row,
        )
        print()
        print("=" * 60)
        print(f"Feature:           {result['feature']}")
        print(f"Line prefix:       {result['line_prefix']!r}")
        print(f"Value (snippet):   {result['value_snippet']!r}")
        print(f"Sequence length:   {result['seq_len']} tokens")
        print(f"Token index:       {result['token_index']} (0-based)")
        print(f"Decoded token:     {result['decoded_token']!r}")
        print(f"Position from end: {result['position_from_end']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
