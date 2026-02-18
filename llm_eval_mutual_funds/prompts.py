# -*- coding: utf-8 -*-
"""
Prompt builders for LLM Mutual Fund Comparison Experiment
"""

from pathlib import Path
from utils import clean_str

PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"

ZERO_SHOT_TEMPLATE = (PROMPT_DIR / "zero_shot_prompt_template.txt").read_text(encoding="utf-8")
SINGLE_SHOT_TEMPLATE = (PROMPT_DIR / "single_shot_prompt_template.txt").read_text(encoding="utf-8")

def _build_values(row) -> dict:
    """
    Map dataset fields to prompt template placeholders.
    Fund names are intentionally left blank to avoid brand leakage.
    """
    return {
        "name_1": "",
        "name_2": "",
        "expense_ratio_net_1": clean_str(row["Expense Ratio - Net_1"]),
        "sharpe_3y_1": clean_str(row["3 Year Sharpe Ratio_1"]),
        "std_dev_1": clean_str(row["Standard Deviation_1"]),
        "return_3y_1": clean_str(row["3 Yr_1"]),
        "beta_1": clean_str(row["Beta_1"]),
        "manager_tenure_1": clean_str(row["Manager Tenure_1"]),
        "inception_date_1": clean_str(row["Inception Date_1"]),
        "assets_millions_1": clean_str(row["Assets (Millions)_1"]),
        "turnover_rates_1": clean_str(row["Turnover Rates_1"]),
        "load_yn_1": clean_str(row["Load (Y/N)_1"]),
        "ntf_1": clean_str(row["NTF_1"]),
        "expense_ratio_net_2": clean_str(row["Expense Ratio - Net_2"]),
        "sharpe_3y_2": clean_str(row["3 Year Sharpe Ratio_2"]),
        "std_dev_2": clean_str(row["Standard Deviation_2"]),
        "return_3y_2": clean_str(row["3 Yr_2"]),
        "beta_2": clean_str(row["Beta_2"]),
        "manager_tenure_2": clean_str(row["Manager Tenure_2"]),
        "inception_date_2": clean_str(row["Inception Date_2"]),
        "assets_millions_2": clean_str(row["Assets (Millions)_2"]),
        "turnover_rates_2": clean_str(row["Turnover Rates_2"]),
        "load_yn_2": clean_str(row["Load (Y/N)_2"]),
        "ntf_2": clean_str(row["NTF_2"]),
    }


def build_prompt_baseline(row) -> str:
    """
    Baseline prompt uses the original zero-shot template.
    """
    return ZERO_SHOT_TEMPLATE.format(**_build_values(row))


def build_prompt_zero_shot_cot(row) -> str:
    """
    Zero-shot condition uses the original zero-shot template.
    """
    return ZERO_SHOT_TEMPLATE.format(**_build_values(row))


def build_prompt_few_shot_cot(row) -> str:
    """
    Few-shot condition uses the original single-shot template.
    """
    return SINGLE_SHOT_TEMPLATE.format(**_build_values(row))


# ============================================================================
# PROMPT FUNCTION REGISTRY
# ============================================================================

PROMPT_BUILDERS = {
    "build_prompt_baseline": build_prompt_baseline,
    "build_prompt_zero_shot_cot": build_prompt_zero_shot_cot,
    "build_prompt_few_shot_cot": build_prompt_few_shot_cot,
}
