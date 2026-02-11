# -*- coding: utf-8 -*-
"""
Utility functions for LLM Mutual Fund Comparison Experiment
"""

import re
import numpy as np
import pandas as pd

# ============================================================================
# STRING CLEANING
# ============================================================================


def clean_str(x):
    """Clean and format string values, handle NaN."""
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    return s if s else "NA"


# ============================================================================
# MEDALIST GROUND TRUTH
# ============================================================================

# Medalist rating hierarchy (higher is better)
MEDALIST_HIERARCHY = {
    "Negative": 0,
    "Neutral": 1,
    "Bronze": 2,
    "Silver": 3,
    "Gold": 4,
}


def get_medalist_value(medalist: str) -> int:
    """Get numeric value for medalist rating."""
    if medalist is None:
        return -1
    return MEDALIST_HIERARCHY.get(str(medalist).strip(), -1)


def compare_medalist(medalist_1: str, medalist_2: str):
    """
    Compare two medalist ratings.

    Returns:
        1 if fund 1 higher, 2 if fund 2 higher, np.nan if tie or unknown.
    """
    val_1 = get_medalist_value(medalist_1)
    val_2 = get_medalist_value(medalist_2)

    if val_1 < 0 or val_2 < 0:
        return np.nan
    if val_1 > val_2:
        return 1
    if val_2 > val_1:
        return 2
    return np.nan


# ============================================================================
# RESULT PARSING
# ============================================================================

# Regex patterns to extract choice from model output
CHOICE_RE = re.compile(r"CHOICE:\s*([12])", re.IGNORECASE)
ANSWER_RE = re.compile(r"ANSWER:\s*([12])", re.IGNORECASE)
# "mutual fund 1" or "mutual fund 2" (case insensitive)
MUTUAL_FUND_RE = re.compile(r"mutual\s+fund\s+([12])", re.IGNORECASE)
# Fallback: "fund 1" / "fund 2" as final answer (word boundaries)
FUND_NUM_RE = re.compile(r"\bfund\s+([12])\b", re.IGNORECASE)

# Prefer matches in the last N chars of the response (model often puts answer at end)
PREFER_TAIL_CHARS = 500
FALLBACK_TAIL_CHARS = 200


def _last_mutual_fund_match(search_text: str):
    """Return the last 'mutual fund 1' or 'mutual fund 2' match in search_text, or None."""
    matches = list(MUTUAL_FUND_RE.finditer(search_text))
    return matches[-1] if matches else None


def _last_fund_num_match(search_text: str):
    """Return the last 'fund 1' or 'fund 2' match in search_text, or None."""
    matches = list(FUND_NUM_RE.finditer(search_text))
    return matches[-1] if matches else None


def parse_choice(text: str):
    """
    Extract which fund the model picked (1 or 2) from a long explanation.
    Prefers "mutual fund 1" / "mutual fund 2" toward the end of the response,
    then falls back to "fund 1" / "fund 2", CHOICE:/ANSWER:, or standalone 1/2 at end.

    Args:
        text: Model's full text response (may include reasoning).

    Returns:
        int (1 or 2) or np.nan if unable to parse
    """
    if not text or not isinstance(text, str):
        return np.nan

    text = text.strip()
    if not text:
        return np.nan

    # 1) Prefer "mutual fund 1" or "mutual fund 2" in the tail (last PREFER_TAIL_CHARS chars)
    tail = text[-PREFER_TAIL_CHARS:] if len(text) > PREFER_TAIL_CHARS else text
    match = _last_mutual_fund_match(tail)
    if match:
        return int(match.group(1))

    # 2) Else any "mutual fund 1" / "mutual fund 2" in the full text (take last occurrence)
    match = _last_mutual_fund_match(text)
    if match:
        return int(match.group(1))

    # 3) "fund 1" / "fund 2" in the last FALLBACK_TAIL_CHARS (model sometimes drops "mutual")
    small_tail = text[-FALLBACK_TAIL_CHARS:] if len(text) > FALLBACK_TAIL_CHARS else text
    match = _last_fund_num_match(small_tail)
    if match:
        return int(match.group(1))

    # 4) CHOICE: 1 or ANSWER: 2 style (take last occurrence in tail)
    for pattern in (CHOICE_RE, ANSWER_RE):
        matches = list(pattern.finditer(tail))
        if matches:
            return int(matches[-1].group(1))

    # 5) Standalone 1 or 2 only in the last 100 chars
    last_100 = text[-100:]
    if re.search(r"\b1\b", last_100) and not re.search(r"\b2\b", last_100):
        return 1
    if re.search(r"\b2\b", last_100) and not re.search(r"\b1\b", last_100):
        return 2

    return np.nan
