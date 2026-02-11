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
# Pattern to find "mutual fund 1" or "mutual fund 2" (case insensitive, allows words before/after)
MUTUAL_FUND_RE = re.compile(r"mutual\s+fund\s+([12])", re.IGNORECASE)


def parse_choice(text: str):
    """
    Extract choice (1 or 2) from model response.
    Looks for "mutual fund 1" or "mutual fund 2" anywhere in the text (case insensitive).

    Args:
        text: Model's text response

    Returns:
        int (1 or 2) or np.nan if unable to parse
    """
    if not text:
        return np.nan

    text_lower = text.lower()

    # First priority: Look for "mutual fund 1" or "mutual fund 2" anywhere in text
    # Find all matches and take the last one (most likely the final answer)
    mutual_fund_matches = list(MUTUAL_FUND_RE.finditer(text))
    if mutual_fund_matches:
        # Take the last match (most likely the final answer)
        last_match = mutual_fund_matches[-1]
        return int(last_match.group(1))

    # Second priority: Try CHOICE: pattern
    m = CHOICE_RE.search(text)
    if m:
        return int(m.group(1))

    # Third priority: Try ANSWER: pattern
    m = ANSWER_RE.search(text)
    if m:
        return int(m.group(1))

    # Fallback: look for standalone 1 or 2 at end
    last_part = text[-100:]
    if re.search(r"\b1\b", last_part) and not re.search(r"\b2\b", last_part):
        return 1
    if re.search(r"\b2\b", last_part) and not re.search(r"\b1\b", last_part):
        return 2

    return np.nan
