# -*- coding: utf-8 -*-
"""
Data loading and sampling for LLM Mutual Fund Comparison Experiment
"""

import pandas as pd
import numpy as np
from config import DATA_PATH, SAMPLE_SIZE, RANDOM_STATE
from utils import compare_medalist


def load_data():
    """
    Load the mutual funds pairs dataset from CSV.

    Returns:
        DataFrame with mutual fund pair features
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} pairs from {DATA_PATH}")
    return df


def sample_pairs(df, sample_size=None, random_state=None):
    """
    Sample a subset of fund pairs for the experiment.
    Filters out Medalist ties before sampling.

    Args:
        df: Full DataFrame of fund pairs
        sample_size: Number of pairs to sample (uses config if None)
        random_state: Random seed (uses config if None)

    Returns:
        Sampled DataFrame (with Medalist ties excluded)
    """
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    # Filter out Medalist ties
    if "Medalist_1" in df.columns and "Medalist_2" in df.columns:
        # Create a mask: keep rows where compare_medalist returns a valid value (not NaN)
        valid_mask = df.apply(
            lambda row: not pd.isna(compare_medalist(row["Medalist_1"], row["Medalist_2"])),
            axis=1
        )
        df_filtered = df[valid_mask].copy()
        n_ties = len(df) - len(df_filtered)
        if n_ties > 0:
            print(f"Filtered out {n_ties} pairs with Medalist ties ({len(df_filtered)} pairs remaining)")
    else:
        df_filtered = df.copy()
        print("Warning: Medalist columns not found, skipping tie filtering")

    if sample_size >= len(df_filtered):
        print(f"Using all {len(df_filtered)} pairs (requested {sample_size})")
        return df_filtered.reset_index(drop=True)

    sampled = df_filtered.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    print(f"Sampled {len(sampled)} pairs (from {len(df_filtered)} non-tie pairs)")
    return sampled


def get_experiment_data():
    """
    Convenience function: load and sample data in one step.

    Returns:
        Sampled DataFrame ready for experiment
    """
    df = load_data()
    return sample_pairs(df)
