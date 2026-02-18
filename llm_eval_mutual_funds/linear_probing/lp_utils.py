# -*- coding: utf-8 -*-
"""
Utility functions for Linear Probing Pipeline (Mutual Funds)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


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


def create_feature_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary feature labels for probing.

    These labels represent intermediate features the model might use
    to make its decision.

    Args:
        df: DataFrame with fund pair data

    Returns:
        DataFrame with binary feature labels
    """
    labels = pd.DataFrame(index=df.index)

    def safe_float(val):
        try:
            if pd.isna(val):
                return np.nan
            return float(str(val).replace(",", "").strip())
        except Exception:
            return np.nan

    def safe_years(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(val).lower().replace("years", "").replace("year", "").strip()
            return float(s)
        except Exception:
            return np.nan

    def safe_date(val):
        try:
            if pd.isna(val):
                return pd.NaT
            return pd.to_datetime(val, errors="coerce")
        except Exception:
            return pd.NaT

    def safe_yes_no(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s == "Y":
            return 1
        if s == "N":
            return 0
        return np.nan

    def compare(a, b, op):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return 1 if op else 0

    # Expense ratio (lower is better)
    er_1 = df["Expense Ratio - Net_1"].apply(safe_float)
    er_2 = df["Expense Ratio - Net_2"].apply(safe_float)
    labels["expense_ratio_f1_lower"] = [compare(a, b, a < b) for a, b in zip(er_1, er_2)]

    # Sharpe ratio (higher is better)
    sh_1 = df["3 Year Sharpe Ratio_1"].apply(safe_float)
    sh_2 = df["3 Year Sharpe Ratio_2"].apply(safe_float)
    labels["sharpe_f1_higher"] = [compare(a, b, a > b) for a, b in zip(sh_1, sh_2)]

    # Standard deviation (lower is better)
    sd_1 = df["Standard Deviation_1"].apply(safe_float)
    sd_2 = df["Standard Deviation_2"].apply(safe_float)
    labels["stdev_f1_lower"] = [compare(a, b, a < b) for a, b in zip(sd_1, sd_2)]

    # 3-year return (higher is better)
    r3_1 = df["3 Yr_1"].apply(safe_float)
    r3_2 = df["3 Yr_2"].apply(safe_float)
    labels["return_3yr_f1_higher"] = [compare(a, b, a > b) for a, b in zip(r3_1, r3_2)]

    # Beta
    b_1 = df["Beta_1"].apply(safe_float)
    b_2 = df["Beta_2"].apply(safe_float)
    # Linear comparison: fund 1 beta lower than fund 2 (easy for linear probe)
    labels["beta_f1_lower"] = [compare(a, b, a < b) for a, b in zip(b_1, b_2)]
    # Nonlinear: fund 1 beta closer to 1 (kept for optional use)
    beta_closer = []
    for a, b in zip(b_1, b_2):
        if pd.isna(a) or pd.isna(b):
            beta_closer.append(np.nan)
        else:
            beta_closer.append(1 if abs(a - 1.0) < abs(b - 1.0) else 0)
    labels["beta_f1_closer_to_1"] = beta_closer

    # Manager tenure (longer is better)
    t_1 = df["Manager Tenure_1"].apply(safe_years)
    t_2 = df["Manager Tenure_2"].apply(safe_years)
    labels["tenure_f1_longer"] = [compare(a, b, a > b) for a, b in zip(t_1, t_2)]

    # Inception date (older = earlier date)
    d_1 = df["Inception Date_1"].apply(safe_date)
    d_2 = df["Inception Date_2"].apply(safe_date)
    labels["inception_f1_older"] = [compare(a, b, a < b) for a, b in zip(d_1, d_2)]

    # Assets (higher is better)
    a_1 = df["Assets (Millions)_1"].apply(safe_float)
    a_2 = df["Assets (Millions)_2"].apply(safe_float)
    labels["assets_f1_higher"] = [compare(a, b, a > b) for a, b in zip(a_1, a_2)]

    # Turnover (lower is better)
    tr_1 = df["Turnover Rates_1"].apply(safe_float)
    tr_2 = df["Turnover Rates_2"].apply(safe_float)
    labels["turnover_f1_lower"] = [compare(a, b, a < b) for a, b in zip(tr_1, tr_2)]

    # Load (No load is better)
    l_1 = df["Load (Y/N)_1"].apply(safe_yes_no)
    l_2 = df["Load (Y/N)_2"].apply(safe_yes_no)
    load_labels = []
    for a, b in zip(l_1, l_2):
        if pd.isna(a) or pd.isna(b):
            load_labels.append(np.nan)
        else:
            load_labels.append(1 if (a == 0 and b == 1) else 0)
    labels["load_f1_no"] = load_labels

    # NTF (Yes is better)
    n_1 = df["NTF_1"].apply(safe_yes_no)
    n_2 = df["NTF_2"].apply(safe_yes_no)
    ntf_labels = []
    for a, b in zip(n_1, n_2):
        if pd.isna(a) or pd.isna(b):
            ntf_labels.append(np.nan)
        else:
            ntf_labels.append(1 if (a == 1 and b == 0) else 0)
    labels["ntf_f1_yes"] = ntf_labels

    # Ground truth: which fund has higher Medalist rating
    medalist_labels = []
    for m1, m2 in zip(df["Medalist_1"], df["Medalist_2"]):
        v1 = get_medalist_value(m1)
        v2 = get_medalist_value(m2)
        if v1 < 0 or v2 < 0 or v1 == v2:
            medalist_labels.append(np.nan)
        else:
            medalist_labels.append(1 if v1 > v2 else 0)
    labels["medalist_f1_higher"] = medalist_labels

    return labels


def create_feature_raw_values(df: pd.DataFrame) -> Dict[str, tuple]:
    """
    Create (value_1, value_2) per sample for each feature, for use in Scrambled Hierarchy control.
    Returns dict: feature_name -> (array_1, array_2) of length n_samples. Values are numeric
    (dates as ordinal). NaN where not comparable.
    """
    def safe_float(val):
        try:
            if pd.isna(val):
                return np.nan
            return float(str(val).replace(",", "").strip())
        except Exception:
            return np.nan

    def safe_years(val):
        try:
            if pd.isna(val):
                return np.nan
            s = str(val).lower().replace("years", "").replace("year", "").strip()
            return float(s)
        except Exception:
            return np.nan

    def safe_date(val):
        try:
            if pd.isna(val):
                return np.nan
            dt = pd.to_datetime(val, errors="coerce")
            return np.nan if pd.isna(dt) else dt.toordinal()
        except Exception:
            return np.nan

    def safe_yes_no(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s == "Y":
            return 1
        if s == "N":
            return 0
        return np.nan

    n = len(df)
    out = {}

    er_1 = np.array(df["Expense Ratio - Net_1"].apply(safe_float))
    er_2 = np.array(df["Expense Ratio - Net_2"].apply(safe_float))
    out["expense_ratio_f1_lower"] = (er_1, er_2)

    sh_1 = np.array(df["3 Year Sharpe Ratio_1"].apply(safe_float))
    sh_2 = np.array(df["3 Year Sharpe Ratio_2"].apply(safe_float))
    out["sharpe_f1_higher"] = (sh_1, sh_2)

    sd_1 = np.array(df["Standard Deviation_1"].apply(safe_float))
    sd_2 = np.array(df["Standard Deviation_2"].apply(safe_float))
    out["stdev_f1_lower"] = (sd_1, sd_2)

    r3_1 = np.array(df["3 Yr_1"].apply(safe_float))
    r3_2 = np.array(df["3 Yr_2"].apply(safe_float))
    out["return_3yr_f1_higher"] = (r3_1, r3_2)

    b_1 = np.array(df["Beta_1"].apply(safe_float))
    b_2 = np.array(df["Beta_2"].apply(safe_float))
    out["beta_f1_lower"] = (b_1, b_2)
    out["beta_f1_closer_to_1"] = (b_1, b_2)

    t_1 = np.array(df["Manager Tenure_1"].apply(safe_years))
    t_2 = np.array(df["Manager Tenure_2"].apply(safe_years))
    out["tenure_f1_longer"] = (t_1, t_2)

    d_1 = np.array(df["Inception Date_1"].apply(safe_date))
    d_2 = np.array(df["Inception Date_2"].apply(safe_date))
    out["inception_f1_older"] = (d_1, d_2)

    a_1 = np.array(df["Assets (Millions)_1"].apply(safe_float))
    a_2 = np.array(df["Assets (Millions)_2"].apply(safe_float))
    out["assets_f1_higher"] = (a_1, a_2)

    tr_1 = np.array(df["Turnover Rates_1"].apply(safe_float))
    tr_2 = np.array(df["Turnover Rates_2"].apply(safe_float))
    out["turnover_f1_lower"] = (tr_1, tr_2)

    l_1 = np.array(df["Load (Y/N)_1"].apply(safe_yes_no))
    l_2 = np.array(df["Load (Y/N)_2"].apply(safe_yes_no))
    out["load_f1_no"] = (l_1, l_2)

    n_1 = np.array(df["NTF_1"].apply(safe_yes_no))
    n_2 = np.array(df["NTF_2"].apply(safe_yes_no))
    out["ntf_f1_yes"] = (n_1, n_2)

    m1_vals = np.array([get_medalist_value(m) for m in df["Medalist_1"]])
    m2_vals = np.array([get_medalist_value(m) for m in df["Medalist_2"]])
    out["medalist_f1_higher"] = (m1_vals, m2_vals)

    return out


def save_checkpoint(data: Dict, checkpoint_path: Path, prefix: str = "extraction"):
    """Save a checkpoint during processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pkl"
    filepath = checkpoint_path / filename

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_latest_checkpoint(checkpoint_path: Path, prefix: str = "extraction") -> Optional[Dict]:
    """Load the most recent checkpoint."""
    checkpoints = sorted(checkpoint_path.glob(f"{prefix}_*.pkl"))

    if not checkpoints:
        return None

    latest = checkpoints[-1]
    print(f"Loading checkpoint: {latest}")

    with open(latest, "rb") as f:
        return pickle.load(f)


def get_activation_path(activations_dir: Path, model_name: str, condition: str) -> Path:
    """Get the path for storing activations."""
    return activations_dir / f"{model_name}_{condition}_activations.npz"


def save_activations(
    activations: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    feature_labels: pd.DataFrame,
    path: Path,
    metadata: Optional[Dict] = None,
    feature_raw_values: Optional[Dict[str, tuple]] = None,
):
    """
    Save extracted activations to disk.
    feature_raw_values: optional dict feature_name -> (array_1, array_2) for Scrambled Hierarchy control.
    """
    save_dict = {
        "activations": activations,
        "sample_indices": sample_indices,
        "labels": labels,
        "feature_labels": feature_labels.values,
        "feature_label_columns": feature_labels.columns.tolist(),
        "metadata": metadata or {},
    }
    if feature_raw_values is not None and len(feature_raw_values) > 0:
        names = list(feature_raw_values.keys())
        n_samples = len(activations)
        n_f = len(names)
        raw_val1 = np.full((n_samples, n_f), np.nan, dtype=np.float64)
        raw_val2 = np.full((n_samples, n_f), np.nan, dtype=np.float64)
        for i, name in enumerate(names):
            v1, v2 = feature_raw_values[name]
            raw_val1[:, i] = np.asarray(v1, dtype=np.float64)
            raw_val2[:, i] = np.asarray(v2, dtype=np.float64)
        save_dict["raw_val1"] = raw_val1
        save_dict["raw_val2"] = raw_val2
        save_dict["raw_feature_names"] = np.array(names, dtype=object)
    np.savez_compressed(path, **save_dict)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved activations to {path} ({size_mb:.1f} MB)")


def load_activations(path: Path) -> Dict[str, Any]:
    """
    Load activations from disk.
    If file contains raw_val1/raw_val2/raw_feature_names, adds "feature_raw_values" dict for Scrambled Hierarchy.
    """
    data = np.load(path, allow_pickle=True)

    feature_labels = pd.DataFrame(
        data["feature_labels"],
        columns=data["feature_label_columns"].tolist(),
    )

    result = {
        "activations": data["activations"],
        "sample_indices": data["sample_indices"],
        "labels": data["labels"],
        "feature_labels": feature_labels,
        "metadata": data["metadata"].item() if data["metadata"].ndim == 0 else data["metadata"],
    }
    if "raw_val1" in data and "raw_val2" in data and "raw_feature_names" in data:
        names = data["raw_feature_names"].tolist()
        raw_val1 = data["raw_val1"]
        raw_val2 = data["raw_val2"]
        result["feature_raw_values"] = {
            name: (raw_val1[:, i].copy(), raw_val2[:, i].copy())
            for i, name in enumerate(names)
        }
    else:
        result["feature_raw_values"] = None
    return result


def print_banner(text: str, char: str = "=", width: int = 60):
    """Print a formatted banner."""
    print(char * width)
    print(f" {text}")
    print(char * width)
