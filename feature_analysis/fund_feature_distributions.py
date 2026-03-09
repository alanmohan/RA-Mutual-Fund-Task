from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_pair_level_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "input_csvs" / "mutual_funds_pairs_no_date.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")
    return pd.read_csv(csv_path)


def build_fund_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = sorted({c[:-2] for c in df.columns if c.endswith("_1")})
    fund_data = {}

    for base in base_cols:
        col1 = f"{base}_1"
        col2 = f"{base}_2"
        if col1 not in df.columns or col2 not in df.columns:
            continue

        s1 = pd.to_numeric(df[col1], errors="coerce")
        s2 = pd.to_numeric(df[col2], errors="coerce")
        combined = pd.concat([s1, s2], ignore_index=True)

        if combined.notna().sum() == 0:
            continue

        fund_data[base] = combined

    fund_df = pd.DataFrame(fund_data)

    min_non_na = max(50, int(0.05 * len(fund_df)))
    numeric_cols = [
        col for col in fund_df.columns if fund_df[col].notna().sum() >= min_non_na
    ]
    return fund_df[numeric_cols]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def plot_feature_distributions(fund_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in fund_df.columns:
        series = pd.to_numeric(fund_df[col], errors="coerce").dropna()
        if series.nunique() < 5:
            continue

        plt.figure(figsize=(6, 4))
        try:
            sns.kdeplot(series, fill=True)
        except Exception:
            sns.histplot(series, kde=True)

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()

        filename = output_dir / f"density_{sanitize_name(col)}.png"
        plt.savefig(filename, dpi=150)
        plt.close()


def plot_correlation_heatmap(fund_df: pd.DataFrame, output_dir: Path) -> None:
    numeric_df = fund_df.apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.empty:
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap of Fund-Level Features")
    plt.tight_layout()

    filename = output_dir / "correlation_heatmap.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_pairwise_differences(df_pairs: pd.DataFrame, output_dir: Path) -> None:
    """
    For each numeric feature with *_1 and *_2 columns, plot the distribution of
    absolute differences |feature_1 - feature_2|.
    """
    sns.set(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cols = sorted({c[:-2] for c in df_pairs.columns if c.endswith("_1")})

    for base in base_cols:
        col1 = f"{base}_1"
        col2 = f"{base}_2"
        if col1 not in df_pairs.columns or col2 not in df_pairs.columns:
            continue

        s1 = pd.to_numeric(df_pairs[col1], errors="coerce")
        s2 = pd.to_numeric(df_pairs[col2], errors="coerce")
        diff = (s1 - s2).abs().dropna()

        # Skip non-numeric or effectively constant differences
        if diff.empty or diff.nunique() < 5:
            continue

        plt.figure(figsize=(6, 4))
        try:
            sns.kdeplot(diff, fill=True)
        except Exception:
            sns.histplot(diff, kde=True)

        pretty_name = base.replace("_", " ")
        plt.title(f"Distribution of |{pretty_name} fund1 - fund2|")
        plt.xlabel(f"|{pretty_name} fund1 - fund2|")
        plt.ylabel("Density")
        plt.tight_layout()

        filename = output_dir / f"diff_{sanitize_name(base)}.png"
        plt.savefig(filename, dpi=150)
        plt.close()


def main() -> None:
    df_pairs = load_pair_level_data()
    fund_df = build_fund_level_dataframe(df_pairs)

    script_dir = Path(__file__).resolve().parent
    plots_dir = script_dir / "plots"

    plot_feature_distributions(fund_df, plots_dir)
    plot_correlation_heatmap(fund_df, plots_dir)
    plot_pairwise_differences(df_pairs, plots_dir)


if __name__ == "__main__":
    main()

