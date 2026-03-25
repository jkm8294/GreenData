"""
GreenData — Feature Correlation Analysis (Phase 3A)

BEFORE assigning weights, check:
  1. Multicollinearity — if two features are |r| > 0.7, they measure
     the same thing. Drop or combine one.
  2. Which features actually correlate with data center presence?

Outputs:
  - outputs/correlation_matrix.png — heatmap
  - List of highly correlated feature pairs to address
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_feature_correlations(
    features: pd.DataFrame,
    threshold: float = 0.7,
    save_path: str = "outputs/correlation_matrix.png",
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """
    Compute and visualize the feature correlation matrix.

    Args:
        features: DataFrame with numeric feature columns.
        threshold: |r| threshold for flagging multicollinear pairs.
        save_path: Where to save the heatmap PNG.

    Returns:
        (correlation_matrix, high_correlation_pairs)
        where each pair is (feature_1, feature_2, r_value).
    """
    print("=" * 60)
    print("Feature Correlation Analysis")
    print("=" * 60)

    numeric = features.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        print("WARNING: Fewer than 2 numeric columns. Nothing to correlate.")
        return pd.DataFrame(), []

    # Drop columns that are all-null
    numeric = numeric.dropna(axis=1, how="all")
    print(f"Analyzing {numeric.shape[1]} numeric features across {numeric.shape[0]} rows")

    # Compute correlation matrix
    corr = numeric.corr()

    # --- Heatmap ---
    fig_width = max(10, numeric.shape[1] * 0.8)
    fig_height = max(8, numeric.shape[1] * 0.6)
    plt.figure(figsize=(fig_width, fig_height))

    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i in range(len(corr)):
        for j in range(i):
            mask.iloc[i, j] = True  # mask lower triangle for cleaner viz

    sns.heatmap(
        corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        mask=None,  # show full matrix
        vmin=-1,
        vmax=1,
        annot_kws={"size": 8},
    )
    plt.title("Feature Correlation Matrix", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {save_path}")

    # --- Flag multicollinear pairs ---
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                high_corr.append((corr.columns[i], corr.columns[j], round(r, 4)))

    print(f"\nHighly correlated feature pairs (|r| > {threshold}):")
    if high_corr:
        for f1, f2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            action = "CONSIDER DROPPING ONE" if abs(r) > 0.85 else "monitor"
            print(f"  {f1} <-> {f2}: r={r:+.3f}  [{action}]")
    else:
        print("  None found — features are sufficiently independent.")

    # --- Feature variance report ---
    print(f"\nFeature statistics:")
    stats = numeric.describe().T[["mean", "std", "min", "max"]]
    stats["null_pct"] = (numeric.isnull().sum() / len(numeric) * 100).round(1)
    stats["skew"] = numeric.skew().round(2)
    print(stats.to_string())

    return corr, high_corr


def recommend_feature_actions(
    high_corr: list[tuple[str, str, float]],
) -> list[dict]:
    """
    Given multicollinear pairs, recommend actions (drop, combine, or keep).

    Returns list of {feature_1, feature_2, r, recommendation} dicts.
    """
    recommendations = []

    # Features to prefer keeping (higher data quality / more direct relevance)
    priority_features = {
        "industrial_rate_cents_kwh",
        "renewable_pct",
        "solar_ghi_kwh_m2_day",
        "social_resistance_score",
        "avg_temp_f",
        "cooling_degree_days",
    }

    for f1, f2, r in high_corr:
        if abs(r) > 0.9:
            # Near-duplicate — drop one
            keep = f1 if f1 in priority_features else f2
            drop = f2 if keep == f1 else f1
            rec = f"DROP '{drop}' (near-duplicate of '{keep}', r={r:+.3f})"
        elif abs(r) > 0.7:
            rec = f"COMBINE '{f1}' and '{f2}' into composite feature, or monitor"
        else:
            rec = "KEEP both"

        recommendations.append({
            "feature_1": f1,
            "feature_2": f2,
            "r": r,
            "recommendation": rec,
        })
        print(f"  → {rec}")

    return recommendations


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("Run data collection first: python data/collection/government_data.py")
    else:
        features = pd.read_parquet(features_path)
        corr, pairs = analyze_feature_correlations(features)
        if pairs:
            print("\nRecommendations:")
            recommend_feature_actions(pairs)
