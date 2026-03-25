"""
GreenData — Feature Normalization (Phase 3B)

Z-score normalization, NOT min-max. Min-max is fragile with outliers —
Loudoun County squashes everything else to zero.

Strategy:
  - Roughly normal features → z-score, clipped to [-3, +3], mapped to 0–100
  - Highly skewed features (|skewness| > 2) → percentile rank × 100

Both methods produce 0–100 scores that are comparable across features.
"""

import numpy as np
import pandas as pd
from scipy import stats


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a single series to 0–100.

    Uses z-score for roughly normal distributions, percentile rank
    for highly skewed distributions (|skewness| > 2).

    Args:
        series: Raw numeric series.

    Returns:
        Normalized series in [0, 100] range.
    """
    clean = series.dropna()

    if len(clean) < 2:
        return series.fillna(50.0)

    skewness = clean.skew()

    if abs(skewness) > 2:
        # Highly skewed — percentile rank
        return series.rank(pct=True) * 100
    else:
        # Roughly normal — z-score
        z = stats.zscore(series, nan_policy="omit")
        clipped = np.clip(z, -3, 3)
        return (clipped + 3) / 6 * 100


def normalize_features(
    features: pd.DataFrame,
    exclude_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Normalize all numeric columns in a DataFrame to 0–100.

    Automatically selects z-score or percentile rank per column
    based on skewness. Non-numeric and excluded columns are
    passed through unchanged.

    Args:
        features: DataFrame with numeric feature columns.
        exclude_cols: Column names to skip (e.g., identifiers like
                      'county_fips', 'state').

    Returns:
        DataFrame with normalized numeric columns, others unchanged.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Auto-exclude common identifier columns
    auto_exclude = {"county_fips", "state_fips", "state", "county_name",
                    "lat", "lon", "model_version", "weights_used"}
    skip = set(exclude_cols) | auto_exclude

    numeric_cols = features.select_dtypes(include="number").columns
    cols_to_normalize = [c for c in numeric_cols if c not in skip]

    normalized = features.copy()

    print("Normalizing features:")
    for col in cols_to_normalize:
        raw = features[col]
        clean = raw.dropna()

        if len(clean) < 2:
            print(f"  {col}: SKIPPED (insufficient non-null values)")
            continue

        skewness = clean.skew()
        method = "percentile" if abs(skewness) > 2 else "z-score"
        normalized[col] = normalize_series(raw)

        print(f"  {col}: {method} (skewness={skewness:.2f}, "
              f"range {clean.min():.2f}–{clean.max():.2f} → "
              f"{normalized[col].min():.1f}–{normalized[col].max():.1f})")

    return normalized


def normalize_dimension_score(scores: pd.Series, invert: bool = False) -> pd.Series:
    """
    Normalize a dimension score (e.g., power_score, social_score) to 0–100.

    Args:
        scores: Raw dimension scores.
        invert: If True, higher raw values map to LOWER normalized scores.
                Use for features where higher = worse (e.g., resistance score,
                cooling degree days).

    Returns:
        Normalized series in [0, 100].
    """
    result = normalize_series(scores)
    if invert:
        result = 100 - result
    return result


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    from pathlib import Path

    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("Run data collection first: python data/collection/government_data.py")
    else:
        features = pd.read_parquet(features_path)
        normalized = normalize_features(features)
        out_path = Path("data/processed/features_normalized.parquet")
        normalized.to_parquet(out_path, index=False)
        print(f"\nNormalized features saved → {out_path}")
