"""
GreenData — Predictive Suitability Index (PSI) Calculation (Phase 3D)

Composite score combining power, environmental, and social dimensions
with data-derived weights. Each location gets a 0–100 PSI score plus
per-dimension breakdowns and a confidence rating based on data completeness.
"""

from pathlib import Path

import pandas as pd

from analytics.scoring.normalize import normalize_series

MODEL_VERSION = "1.0.0"  # Increment on every weight/feature change

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Default dimension → feature column mapping
DIMENSION_COLS = {
    "power": [
        "industrial_rate_cents_kwh",
        "renewable_pct",
        "total_generation_mwh",
        "renewable_potential_score",
    ],
    "environmental": [
        "avg_temp_f",
        "cooling_degree_days",
        "solar_ghi_kwh_m2_day",
        "groundwater_level_ft",
        "surface_water_flow_cfs",
        "drought_severity_index",
    ],
    "social": [
        "social_resistance_score",
        "posts_per_capita",
    ],
}

# Which features should be inverted (higher raw = worse for suitability)
INVERT_FEATURES = {
    "industrial_rate_cents_kwh",    # cheaper is better
    "avg_temp_f",                   # cooler is better (less cooling cost)
    "cooling_degree_days",          # fewer is better
    "drought_severity_index",       # lower is better
    "social_resistance_score",      # lower resistance is better
    "posts_per_capita",             # less discourse intensity is better
}


def calculate_psi(
    features: pd.DataFrame,
    weights: dict,
    dimension_cols: dict = None,
    invert_features: set = None,
) -> pd.DataFrame:
    """
    Calculate the Predictive Suitability Index for all locations.

    Args:
        features: DataFrame with county_fips and feature columns.
        weights: Dict of dimension name → weight (should sum to ~1.0).
        dimension_cols: Dict mapping dimension → list of feature column names.
        invert_features: Set of feature names where higher = worse.

    Returns:
        DataFrame sorted by PSI (descending) with columns:
          county_fips, state, power_score, environmental_score,
          social_score, psi, confidence, model_version, weights_used
    """
    if dimension_cols is None:
        dimension_cols = DIMENSION_COLS
    if invert_features is None:
        invert_features = INVERT_FEATURES

    print("=" * 60)
    print(f"Calculating PSI (model v{MODEL_VERSION})")
    print("=" * 60)
    print(f"Weights: {weights}")

    # --- Build result frame with identifiers ---
    id_cols = ["county_fips"]
    for col in ["state", "county_name", "lat", "lon", "state_fips"]:
        if col in features.columns:
            id_cols.append(col)

    results = features[id_cols].copy()

    # --- Compute per-dimension scores ---
    for dim, cols in dimension_cols.items():
        available = [c for c in cols if c in features.columns]
        missing = [c for c in cols if c not in features.columns]

        if missing:
            print(f"  {dim}: missing columns {missing}")

        if not available:
            print(f"  WARNING: No data for '{dim}' — defaulting to 50")
            results[f"{dim}_score"] = 50.0
            continue

        # Normalize each contributing feature to 0–100
        dim_scores = pd.DataFrame(index=features.index)
        for col in available:
            normalized = normalize_series(features[col])
            if col in invert_features:
                normalized = 100 - normalized
            dim_scores[col] = normalized

        # Dimension score = mean of its normalized features
        results[f"{dim}_score"] = dim_scores.mean(axis=1)

        stats = results[f"{dim}_score"].describe()
        print(f"  {dim}_score: mean={stats['mean']:.1f}, "
              f"std={stats['std']:.1f}, "
              f"range=[{stats['min']:.1f}, {stats['max']:.1f}], "
              f"features={len(available)}/{len(cols)}")

    # --- Compute composite PSI ---
    score_cols = [f"{dim}_score" for dim in dimension_cols.keys()]
    psi_components = []

    for dim in dimension_cols.keys():
        score_col = f"{dim}_score"
        w = weights.get(dim, 0)
        if score_col in results.columns:
            psi_components.append(w * results[score_col])

    if psi_components:
        results["psi"] = sum(psi_components)
    else:
        results["psi"] = 50.0

    # --- Data completeness confidence ---
    total_features = sum(len(v) for v in dimension_cols.values())
    available_features = sum(
        len([c for c in v if c in features.columns])
        for v in dimension_cols.values()
    )
    base_confidence = available_features / total_features if total_features > 0 else 0

    # Per-row confidence: penalize rows with many nulls
    feature_cols_flat = [c for cols in dimension_cols.values() for c in cols
                         if c in features.columns]
    if feature_cols_flat:
        row_completeness = features[feature_cols_flat].notna().mean(axis=1)
        results["confidence"] = round(base_confidence * row_completeness, 3)
    else:
        results["confidence"] = base_confidence

    # --- Metadata ---
    results["model_version"] = MODEL_VERSION
    results["weights_used"] = str(weights)

    # --- Sort by PSI ---
    results = results.sort_values("psi", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)

    # --- Report ---
    print(f"\n{'=' * 60}")
    print("PSI RANKINGS")
    print("=" * 60)

    display_cols = ["rank", "county_fips"]
    if "state" in results.columns:
        display_cols.append("state")
    display_cols.extend([f"{dim}_score" for dim in dimension_cols.keys()])
    display_cols.extend(["psi", "confidence"])

    # Top 10
    print("\nTop 10 most suitable locations:")
    top10 = results.head(10)[display_cols].copy()
    for col in top10.select_dtypes(include="number").columns:
        top10[col] = top10[col].round(1)
    print(top10.to_string(index=False))

    # Bottom 5
    print("\nBottom 5 (least suitable):")
    bottom5 = results.tail(5)[display_cols].copy()
    for col in bottom5.select_dtypes(include="number").columns:
        bottom5[col] = bottom5[col].round(1)
    print(bottom5.to_string(index=False))

    # PSI distribution
    print(f"\nPSI distribution:")
    print(f"  Mean: {results['psi'].mean():.1f}")
    print(f"  Std:  {results['psi'].std():.1f}")
    print(f"  Min:  {results['psi'].min():.1f}")
    print(f"  Max:  {results['psi'].max():.1f}")

    # --- Save ---
    output_path = OUTPUTS_DIR / "psi_scores.csv"
    results.to_csv(output_path, index=False)
    print(f"\nPSI scores saved → {output_path}")

    parquet_path = Path("data/processed/psi_scores.parquet")
    results.to_parquet(parquet_path, index=False)
    print(f"PSI scores saved → {parquet_path}")

    return results


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import json

    features_path = Path("data/processed/features.parquet")
    weights_path = Path("outputs/derived_weights.json")

    if not features_path.exists():
        print("Run data collection first: python data/collection/government_data.py")
    else:
        features = pd.read_parquet(features_path)

        if weights_path.exists():
            with open(weights_path) as f:
                weights = json.load(f)["weights"]
            print(f"Loaded derived weights: {weights}")
        else:
            from analytics.scoring.weights import FALLBACK_WEIGHTS
            weights = FALLBACK_WEIGHTS
            print(f"Using fallback weights: {weights}")

        scores = calculate_psi(features, weights)
        print(f"\nDone. {len(scores)} locations scored.")
