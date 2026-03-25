"""
GreenData — Weight Derivation via Logistic Regression (Phase 3C)

HIDDEN BOSS #2: Class imbalance. You'll find 50+ successful sites but
maybe only 15–25 blocked/cancelled ones. If training data is 80% positive,
logistic regression predicts "Successful" every time for 80% accuracy.

Four-layer defense:
  1. class_weight='balanced' — penalizes minority misclassification
  2. SMOTE oversampling — generates synthetic minority examples
  3. Stratified K-Fold — preserves class ratio in each fold
  4. F1 score, not accuracy — forces performance on BOTH classes
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dimension definitions: which raw features map to each scoring dimension
# ---------------------------------------------------------------------------
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

# Fallback weights (literature-informed) used when insufficient training data
FALLBACK_WEIGHTS = {
    "power": 0.45,
    "environmental": 0.30,
    "social": 0.25,
}


def derive_weights(
    features: pd.DataFrame,
    labels: pd.Series,
    dimension_cols: dict = None,
) -> dict:
    """
    Derive dimension weights from logistic regression on historical
    site outcomes.

    Args:
        features: Normalized feature DataFrame (one row per county).
        labels: Binary series — 1 = successful, 0 = blocked/cancelled.
        dimension_cols: Dict mapping dimension name → list of feature columns.

    Returns:
        Dict of dimension name → weight (sums to 1.0).
    """
    if dimension_cols is None:
        dimension_cols = DIMENSION_COLS

    print("=" * 60)
    print("Weight Derivation — Logistic Regression")
    print("=" * 60)

    # --- Aggregate features into dimension scores ---
    X = pd.DataFrame(index=features.index)
    for dim, cols in dimension_cols.items():
        available = [c for c in cols if c in features.columns]
        if available:
            X[dim] = features[available].mean(axis=1)
        else:
            print(f"  WARNING: No columns available for '{dim}' dimension")
            X[dim] = 50.0  # neutral default

    # Fill remaining NaN with column median
    X = X.fillna(X.median())

    # --- Class balance report ---
    n_success = (labels == 1).sum()
    n_blocked = (labels == 0).sum()
    total = len(labels)

    print(f"\n=== Class Balance ===")
    print(f"  Successful sites: {n_success} ({n_success/total*100:.1f}%)")
    print(f"  Blocked sites:    {n_blocked} ({n_blocked/total*100:.1f}%)")

    if n_success > 0 and n_blocked > 0:
        print(f"  Ratio: {n_success/n_blocked:.1f}:1")

    if total < 10:
        print(f"\nWARNING: Only {total} training samples — using fallback weights.")
        print(f"  Fallback: {FALLBACK_WEIGHTS}")
        return FALLBACK_WEIGHTS.copy()

    # --- Strategy A: class_weight='balanced' ---
    model_balanced = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    n_splits = min(5, min(n_success, n_blocked))
    if n_splits < 2:
        print(f"\nWARNING: Not enough samples for cross-validation. Using fallback.")
        print(f"  Fallback: {FALLBACK_WEIGHTS}")
        return FALLBACK_WEIGHTS.copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores_a = cross_val_score(
        model_balanced, X, labels,
        cv=skf,
        scoring=make_scorer(f1_score, average="macro"),
    )
    print(f"\nStrategy A (balanced class weights):")
    print(f"  Macro F1: {f1_scores_a.mean():.3f} (+/- {f1_scores_a.std():.3f})")

    # --- Strategy B: SMOTE + balanced ---
    use_smote = False
    f1_scores_b = np.array([0.0])

    if n_blocked >= 6:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            k_neighbors = min(5, n_blocked - 1)
            smote_pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                )),
            ])

            f1_scores_b = cross_val_score(
                smote_pipeline, X, labels,
                cv=skf,
                scoring=make_scorer(f1_score, average="macro"),
            )
            print(f"\nStrategy B (SMOTE + balanced):")
            print(f"  Macro F1: {f1_scores_b.mean():.3f} (+/- {f1_scores_b.std():.3f})")

            if f1_scores_b.mean() > f1_scores_a.mean():
                use_smote = True
                print("  → Using SMOTE strategy (better F1)")
            else:
                print("  → Using balanced-weights strategy (better F1)")

        except ImportError:
            print("\nSMOTE unavailable (install imbalanced-learn). Using Strategy A.")
    else:
        print(f"\nSkipping SMOTE: only {n_blocked} minority samples (need >= 6)")

    # --- Fit final model ---
    if use_smote:
        smote = SMOTE(random_state=42, k_neighbors=min(5, n_blocked - 1))
        X_resampled, y_resampled = smote.fit_resample(X, labels)
        model_balanced.fit(X_resampled, y_resampled)
    else:
        model_balanced.fit(X, labels)

    # --- Extract weights ---
    raw_coefficients = model_balanced.coef_[0]
    abs_weights = np.abs(raw_coefficients)
    normalized_weights = abs_weights / abs_weights.sum()

    dim_names = list(dimension_cols.keys())
    weights = {}
    for i, dim in enumerate(dim_names):
        if i < len(normalized_weights):
            weights[dim] = round(float(normalized_weights[i]), 3)
        else:
            weights[dim] = round(1.0 / len(dim_names), 3)

    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        for dim in weights:
            weights[dim] = round(weights[dim] / total_weight, 3)

    print(f"\n{'=' * 60}")
    print("DERIVED WEIGHTS")
    print("=" * 60)
    for dim, w in weights.items():
        bar = "█" * int(w * 40)
        print(f"  {dim:<15s} {w:.3f}  {bar}")
    print(f"\n  (Fallback was: {FALLBACK_WEIGHTS})")

    # --- Coefficient direction report ---
    print(f"\nRaw coefficients (sign indicates direction):")
    for i, dim in enumerate(dim_names):
        if i < len(raw_coefficients):
            direction = "→ HIGHER score helps success" if raw_coefficients[i] > 0 else "→ HIGHER score hurts success"
            print(f"  {dim:<15s} {raw_coefficients[i]:+.4f}  {direction}")

    # --- Classification report on training data ---
    y_pred = model_balanced.predict(X)
    print(f"\n=== Full Classification Report (Training Data) ===")
    print(classification_report(
        labels, y_pred,
        target_names=["Blocked/Cancelled", "Successful"],
    ))

    # --- Save weights ---
    import json
    weights_path = OUTPUTS_DIR / "derived_weights.json"
    output = {
        "weights": weights,
        "fallback_weights": FALLBACK_WEIGHTS,
        "strategy": "SMOTE + balanced" if use_smote else "balanced class weights",
        "macro_f1": round(float(max(f1_scores_a.mean(), f1_scores_b.mean())), 3),
        "n_successful": int(n_success),
        "n_blocked": int(n_blocked),
        "raw_coefficients": {dim: round(float(raw_coefficients[i]), 4)
                             for i, dim in enumerate(dim_names)
                             if i < len(raw_coefficients)},
    }
    with open(weights_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Weights saved → {weights_path}")

    return weights


def load_weights(path: str = "outputs/derived_weights.json") -> dict:
    """Load previously derived weights from disk."""
    import json
    weights_path = Path(path)
    if weights_path.exists():
        with open(weights_path) as f:
            data = json.load(f)
        return data["weights"]
    else:
        print(f"No weights found at {weights_path}. Using fallback.")
        return FALLBACK_WEIGHTS.copy()
