"""
GreenData — Model Validation (Phase 4)

Three validation tests:
  1. Can the PSI model distinguish successful from blocked sites?
  2. Do social resistance scores predict actual cancellations?
  3. Sensitivity analysis — how stable are rankings when weights change?

Also includes the validation dataset of known successful and
cancelled/blocked data center projects.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from analytics.scoring.psi import calculate_psi

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Validation dataset — manually curated from public records & news
# ===================================================================

CANCELLED_PROJECTS = [
    {
        "name": "QTS Mt. Prospect",
        "county_fips": "17031",
        "state": "IL",
        "year_proposed": 2022,
        "outcome": "blocked",
        "reason": "zoning_denial",
    },
    {
        "name": "Prince William County DC Moratorium",
        "county_fips": "51153",
        "state": "VA",
        "year_proposed": 2023,
        "outcome": "moratorium",
        "reason": "community_opposition",
    },
    {
        "name": "Haymarket Data Center (PW Digital Gateway)",
        "county_fips": "51153",
        "state": "VA",
        "year_proposed": 2021,
        "outcome": "blocked",
        "reason": "community_opposition",
    },
    {
        "name": "Compass Datacenters Chandler AZ",
        "county_fips": "04013",
        "state": "AZ",
        "year_proposed": 2022,
        "outcome": "blocked",
        "reason": "water_concerns",
    },
    {
        "name": "Vantage Mesa AZ",
        "county_fips": "04013",
        "state": "AZ",
        "year_proposed": 2022,
        "outcome": "delayed",
        "reason": "water_concerns",
    },
    {
        "name": "Meta Zeewolde Netherlands",
        "county_fips": None,  # international — excluded from scoring
        "state": None,
        "year_proposed": 2022,
        "outcome": "cancelled",
        "reason": "community_opposition",
    },
    {
        "name": "EdgeCore Sterling VA",
        "county_fips": "51107",
        "state": "VA",
        "year_proposed": 2023,
        "outcome": "delayed",
        "reason": "grid_capacity",
    },
    {
        "name": "CloudHQ Manassas VA",
        "county_fips": "51153",
        "state": "VA",
        "year_proposed": 2022,
        "outcome": "blocked",
        "reason": "zoning_denial",
    },
    {
        "name": "Elk Grove Village IL Proposal",
        "county_fips": "17031",
        "state": "IL",
        "year_proposed": 2021,
        "outcome": "blocked",
        "reason": "noise_concerns",
    },
    {
        "name": "Goodyear AZ Water Restriction",
        "county_fips": "04013",
        "state": "AZ",
        "year_proposed": 2023,
        "outcome": "moratorium",
        "reason": "water_concerns",
    },
    {
        "name": "South Holland IL",
        "county_fips": "17031",
        "state": "IL",
        "year_proposed": 2023,
        "outcome": "blocked",
        "reason": "community_opposition",
    },
    {
        "name": "Gainesville VA Proposal",
        "county_fips": "51153",
        "state": "VA",
        "year_proposed": 2022,
        "outcome": "blocked",
        "reason": "community_opposition",
    },
    {
        "name": "Warrenton VA Fauquier County",
        "county_fips": "51061",
        "state": "VA",
        "year_proposed": 2023,
        "outcome": "moratorium",
        "reason": "community_opposition",
    },
    {
        "name": "Culpeper County VA",
        "county_fips": "51047",
        "state": "VA",
        "year_proposed": 2023,
        "outcome": "moratorium",
        "reason": "community_opposition",
    },
    {
        "name": "The Dalles OR Water Limits",
        "county_fips": "41065",
        "state": "OR",
        "year_proposed": 2022,
        "outcome": "restricted",
        "reason": "water_concerns",
    },
]

SUCCESSFUL_PROJECTS = [
    {
        "name": "Microsoft Quincy WA",
        "county_fips": "53025",
        "state": "WA",
        "year_built": 2007,
        "outcome": "operational",
    },
    {
        "name": "Google The Dalles OR",
        "county_fips": "41065",
        "state": "OR",
        "year_built": 2006,
        "outcome": "operational",
    },
    {
        "name": "Google Council Bluffs IA",
        "county_fips": "19155",
        "state": "IA",
        "year_built": 2009,
        "outcome": "operational",
    },
    {
        "name": "Facebook Prineville OR",
        "county_fips": "41013",
        "state": "OR",
        "year_built": 2011,
        "outcome": "operational",
    },
    {
        "name": "Amazon US-East-1 (Ashburn VA)",
        "county_fips": "51107",
        "state": "VA",
        "year_built": 2006,
        "outcome": "operational",
    },
    {
        "name": "Equinix Ashburn VA Campus",
        "county_fips": "51107",
        "state": "VA",
        "year_built": 2010,
        "outcome": "operational",
    },
    {
        "name": "Digital Realty Dallas TX",
        "county_fips": "48113",
        "state": "TX",
        "year_built": 2008,
        "outcome": "operational",
    },
    {
        "name": "CyrusOne San Antonio TX",
        "county_fips": "48029",
        "state": "TX",
        "year_built": 2012,
        "outcome": "operational",
    },
    {
        "name": "QTS Atlanta GA (Suwanee)",
        "county_fips": "13121",
        "state": "GA",
        "year_built": 2010,
        "outcome": "operational",
    },
    {
        "name": "Google Lenoir NC",
        "county_fips": "37027",
        "state": "NC",
        "year_built": 2009,
        "outcome": "operational",
    },
    {
        "name": "Apple Maiden NC",
        "county_fips": "37035",
        "state": "NC",
        "year_built": 2010,
        "outcome": "operational",
    },
    {
        "name": "Facebook New Albany OH",
        "county_fips": "39049",
        "state": "OH",
        "year_built": 2019,
        "outcome": "operational",
    },
    {
        "name": "Microsoft Cheyenne WY",
        "county_fips": "56021",
        "state": "WY",
        "year_built": 2012,
        "outcome": "operational",
    },
    {
        "name": "Microsoft San Antonio TX",
        "county_fips": "48029",
        "state": "TX",
        "year_built": 2013,
        "outcome": "operational",
    },
    {
        "name": "Apple Mesa AZ",
        "county_fips": "04013",
        "state": "AZ",
        "year_built": 2018,
        "outcome": "operational",
    },
    {
        "name": "Google Mesa AZ",
        "county_fips": "04013",
        "state": "AZ",
        "year_built": 2020,
        "outcome": "operational",
    },
    {
        "name": "Microsoft Des Moines IA",
        "county_fips": "19153",
        "state": "IA",
        "year_built": 2014,
        "outcome": "operational",
    },
    {
        "name": "CoreSite Reston VA",
        "county_fips": "51059",
        "state": "VA",
        "year_built": 2003,
        "outcome": "operational",
    },
    {
        "name": "Equinix Chicago IL",
        "county_fips": "17031",
        "state": "IL",
        "year_built": 2002,
        "outcome": "operational",
    },
    {
        "name": "Digital Realty Phoenix AZ",
        "county_fips": "04013",
        "state": "AZ",
        "year_built": 2015,
        "outcome": "operational",
    },
    {
        "name": "Switch Las Vegas NV",
        "county_fips": "32003",
        "state": "NV",
        "year_built": 2000,
        "outcome": "operational",
    },
    {
        "name": "T5 DFW Dallas TX",
        "county_fips": "48113",
        "state": "TX",
        "year_built": 2017,
        "outcome": "operational",
    },
    {
        "name": "Aligned Salt Lake City UT",
        "county_fips": "49035",
        "state": "UT",
        "year_built": 2020,
        "outcome": "operational",
    },
    {
        "name": "NTT Nashville TN",
        "county_fips": "47037",
        "state": "TN",
        "year_built": 2019,
        "outcome": "operational",
    },
    {
        "name": "Iron Mountain Manassas VA",
        "county_fips": "51153",
        "state": "VA",
        "year_built": 2016,
        "outcome": "operational",
    },
]


def get_validation_dataset() -> pd.DataFrame:
    """
    Build a combined validation DataFrame from successful and blocked projects.

    Returns DataFrame with columns:
      name, county_fips, state, outcome, label (1=successful, 0=blocked)
    """
    records = []

    for p in CANCELLED_PROJECTS:
        if p.get("county_fips"):  # skip international projects
            records.append({
                "name": p["name"],
                "county_fips": p["county_fips"],
                "state": p.get("state"),
                "outcome": p["outcome"],
                "reason": p.get("reason"),
                "label": 0,
            })

    for p in SUCCESSFUL_PROJECTS:
        records.append({
            "name": p["name"],
            "county_fips": p["county_fips"],
            "state": p.get("state"),
            "outcome": p["outcome"],
            "reason": None,
            "label": 1,
        })

    df = pd.DataFrame(records)
    print(f"Validation dataset: {len(df)} projects "
          f"({(df['label']==1).sum()} successful, {(df['label']==0).sum()} blocked)")
    return df


# ===================================================================
# Test 1: PSI classification accuracy
# ===================================================================

def validate_model(
    scores: pd.DataFrame,
    validation_set: pd.DataFrame = None,
    psi_threshold: float = 60.0,
) -> dict:
    """
    Validate the PSI model against known site outcomes.

    Test 1: Can PSI distinguish successful from blocked sites?
    Test 2: Do social resistance scores predict actual cancellations?
    Test 3: ROC curve and AUC.

    Args:
        scores: PSI results from calculate_psi().
        validation_set: DataFrame with county_fips, outcome, label columns.
        psi_threshold: PSI cutoff for predicting viable vs non-viable.

    Returns:
        Dict with AUC, p-value, and classification metrics.
    """
    if validation_set is None:
        validation_set = get_validation_dataset()

    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)

    # Merge scores with validation set on county_fips
    # Some counties appear multiple times (multiple projects) — take first
    merged = validation_set.merge(
        scores, on="county_fips", how="inner",
    )

    if len(merged) < 5:
        print(f"WARNING: Only {len(merged)} projects matched to scored counties.")
        print("Need more overlap between validation set and scored locations.")
        return {"auc": None, "p_value": None}

    print(f"Matched {len(merged)} projects to scored counties")
    print(f"  Successful: {(merged['label']==1).sum()}")
    print(f"  Blocked:    {(merged['label']==0).sum()}")

    # --- Test 1: Classification ---
    merged["predicted_viable"] = (merged["psi"] >= psi_threshold).astype(int)

    print(f"\n=== Test 1: Classification (PSI threshold = {psi_threshold}) ===")
    print(classification_report(
        merged["label"],
        merged["predicted_viable"],
        target_names=["Blocked/Cancelled", "Successful"],
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(merged["label"], merged["predicted_viable"])
    print("Confusion Matrix:")
    print(f"  {'':>20s} Predicted Block  Predicted Success")
    print(f"  {'Actual Blocked':<20s} {cm[0][0]:>15d}  {cm[0][1]:>17d}")
    print(f"  {'Actual Successful':<20s} {cm[1][0]:>15d}  {cm[1][1]:>17d}")

    # --- Test 2: Social resistance vs actual cancellations ---
    results = {}

    if "social_score" in merged.columns:
        blocked = merged[merged["label"] == 0]
        successful = merged[merged["label"] == 1]

        print(f"\n=== Test 2: Social Resistance Score Distribution ===")
        print(f"  Blocked projects   — mean: {blocked['social_score'].mean():.1f}, "
              f"median: {blocked['social_score'].median():.1f}")
        print(f"  Successful projects — mean: {successful['social_score'].mean():.1f}, "
              f"median: {successful['social_score'].median():.1f}")

        if len(blocked) >= 2 and len(successful) >= 2:
            stat, p_value = mannwhitneyu(
                blocked["social_score"],
                successful["social_score"],
                alternative="greater",
            )
            sig = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
            print(f"  Mann-Whitney U: p={p_value:.4f} ({sig} at α=0.05)")
            results["p_value"] = round(float(p_value), 4)
        else:
            results["p_value"] = None
    else:
        print("\n  social_score not available in PSI results — skipping Test 2")
        results["p_value"] = None

    # --- Test 3: ROC Curve ---
    print(f"\n=== Test 3: ROC Curve ===")
    try:
        fpr, tpr, thresholds = roc_curve(merged["label"], merged["psi"])
        auc = roc_auc_score(merged["label"], merged["psi"])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, "b-", linewidth=2, label=f"PSI Model (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
        plt.fill_between(fpr, tpr, alpha=0.1, color="blue")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve: PSI Predicting Site Viability", fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=150)
        plt.close()
        print(f"  AUC: {auc:.3f}")
        print(f"  ROC curve saved → outputs/roc_curve.png")
        results["auc"] = round(float(auc), 3)
    except ValueError as e:
        print(f"  Could not compute ROC: {e}")
        results["auc"] = None

    return results


# ===================================================================
# Sensitivity analysis
# ===================================================================

def sensitivity_analysis(
    features: pd.DataFrame,
    dimension_cols: dict,
    derived_weights: dict,
) -> pd.DataFrame:
    """
    Vary weights across scenarios to test ranking stability.

    If small weight changes cause massive rank shifts, the model is
    fragile and the weights need more justification.

    Reports Kendall's tau rank correlation between all scenario pairs.
    """
    print(f"\n{'=' * 60}")
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    scenarios = [
        ("Power-heavy",  {"power": 0.60, "environmental": 0.25, "social": 0.15}),
        ("Balanced",     {"power": 0.33, "environmental": 0.34, "social": 0.33}),
        ("Social-heavy", {"power": 0.30, "environmental": 0.25, "social": 0.45}),
        ("Env-heavy",    {"power": 0.25, "environmental": 0.50, "social": 0.25}),
        ("Derived",      derived_weights),
    ]

    rankings = {}
    for name, weights in scenarios:
        scores = calculate_psi(features, weights, dimension_cols)
        rankings[name] = scores.set_index("county_fips")["psi"].rank(ascending=False)

    # Kendall's tau between each pair
    print(f"\nRank Correlation (Kendall's τ):")
    print(f"  τ > 0.8 = stable, τ < 0.5 = fragile")
    print()

    names = list(rankings.keys())
    tau_results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Align on common indices
            common = rankings[names[i]].index.intersection(rankings[names[j]].index)
            if len(common) < 3:
                continue
            tau, p = kendalltau(
                rankings[names[i]].loc[common],
                rankings[names[j]].loc[common],
            )
            stability = "STABLE" if tau > 0.8 else ("moderate" if tau > 0.5 else "FRAGILE")
            print(f"  {names[i]:<14s} vs {names[j]:<14s}: τ={tau:.3f}, p={p:.4f} [{stability}]")
            tau_results.append({
                "scenario_1": names[i],
                "scenario_2": names[j],
                "kendall_tau": round(tau, 3),
                "p_value": round(p, 4),
                "stability": stability,
            })

    # Save sensitivity results
    results_df = pd.DataFrame(tau_results)
    results_path = OUTPUTS_DIR / "sensitivity_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSensitivity results saved → {results_path}")

    # Rank comparison table
    print(f"\nRank Comparison (top 10 by derived weights):")
    rank_table = pd.DataFrame(rankings)
    rank_table = rank_table.sort_values("Derived").head(10)
    print(rank_table.round(0).astype(int).to_string())

    return results_df


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import json

    scores_path = Path("data/processed/psi_scores.parquet")
    weights_path = Path("outputs/derived_weights.json")

    if not scores_path.exists():
        print("Run PSI calculation first: python -m analytics.scoring.psi")
    else:
        scores = pd.read_parquet(scores_path)

        # Validate
        validation_set = get_validation_dataset()
        results = validate_model(scores, validation_set)
        print(f"\nValidation results: {results}")

        # Sensitivity
        features_path = Path("data/processed/features.parquet")
        if features_path.exists() and weights_path.exists():
            features = pd.read_parquet(features_path)
            with open(weights_path) as f:
                derived = json.load(f)["weights"]
            from analytics.scoring.psi import DIMENSION_COLS
            sensitivity_analysis(features, DIMENSION_COLS, derived)
