"""
GreenData — Targeted Sentiment Analysis with Vocal Minority Correction (Phase 2D)

HIDDEN BOSS #3: Reddit represents a vocal minority. 500 angry posts in a county
of 1,000,000 people is noise. 20 angry posts in a county of 2,000 people is a
signal of organized local opposition that can actually kill a project.

This module computes a social resistance score per county that accounts for:
  1. Topic relevance — only resistance-correlated topics contribute
  2. Vocal minority correction — normalize by county population
  3. Volume factor — more posts = higher confidence (diminishing returns)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# VADER with domain-specific lexicon
# ---------------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

DOMAIN_LEXICON = {
    # Strong negative signals (opposition keywords)
    "moratorium": -2.5,
    "oppose": -2.0,
    "opposed": -2.0,
    "opposition": -2.0,
    "protest": -2.0,
    "protesting": -2.0,
    "protesters": -1.5,
    "noise pollution": -2.5,
    "water shortage": -3.0,
    "water usage": -1.5,
    "brownout": -2.0,
    "blackout": -2.0,
    "grid strain": -2.0,
    "power outage": -2.0,
    "outages": -1.5,
    "eyesore": -2.0,
    "destroy": -2.0,
    "ruining": -2.5,
    "petition": -1.5,
    "lawsuit": -2.0,
    "block": -1.0,
    "blocked": -1.5,
    "denied": -1.5,
    "rejected": -1.5,
    "environmental impact": -1.0,

    # Moderate negative
    "tax break": -0.5,
    "tax incentive": -0.3,
    "property value": -1.0,
    "property values": -1.0,
    "congestion": -1.0,
    "traffic": -0.5,
    "construction noise": -1.5,
    "generator noise": -2.0,
    "diesel generators": -1.5,
    "zoning change": -0.5,
    "rezoning": -0.5,

    # Positive signals
    "jobs": 1.5,
    "employment": 1.5,
    "economic development": 1.5,
    "investment": 1.0,
    "revenue": 1.0,
    "tax revenue": 1.0,
    "infrastructure improvement": 1.5,
    "broadband": 1.0,
    "modernize": 1.0,
}

for word, score in DOMAIN_LEXICON.items():
    analyzer.lexicon[word] = score


from analytics.scoring.normalize import normalize_series


# ===================================================================
# Per-post sentiment scoring
# ===================================================================

def compute_post_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VADER compound sentiment for each post.

    Adds 'sentiment' column (range -1 to +1, where negative = opposition).
    """
    posts = posts.copy()
    posts["sentiment"] = posts["clean_text"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    return posts


# ===================================================================
# County-level resistance scoring
# ===================================================================

def compute_resistance_score(
    posts: pd.DataFrame,
    county_populations: pd.DataFrame,
    resistance_topics: list[int],
) -> pd.DataFrame:
    """
    Compute social resistance score per county.

    The score is NOT just average sentiment. It accounts for:
      1. Topic relevance (only resistance-correlated topics contribute)
      2. Vocal minority correction (normalize by county population)
      3. Volume factor (more posts = higher confidence, diminishing returns)

    Args:
        posts: DataFrame with columns: matched_county_fips, clean_text,
               sentiment, topic_0_weight ... topic_N_weight
        county_populations: DataFrame with columns: county_fips, population
        resistance_topics: List of topic IDs that correlate with resistance

    Returns:
        DataFrame with per-county resistance scores and diagnostics.
    """
    print("=" * 60)
    print("Computing social resistance scores")
    print("=" * 60)

    posts = posts.copy()

    # Ensure sentiment is computed
    if "sentiment" not in posts.columns:
        posts = compute_post_sentiment(posts)

    # --- Topic relevance filter ---
    # Sum the weights of resistance-relevant topics per post
    topic_cols = [f"topic_{i}_weight" for i in resistance_topics
                  if f"topic_{i}_weight" in posts.columns]

    if topic_cols:
        posts["resistance_relevance"] = posts[topic_cols].sum(axis=1)
    else:
        # If no topic weights, treat all posts as equally relevant
        print("  WARNING: No topic weight columns found. Using uniform relevance.")
        posts["resistance_relevance"] = 1.0

    # --- Raw resistance signal per post ---
    # Invert sentiment (negative sentiment → positive resistance signal)
    # and weight by topic relevance
    posts["resistance_signal"] = (
        posts["sentiment"] * -1
        * posts["resistance_relevance"]
    )

    # ---------------------------------------------------------------
    # AGGREGATE BY COUNTY
    # ---------------------------------------------------------------
    county_agg = posts.groupby("matched_county_fips").agg(
        raw_resistance_mean=("resistance_signal", "mean"),
        raw_resistance_max=("resistance_signal", "max"),
        post_count=("resistance_signal", "count"),
        std_resistance=("resistance_signal", "std"),
        avg_sentiment=("sentiment", "mean"),
        negative_pct=("sentiment", lambda x: (x < -0.05).mean() * 100),
    ).reset_index()

    # ---------------------------------------------------------------
    # VOCAL MINORITY CORRECTION
    # ---------------------------------------------------------------
    county_agg = county_agg.merge(
        county_populations[["county_fips", "population"]],
        left_on="matched_county_fips",
        right_on="county_fips",
        how="left",
    )

    # Posts-per-capita intensity:
    # A county with 100 posts and 1M people (intensity = 0.0001)
    # is LESS risky than a county with 20 posts and 2,000 people (intensity = 0.01)
    county_agg["posts_per_capita"] = (
        county_agg["post_count"]
        / county_agg["population"].clip(lower=1)
    )

    # Intensity factor: log-scale to prevent extreme outliers
    county_agg["intensity_factor"] = np.log1p(
        county_agg["posts_per_capita"] * 100_000  # scale to readable range
    )

    # Volume factor: more posts = higher confidence, diminishing returns
    county_agg["volume_factor"] = (
        np.log1p(county_agg["post_count"]) / np.log1p(50)
    )
    county_agg["volume_factor"] = county_agg["volume_factor"].clip(upper=1.5)

    # --- Final social resistance score ---
    county_agg["social_resistance_score_raw"] = (
        county_agg["raw_resistance_mean"]
        * county_agg["intensity_factor"]
        * county_agg["volume_factor"]
    )

    # Normalize to 0-100
    county_agg["social_resistance_score"] = normalize_series(
        county_agg["social_resistance_score_raw"]
    )

    # ---------------------------------------------------------------
    # REPORT — these stats go on the methodology page
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("VOCAL MINORITY CORRECTION REPORT")
    print("=" * 60)
    print(f"Counties with Reddit data: {len(county_agg)}")
    print(f"Mean posts per county: {county_agg['post_count'].mean():.1f}")
    print(f"Median posts per county: {county_agg['post_count'].median():.1f}")

    print(f"\nTop 5 by RAW resistance (before correction):")
    top_raw = county_agg.nlargest(5, "raw_resistance_mean")[
        ["matched_county_fips", "raw_resistance_mean", "post_count", "population"]
    ]
    print(top_raw.to_string(index=False))

    print(f"\nTop 5 by CORRECTED resistance (after vocal minority adjustment):")
    top_corrected = county_agg.nlargest(5, "social_resistance_score")[
        ["matched_county_fips", "social_resistance_score", "post_count", "population"]
    ]
    print(top_corrected.to_string(index=False))

    # Show before/after comparison for methodology page
    print(f"\n{'=' * 60}")
    print("BEFORE vs AFTER correction (for methodology page):")
    print("=" * 60)
    comparison = county_agg[["matched_county_fips", "raw_resistance_mean",
                              "social_resistance_score", "post_count",
                              "population", "posts_per_capita"]].copy()
    comparison["raw_rank"] = comparison["raw_resistance_mean"].rank(ascending=False)
    comparison["corrected_rank"] = comparison["social_resistance_score"].rank(ascending=False)
    comparison["rank_change"] = comparison["raw_rank"] - comparison["corrected_rank"]

    biggest_movers = comparison.nlargest(5, "rank_change", keep="all")
    print("\nBiggest rank INCREASES after correction (small counties amplified):")
    print(biggest_movers[["matched_county_fips", "raw_rank", "corrected_rank",
                          "rank_change", "post_count", "population"]].to_string(index=False))

    # Save output
    output_path = PROCESSED_DIR / "county_resistance_scores.parquet"
    county_agg.to_parquet(output_path, index=False)
    print(f"\nResistance scores saved → {output_path}")

    return county_agg


# ===================================================================
# County population data (fallback when no external source)
# ===================================================================

def get_county_populations() -> pd.DataFrame:
    """
    Return county population data for the target counties.

    Uses Census Bureau estimates. In production, this would be loaded
    from a downloaded Census file. This provides hardcoded values for
    the target counties as a working fallback.
    """
    # 2023 Census estimates for target counties
    populations = {
        "51107": 434_741,   # Loudoun County, VA
        "51059": 1_150_309, # Fairfax County, VA
        "51153": 495_601,   # Prince William County, VA
        "51013": 238_643,   # Arlington County, VA
        "48113": 2_613_539, # Dallas County, TX
        "48439": 2_110_640, # Tarrant County, TX
        "48029": 2_088_159, # Bexar County, TX
        "41065": 27_317,    # Wasco County, OR
        "41013": 25_076,    # Crook County, OR
        "19155": 93_791,    # Pottawattamie County, IA
        "04013": 4_496_588, # Maricopa County, AZ
        "37183": 1_175_021, # Wake County, NC
        "37063": 332_680,   # Durham County, NC
        "13121": 1_066_677, # Fulton County, GA
        "13089": 764_382,   # DeKalb County, GA
        "39049": 1_323_807, # Franklin County, OH
        "39035": 1_264_817, # Cuyahoga County, OH
        "18097": 990_987,   # Marion County, IN
        "53025": 100_621,   # Grant County, WA
        "53033": 2_269_675, # King County, WA
        "17031": 5_275_541, # Cook County, IL
        "17043": 932_877,   # DuPage County, IL
        "32003": 2_265_461, # Clark County, NV
        "45083": 327_997,   # Spartanburg County, SC
        "47037": 715_884,   # Davidson County, TN
        "49035": 1_185_238, # Salt Lake County, UT
    }

    records = [{"county_fips": fips, "population": pop}
               for fips, pop in populations.items()]
    return pd.DataFrame(records)


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    posts_path = Path("data/processed/posts_with_topics.parquet")
    if not posts_path.exists():
        print("Run topic modeling first: python -m analytics.nlp.topic_modeling")
    else:
        posts = pd.read_parquet(posts_path)
        populations = get_county_populations()

        # Default: assume topics 0-4 are resistance topics
        # (refine after inspecting topic_summary.csv)
        topic_cols = [c for c in posts.columns if c.startswith("topic_") and c.endswith("_weight")]
        n_topics = len(topic_cols)
        resistance_topics = list(range(min(n_topics, 5)))

        print(f"Using resistance topics: {resistance_topics}")
        scores = compute_resistance_score(posts, populations, resistance_topics)
        print(f"\nDone. {len(scores)} counties scored.")
