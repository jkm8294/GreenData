"""
GreenData — End-to-End Pipeline Runner

Runs the full analytics pipeline in order:
  1. Data collection (government APIs + Reddit scrape)
  2. NLP preprocessing (clean, NER, geocode)
  3. Topic modeling (LDA)
  4. Sentiment analysis (VADER + vocal minority correction)
  5. Feature correlation analysis
  6. Feature normalization
  7. PSI scoring
  8. Validation

Each step checks for existing output files and skips if already present
(use --force to re-run everything).

Usage:
  python scripts/run_pipeline.py             # run all steps, skip existing
  python scripts/run_pipeline.py --force     # re-run everything
  python scripts/run_pipeline.py --step 3    # run only step 3
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_1_data_collection(force: bool = False):
    """Fetch government data and scrape Reddit."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)

    features_path = Path("data/processed/features.parquet")
    corpus_path = Path("data/raw/reddit_corpus.jsonl")

    # Government data
    if features_path.exists() and not force:
        print(f"  SKIP: {features_path} already exists")
    else:
        from data.collection.government_data import build_master_features
        build_master_features()

    # Reddit corpus
    if corpus_path.exists() and not force:
        print(f"  SKIP: {corpus_path} already exists")
    else:
        from data.collection.reddit_scraper import scrape_reddit_corpus
        scrape_reddit_corpus()


def step_2_preprocessing(force: bool = False):
    """Clean text, extract entities, geocode to county FIPS."""
    print("\n" + "=" * 60)
    print("STEP 2: NLP Preprocessing")
    print("=" * 60)

    output_path = Path("data/processed/matched_posts.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    from analytics.nlp.preprocessing import preprocess_reddit_corpus
    preprocess_reddit_corpus()


def step_3_topic_modeling(force: bool = False):
    """Run LDA topic modeling on preprocessed posts."""
    print("\n" + "=" * 60)
    print("STEP 3: Topic Modeling")
    print("=" * 60)

    output_path = Path("data/processed/posts_with_topics.parquet")
    input_path = Path("data/processed/matched_posts.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    if not input_path.exists():
        print(f"  ERROR: {input_path} not found. Run step 2 first.")
        return

    import pandas as pd
    from analytics.nlp.topic_modeling import extract_resistance_topics

    posts = pd.read_parquet(input_path)
    posts_with_topics, _, _ = extract_resistance_topics(posts)
    posts_with_topics.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")


def step_4_sentiment(force: bool = False):
    """Compute sentiment and social resistance scores."""
    print("\n" + "=" * 60)
    print("STEP 4: Sentiment Analysis + Vocal Minority Correction")
    print("=" * 60)

    output_path = Path("data/processed/county_resistance_scores.parquet")
    input_path = Path("data/processed/posts_with_topics.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    if not input_path.exists():
        print(f"  ERROR: {input_path} not found. Run step 3 first.")
        return

    import pandas as pd
    from analytics.nlp.sentiment import (
        compute_post_sentiment,
        compute_resistance_score,
        get_county_populations,
    )

    posts = pd.read_parquet(input_path)
    posts = compute_post_sentiment(posts)

    populations = get_county_populations()

    # Determine resistance topics (all non-economic topics)
    topic_cols = [c for c in posts.columns
                  if c.startswith("topic_") and c.endswith("_weight")]
    n_topics = len(topic_cols)
    resistance_topics = list(range(min(n_topics, 5)))

    compute_resistance_score(posts, populations, resistance_topics)


def step_5_correlation(force: bool = False):
    """Analyze feature correlations."""
    print("\n" + "=" * 60)
    print("STEP 5: Feature Correlation Analysis")
    print("=" * 60)

    output_path = Path("outputs/correlation_matrix.png")
    input_path = Path("data/processed/features.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    if not input_path.exists():
        print(f"  ERROR: {input_path} not found. Run step 1 first.")
        return

    import pandas as pd
    from analytics.scoring.correlation import analyze_feature_correlations

    features = pd.read_parquet(input_path)
    analyze_feature_correlations(features)


def step_6_normalize(force: bool = False):
    """Normalize features."""
    print("\n" + "=" * 60)
    print("STEP 6: Feature Normalization")
    print("=" * 60)

    output_path = Path("data/processed/features_normalized.parquet")
    input_path = Path("data/processed/features.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    if not input_path.exists():
        print(f"  ERROR: {input_path} not found. Run step 1 first.")
        return

    import pandas as pd
    from analytics.scoring.normalize import normalize_features

    features = pd.read_parquet(input_path)
    normalized = normalize_features(features)
    normalized.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")


def step_7_psi(force: bool = False):
    """Calculate PSI scores."""
    print("\n" + "=" * 60)
    print("STEP 7: PSI Calculation")
    print("=" * 60)

    output_path = Path("data/processed/psi_scores.parquet")
    input_path = Path("data/processed/features.parquet")

    if output_path.exists() and not force:
        print(f"  SKIP: {output_path} already exists")
        return

    if not input_path.exists():
        print(f"  ERROR: {input_path} not found. Run step 1 first.")
        return

    import json
    import pandas as pd
    from analytics.scoring.psi import calculate_psi
    from analytics.scoring.weights import FALLBACK_WEIGHTS

    features = pd.read_parquet(input_path)

    # Merge resistance scores into features if available
    resistance_path = Path("data/processed/county_resistance_scores.parquet")
    if resistance_path.exists():
        resistance = pd.read_parquet(resistance_path)
        fips_col = ("matched_county_fips" if "matched_county_fips" in resistance.columns
                     else "county_fips")
        merge_cols = [fips_col]
        for col in ["social_resistance_score", "posts_per_capita",
                     "post_count", "avg_sentiment"]:
            if col in resistance.columns:
                merge_cols.append(col)
        features = features.merge(
            resistance[merge_cols],
            left_on="county_fips",
            right_on=fips_col,
            how="left",
        )
        print(f"  Merged resistance scores from {resistance_path}")

    # Load weights
    weights_path = Path("outputs/derived_weights.json")
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)["weights"]
        print(f"  Using derived weights: {weights}")
    else:
        weights = FALLBACK_WEIGHTS
        print(f"  Using fallback weights: {weights}")

    calculate_psi(features, weights)


def step_8_validation(force: bool = False):
    """Run model validation and sensitivity analysis."""
    print("\n" + "=" * 60)
    print("STEP 8: Model Validation")
    print("=" * 60)

    scores_path = Path("data/processed/psi_scores.parquet")

    if not scores_path.exists():
        print(f"  ERROR: {scores_path} not found. Run step 7 first.")
        return

    import json
    import pandas as pd
    from analytics.validation.validate import (
        get_validation_dataset,
        sensitivity_analysis,
        validate_model,
    )
    from analytics.scoring.psi import DIMENSION_COLS

    scores = pd.read_parquet(scores_path)
    validation_set = get_validation_dataset()

    # Test 1-3: Classification, Mann-Whitney U, ROC
    results = validate_model(scores, validation_set)

    # Sensitivity analysis
    features_path = Path("data/processed/features.parquet")
    weights_path = Path("outputs/derived_weights.json")

    if features_path.exists():
        features = pd.read_parquet(features_path)
        if weights_path.exists():
            with open(weights_path) as f:
                derived = json.load(f)["weights"]
        else:
            from analytics.scoring.weights import FALLBACK_WEIGHTS
            derived = FALLBACK_WEIGHTS

        sensitivity_analysis(features, DIMENSION_COLS, derived)


# ===================================================================
# Main
# ===================================================================

STEPS = {
    1: ("Data Collection", step_1_data_collection),
    2: ("NLP Preprocessing", step_2_preprocessing),
    3: ("Topic Modeling", step_3_topic_modeling),
    4: ("Sentiment Analysis", step_4_sentiment),
    5: ("Feature Correlation", step_5_correlation),
    6: ("Normalization", step_6_normalize),
    7: ("PSI Calculation", step_7_psi),
    8: ("Validation", step_8_validation),
}


def main():
    parser = argparse.ArgumentParser(description="GreenData Pipeline Runner")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if output exists")
    parser.add_argument("--step", type=int, choices=list(STEPS.keys()),
                        help="Run only a specific step")
    parser.add_argument("--from-step", type=int, choices=list(STEPS.keys()),
                        help="Start from a specific step")
    args = parser.parse_args()

    print("=" * 60)
    print("GreenData — Full Analytics Pipeline")
    print("=" * 60)

    start_time = time.time()

    if args.step:
        name, fn = STEPS[args.step]
        print(f"\nRunning step {args.step}: {name}")
        fn(force=args.force)
    else:
        start = args.from_step or 1
        for step_num in range(start, max(STEPS.keys()) + 1):
            name, fn = STEPS[step_num]
            fn(force=args.force)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
