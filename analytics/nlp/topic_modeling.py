"""
GreenData — Topic Modeling (Phase 2C)

Runs LDA topic modeling on the preprocessed Reddit corpus to discover
resistance categories. "Negative sentiment" is meaningless without
knowing WHAT people are negative about.

Expected topic clusters:
  - Noise / quality of life
  - Water consumption / drought
  - Power grid strain
  - Property values / land use
  - Tax incentive fairness
  - Environmental / climate impact
"""

from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain-specific stop words to exclude from topic modeling
# (generic terms that appear in every data center discussion)
# ---------------------------------------------------------------------------
DOMAIN_STOP_WORDS = [
    "data", "center", "centers", "data center", "data centers",
    "server", "servers", "server farm", "facility", "facilities",
    "company", "companies", "build", "building", "project",
    "said", "would", "could", "also", "people", "like",
    "just", "really", "think", "know", "going",
]


def extract_resistance_topics(
    posts: pd.DataFrame,
    n_topics: int = 6,
    max_features: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, LatentDirichletAllocation, TfidfVectorizer]:
    """
    Run LDA topic modeling to discover resistance categories.

    Args:
        posts: DataFrame with 'clean_text' column
        n_topics: Number of LDA topics to extract
        max_features: Max vocabulary size for TF-IDF
        random_state: For reproducibility

    Returns:
        (posts_with_topics, lda_model, vectorizer)
        Posts DataFrame gets new columns: topic_0_weight ... topic_N_weight
        and a 'dominant_topic' column.
    """
    print("=" * 60)
    print(f"Topic Modeling — extracting {n_topics} topics via LDA")
    print("=" * 60)

    if "clean_text" not in posts.columns:
        raise ValueError("posts DataFrame must have a 'clean_text' column")

    texts = posts["clean_text"].fillna("").tolist()

    # Build TF-IDF matrix
    all_stop_words = list(DOMAIN_STOP_WORDS)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # alpha only, 2+ chars
    )

    # Add domain stop words to sklearn's English list
    if vectorizer.stop_words == "english":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        vectorizer.stop_words = list(ENGLISH_STOP_WORDS) + all_stop_words

    print(f"Building TF-IDF matrix ({len(texts)} documents)...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(feature_names)}")
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    # Fit LDA
    print(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20,
        learning_method="online",
        batch_size=128,
        n_jobs=-1,
    )
    topic_distribution = lda.fit_transform(tfidf_matrix)

    # Display topics
    print(f"\n{'=' * 60}")
    print("DISCOVERED TOPICS")
    print("=" * 60)
    topic_summaries = []

    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]

        # Auto-label based on top words (heuristic)
        label = _auto_label_topic(top_words)
        topic_summaries.append({
            "topic_id": idx,
            "label": label,
            "top_words": top_words,
        })

        print(f"\nTopic {idx}: {label}")
        for word, weight in zip(top_words[:10], top_weights[:10]):
            bar = "█" * int(weight / max(top_weights) * 20)
            print(f"  {word:<25s} {bar} ({weight:.2f})")

    # Add topic weights to posts DataFrame
    posts = posts.copy()
    for i in range(n_topics):
        posts[f"topic_{i}_weight"] = topic_distribution[:, i]

    # Dominant topic per post
    posts["dominant_topic"] = topic_distribution.argmax(axis=1)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("TOPIC DISTRIBUTION")
    print("=" * 60)
    topic_counts = posts["dominant_topic"].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        label = topic_summaries[topic_id]["label"]
        pct = count / len(posts) * 100
        bar = "█" * int(pct / 2)
        print(f"  Topic {topic_id} ({label}): {count} posts ({pct:.1f}%) {bar}")

    # Save topic summary
    summary_df = pd.DataFrame(topic_summaries)
    summary_path = OUTPUTS_DIR / "topic_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nTopic summary saved → {summary_path}")

    return posts, lda, vectorizer


def _auto_label_topic(top_words: list[str]) -> str:
    """
    Heuristic auto-labeling based on keyword presence in top words.
    Falls back to the top word if no pattern matches.
    """
    words_set = set(w.lower() for w in top_words)

    label_patterns = [
        ({"noise", "loud", "sound", "quiet", "residential", "neighborhood", "quality life"},
         "Noise / Quality of Life"),
        ({"water", "drought", "consumption", "gallons", "aquifer", "shortage", "cooling"},
         "Water Consumption"),
        ({"power", "grid", "electricity", "energy", "outage", "brownout", "blackout", "strain"},
         "Power Grid Strain"),
        ({"property", "home", "value", "housing", "land", "zoning", "residential", "real estate"},
         "Property Values / Land Use"),
        ({"tax", "incentive", "subsidy", "break", "revenue", "taxpayer", "abatement", "deal"},
         "Tax Incentive Fairness"),
        ({"environment", "climate", "carbon", "emissions", "pollution", "green", "sustainable"},
         "Environmental Impact"),
        ({"jobs", "employment", "hire", "workers", "economic", "economy", "workforce"},
         "Economic Development"),
        ({"traffic", "construction", "road", "truck", "congestion", "infrastructure"},
         "Infrastructure / Traffic"),
    ]

    best_match = None
    best_overlap = 0

    for keywords, label in label_patterns:
        overlap = len(words_set & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = label

    if best_match and best_overlap >= 2:
        return best_match

    return f"Cluster: {top_words[0]}"


def identify_resistance_topics(
    topic_summaries: list[dict],
) -> list[int]:
    """
    Identify which topic IDs correspond to resistance/opposition
    (vs. neutral or positive economic development topics).

    Resistance topics: noise, water, power grid, property values,
    tax fairness, environmental impact.

    Non-resistance: economic development, general discussion.

    Returns list of resistance topic IDs for use in sentiment scoring.
    """
    resistance_labels = {
        "Noise / Quality of Life",
        "Water Consumption",
        "Power Grid Strain",
        "Property Values / Land Use",
        "Tax Incentive Fairness",
        "Environmental Impact",
        "Infrastructure / Traffic",
    }

    resistance_ids = []
    for summary in topic_summaries:
        if summary["label"] in resistance_labels:
            resistance_ids.append(summary["topic_id"])

    print(f"Resistance topic IDs: {resistance_ids}")
    return resistance_ids


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    processed_path = Path("data/processed/matched_posts.parquet")
    if not processed_path.exists():
        print("Run preprocessing first: python -m analytics.nlp.preprocessing")
    else:
        posts = pd.read_parquet(processed_path)
        posts_with_topics, lda, vectorizer = extract_resistance_topics(posts)
        posts_with_topics.to_parquet(
            "data/processed/posts_with_topics.parquet", index=False
        )
        print("\nSaved posts with topic weights → data/processed/posts_with_topics.parquet")
