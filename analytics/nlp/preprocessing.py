"""
GreenData — Text Preprocessing (Phase 2B)

Loads the frozen Reddit corpus → cleans text → extracts location
entities via SpaCy NER → geocodes each to a county FIPS code.

Outputs:
  - DataFrame of matched posts with county_fips assignments
  - data/processed/unmatched_posts.csv for manual review / alias expansion
  - Match rate metric for the methodology page
"""

import re
from pathlib import Path

import pandas as pd
import spacy

from analytics.nlp.geocoding import geocode_to_county_fips

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load SpaCy model (small English model — install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    raise

# ---------------------------------------------------------------------------
# Subreddit → state mapping
# ---------------------------------------------------------------------------
SUBREDDIT_TO_STATE = {
    "virginia": "VA",
    "nova": "VA",
    "northernvirginia": "VA",
    "loudouncounty": "VA",
    "oregon": "OR",
    "portland": "OR",
    "texas": "TX",
    "dallas": "TX",
    "sanantonio": "TX",
    "iowa": "IA",
    "arizona": "AZ",
    "phoenix": "AZ",
    "northcarolina": "NC",
    "georgia": "GA",
    "atlanta": "GA",
    "ohio": "OH",
    "indiana": "IN",
    "seattle": "WA",
    "washington": "WA",
    "chicago": "IL",
    "illinois": "IL",
    "lasvegas": "NV",
    "nevada": "NV",
    "nashville": "TN",
    "tennessee": "TN",
    "utah": "UT",
    "saltlakecity": "UT",
    "southcarolina": "SC",
}


def extract_state_hint(post: dict) -> str | None:
    """
    Extract state context from subreddit name or URL.

    r/virginia → "VA", r/NoVA → "VA", r/oregon → "OR"
    """
    url = post.get("url", "")
    match = re.search(r"reddit\.com/r/(\w+)", url)
    if match:
        sub = match.group(1).lower()
        return SUBREDDIT_TO_STATE.get(sub)
    return None


def clean_text(raw: str) -> str:
    """
    Clean raw markdown content from Reddit scrape.

    Removes URLs, markdown formatting, excessive whitespace.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown image/link syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    # Remove markdown headers, bold, italic markers
    text = re.sub(r"[#*_~`>]", "", text)
    # Remove Reddit-specific formatting
    text = re.sub(r"/?(u|r)/\w+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_locations(text: str) -> list[str]:
    """
    Extract geographic entities (GPE, LOC, FAC) from text using SpaCy NER.

    Returns deduplicated list of location strings.
    """
    if not text:
        return []

    doc = nlp(text[:100_000])  # cap to avoid memory issues on very long posts
    locations = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            name = ent.text.strip()
            name_lower = name.lower()
            # Skip very short or obviously wrong entities
            if len(name) < 2:
                continue
            if name_lower in ("us", "usa", "u.s.", "u.s.a.", "america", "united states"):
                continue
            if name_lower not in seen:
                seen.add(name_lower)
                locations.append(name)

    return locations


def match_post_to_county(row: pd.Series) -> str | None:
    """
    Try each extracted location to find a county FIPS match.

    Iterates locations in order (SpaCy extracts them in document order,
    so more prominent mentions come first).
    """
    for loc in row["raw_locations"]:
        fips = geocode_to_county_fips(loc, state_hint=row["state_hint"])
        if fips:
            return fips
    return None


# ===================================================================
# Main preprocessing pipeline
# ===================================================================

def preprocess_reddit_corpus(corpus_path: str = "data/raw/reddit_corpus.jsonl") -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Load frozen corpus from JSONL
      2. Clean text (remove URLs, markdown, whitespace)
      3. Extract location entities via SpaCy NER
      4. Extract state hint from subreddit context
      5. Geocode each location to county FIPS
      6. Report match rate
      7. Save unmatched posts for manual review

    Returns:
        DataFrame of successfully matched posts with county_fips column.
    """
    print("=" * 60)
    print("Preprocessing Reddit corpus")
    print("=" * 60)

    # Load corpus
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        print(f"ERROR: Corpus not found at {corpus_file}")
        print("Run data/collection/reddit_scraper.py first.")
        return pd.DataFrame()

    posts = pd.read_json(corpus_file, lines=True)
    print(f"Loaded {len(posts)} posts from corpus")

    # Clean text
    print("Cleaning text...")
    posts["clean_text"] = posts["content"].apply(clean_text)

    # Remove empty posts
    posts = posts[posts["clean_text"].str.len() > 50].copy()
    print(f"  {len(posts)} posts after removing short/empty content")

    # Extract location entities
    print("Extracting location entities via SpaCy NER...")
    posts["raw_locations"] = posts["clean_text"].apply(extract_locations)

    locations_found = posts["raw_locations"].apply(len).sum()
    posts_with_locations = (posts["raw_locations"].apply(len) > 0).sum()
    print(f"  Found {locations_found} location mentions across {posts_with_locations} posts")

    # Extract state hint from subreddit
    posts["state_hint"] = posts.apply(
        lambda row: extract_state_hint(row.to_dict()), axis=1
    )
    hints_found = posts["state_hint"].notna().sum()
    print(f"  State hints extracted for {hints_found} posts")

    # Geocode locations to county FIPS
    print("Geocoding locations to county FIPS codes...")
    posts["matched_county_fips"] = posts.apply(match_post_to_county, axis=1)

    # ---------------------------------------------------------------------------
    # Match rate report — THIS GOES ON THE METHODOLOGY PAGE
    # ---------------------------------------------------------------------------
    total = len(posts)
    matched = posts["matched_county_fips"].notna().sum()
    rate = (matched / total * 100) if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"GEOCODING MATCH RATE: {matched}/{total} ({rate:.1f}%)")
    print(f"{'=' * 60}")

    if rate < 60:
        print("WARNING: Match rate below 60%. Consider adding more aliases")
        print("to analytics/nlp/geocoding.py REDDIT_ALIASES dict.")

    # County distribution
    if matched > 0:
        county_counts = posts["matched_county_fips"].value_counts()
        print(f"\nTop 10 counties by post count:")
        for fips, count in county_counts.head(10).items():
            print(f"  {fips}: {count} posts")

    # Save unmatched posts for manual review
    unmatched = posts[posts["matched_county_fips"].isna()]
    if len(unmatched) > 0:
        unmatched_path = PROCESSED_DIR / "unmatched_posts.csv"
        unmatched[["url", "raw_locations", "state_hint", "clean_text"]].to_csv(
            unmatched_path, index=False
        )
        print(f"\nUnmatched posts saved → {unmatched_path}")
        print("Review these to find missing aliases for geocoding.py")

    # Save matched posts
    matched_posts = posts[posts["matched_county_fips"].notna()].copy()
    matched_path = PROCESSED_DIR / "matched_posts.parquet"
    matched_posts.to_parquet(matched_path, index=False)
    print(f"Matched posts saved → {matched_path}")

    return matched_posts


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    df = preprocess_reddit_corpus()
    if not df.empty:
        print(f"\nPreprocessing complete: {len(df)} posts matched to counties")
        print(f"Columns: {list(df.columns)}")
    else:
        print("\nNo posts matched. Check corpus and geocoding setup.")
