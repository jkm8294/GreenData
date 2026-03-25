"""
GreenData — Reddit Corpus Scraper (One-Time Frozen Snapshot)

Uses Firecrawl to search for Reddit posts/comments about data center
community sentiment. Saves results as a JSONL corpus for NLP processing.

IMPORTANT: This is a one-time scrape. Run once, freeze the corpus,
and work with the static dataset. Reproducibility > freshness.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = RAW_DIR / "reddit_corpus.jsonl"

# ---------------------------------------------------------------------------
# Search queries — designed to capture the full spectrum of sentiment
# ---------------------------------------------------------------------------
SEARCH_QUERIES = [
    # Opposition / concern
    '"data center" opposition',
    '"data center" community concern',
    '"data center" noise complaint',
    '"data center" water usage',
    '"data center" power grid',
    '"data center" zoning',
    '"data center" construction protest',
    '"data center" moratorium',
    '"server farm" neighborhood',
    # Positive / economic
    '"data center" jobs economic development',
    '"data center" tax revenue benefit',
    # Neutral / informational
    '"data center" construction project',
]

# State/regional subreddits with known data center activity
STATE_SUBS = [
    "r/virginia", "r/oregon", "r/texas", "r/iowa", "r/arizona",
    "r/northcarolina", "r/georgia", "r/ohio", "r/indiana",
    "r/NoVA",          # Northern Virginia — data center capital
    "r/nova",          # lowercase variant
    "r/LoudounCounty",
    "r/Dallas", "r/sanantonio",
    "r/Portland", "r/Seattle",
    "r/phoenix",
]

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


def scrape_reddit_corpus(max_per_query: int = 50) -> list[dict]:
    """
    One-time scrape using Firecrawl search API.

    Goal: 1,000–5,000 posts/comments mentioning data centers.
    Each result includes the URL, markdown content, query used,
    and timestamp.

    Args:
        max_per_query: Maximum results per search query (Firecrawl limit).

    Returns:
        List of scraped result dicts.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        print("ERROR: FIRECRAWL_API_KEY not set. Cannot scrape.")
        return []

    from firecrawl import FirecrawlApp
    firecrawl = FirecrawlApp(api_key=api_key)

    all_results = []
    seen_urls = set()  # deduplicate across queries

    print("=" * 60)
    print("Reddit Corpus Scrape — One-Time Frozen Snapshot")
    print("=" * 60)

    # --- Phase 1: General search queries ---
    for i, query in enumerate(SEARCH_QUERIES, 1):
        search_term = f"{query} site:reddit.com"
        print(f"\n[{i}/{len(SEARCH_QUERIES)}] Searching: {search_term}")

        try:
            results = firecrawl.search(
                search_term,
                limit=max_per_query,
                scrape_options={"formats": ["markdown"]},
            )

            data = results if isinstance(results, list) else results.get("data", [])
            new_count = 0

            for r in data:
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                if "reddit.com" not in url:
                    continue

                seen_urls.add(url)
                all_results.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "content": r.get("markdown", r.get("content", "")),
                    "query": query,
                    "source": "firecrawl_search",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })
                new_count += 1

            print(f"  → {new_count} new results (total: {len(all_results)})")

        except Exception as e:
            print(f"  ERROR: {e}")

        # Be polite to the API
        time.sleep(1.0)

    # --- Phase 2: Targeted subreddit searches ---
    print(f"\n{'=' * 60}")
    print("Phase 2: Targeted subreddit searches")
    print("=" * 60)

    subreddit_queries = [
        "data center",
        "server farm",
        "hyperscale",
    ]

    for sub in STATE_SUBS:
        for sq in subreddit_queries:
            search_term = f'"{sq}" site:reddit.com/{sub}'
            print(f"  Searching: {search_term}")

            try:
                results = firecrawl.search(
                    search_term,
                    limit=20,
                    scrape_options={"formats": ["markdown"]},
                )

                data = results if isinstance(results, list) else results.get("data", [])
                new_count = 0

                for r in data:
                    url = r.get("url", "")
                    if url in seen_urls or "reddit.com" not in url:
                        continue

                    seen_urls.add(url)
                    all_results.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "content": r.get("markdown", r.get("content", "")),
                        "query": f"{sq} ({sub})",
                        "source": "firecrawl_subreddit",
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    })
                    new_count += 1

                if new_count > 0:
                    print(f"    → {new_count} new results")

            except Exception as e:
                print(f"    ERROR: {e}")

            time.sleep(0.5)

    # --- Save corpus ---
    print(f"\n{'=' * 60}")
    print(f"Corpus complete: {len(all_results)} total results")
    print("=" * 60)

    _save_corpus(all_results)
    return all_results


def _save_corpus(results: list[dict]):
    """Save results as JSON Lines file."""
    with open(CORPUS_PATH, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved → {CORPUS_PATH}")


def load_corpus() -> list[dict]:
    """Load the frozen corpus from disk."""
    if not CORPUS_PATH.exists():
        print(f"No corpus found at {CORPUS_PATH}. Run scrape_reddit_corpus() first.")
        return []

    results = []
    with open(CORPUS_PATH) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loaded {len(results)} posts from frozen corpus")
    return results


def corpus_stats():
    """Print summary statistics of the frozen corpus."""
    results = load_corpus()
    if not results:
        return

    queries = {}
    sources = {}
    for r in results:
        q = r.get("query", "unknown")
        queries[q] = queries.get(q, 0) + 1
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    print(f"\nCorpus Statistics:")
    print(f"  Total posts: {len(results)}")
    print(f"\n  By source:")
    for s, count in sorted(sources.items()):
        print(f"    {s}: {count}")
    print(f"\n  By query:")
    for q, count in sorted(queries.items(), key=lambda x: -x[1]):
        print(f"    {q}: {count}")

    # Content length distribution
    lengths = [len(r.get("content", "")) for r in results]
    print(f"\n  Content length (chars):")
    print(f"    Min: {min(lengths)}, Max: {max(lengths)}, "
          f"Median: {sorted(lengths)[len(lengths)//2]}")


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        corpus_stats()
    else:
        print("This will perform a ONE-TIME scrape of Reddit via Firecrawl.")
        print(f"Results will be saved to {CORPUS_PATH}")
        print()
        scrape_reddit_corpus()
