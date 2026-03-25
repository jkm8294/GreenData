"""
Convenience wrapper: scrape Reddit corpus via Firecrawl.

Usage:
  python scripts/scrape_reddit.py          # run one-time scrape
  python scripts/scrape_reddit.py stats    # print corpus statistics
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collection.reddit_scraper import scrape_reddit_corpus, corpus_stats

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        corpus_stats()
    else:
        scrape_reddit_corpus()
