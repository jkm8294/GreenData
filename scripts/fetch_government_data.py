"""
Convenience wrapper: fetch all government data and build master features.

Usage:
  python scripts/fetch_government_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collection.government_data import build_master_features

if __name__ == "__main__":
    master = build_master_features()
    print(f"\nDone. {master.shape[0]} counties × {master.shape[1]} features")
    print(master.head(10).to_string())
