"""
GreenData — Export Validation Datasets to CSV

Exports the manually curated successful and cancelled data center project
lists from analytics/validation/validate.py into CSV files at
data/validation/ for reproducibility and easy review.

Usage:
  python scripts/export_validation_data.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analytics.validation.validate import CANCELLED_PROJECTS, SUCCESSFUL_PROJECTS

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export():
    # Cancelled / blocked sites
    cancelled = []
    for p in CANCELLED_PROJECTS:
        cancelled.append({
            "name": p["name"],
            "county_fips": p.get("county_fips"),
            "state": p.get("state"),
            "year_proposed": p.get("year_proposed"),
            "outcome": p["outcome"],
            "reason": p.get("reason"),
        })
    cancelled_df = pd.DataFrame(cancelled)
    cancelled_path = OUTPUT_DIR / "cancelled_sites.csv"
    cancelled_df.to_csv(cancelled_path, index=False)
    print(f"Exported {len(cancelled_df)} cancelled/blocked sites → {cancelled_path}")

    # Successful sites
    successful = []
    for p in SUCCESSFUL_PROJECTS:
        successful.append({
            "name": p["name"],
            "county_fips": p["county_fips"],
            "state": p.get("state"),
            "year_built": p.get("year_built"),
            "outcome": p["outcome"],
        })
    successful_df = pd.DataFrame(successful)
    successful_path = OUTPUT_DIR / "successful_sites.csv"
    successful_df.to_csv(successful_path, index=False)
    print(f"Exported {len(successful_df)} successful sites → {successful_path}")

    # Combined for quick reference
    cancelled_df["label"] = 0
    cancelled_df["year"] = cancelled_df["year_proposed"]
    successful_df["label"] = 1
    successful_df["year"] = successful_df["year_built"]

    combined = pd.concat([
        cancelled_df[["name", "county_fips", "state", "outcome", "label", "year"]],
        successful_df[["name", "county_fips", "state", "outcome", "label", "year"]],
    ], ignore_index=True)

    combined_path = OUTPUT_DIR / "all_validation_sites.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Exported {len(combined)} total validation sites → {combined_path}")
    print(f"  Successful: {(combined['label']==1).sum()}")
    print(f"  Blocked:    {(combined['label']==0).sum()}")


if __name__ == "__main__":
    export()
