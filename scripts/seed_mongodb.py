"""
GreenData — MongoDB Atlas Seeding Script (Versioned)

Seeds pre-calculated PSI scores and location data into MongoDB Atlas
with full version tracking. Idempotent — safe to run multiple times.

Version tracking:
  - Deterministic hash of the scores DataFrame
  - model_versions collection tracks: version, hash, timestamp, weights
  - Old versions deactivated, new version set as active
  - Every document tagged with _model_version and _data_hash

Usage:
  python scripts/seed_mongodb.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_VERSION = "1.0.0"


def compute_data_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of the DataFrame for version tracking."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:12]


def seed_database():
    """Seed MongoDB Atlas with versioned PSI scores and location data."""
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        print("ERROR: MONGODB_URI not set in environment.")
        print("Set it in .env or export MONGODB_URI=mongodb+srv://...")
        return

    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pymongo not installed. Run: pip install pymongo")
        return

    # --- Load data ---
    scores_path = PROJECT_ROOT / "data" / "processed" / "psi_scores.parquet"
    final_path = PROJECT_ROOT / "data" / "processed" / "final_scores.parquet"

    if final_path.exists():
        scores = pd.read_parquet(final_path)
        print(f"Loaded final_scores.parquet ({len(scores)} rows)")
    elif scores_path.exists():
        scores = pd.read_parquet(scores_path)
        print(f"Loaded psi_scores.parquet ({len(scores)} rows)")
    else:
        print("ERROR: No scores parquet found. Run the scoring pipeline first.")
        return

    # Ensure FIPS is string
    scores["county_fips"] = scores["county_fips"].astype(str).str.zfill(5)

    # Load features for location data
    features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    locations = None
    if features_path.exists():
        locations = pd.read_parquet(features_path)
        locations["county_fips"] = locations["county_fips"].astype(str).str.zfill(5)
        print(f"Loaded features.parquet ({len(locations)} rows) for location data")

    # Load weights
    weights = None
    weights_path = PROJECT_ROOT / "outputs" / "derived_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)

    # --- Compute version hash ---
    data_hash = compute_data_hash(scores)
    print(f"Data hash: {data_hash}")
    print(f"Model version: {MODEL_VERSION}")

    # --- Connect to MongoDB ---
    print(f"Connecting to MongoDB...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)

    # Test connection
    try:
        client.admin.command("ping")
        print("  Connected successfully")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    db = client["greendata"]

    # --- Check for existing version ---
    existing = db.model_versions.find_one({
        "model_version": MODEL_VERSION,
        "data_hash": data_hash,
    })
    if existing:
        print(f"\nVersion {MODEL_VERSION} ({data_hash}) already seeded at "
              f"{existing.get('seeded_at')}. Skipping.")
        client.close()
        return

    # --- Register model version ---
    print(f"\nSeeding version {MODEL_VERSION}...")

    # Parse weights from scores if not loaded separately
    weights_dict = {}
    if weights:
        weights_dict = weights.get("weights", {})
    elif "weights_used" in scores.columns:
        try:
            weights_dict = json.loads(
                scores.iloc[0]["weights_used"].replace("'", '"')
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    version_record = {
        "model_version": MODEL_VERSION,
        "data_hash": data_hash,
        "seeded_at": datetime.now(timezone.utc),
        "n_locations": len(scores),
        "weights": weights_dict,
        "is_active": True,
    }

    # Deactivate all previous versions
    db.model_versions.update_many({}, {"$set": {"is_active": False}})
    db.model_versions.insert_one(version_record)
    print(f"  Registered model version {MODEL_VERSION}")

    # --- Seed scores ---
    scores_records = scores.to_dict(orient="records")

    # Clean NaN values (MongoDB doesn't handle NaN well)
    import math
    for r in scores_records:
        r["_model_version"] = MODEL_VERSION
        r["_data_hash"] = data_hash
        for k, v in list(r.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None

    # Replace scores for this version
    db.scores.delete_many({"_model_version": MODEL_VERSION})
    if scores_records:
        db.scores.insert_many(scores_records)
    print(f"  Inserted {len(scores_records)} score documents")

    # --- Seed locations (from features) ---
    if locations is not None and not locations.empty:
        loc_records = locations.to_dict(orient="records")
        for r in loc_records:
            r["_model_version"] = MODEL_VERSION
            for k, v in list(r.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    r[k] = None
            # Add GeoJSON point if lat/lon available
            if "lat" in r and "lon" in r and r["lat"] and r["lon"]:
                r["coordinates"] = {
                    "type": "Point",
                    "coordinates": [r["lon"], r["lat"]],
                }

        db.locations.delete_many({"_model_version": MODEL_VERSION})
        if loc_records:
            db.locations.insert_many(loc_records)
        print(f"  Inserted {len(loc_records)} location documents")

    # --- Create indexes ---
    print("  Creating indexes...")
    db.scores.create_index([("psi", -1)])
    db.scores.create_index([("state", 1)])
    db.scores.create_index([("county_fips", 1)])
    db.scores.create_index([("_model_version", 1)])
    db.locations.create_index([("county_fips", 1)])
    db.locations.create_index([("_model_version", 1)])

    # 2dsphere index for geospatial queries
    try:
        db.locations.create_index([("coordinates", "2dsphere")])
    except Exception as e:
        print(f"  Note: 2dsphere index skipped ({e})")

    # --- Summary ---
    n_scores = db.scores.count_documents({"_model_version": MODEL_VERSION})
    n_locations = db.locations.count_documents({"_model_version": MODEL_VERSION})

    print(f"\n{'=' * 60}")
    print(f"Seeding complete!")
    print(f"  Version: {MODEL_VERSION}")
    print(f"  Hash: {data_hash}")
    print(f"  Scores: {n_scores}")
    print(f"  Locations: {n_locations}")
    print(f"{'=' * 60}")

    client.close()


if __name__ == "__main__":
    seed_database()
