"""
GreenData — FastAPI Backend (Phase 5)

Serves pre-calculated PSI scores from parquet files.
All heavy computation happens offline in the analytics pipeline —
this API just reads and filters the results.

Endpoints:
  GET /api/scores          — list with filters (min_psi, state, limit, sort)
  GET /api/scores/{fips}   — single location with all sub-scores
  GET /api/dimensions      — dimension breakdown for all locations
  GET /api/validation      — model metrics for the methodology page
  GET /api/topics          — LDA topic summaries
  GET /api/health          — health check
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ===================================================================
# App setup
# ===================================================================

app = FastAPI(
    title="GreenData PSI API",
    description="Predictive Suitability Index for data center site selection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ===================================================================
# Data loading — parquets loaded once at startup
# ===================================================================

DATA_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")


def _load_parquet(name: str) -> pd.DataFrame | None:
    """Load a parquet file, returning None if missing."""
    path = DATA_DIR / name
    if path.exists():
        return pd.read_parquet(path)
    return None


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# Load all data sources
SCORES: pd.DataFrame | None = None
FEATURES: pd.DataFrame | None = None
RESISTANCE: pd.DataFrame | None = None
WEIGHTS: dict | None = None
TOPIC_SUMMARY: pd.DataFrame | None = None


@app.on_event("startup")
def load_data():
    """Load pre-calculated data at startup."""
    global SCORES, FEATURES, RESISTANCE, WEIGHTS, TOPIC_SUMMARY

    # PSI scores — try final_scores first, fall back to psi_scores
    SCORES = _load_parquet("final_scores.parquet")
    if SCORES is None:
        SCORES = _load_parquet("psi_scores.parquet")

    if SCORES is not None:
        print(f"Loaded {len(SCORES)} PSI scores")
        # Ensure county_fips is string
        SCORES["county_fips"] = SCORES["county_fips"].astype(str).str.zfill(5)
    else:
        print("WARNING: No scores parquet found. API will return empty results.")
        SCORES = pd.DataFrame()

    # Features
    FEATURES = _load_parquet("features.parquet")
    if FEATURES is not None:
        FEATURES["county_fips"] = FEATURES["county_fips"].astype(str).str.zfill(5)
        print(f"Loaded {len(FEATURES)} feature rows")

    # Resistance scores
    RESISTANCE = _load_parquet("county_resistance_scores.parquet")
    if RESISTANCE is not None:
        print(f"Loaded {len(RESISTANCE)} resistance scores")

    # Derived weights
    WEIGHTS = _load_json(OUTPUTS_DIR / "derived_weights.json")
    if WEIGHTS:
        print(f"Loaded derived weights: {WEIGHTS.get('weights')}")

    # Topic summary
    topic_path = OUTPUTS_DIR / "topic_summary.csv"
    if topic_path.exists():
        TOPIC_SUMMARY = pd.read_csv(topic_path)
        print(f"Loaded {len(TOPIC_SUMMARY)} topic summaries")


# ===================================================================
# Helper
# ===================================================================

def _clean_record(record: dict) -> dict:
    """Replace NaN/inf with None for JSON serialization."""
    import math
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of JSON-safe dicts."""
    return [_clean_record(r) for r in df.to_dict(orient="records")]


# ===================================================================
# Endpoints
# ===================================================================

@app.get("/api/health")
def health_check():
    """Health check with data availability status."""
    return {
        "status": "ok",
        "scores_loaded": len(SCORES) if SCORES is not None else 0,
        "features_loaded": len(FEATURES) if FEATURES is not None else 0,
        "resistance_loaded": len(RESISTANCE) if RESISTANCE is not None else 0,
        "weights_loaded": WEIGHTS is not None,
    }


@app.get("/api/scores")
def get_scores(
    min_psi: float = Query(0, ge=0, le=100, description="Minimum PSI score"),
    state: str | None = Query(None, description="Filter by state abbreviation"),
    limit: int = Query(50, ge=1, le=500, description="Max results to return"),
    sort: str = Query(
        "psi",
        description="Sort column",
    ),
    order: str = Query("desc", description="Sort order: asc or desc"),
):
    """
    List PSI scores with optional filters.

    Returns ranked locations with PSI and dimension sub-scores.
    """
    if SCORES is None or SCORES.empty:
        return {"data": [], "count": 0, "total": 0}

    df = SCORES.copy()

    # Filter by minimum PSI
    if min_psi > 0:
        df = df[df["psi"] >= min_psi]

    # Filter by state
    if state:
        state_upper = state.upper()
        if "state" in df.columns:
            df = df[df["state"] == state_upper]
        elif "state_fips" in df.columns:
            from data.collection.government_data import STATE_FIPS
            fips_prefix = STATE_FIPS.get(state_upper, "")
            if fips_prefix:
                df = df[df["county_fips"].str.startswith(fips_prefix)]

    # Sort
    valid_sort_cols = {"psi", "power_score", "environmental_score",
                       "social_score", "confidence", "rank"}
    sort_col = sort if sort in valid_sort_cols and sort in df.columns else "psi"
    ascending = order.lower() == "asc"
    df = df.sort_values(sort_col, ascending=ascending)

    total = len(df)
    df = df.head(limit)

    return {
        "data": _df_to_records(df),
        "count": len(df),
        "total": total,
    }


@app.get("/api/scores/{county_fips}")
def get_score_detail(county_fips: str):
    """
    Get full score detail for a single county.

    Includes PSI, all dimension sub-scores, raw features,
    and resistance data if available.
    """
    if SCORES is None or SCORES.empty:
        raise HTTPException(status_code=404, detail=f"No scores loaded")

    fips = county_fips.zfill(5)
    row = SCORES[SCORES["county_fips"] == fips]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"No score for county {fips}")

    result = _clean_record(row.iloc[0].to_dict())

    # Enrich with raw features if available
    if FEATURES is not None and not FEATURES.empty:
        feat_row = FEATURES[FEATURES["county_fips"] == fips]
        if not feat_row.empty:
            result["raw_features"] = _clean_record(feat_row.iloc[0].to_dict())

    # Enrich with resistance details if available
    if RESISTANCE is not None and not RESISTANCE.empty:
        fips_col = "matched_county_fips" if "matched_county_fips" in RESISTANCE.columns else "county_fips"
        res_row = RESISTANCE[RESISTANCE[fips_col] == fips]
        if not res_row.empty:
            result["resistance_detail"] = _clean_record(res_row.iloc[0].to_dict())

    return {"data": result}


@app.get("/api/dimensions")
def get_dimensions():
    """
    Get dimension score breakdowns for all locations.

    Returns power, environmental, and social sub-scores.
    """
    if SCORES is None or SCORES.empty:
        return {"data": [], "count": 0}

    dim_cols = ["county_fips"]
    if "state" in SCORES.columns:
        dim_cols.append("state")

    for col in ["power_score", "environmental_score", "social_score",
                "psi", "confidence", "rank"]:
        if col in SCORES.columns:
            dim_cols.append(col)

    df = SCORES[dim_cols].copy()
    df = df.sort_values("psi", ascending=False) if "psi" in df.columns else df

    return {
        "data": _df_to_records(df),
        "count": len(df),
    }


@app.get("/api/validation")
def get_validation_results():
    """
    Model validation metrics for the methodology page.

    Returns everything needed to populate the methodology page:
    AUC, F1, p-value, weights, class balance, strategy, match rate.
    """
    result = {
        "model_version": None,
        "auc": None,
        "macro_f1": None,
        "social_test_p_value": None,
        "n_validation_sites": None,
        "class_balance": None,
        "imbalance_strategy": None,
        "geocoding_match_rate": None,
        "weights": None,
        "sensitivity": None,
    }

    # Model version from scores
    if SCORES is not None and "model_version" in SCORES.columns:
        result["model_version"] = str(SCORES["model_version"].iloc[0])

    # Weights
    if WEIGHTS:
        result["weights"] = WEIGHTS.get("weights")
        result["macro_f1"] = WEIGHTS.get("macro_f1")
        result["imbalance_strategy"] = WEIGHTS.get("strategy")
        result["n_validation_sites"] = (
            (WEIGHTS.get("n_successful", 0) or 0)
            + (WEIGHTS.get("n_blocked", 0) or 0)
        )
        result["class_balance"] = {
            "successful": WEIGHTS.get("n_successful"),
            "blocked": WEIGHTS.get("n_blocked"),
        }

    # Sensitivity analysis
    sensitivity_path = OUTPUTS_DIR / "sensitivity_analysis.csv"
    if sensitivity_path.exists():
        sens_df = pd.read_csv(sensitivity_path)
        result["sensitivity"] = sens_df.to_dict(orient="records")

    return result


@app.get("/api/topics")
def get_topics():
    """
    LDA topic summaries for the methodology page.

    Returns topic IDs, labels, and top keywords.
    """
    if TOPIC_SUMMARY is None or TOPIC_SUMMARY.empty:
        return {"data": [], "count": 0}

    records = []
    for _, row in TOPIC_SUMMARY.iterrows():
        record = {
            "topic_id": int(row.get("topic_id", 0)),
            "label": row.get("label", ""),
            "top_words": row.get("top_words", ""),
        }
        records.append(record)

    return {"data": records, "count": len(records)}


@app.get("/api/states")
def get_states():
    """
    List all states that have scored counties.

    Useful for populating filter dropdowns in the frontend.
    """
    if SCORES is None or SCORES.empty:
        return {"data": []}

    if "state" in SCORES.columns:
        states = sorted(SCORES["state"].dropna().unique().tolist())
    else:
        # Derive from county_fips prefix
        from data.collection.government_data import FIPS_PREFIX_TO_STATE
        prefixes = SCORES["county_fips"].str[:2].unique()
        fips_to_state = {v: k for k, v in __import__(
            "data.collection.government_data", fromlist=["STATE_FIPS"]
        ).STATE_FIPS.items()}
        states = sorted([fips_to_state.get(p, p) for p in prefixes])

    return {"data": states}


@app.get("/api/compare")
def compare_locations(
    fips: list[str] = Query(..., description="County FIPS codes to compare (2-5)"),
):
    """
    Compare 2-5 locations side by side.

    Returns dimension scores for radar chart overlay.
    """
    if SCORES is None or SCORES.empty:
        raise HTTPException(status_code=404, detail="No scores loaded")

    if len(fips) < 2 or len(fips) > 5:
        raise HTTPException(
            status_code=400,
            detail="Provide 2-5 FIPS codes to compare",
        )

    padded = [f.zfill(5) for f in fips]
    df = SCORES[SCORES["county_fips"].isin(padded)]

    if df.empty:
        raise HTTPException(status_code=404, detail="None of the provided FIPS codes found")

    return {
        "data": _df_to_records(df),
        "count": len(df),
        "requested": padded,
        "found": df["county_fips"].tolist(),
    }


# ===================================================================
# Run with: uvicorn api.main:app --reload --port 8000
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
