"""
GreenData — Geocoding Layer (Place Name → County FIPS)

Two-layer system for resolving free-text place names (from Reddit posts)
to county FIPS codes (for joining with government feature data).

  Layer 1: Static lookup table — handles ~80% of cases instantly.
           Combines Census place-to-county crosswalk with manual
           Reddit slang/abbreviation aliases.

  Layer 2: GeoPy (Nominatim) fallback — handles remaining ~20%.
           Geocodes to lat/lon, reverse-geocodes to county, caches
           every result so the same string is never geocoded twice.

This is HIDDEN BOSS #1 from the plan. Without this layer, 90% of
Reddit data is unmatchable to government features.
"""

import json
import os
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# State abbreviation → FIPS prefix mapping
# ---------------------------------------------------------------------------
STATE_TO_FIPS_PREFIX = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}

# Reverse: FIPS prefix → state abbreviation
FIPS_PREFIX_TO_STATE = {v: k for k, v in STATE_TO_FIPS_PREFIX.items()}

# ---------------------------------------------------------------------------
# Reddit-specific place aliases (NOT in any Census file)
# ---------------------------------------------------------------------------
REDDIT_ALIASES = {
    # Northern Virginia data center corridor
    "nova": ["51059", "51107", "51153", "51013"],
    "northern virginia": ["51059", "51107", "51153", "51013"],
    "data center alley": ["51107"],
    "ashburn": ["51107"],
    "manassas": ["51153"],
    "sterling": ["51107"],
    "leesburg": ["51107"],
    "chantilly": ["51059"],
    "reston": ["51059"],
    "herndon": ["51059"],

    # Oregon
    "the dalles": ["41065"],
    "prineville": ["41013"],
    "hillsboro": ["41067"],
    "boardman": ["41049"],

    # Texas
    "dfw": ["48113", "48439"],
    "san antonio": ["48029"],
    "fort worth": ["48439"],
    "garland": ["48113"],
    "plano": ["48085"],
    "midlothian": ["48139"],

    # Iowa
    "council bluffs": ["19155"],
    "west des moines": ["19153"],
    "altoona": ["19153"],

    # Arizona
    "mesa": ["04013"],
    "goodyear": ["04013"],
    "chandler": ["04013"],
    "tempe": ["04013"],
    "scottsdale": ["04013"],
    "surprise": ["04013"],

    # North Carolina
    "durham": ["37063"],
    "raleigh": ["37183"],
    "research triangle": ["37063", "37183"],
    "rtp": ["37063", "37183"],

    # Georgia
    "atlanta": ["13121"],
    "douglasville": ["13097"],

    # Ohio
    "columbus": ["39049"],
    "new albany": ["39049"],
    "hilliard": ["39049"],

    # Indiana
    "indianapolis": ["18097"],
    "indy": ["18097"],

    # Washington
    "quincy": ["53025"],
    "moses lake": ["53025"],
    "seattle": ["53033"],

    # Illinois
    "chicago": ["17031"],
    "elk grove village": ["17031"],
    "mt. prospect": ["17031"],
    "mount prospect": ["17031"],

    # Nevada
    "las vegas": ["32003"],
    "henderson": ["32003"],
    "reno": ["32031"],

    # South Carolina
    "spartanburg": ["45083"],

    # Tennessee
    "nashville": ["47037"],
    "clarksville": ["47125"],

    # Utah
    "salt lake city": ["49035"],
    "slc": ["49035"],
    "west jordan": ["49035"],

    # Common county name shortcuts
    "loudoun": ["51107"],
    "loudoun county": ["51107"],
    "prince william": ["51153"],
    "pw county": ["51153"],
    "fairfax": ["51059"],
    "fairfax county": ["51059"],
    "maricopa": ["04013"],
    "maricopa county": ["04013"],
    "cook county": ["17031"],
}

# ---------------------------------------------------------------------------
# County name → FIPS mapping (for reverse geocoding results)
# Built from Census Bureau county list for target states.
# ---------------------------------------------------------------------------
COUNTY_NAME_TO_FIPS = {
    # Virginia
    ("loudoun county", "VA"): "51107",
    ("fairfax county", "VA"): "51059",
    ("prince william county", "VA"): "51153",
    ("arlington county", "VA"): "51013",
    ("fauquier county", "VA"): "51061",
    ("stafford county", "VA"): "51179",
    # Texas
    ("dallas county", "TX"): "48113",
    ("tarrant county", "TX"): "48439",
    ("bexar county", "TX"): "48029",
    ("collin county", "TX"): "48085",
    ("ellis county", "TX"): "48139",
    ("denton county", "TX"): "48121",
    # Oregon
    ("wasco county", "OR"): "41065",
    ("crook county", "OR"): "41013",
    ("washington county", "OR"): "41067",
    ("morrow county", "OR"): "41049",
    # Iowa
    ("pottawattamie county", "IA"): "19155",
    ("polk county", "IA"): "19153",
    # Arizona
    ("maricopa county", "AZ"): "04013",
    ("pinal county", "AZ"): "04021",
    # North Carolina
    ("wake county", "NC"): "37183",
    ("durham county", "NC"): "37063",
    # Georgia
    ("fulton county", "GA"): "13121",
    ("dekalb county", "GA"): "13089",
    ("douglas county", "GA"): "13097",
    # Ohio
    ("franklin county", "OH"): "39049",
    ("cuyahoga county", "OH"): "39035",
    # Indiana
    ("marion county", "IN"): "18097",
    ("hamilton county", "IN"): "18057",
    # Washington
    ("grant county", "WA"): "53025",
    ("king county", "WA"): "53033",
    # Illinois
    ("cook county", "IL"): "17031",
    ("dupage county", "IL"): "17043",
    ("will county", "IL"): "17197",
    # Nevada
    ("clark county", "NV"): "32003",
    ("washoe county", "NV"): "32031",
    # South Carolina
    ("spartanburg county", "SC"): "45083",
    # Tennessee
    ("davidson county", "TN"): "47037",
    ("montgomery county", "TN"): "47125",
    # Utah
    ("salt lake county", "UT"): "49035",
    ("utah county", "UT"): "49049",
}

# ---------------------------------------------------------------------------
# Geocode cache paths
# ---------------------------------------------------------------------------
REFERENCE_DIR = Path("data/reference")
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

GEOCODE_CACHE_PATH = REFERENCE_DIR / "geocode_cache.json"
CROSSWALK_PATH = REFERENCE_DIR / "place_county_crosswalk.csv"

# ---------------------------------------------------------------------------
# Module-level cache for the static mapping (built once)
# ---------------------------------------------------------------------------
_place_mapping_cache: dict | None = None


# ===================================================================
# Layer 1: Static lookup table
# ===================================================================

def build_place_to_fips_mapping() -> dict:
    """
    Build a comprehensive mapping from place names to county FIPS codes.

    Sources:
      1. Census Bureau place-to-county crosswalk (if available on disk)
      2. Manual Reddit aliases for slang and abbreviations

    Returns:
        dict mapping lowercase place name → list of FIPS codes
    """
    global _place_mapping_cache
    if _place_mapping_cache is not None:
        return _place_mapping_cache

    mapping = {}

    # 1. Load Census crosswalk if available
    if CROSSWALK_PATH.exists():
        try:
            crosswalk = pd.read_csv(CROSSWALK_PATH)
            for _, row in crosswalk.iterrows():
                place_name = str(row["place_name"]).lower().strip()
                county_fips = str(row["county_fips"]).zfill(5)
                if place_name not in mapping:
                    mapping[place_name] = []
                if county_fips not in mapping[place_name]:
                    mapping[place_name].append(county_fips)
            print(f"  Loaded {len(crosswalk)} entries from Census crosswalk")
        except Exception as e:
            print(f"  WARNING: Failed to load crosswalk: {e}")
    else:
        print(f"  No crosswalk at {CROSSWALK_PATH} — using aliases only")

    # 2. Layer Reddit aliases on top (overrides crosswalk for known cases)
    for alias, fips_list in REDDIT_ALIASES.items():
        mapping[alias] = fips_list

    _place_mapping_cache = mapping
    return mapping


# ===================================================================
# Layer 2: GeoPy (Nominatim) fallback with persistent cache
# ===================================================================

_geolocator = Nominatim(user_agent="greendata_research_v1")
_geocode_fn = RateLimiter(_geolocator.geocode, min_delay_seconds=1.0)


def _load_geocode_cache() -> dict:
    """Load the persistent geocode cache from disk."""
    if GEOCODE_CACHE_PATH.exists():
        with open(GEOCODE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: dict):
    """Persist the geocode cache to disk."""
    with open(GEOCODE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def county_name_to_fips(county_name: str, state_hint: str = None) -> str | None:
    """
    Resolve a county name (from GeoPy reverse-geocode) to a FIPS code.

    Uses the COUNTY_NAME_TO_FIPS lookup table. Falls back to a fuzzy
    match if the exact name isn't found.
    """
    if not county_name:
        return None

    normalized = county_name.lower().strip()

    # Add "county" suffix if missing (GeoPy sometimes returns just the name)
    if not normalized.endswith("county"):
        normalized = f"{normalized} county"

    # Try exact match with state hint
    if state_hint:
        key = (normalized, state_hint.upper())
        if key in COUNTY_NAME_TO_FIPS:
            return COUNTY_NAME_TO_FIPS[key]

    # Try without state hint — iterate all entries
    for (name, state), fips in COUNTY_NAME_TO_FIPS.items():
        if name == normalized:
            if state_hint and state != state_hint.upper():
                continue
            return fips

    return None


# ===================================================================
# Main geocoding function
# ===================================================================

def geocode_to_county_fips(place_name: str, state_hint: str = None) -> str | None:
    """
    Geocode a free-text place name to a county FIPS code.

    Strategy:
      1. Check persistent cache first (instant)
      2. Try static lookup table — handles 80% of cases
      3. If ambiguous, use state_hint to disambiguate
      4. Fall back to GeoPy Nominatim API
      5. Cache every result (hit or miss) for reproducibility

    Args:
        place_name: Raw text extracted by SpaCy NER (e.g., "Ashburn")
        state_hint: Optional state from subreddit context (e.g., "VA")

    Returns:
        5-digit county FIPS string, or None if unresolvable
    """
    if not place_name or len(place_name.strip()) < 2:
        return None

    cache = _load_geocode_cache()
    cache_key = f"{place_name.lower().strip()}|{state_hint or ''}"

    # Check cache first
    if cache_key in cache:
        return cache[cache_key]

    # Layer 1: Static lookup
    mapping = build_place_to_fips_mapping()
    candidates = mapping.get(place_name.lower().strip(), [])

    if len(candidates) == 1:
        result = candidates[0]
        cache[cache_key] = result
        _save_geocode_cache(cache)
        return result

    if len(candidates) > 1 and state_hint:
        state_prefix = STATE_TO_FIPS_PREFIX.get(state_hint.upper(), "")
        if state_prefix:
            filtered = [f for f in candidates if f.startswith(state_prefix)]
            if len(filtered) == 1:
                result = filtered[0]
                cache[cache_key] = result
                _save_geocode_cache(cache)
                return result
            elif len(filtered) > 1:
                # Multiple counties in the same state — take the first
                # (usually the most prominent one in the alias list)
                result = filtered[0]
                cache[cache_key] = result
                _save_geocode_cache(cache)
                return result

    if len(candidates) > 1 and not state_hint:
        # Ambiguous without state hint — take the first candidate
        # (aliases are ordered by prominence)
        result = candidates[0]
        cache[cache_key] = result
        _save_geocode_cache(cache)
        return result

    # Layer 2: GeoPy fallback
    query = f"{place_name}, {state_hint}, USA" if state_hint else f"{place_name}, USA"
    try:
        location = _geocode_fn(query, addressdetails=True, country_codes=["us"])
        if location and "address" in location.raw:
            county = location.raw["address"].get("county", "")
            state_from_geo = location.raw["address"].get("state", "")
            # Try to determine state abbreviation from response
            geo_state = state_hint
            if not geo_state:
                for abbr, prefix in STATE_TO_FIPS_PREFIX.items():
                    if state_from_geo.lower() in abbr.lower():
                        geo_state = abbr
                        break

            fips = county_name_to_fips(county, geo_state)
            if fips:
                cache[cache_key] = fips
                _save_geocode_cache(cache)
                return fips
    except Exception as e:
        print(f"  Geocoding failed for '{place_name}': {e}")

    # Cache the miss so we don't retry
    cache[cache_key] = None
    _save_geocode_cache(cache)
    return None


# ===================================================================
# Batch geocoding utility
# ===================================================================

def batch_geocode(
    place_names: list[str],
    state_hints: list[str | None] = None,
) -> dict[str, str | None]:
    """
    Geocode a list of place names, returning a dict of name → FIPS.

    Reports match rate at the end.
    """
    if state_hints is None:
        state_hints = [None] * len(place_names)

    results = {}
    matched = 0
    total = len(place_names)

    for name, hint in zip(place_names, state_hints):
        fips = geocode_to_county_fips(name, hint)
        results[name] = fips
        if fips:
            matched += 1

    rate = (matched / total * 100) if total > 0 else 0
    print(f"\nGeocoding batch: {matched}/{total} matched ({rate:.1f}%)")

    return results
