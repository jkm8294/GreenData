"""
GreenData — Government Data Collection Module

Fetches data from four federal APIs:
  - EIA Open Data API v2 (electricity pricing, generation, capacity)
  - NREL Developer API (solar irradiance, wind capacity factor)
  - NOAA Climate Data Online (temperature, extreme weather)
  - USGS National Water Information System (water availability)

All raw responses are saved to data/raw/ for reproducibility.
Results are returned as DataFrames keyed by county FIPS.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Target scope: states with significant data center activity
# ---------------------------------------------------------------------------
TARGET_STATES = ["VA", "TX", "OR", "IA", "AZ", "NC", "GA", "OH", "IN", "WA",
                 "IL", "NV", "SC", "TN", "UT"]

STATE_FIPS = {
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

# Key counties where data centers exist or have been proposed
TARGET_COUNTIES = [
    "51107",  # Loudoun County, VA (Data Center Alley)
    "51059",  # Fairfax County, VA
    "51153",  # Prince William County, VA
    "51013",  # Arlington County, VA
    "48113",  # Dallas County, TX
    "48439",  # Tarrant County, TX
    "48029",  # Bexar County, TX
    "41065",  # Wasco County, OR (The Dalles — Google)
    "41013",  # Crook County, OR (Prineville — Facebook)
    "19155",  # Pottawattamie County, IA (Council Bluffs — Google/Facebook)
    "04013",  # Maricopa County, AZ (Mesa/Goodyear — multiple)
    "37183",  # Wake County, NC
    "37063",  # Durham County, NC
    "13121",  # Fulton County, GA
    "13089",  # DeKalb County, GA
    "39049",  # Franklin County, OH
    "39035",  # Cuyahoga County, OH
    "18097",  # Marion County, IN
    "53025",  # Grant County, WA (Quincy — Microsoft)
    "53033",  # King County, WA
    "17031",  # Cook County, IL
    "17043",  # DuPage County, IL
    "32003",  # Clark County, NV (Las Vegas)
    "45083",  # Spartanburg County, SC
    "47037",  # Davidson County, TN
    "49035",  # Salt Lake County, UT
]

# Lat/lon pairs for NREL solar/wind queries (county centroids)
TARGET_LOCATIONS = [
    ("51107", 39.08, -77.64),   # Loudoun County, VA
    ("51059", 38.83, -77.28),   # Fairfax County, VA
    ("51153", 38.69, -77.48),   # Prince William County, VA
    ("48113", 32.77, -96.77),   # Dallas County, TX
    ("48439", 32.76, -97.29),   # Tarrant County, TX
    ("48029", 29.45, -98.52),   # Bexar County, TX
    ("41065", 45.16, -121.17),  # Wasco County, OR
    ("41013", 44.29, -120.83),  # Crook County, OR
    ("19155", 41.34, -95.54),   # Pottawattamie County, IA
    ("04013", 33.35, -112.49),  # Maricopa County, AZ
    ("37183", 35.79, -78.64),   # Wake County, NC
    ("13121", 33.79, -84.39),   # Fulton County, GA
    ("39049", 39.97, -82.99),   # Franklin County, OH
    ("18097", 39.78, -86.15),   # Marion County, IN
    ("53025", 47.20, -119.52),  # Grant County, WA
    ("17031", 41.84, -87.82),   # Cook County, IL
    ("32003", 36.22, -115.27),  # Clark County, NV
    ("45083", 34.94, -81.99),   # Spartanburg County, SC
    ("47037", 36.17, -86.78),   # Davidson County, TN
    ("49035", 40.67, -111.93),  # Salt Lake County, UT
]

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_last_request_time = 0.0
RATE_LIMIT_SECONDS = 0.5  # max 2 requests/sec per API


def _rate_limited_get(url: str, params: dict = None, timeout: int = 30) -> requests.Response:
    """GET with rate limiting and retries."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            _last_request_time = time.time()
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise

    return resp


def _save_raw(data, filename: str):
    """Save raw API response to data/raw/ for reproducibility."""
    path = RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved raw data → {path}")


# ===================================================================
# 1. EIA — Electricity pricing, generation, renewable share
# ===================================================================

def fetch_eia_electricity(states: list[str] = None) -> pd.DataFrame:
    """
    Fetch electricity data from EIA Open Data API v2.

    Endpoints used:
      - ELEC.PRICE — average retail electricity price by state/sector
      - ELEC.GEN — net generation by state and energy source

    Returns DataFrame with columns:
      state, state_fips, industrial_rate_cents_kwh, renewable_pct,
      total_generation_mwh
    """
    if states is None:
        states = TARGET_STATES

    if not EIA_API_KEY:
        print("WARNING: EIA_API_KEY not set. Returning empty DataFrame.")
        return pd.DataFrame()

    print("Fetching EIA electricity data...")
    base_url = "https://api.eia.gov/v2"
    records = []

    for state in states:
        print(f"  EIA → {state}")

        # --- Industrial electricity price ---
        price_params = {
            "api_key": EIA_API_KEY,
            "frequency": "annual",
            "data[0]": "price",
            "facets[stateid][]": state,
            "facets[sectorid][]": "IND",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 1,
        }
        try:
            resp = _rate_limited_get(f"{base_url}/electricity/retail-sales/data/", params=price_params)
            price_data = resp.json()
            price_value = None
            if price_data.get("response", {}).get("data"):
                price_value = price_data["response"]["data"][0].get("price")
        except Exception as e:
            print(f"    Price fetch failed for {state}: {e}")
            price_data = {}
            price_value = None

        # --- Generation by source (to compute renewable %) ---
        gen_params = {
            "api_key": EIA_API_KEY,
            "frequency": "annual",
            "data[0]": "generation",
            "facets[stateid][]": state,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 50,
        }
        try:
            resp = _rate_limited_get(f"{base_url}/electricity/electric-power-operational-data/data/", params=gen_params)
            gen_data = resp.json()
            gen_rows = gen_data.get("response", {}).get("data", [])

            total_gen = 0
            renewable_gen = 0
            renewable_sources = {"SUN", "WND", "HYC", "GEO", "WWW"}
            latest_period = None

            for row in gen_rows:
                if latest_period is None:
                    latest_period = row.get("period")
                if row.get("period") != latest_period:
                    continue
                gen_val = float(row.get("generation", 0) or 0)
                fuel = row.get("fueltypeid", "")
                if fuel == "ALL":
                    total_gen = gen_val
                elif fuel in renewable_sources:
                    renewable_gen += gen_val

        except Exception as e:
            print(f"    Generation fetch failed for {state}: {e}")
            gen_data = {}
            total_gen = 0
            renewable_gen = 0

        renewable_pct = (renewable_gen / total_gen * 100) if total_gen > 0 else None

        records.append({
            "state": state,
            "state_fips": STATE_FIPS.get(state, ""),
            "industrial_rate_cents_kwh": float(price_value) if price_value else None,
            "renewable_pct": round(renewable_pct, 2) if renewable_pct is not None else None,
            "total_generation_mwh": total_gen if total_gen > 0 else None,
        })

    _save_raw(records, "eia_electricity.json")

    df = pd.DataFrame(records)
    print(f"  EIA: collected {len(df)} state records")
    return df


# ===================================================================
# 2. NREL — Solar irradiance & wind capacity factor
# ===================================================================

def fetch_nrel_renewable(locations: list[tuple] = None) -> pd.DataFrame:
    """
    Fetch solar GHI and wind capacity factor from NREL PVWatts and
    Wind Toolkit APIs for each county centroid.

    Returns DataFrame with columns:
      county_fips, lat, lon, solar_ghi_kwh_m2_day, wind_capacity_factor,
      renewable_potential_score
    """
    if locations is None:
        locations = TARGET_LOCATIONS

    if not NREL_API_KEY:
        print("WARNING: NREL_API_KEY not set. Returning empty DataFrame.")
        return pd.DataFrame()

    print("Fetching NREL renewable potential data...")
    records = []

    for county_fips, lat, lon in locations:
        print(f"  NREL → {county_fips} ({lat}, {lon})")

        # --- Solar (PVWatts v8) ---
        solar_ghi = None
        try:
            solar_params = {
                "api_key": NREL_API_KEY,
                "lat": lat,
                "lon": lon,
                "system_capacity": 4,
                "azimuth": 180,
                "tilt": lat,  # optimal tilt ≈ latitude
                "array_type": 1,
                "module_type": 0,
                "losses": 14,
            }
            resp = _rate_limited_get(
                "https://developer.nrel.gov/api/pvwatts/v8.json",
                params=solar_params,
            )
            solar_data = resp.json()
            outputs = solar_data.get("outputs", {})
            # solrad_annual = avg daily GHI in kWh/m²/day
            solar_ghi = outputs.get("solrad_annual")
        except Exception as e:
            print(f"    Solar fetch failed for {county_fips}: {e}")

        # --- Wind (Wind Toolkit — closest site) ---
        wind_cf = None
        try:
            wind_params = {
                "api_key": NREL_API_KEY,
                "lat": lat,
                "lon": lon,
                "hub_height": 100,
            }
            resp = _rate_limited_get(
                "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download.json",
                params=wind_params,
            )
            wind_data = resp.json()
            # Extract capacity factor if available
            if "outputs" in wind_data:
                wind_cf = wind_data["outputs"].get("capacity_factor")
        except Exception as e:
            print(f"    Wind fetch failed for {county_fips}: {e}")

        # Composite renewable potential score (simple average of normalized values)
        potential = None
        if solar_ghi is not None:
            # Solar GHI range: ~3 (cloudy) to ~6.5 (desert) kWh/m²/day
            solar_norm = min(max((solar_ghi - 3.0) / 3.5, 0), 1) * 100
            potential = solar_norm
            if wind_cf is not None:
                # Wind CF range: ~0.15 (poor) to ~0.45 (excellent)
                wind_norm = min(max((wind_cf - 0.15) / 0.30, 0), 1) * 100
                potential = (solar_norm + wind_norm) / 2

        records.append({
            "county_fips": county_fips,
            "lat": lat,
            "lon": lon,
            "solar_ghi_kwh_m2_day": solar_ghi,
            "wind_capacity_factor": wind_cf,
            "renewable_potential_score": round(potential, 2) if potential else None,
        })

    _save_raw(records, "nrel_renewable.json")

    df = pd.DataFrame(records)
    print(f"  NREL: collected {len(df)} location records")
    return df


# ===================================================================
# 3. NOAA — Climate & extreme weather
# ===================================================================

def fetch_noaa_climate(county_fips_list: list[str] = None) -> pd.DataFrame:
    """
    Fetch climate normals and extreme weather event counts from NOAA
    Climate Data Online (CDO) Web Services.

    API: https://www.ncdc.noaa.gov/cdo-web/api/v2/

    Note: NOAA CDO uses FIPS codes prefixed with "FIPS:" for location queries.

    Returns DataFrame with columns:
      county_fips, avg_temp_f, temp_variance, cooling_degree_days,
      extreme_weather_events_yr
    """
    if county_fips_list is None:
        county_fips_list = TARGET_COUNTIES

    # NOAA CDO requires a token in header (free, register at ncdc.noaa.gov)
    noaa_token = os.environ.get("NOAA_CDO_TOKEN", "")
    if not noaa_token:
        print("WARNING: NOAA_CDO_TOKEN not set. Using fallback climate estimates.")
        return _noaa_fallback(county_fips_list)

    print("Fetching NOAA climate data...")
    headers = {"token": noaa_token}
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    records = []

    for fips in county_fips_list:
        print(f"  NOAA → FIPS:{fips}")
        try:
            params = {
                "datasetid": "NORMAL_ANN",
                "locationid": f"FIPS:{fips}",
                "datatypeid": "ANN-TAVG-NORMAL,ANN-CLDD-NORMAL",
                "limit": 10,
            }
            resp = _rate_limited_get(f"{base_url}/data", params=params)
            resp.headers  # access to trigger potential errors
            data = resp.json()

            avg_temp = None
            cdd = None
            for item in data.get("results", []):
                if item["datatype"] == "ANN-TAVG-NORMAL":
                    avg_temp = item["value"] / 10.0  # tenths of degree F
                elif item["datatype"] == "ANN-CLDD-NORMAL":
                    cdd = item["value"]

            records.append({
                "county_fips": fips,
                "avg_temp_f": avg_temp,
                "cooling_degree_days": cdd,
                "temp_variance": None,  # requires monthly data
                "extreme_weather_events_yr": None,  # requires storm events dataset
            })
        except Exception as e:
            print(f"    NOAA fetch failed for {fips}: {e}")
            records.append({
                "county_fips": fips,
                "avg_temp_f": None,
                "cooling_degree_days": None,
                "temp_variance": None,
                "extreme_weather_events_yr": None,
            })

    _save_raw(records, "noaa_climate.json")

    df = pd.DataFrame(records)
    print(f"  NOAA: collected {len(df)} county records")
    return df


def _noaa_fallback(county_fips_list: list[str]) -> pd.DataFrame:
    """
    Fallback climate estimates based on state-level averages when NOAA
    CDO token is not available. These are approximate and should be
    replaced with real API data for production use.
    """
    # State-level avg temps (°F) and cooling degree days (approx annual)
    state_climate = {
        "51": {"avg_temp_f": 55.1, "cooling_degree_days": 1100},  # VA
        "48": {"avg_temp_f": 64.8, "cooling_degree_days": 2700},  # TX
        "41": {"avg_temp_f": 48.4, "cooling_degree_days": 300},   # OR
        "19": {"avg_temp_f": 47.0, "cooling_degree_days": 700},   # IA
        "04": {"avg_temp_f": 60.3, "cooling_degree_days": 3400},  # AZ
        "37": {"avg_temp_f": 59.0, "cooling_degree_days": 1500},  # NC
        "13": {"avg_temp_f": 63.5, "cooling_degree_days": 1800},  # GA
        "39": {"avg_temp_f": 50.7, "cooling_degree_days": 700},   # OH
        "18": {"avg_temp_f": 51.7, "cooling_degree_days": 800},   # IN
        "53": {"avg_temp_f": 48.3, "cooling_degree_days": 200},   # WA
        "17": {"avg_temp_f": 51.8, "cooling_degree_days": 800},   # IL
        "32": {"avg_temp_f": 49.9, "cooling_degree_days": 2500},  # NV
        "45": {"avg_temp_f": 62.4, "cooling_degree_days": 1700},  # SC
        "47": {"avg_temp_f": 57.6, "cooling_degree_days": 1300},  # TN
        "49": {"avg_temp_f": 48.6, "cooling_degree_days": 700},   # UT
    }

    records = []
    for fips in county_fips_list:
        state_fp = fips[:2]
        climate = state_climate.get(state_fp, {"avg_temp_f": None, "cooling_degree_days": None})
        records.append({
            "county_fips": fips,
            "avg_temp_f": climate["avg_temp_f"],
            "cooling_degree_days": climate["cooling_degree_days"],
            "temp_variance": None,
            "extreme_weather_events_yr": None,
        })

    print(f"  NOAA fallback: estimated {len(records)} county records from state averages")
    return pd.DataFrame(records)


# ===================================================================
# 4. USGS — Water availability
# ===================================================================

def fetch_usgs_water(county_fips_list: list[str] = None) -> pd.DataFrame:
    """
    Fetch water availability indicators from USGS National Water
    Information System (NWIS).

    API: https://waterservices.usgs.gov/rest/IV/

    Uses instantaneous values for groundwater levels and streamflow
    at representative sites per county.

    Returns DataFrame with columns:
      county_fips, groundwater_level_ft, surface_water_flow_cfs,
      drought_severity_index
    """
    if county_fips_list is None:
        county_fips_list = TARGET_COUNTIES

    print("Fetching USGS water data...")

    # USGS queries by state + county FIPS. We use the IV (instantaneous
    # values) service to get recent groundwater and streamflow readings.
    records = []

    for fips in county_fips_list:
        state_fp = fips[:2]
        county_fp = fips[2:]
        print(f"  USGS → {fips}")

        gw_level = None
        sw_flow = None

        # --- Groundwater levels (parameter code 72019 = depth to water) ---
        try:
            gw_params = {
                "format": "json",
                "stateCd": state_fp,
                "countyCd": county_fp,
                "parameterCd": "72019",
                "siteType": "GW",
                "siteStatus": "active",
            }
            resp = _rate_limited_get(
                "https://waterservices.usgs.gov/nwis/iv/",
                params=gw_params,
                timeout=15,
            )
            gw_data = resp.json()
            ts_list = gw_data.get("value", {}).get("timeSeries", [])
            if ts_list:
                values = ts_list[0].get("values", [{}])[0].get("value", [])
                if values:
                    gw_level = float(values[-1].get("value", 0))
        except Exception as e:
            print(f"    GW fetch failed for {fips}: {e}")

        # --- Surface water flow (parameter code 00060 = discharge cfs) ---
        try:
            sw_params = {
                "format": "json",
                "stateCd": state_fp,
                "countyCd": county_fp,
                "parameterCd": "00060",
                "siteType": "ST",
                "siteStatus": "active",
            }
            resp = _rate_limited_get(
                "https://waterservices.usgs.gov/nwis/iv/",
                params=sw_params,
                timeout=15,
            )
            sw_data = resp.json()
            ts_list = sw_data.get("value", {}).get("timeSeries", [])
            if ts_list:
                # Average across all monitoring sites in the county
                flows = []
                for ts in ts_list[:10]:  # cap to avoid huge responses
                    vals = ts.get("values", [{}])[0].get("value", [])
                    if vals:
                        try:
                            flows.append(float(vals[-1]["value"]))
                        except (ValueError, KeyError):
                            pass
                if flows:
                    sw_flow = sum(flows) / len(flows)
        except Exception as e:
            print(f"    SW fetch failed for {fips}: {e}")

        # Drought severity: derive from water availability later in scoring
        records.append({
            "county_fips": fips,
            "groundwater_level_ft": gw_level,
            "surface_water_flow_cfs": round(sw_flow, 2) if sw_flow else None,
            "drought_severity_index": None,  # computed in scoring phase
        })

    _save_raw(records, "usgs_water.json")

    df = pd.DataFrame(records)
    print(f"  USGS: collected {len(df)} county records")
    return df


# ===================================================================
# 5. Master Feature Builder
# ===================================================================

def build_master_features() -> pd.DataFrame:
    """
    Fetch all government datasets and join on county_fips into a single
    master feature DataFrame. Saves to data/processed/features.parquet.

    EIA data is at state level, so it gets broadcast to all counties
    in that state. NREL, NOAA, and USGS are at county level.
    """
    print("\n" + "=" * 60)
    print("Building master feature dataset")
    print("=" * 60)

    # Fetch all sources
    eia_df = fetch_eia_electricity(TARGET_STATES)
    nrel_df = fetch_nrel_renewable(TARGET_LOCATIONS)
    climate_df = fetch_noaa_climate(TARGET_COUNTIES)
    water_df = fetch_usgs_water(TARGET_COUNTIES)

    # Start with the county list as the spine
    master = pd.DataFrame({"county_fips": TARGET_COUNTIES})

    # Add state FIPS prefix for EIA join
    master["state_fips"] = master["county_fips"].str[:2]

    # --- Merge EIA (state-level → broadcast to counties) ---
    if not eia_df.empty:
        master = master.merge(
            eia_df[["state_fips", "state", "industrial_rate_cents_kwh",
                     "renewable_pct", "total_generation_mwh"]],
            on="state_fips",
            how="left",
        )

    # --- Merge NREL (county-level) ---
    if not nrel_df.empty:
        master = master.merge(
            nrel_df[["county_fips", "solar_ghi_kwh_m2_day",
                      "wind_capacity_factor", "renewable_potential_score"]],
            on="county_fips",
            how="left",
        )

    # --- Merge NOAA (county-level) ---
    if not climate_df.empty:
        master = master.merge(
            climate_df[["county_fips", "avg_temp_f", "cooling_degree_days",
                         "temp_variance", "extreme_weather_events_yr"]],
            on="county_fips",
            how="left",
        )

    # --- Merge USGS (county-level) ---
    if not water_df.empty:
        master = master.merge(
            water_df[["county_fips", "groundwater_level_ft",
                       "surface_water_flow_cfs", "drought_severity_index"]],
            on="county_fips",
            how="left",
        )

    # Save
    out_path = Path("data/processed/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out_path, index=False)

    print(f"\nMaster features: {master.shape[0]} counties × {master.shape[1]} columns")
    print(f"Saved → {out_path}")
    print(f"\nColumns: {list(master.columns)}")
    print(f"Null counts:\n{master.isnull().sum().to_string()}")

    return master


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    master = build_master_features()
    print("\nDone. Preview:")
    print(master.head(10).to_string())
