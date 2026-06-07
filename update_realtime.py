"""
update_realtime.py
------------------
FAST PATH: Fetches the last ~6 hours of air quality + weather data and
upserts into `realtime_observations` (a lightweight, self-cleaning collection).

This script is intentionally separate from the main feature pipeline so it
can run at the end of every hourly GitHub Actions job without touching the
historical `processed_features` collection that the training pipeline depends on.

Architecture
─────────────
  feature_pipeline.yml  →  run_feature_pipeline.py  →  processed_features  (training data)
  feature_pipeline.yml  →  update_realtime.py        →  realtime_observations (dashboard)

The `realtime_observations` collection has a TTL index that auto-deletes
documents older than 7 days, keeping your free-tier Atlas storage clean.

Open-Meteo lag note
────────────────────
The air quality API has a ~1–2 h publication lag for the most recent hour.
We fetch from (now - 6h) to today so those hours are always present, even
if the very latest slot is still NaN-filled.
"""

import os
import sys
import certifi
import requests
import pymongo
from pymongo import UpdateOne
from datetime import datetime, timedelta
import pandas as pd

# config.py lives in feature_pipeline/ — add it to the path so we can import it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_pipeline"))
from config import LATITUDE, LONGITUDE

# ── How far back to look on each run ─────────────────────────────────────────
LOOKBACK_HOURS = 6
_API_TIMEOUT   = 45   # seconds — Open-Meteo can be slow; 15s was too tight
_API_RETRIES   = 3    # attempts before giving up

# ── TTL: 7 days in seconds ────────────────────────────────────────────────────
TTL_SECONDS = 7 * 24 * 3600   # 604 800


# ─────────────────────────────────────────────────────────────────────────────
#  API FETCHERS  (lightweight — no missingness checks, no bulk batching)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_realtime_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly="
        "temperature_2m,relative_humidity_2m,pressure_msl,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
        "precipitation,cloud_cover,dew_point_2m,surface_pressure"
        "&wind_speed_unit=kmh"
        "&timezone=Asia%2FKarachi"
    )

    last_exc = None

    for attempt in range(1, _API_RETRIES + 1):
        try:
            r = requests.get(url, timeout=_API_TIMEOUT)

            if r.status_code != 200:
                archive_url = (
                    "https://archive-api.open-meteo.com/v1/archive"
                    f"?latitude={LATITUDE}"
                    f"&longitude={LONGITUDE}"
                    f"&start_date={start_date}"
                    f"&end_date={end_date}"
                    "&hourly="
                    "temperature_2m,relative_humidity_2m,pressure_msl,"
                    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                    "precipitation,cloud_cover,dew_point_2m,surface_pressure"
                    "&wind_speed_unit=kmh"
                    "&timezone=Asia%2FKarachi"
                )

                r = requests.get(archive_url, timeout=_API_TIMEOUT)

            r.raise_for_status()

            data = r.json()

            if "hourly" not in data:
                raise KeyError(f"Unexpected weather response: {data}")

            df = pd.DataFrame(data["hourly"])

            if df.empty:
                raise ValueError("Weather API returned an empty dataframe.")

            df["datetime"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])

            return df

        except Exception as e:
            last_exc = e
            print(f"  [WX fetch] Attempt {attempt}/{_API_RETRIES} failed: {e}")

            if attempt < _API_RETRIES:
                import time as _time
                _time.sleep(10 * attempt)

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE  —  join air-quality + weather on datetime
# ─────────────────────────────────────────────────────────────────────────────

def _merge(aq_df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(aq_df, wx_df, on="datetime", how="inner")
    merged = merged.dropna(subset=["pm25"])   # keep only hours with AQ data
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  ENSURE TTL INDEX  (idempotent — MongoDB ignores duplicate index creation)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_ttl_index(collection: pymongo.collection.Collection) -> None:
    """
    Creates a TTL index on `fetched_at` (ISODate) so documents older than
    7 days are automatically deleted by MongoDB's background TTL thread.
    Safe to call on every run — MongoDB ignores the call if the index exists.
    """
    collection.create_index(
        "fetched_at",
        expireAfterSeconds=TTL_SECONDS,
        background=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    # ── Date window ───────────────────────────────────────────────────────────
    now        = datetime.utcnow()
    start_dt   = now - timedelta(hours=LOOKBACK_HOURS)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = now.strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"REALTIME UPDATE  |  window: {start_date} → {end_date}  (last {LOOKBACK_HOURS}h)")
    print("=" * 70)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        aq_df = _fetch_realtime_air_quality(start_date, end_date)
        print(f"  Air quality rows fetched : {len(aq_df):,}")
    except Exception as e:
        print(f"  WARN: Air quality fetch failed — {e}. Aborting realtime update.")
        return

    try:
        wx_df = _fetch_realtime_weather(start_date, end_date)
        print(f"  Weather rows fetched     : {len(wx_df):,}")
    except Exception as e:
        print(f"  WARN: Weather fetch failed — {e}")
        wx_df = pd.DataFrame()

    # ── Merge ─────────────────────────────────────────────────────────────────
    if wx_df.empty:
        print("  WARN: Proceeding with AQ-only realtime update.")
        merged = aq_df.copy()
    else:
        merged = _merge(aq_df, wx_df)

    if merged.empty:
        print("  WARN: No overlapping rows after merge — nothing to write.")
        return

    # ── Prepare records ───────────────────────────────────────────────────────
    fetched_at = datetime.utcnow()              # ISODate for TTL index
    records = []
    for _, row in merged.iterrows():
        doc = row.to_dict()
        dt_obj = doc.pop("datetime")
        doc["datetime"]   = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        doc["fetched_at"] = fetched_at          # TTL anchor — must be ISODate
        records.append(doc)

    # ── Upsert into MongoDB ───────────────────────────────────────────────────
    client = pymongo.MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=15000,
        tlsCAFile=certifi.where(),
    )
    db         = client["karachi_aqi"]
    collection = db["realtime_observations"]

    # Ensure the TTL index exists (no-op after first run)
    _ensure_ttl_index(collection)

    operations = [
        UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
        for r in records
    ]
    result = collection.bulk_write(operations, ordered=False)
    client.close()

    print(f"  Upserted : {result.upserted_count} new | Modified : {result.modified_count}")
    print("  Realtime update cycle complete.\n")


if __name__ == "__main__":
    main()