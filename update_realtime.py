"""
update_realtime.py  (FULLY ROBUST PRODUCTION VERSION)
"""

import os
import sys
import certifi
import requests
import pymongo
import time as _time
from pymongo import UpdateOne
from datetime import datetime, timedelta
import pandas as pd

# config.py lives in feature_pipeline/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_pipeline"))
from config import LATITUDE, LONGITUDE

LOOKBACK_HOURS = 6
_API_TIMEOUT   = 30    # Increased to 90s
_API_RETRIES   = 3     # Increased to 5 retries
TTL_SECONDS = 7 * 24 * 3600

# ─────────────────────────────────────────────────────────────────────────────
#  API FETCHERS WITH RETRY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_realtime_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches air quality data from Open-Meteo with exponential backoff."""
    # Note: Using correct Open-Meteo param names (pm2_5) and renaming to internal schema
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index"
        "&timezone=Asia%2FKarachi"
    )
    
    last_exc = None
    for attempt in range(1, _API_RETRIES + 1):
        try:
            r = requests.get(url, timeout=_API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data["hourly"])
            df["datetime"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])
            
            # Standardize names to match historical feature pipeline
            df = df.rename(columns={
                "pm2_5": "pm25",
                "carbon_monoxide": "co",
                "nitrogen_dioxide": "no2",
                "sulphur_dioxide": "so2",
            })
            return df
        except Exception as e:
            last_exc = e
            print(f"  [AQ fetch] Attempt {attempt}/{_API_RETRIES} failed: {e}")
            if attempt < _API_RETRIES:
                _time.sleep(15 * attempt)
    raise last_exc

def _fetch_realtime_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches weather data from Open-Meteo with exponential backoff."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
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
                archive_url = url.replace("api.open-meteo.com/v1/forecast", "archive-api.open-meteo.com/v1/archive")
                r = requests.get(archive_url, timeout=_API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data["hourly"])
            df["datetime"] = pd.to_datetime(df["time"])
            return df.drop(columns=["time"])
        except Exception as e:
            last_exc = e
            print(f"  [WX fetch] Attempt {attempt}/{_API_RETRIES} failed: {e}")
            if attempt < _API_RETRIES:
                _time.sleep(15 * attempt)
    raise last_exc

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    now        = datetime.utcnow()
    start_dt   = now - timedelta(hours=LOOKBACK_HOURS)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = now.strftime("%Y-%m-%d")

    print(f"REALTIME UPDATE | window: {start_date} -> {end_date}")

    # 1. Fetch AQ (Critical - Graceful Exit)
    try:
        aq_df = _fetch_realtime_air_quality(start_date, end_date)
    except Exception as e:
        print(f"  WARN: Air quality fetch failed after retries — {e}")
        print("  Skipping realtime update this cycle to prevent pipeline failure.")
        return

    # 2. Fetch Weather (Non-critical fallback)
    try:
        wx_df = _fetch_realtime_weather(start_date, end_date)
    except Exception as e:
        print(f"  WARN: Weather fetch failed — {e}. Proceeding with AQ-only data.")
        wx_df = pd.DataFrame()

    # 3. Merge
    merged = aq_df if wx_df.empty else pd.merge(aq_df, wx_df, on="datetime", how="inner")

    # 4. Drop future hours — Open-Meteo forecast endpoint returns future rows too.
    # We only want rows whose datetime has already passed in PKT (UTC+5) so the
    # dashboard never shows a future timestamp as "current conditions".
    now_pkt = datetime.utcnow() + timedelta(hours=5)
    merged = merged[merged["datetime"] <= now_pkt].reset_index(drop=True)

    if merged.empty:
        print("  No past-hour rows after future-filter. Skipping upsert.")
        return

    # 5. Prepare and Upsert
    fetched_at = datetime.utcnow()
    records = []
    for _, row in merged.iterrows():
        doc = row.to_dict()
        doc["datetime"] = doc["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        doc["fetched_at"] = fetched_at
        records.append(doc)

    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000, tlsCAFile=certifi.where())
    collection = client["karachi_aqi"]["realtime_observations"]
    
    # TTL Index
    collection.create_index("fetched_at", expireAfterSeconds=TTL_SECONDS, background=True)

    operations = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    if operations:
        collection.bulk_write(operations, ordered=False)
        print(f"  Successfully upserted {len(records)} records.")
    
    client.close()
    print("  Realtime update cycle complete.")

if __name__ == "__main__":
    main()