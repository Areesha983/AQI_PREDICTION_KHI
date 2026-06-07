"""
update_realtime.py  (FIXED)
"""

import os
import sys
import certifi
import requests
import pymongo
from pymongo import UpdateOne
from datetime import datetime, timedelta
import pandas as pd

# config.py lives in feature_pipeline/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_pipeline"))
from config import LATITUDE, LONGITUDE

LOOKBACK_HOURS = 6
_API_TIMEOUT   = 45
_API_RETRIES   = 3
TTL_SECONDS = 7 * 24 * 3600

# ─────────────────────────────────────────────────────────────────────────────
#  API FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_realtime_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches air quality data from Open-Meteo."""
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=pm25,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        "&timezone=Asia%2FKarachi"
    )
    r = requests.get(url, timeout=_API_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df

def _fetch_realtime_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches weather data from Open-Meteo."""
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
#  MERGE AND MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _merge(aq_df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(aq_df, wx_df, on="datetime", how="inner")
    merged = merged.dropna(subset=["pm25"])
    return merged

def _ensure_ttl_index(collection: pymongo.collection.Collection) -> None:
    collection.create_index("fetched_at", expireAfterSeconds=TTL_SECONDS, background=True)

def main() -> None:
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    now        = datetime.utcnow()
    start_dt   = now - timedelta(hours=LOOKBACK_HOURS)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = now.strftime("%Y-%m-%d")

    print(f"REALTIME UPDATE | window: {start_date} -> {end_date}")

    try:
        aq_df = _fetch_realtime_air_quality(start_date, end_date)
        print(f"  Air quality rows fetched : {len(aq_df):,}")
    except Exception as e:
        print(f"  WARN: Air quality fetch failed — {e}. Aborting.")
        return

    try:
        wx_df = _fetch_realtime_weather(start_date, end_date)
        print(f"  Weather rows fetched     : {len(wx_df):,}")
    except Exception as e:
        print(f"  WARN: Weather fetch failed — {e}")
        wx_df = pd.DataFrame()

    if wx_df.empty:
        merged = aq_df.copy()
    else:
        merged = _merge(aq_df, wx_df)

    if merged.empty:
        print("  WARN: No overlapping rows — nothing to write.")
        return

    fetched_at = datetime.utcnow()
    records = []
    for _, row in merged.iterrows():
        doc = row.to_dict()
        dt_obj = doc.pop("datetime")
        doc["datetime"]   = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        doc["fetched_at"] = fetched_at
        records.append(doc)

    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000, tlsCAFile=certifi.where())
    collection = client["karachi_aqi"]["realtime_observations"]
    _ensure_ttl_index(collection)

    operations = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    client["karachi_aqi"]["realtime_observations"].bulk_write(operations, ordered=False)
    client.close()
    print("  Realtime update cycle complete.")

if __name__ == "__main__":
    main()