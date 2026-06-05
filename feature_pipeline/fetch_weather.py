"""
fetch_weather.py
----------------
Fetches hourly weather data for Karachi from Open-Meteo Archive API.
OPTIMIZATION: Dynamically scans collection state to isolate data voids
and request incremental catch-up data matrices selectively.
"""

import os
import requests
import pymongo
from pymongo import UpdateOne
from datetime import datetime, timedelta
import pandas as pd
from config import LATITUDE, LONGITUDE, HISTORICAL_START_DATE, get_end_date

BULK_BATCH_SIZE = 1000

def get_dynamic_start_date(collection) -> str:
    """Checks MongoDB for the latest entry to determine incremental pipeline window."""
    try:
        latest_record = collection.find_one(
            filter={},
            projection={"datetime": 1, "_id": 0},
            sort=[("datetime", pymongo.DESCENDING)]
        )
        
        if latest_record and "datetime" in latest_record:
            latest_dt = datetime.strptime(latest_record["datetime"], "%Y-%m-%d %H:%M:%S")
            buffer_dt = latest_dt - timedelta(days=2)
            print(f" -> Found existing records up to {latest_record['datetime']}. Setting buffer lookback.")
            return buffer_dt.strftime("%Y-%m-%d")
            
    except Exception as e:
        print(f" -> Warning while querying collection state: {e}. Falling back to historical initialization.")
        
    print(" -> Collection is empty. Initiating complete historical baseline pull.")
    return HISTORICAL_START_DATE

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly="
        "temperature_2m,"
        "relative_humidity_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m,"
        "precipitation,"
        "cloud_cover,"
        "dew_point_2m,"
        "surface_pressure"
        "&wind_speed_unit=kmh"
        "&timezone=Asia%2FKarachi"
    )

    print("\n" + "=" * 80)
    print(f"WEATHER ARCHIVE REQUEST: {start_date} to {end_date}")
    print("=" * 80)

    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo Archive API returned error code {response.status_code}: {response.text}")

    data = response.json()
    if "hourly" not in data:
        raise KeyError(f"Unexpected API response payload: {data}")

    hourly_data = data["hourly"]
    weather_df = pd.DataFrame(hourly_data)
    weather_df["datetime"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.drop(columns=["time"])

    print(f"Successfully processed {len(weather_df):,} rows.")
    return weather_df

def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client = pymongo.MongoClient(
    mongo_uri,
    serverSelectionTimeoutMS=15000,
    connectTimeoutMS=15000,
    socketTimeoutMS=15000,
    tlsAllowInvalidCertificates=True  # Bypasses local runner certificate validation barriers
)
    db = client["karachi_aqi"]
    collection = db["raw_weather"]

    start_date = get_dynamic_start_date(collection)
    end_date = get_end_date()

    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
        print(" -> Data is already fully synchronized up to the current operational window. Skipping pull.")
        client.close()
        return

    df = fetch_weather(start_date, end_date)

    df_upload = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if not records:
        client.close()
        return

    operations = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    
    print(f"Uploading {len(records):,} weather records via bulk_write...")
    for i in range(0, len(operations), BULK_BATCH_SIZE):
        collection.bulk_write(operations[i : i + BULK_BATCH_SIZE], ordered=False)

    client.close()
    print("Weather data update cycle finalized cleanly.")

if __name__ == "__main__":
    main()