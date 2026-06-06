"""
fetch_air_quality.py
---------------------
Fetches air-quality data for Karachi from Open-Meteo.
OPTIMIZATION: Dynamically determines if it needs a full historical backfill
or an incremental hourly/daily catch-up based on existing database state.
"""

import os
import ssl
import certifi
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

def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly="
        "pm2_5,"
        "pm10,"
        "carbon_monoxide,"
        "nitrogen_dioxide,"
        "sulphur_dioxide,"
        "ozone,"
        "dust,"
        "uv_index"
        "&timezone=Asia%2FKarachi"
    )

    print("\n" + "=" * 80)
    print(f"AIR QUALITY REQUEST: {start_date} to {end_date}")
    print("=" * 80)

    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo API returned error code {response.status_code}: {response.text}")

    data = response.json()
    if "hourly" not in data:
        raise KeyError(f"Unexpected API response payload: {data}")

    hourly_data = data["hourly"]
    aq_df = pd.DataFrame(hourly_data)
    aq_df["datetime"] = pd.to_datetime(aq_df["time"])
    aq_df = aq_df.drop(columns=["time"])

    rename_map = {
        "pm2_5": "pm25",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
    }
    aq_df = aq_df.rename(columns=rename_map)

    total_rows = len(aq_df)
    pm25_nans = aq_df["pm25"].isna().sum()
    pm25_nan_pct = (pm25_nans / total_rows) * 100
    print(f"Retrieved {total_rows:,} intervals. PM2.5 Missingness: {pm25_nan_pct:.2f}%")

    if pm25_nan_pct > 30.0:
        raise RuntimeError(f"FATAL: PM2.5 is {pm25_nan_pct:.1f}% missing.")

    return aq_df

def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    # FIX: removed tlsAllowInvalidCertificates=True — incompatible with Atlas M0 TLS enforcement.
    # Use certifi's CA bundle instead for proper certificate validation on Render.
    client = pymongo.MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=15000,
        tlsCAFile=certifi.where(),
    )
    db = client["karachi_aqi"]
    collection = db["raw_air_quality"]

    start_date = get_dynamic_start_date(collection)
    end_date = get_end_date()
    
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
        print(" -> Data is already fully synchronized up to the current operational window. Skipping pull.")
        client.close()
        return

    df = fetch_air_quality(start_date, end_date)
    
    df_upload = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if not records:
        client.close()
        return

    operations = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

    print(f"Uploading {len(records):,} air quality records via bulk_write...")
    for i in range(0, len(operations), BULK_BATCH_SIZE):
        batch_num = i // BULK_BATCH_SIZE + 1
        collection.bulk_write(operations[i : i + BULK_BATCH_SIZE], ordered=False)
        
    client.close()
    print("Air quality update cycle finalized cleanly.")

if __name__ == "__main__":
    main()