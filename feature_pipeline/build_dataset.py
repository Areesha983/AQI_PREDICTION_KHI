"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data directly from MongoDB, merges them,
handles short-gap interpolation, and stores the aligned records back to MongoDB.

OPTIMIZED LAYER CONFIGURATION:
  Bypasses full historical scans. Isolates query ranges exclusively to 
  the trailing sync windows to prevent network packet read timeouts.
"""

import os
import certifi
import pymongo
from pymongo import UpdateOne
import pandas as pd
from datetime import datetime, timedelta

import fetch_air_quality
import fetch_weather
from config import HISTORICAL_START_DATE, get_end_date

MAX_GAP_FILL_HOURS = 3
BULK_BATCH_SIZE    = 1000

def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    # FIX: removed tlsAllowInvalidCertificates=True — use certifi CA bundle instead.
    client = pymongo.MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=30000,
        tlsCAFile=certifi.where(),
    )
    db = client["karachi_aqi"]

    print("\n" + "=" * 70)
    print(" ORCHESTRATING INCREMENTAL DATA EXTRACTION LAYER")
    print("=" * 70)

    fetch_air_quality.main()
    fetch_weather.main()

    print("\n" + "=" * 70)
    print(" CONSOLIDATING WEATHER AND AIR QUALITY RECORDS")
    print("=" * 70)

    dataset_collection = db["karachi_aqi_dataset"]
    latest_dataset_doc = dataset_collection.find_one(
        filter={},
        projection={"datetime": 1, "_id": 0},
        sort=[("datetime", pymongo.DESCENDING)],
    )
    if latest_dataset_doc and "datetime" in latest_dataset_doc:
        latest_dt = datetime.strptime(latest_dataset_doc["datetime"], "%Y-%m-%d %H:%M:%S")
        lookback_dt = latest_dt - timedelta(days=2)
        tracking_lookback = lookback_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f" -> Dataset collection latest row: {latest_dataset_doc['datetime']}.")
    else:
        tracking_lookback = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        print(" -> Dataset collection is empty. Falling back to 5-day merge window.")
    print(f"Filtering extraction query to active sync frame (>= {tracking_lookback})...")
    
    query_filter = {"datetime": {"$gte": tracking_lookback}}
    projection_filter = {"_id": 0}

    weather_cursor = db["raw_weather"].find(query_filter, projection_filter)
    weather_df = pd.DataFrame(list(weather_cursor))
    
    aq_cursor = db["raw_air_quality"].find(query_filter, projection_filter)
    air_quality_df = pd.DataFrame(list(aq_cursor))

    if weather_df.empty or air_quality_df.empty:
        print(" -> Active window delta frame is empty. System synchronized.")
        client.close()
        return

    weather_df["datetime"]     = pd.to_datetime(weather_df["datetime"])
    air_quality_df["datetime"] = pd.to_datetime(air_quality_df["datetime"])

    weather_df     = weather_df.drop_duplicates(subset=["datetime"])
    air_quality_df = air_quality_df.drop_duplicates(subset=["datetime"])

    dataset = pd.merge(weather_df, air_quality_df, on="datetime", how="left")
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    print(f" -> Delta matrix generated. Aligned slice footprint: {dataset.shape}")

    dataset = dataset.set_index("datetime")
    initial_len = len(dataset)
    pm25_nulls_before = dataset["pm25"].isna().sum()

    dataset = dataset.ffill(limit=MAX_GAP_FILL_HOURS)
    dataset = dataset.reset_index()
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    
    dropped = initial_len - len(dataset)
    print(f" -> Imputed {pm25_nulls_before - dataset['pm25'].isna().sum()} short voids.")

    dataset_upload = dataset.copy()
    dataset_upload["datetime"] = dataset_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = dataset_upload.to_dict(orient="records")

    if records:
        output_collection = db["karachi_aqi_dataset"]
        output_collection.create_index("datetime", unique=True, background=True)
        print(f"Saving {len(records):,} synced entries via fast bulk execution...")

        operations = [
            UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
            for r in records
        ]

        for i in range(0, len(operations), BULK_BATCH_SIZE):
            output_collection.bulk_write(operations[i : i + BULK_BATCH_SIZE], ordered=False)
            
    client.close()
    print("Data alignment and dataset assembly finalized safely.\n")

if __name__ == "__main__":
    main()