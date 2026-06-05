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

    # Set fault-tolerant network fallback parameters
    client = pymongo.MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=30000,           # Elevated to 30s to permit larger batch transfers safely
        tlsAllowInvalidCertificates=True
    )
    db = client["karachi_aqi"]

    print("\n" + "=" * 70)
    print(" ORCHESTRATING INCREMENTAL DATA EXTRACTION LAYER")
    print("=" * 70)

    # 1. Trigger dynamic data catch-up APIs
    fetch_air_quality.main()
    fetch_weather.main()

    print("\n" + "=" * 70)
    print(" CONSOLIDATING WEATHER AND AIR QUALITY RECORDS")
    print("=" * 70)

    # 2. OPTIMIZATION: Isolate connection range to the active sliding tracking window
    # We look back 5 days from today to guarantee we catch the 72 hours just downloaded,
    # avoiding a full collection scan over the remote cloud network.
    tracking_lookback = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
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

    # Enforce standard timestamp datatype transformations
    weather_df["datetime"]     = pd.to_datetime(weather_df["datetime"])
    air_quality_df["datetime"] = pd.to_datetime(air_quality_df["datetime"])

    # Ensure uniqueness across temporal indices before structural merge
    weather_df     = weather_df.drop_duplicates(subset=["datetime"])
    air_quality_df = air_quality_df.drop_duplicates(subset=["datetime"])

    # 3. Apply Left Join on Weather Archive anchors
    dataset = pd.merge(weather_df, air_quality_df, on="datetime", how="left")
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    print(f" -> Delta matrix generated. Aligned slice footprint: {dataset.shape}")

    # 4. Leakage-Free Causal Capping
    dataset = dataset.set_index("datetime")
    initial_len = len(dataset)
    pm25_nulls_before = dataset["pm25"].isna().sum()

    # Apply causal forward-fill capped tightly to handle brief hardware API drops
    dataset = dataset.ffill(limit=MAX_GAP_FILL_HOURS)
    dataset = dataset.reset_index()

    # Drop any severe persistent data voids that forward-filling cannot fix
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    
    dropped = initial_len - len(dataset)
    print(f" -> Imputed {pm25_nulls_before - dataset['pm25'].isna().sum()} short voids.")

    # 5. Push Aligned Matrix straight into 'karachi_aqi_dataset' Collection via upsert
    dataset_upload = dataset.copy()
    dataset_upload["datetime"] = dataset_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = dataset_upload.to_dict(orient="records")

    if records:
        output_collection = db["karachi_aqi_dataset"]
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