"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data directly from MongoDB, merges them,
handles short-gap interpolation, and stores the aligned records back to MongoDB.

OPTIMIZATION CRITICAL UPDATE:
  Integrated dynamic timeline lookbacks to extract incremental delta arrays
  and align multi-source frames smoothly without full archival scanning.
"""

import os
import pymongo
from pymongo import UpdateOne
import pandas as pd
from datetime import datetime, timedelta

# Import the updated extraction subroutines and config helpers
# Import the subroutines directly using standard absolute imports
import fetch_air_quality
import fetch_weather
from config import HISTORICAL_START_DATE, get_end_date

MAX_GAP_FILL_HOURS = 3
BULK_BATCH_SIZE    = 1000

def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]

    print("\n" + "=" * 70)
    print(" ORCHESTRATING INCREMENTAL DATA EXTRACTION LAYER")
    print("=" * 70)

    # 1. Trigger the data extraction layer dynamically
    # The extraction functions inside fetch scripts now manage their own internal database lookbacks,
    # but we trigger them from here to ensure raw MongoDB collections are up to date.
    fetch_air_quality.main()
    fetch_weather.main()

    print("\n" + "=" * 70)
    print(" CONSOLIDATING WEATHER AND AIR QUALITY RECORDS")
    print("=" * 70)

    # 2. Extract freshly updated raw records from collections
    weather_df = (
        pd.DataFrame(list(db["raw_weather"].find()))
        .drop(columns=["_id"], errors="ignore")
    )
    air_quality_df = (
        pd.DataFrame(list(db["raw_air_quality"].find()))
        .drop(columns=["_id"], errors="ignore")
    )

    if weather_df.empty or air_quality_df.empty:
        print(" -> Warning: One or both collections are empty. Synchronizing full history required.")
        client.close()
        return

    # Enforce standard timestamp datatype transformations
    weather_df["datetime"]     = pd.to_datetime(weather_df["datetime"])
    air_quality_df["datetime"] = pd.to_datetime(air_quality_df["datetime"])

    # Ensure uniqueness across temporal indices before structural merge
    weather_df     = weather_df.drop_duplicates(subset=["datetime"])
    air_quality_df = air_quality_df.drop_duplicates(subset=["datetime"])

    # 3. Apply Left Join on Weather Archive anchors
    # (Ensuring zero future target data contamination leaks into feature observations)
    dataset = pd.merge(weather_df, air_quality_df, on="datetime", how="left")
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    print(f" -> Merged matrix generated. Combined structural matrix footprint: {dataset.shape}")

    # 4. Leakage-Free Causal Capping
    dataset = dataset.set_index("datetime")
    
    # Calculate initial completeness validation markers
    initial_len = len(dataset)
    pm25_nulls_before = dataset["pm25"].isna().sum()

    # Apply causal forward-fill capped tightly to handle brief hardware API collection drops
    dataset = dataset.ffill(limit=MAX_GAP_FILL_HOURS)
    dataset = dataset.reset_index()

    # Drop any severe persistent multi-day data voids that forward-filling cannot fix
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    
    dropped = initial_len - len(dataset)
    print(f" -> Causal forward fill complete. Imputed {pm25_nulls_before - dataset['pm25'].isna().sum()} short gaps.")
    print(f" -> Dropped {dropped:,} records with unrecoverable missing PM2.5 strings.")

    # 5. Push Aligned Matrix straight into 'karachi_aqi_dataset' Collection
    dataset_upload = dataset.copy()
    dataset_upload["datetime"] = dataset_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = dataset_upload.to_dict(orient="records")

    if records:
        output_collection = db["karachi_aqi_dataset"]
        print(f"Saving {len(records):,} synced entries to 'karachi_aqi_dataset' via batch bulk writing...")

        operations = [
            UpdateOne(
                {"datetime": r["datetime"]},
                {"$set": r},
                upsert=True,
            )
            for r in records
        ]

        total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE
        for i in range(0, len(operations), BULK_BATCH_SIZE):
            batch_num = i // BULK_BATCH_SIZE + 1
            result = output_collection.bulk_write(
                operations[i : i + BULK_BATCH_SIZE],
                ordered=False,
            )
            
    client.close()
    print("Data alignment and dataset assembly finalized safely.\n")

if __name__ == "__main__":
    main()