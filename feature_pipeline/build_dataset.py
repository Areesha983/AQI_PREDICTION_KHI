"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data directly from MongoDB, merges them,
handles short-gap interpolation, and stores the aligned records back to MongoDB.

FIXES APPLIED:
  1. Gap-fill changed from interpolate(method="linear", limit_direction="forward")
     to ffill(limit=MAX_GAP_FILL_HOURS). The old method still used future anchor
     values; ffill is strictly causal.
  2. MongoDB writes now use bulk_write() in batches of 1000 instead of
     individual update_one() calls. For 33k rows this drops write time from
     ~50 minutes to under 30 seconds.
"""

import os
import pymongo
from pymongo import UpdateOne
import pandas as pd

MAX_GAP_FILL_HOURS = 3
BULK_BATCH_SIZE    = 1000


def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]

    print("Extracting raw data from MongoDB cloud collections...")

    # 1. Download records from MongoDB collections
    weather_df = (
        pd.DataFrame(list(db["raw_weather"].find()))
        .drop(columns=["_id"], errors="ignore")
    )
    aq_df = (
        pd.DataFrame(list(db["raw_air_quality"].find()))
        .drop(columns=["_id"], errors="ignore")
    )

    if weather_df.empty or aq_df.empty:
        raise RuntimeError(
            "One or both MongoDB raw data collections are empty! "
            "Run fetch_weather.py and fetch_air_quality.py first."
        )

    # 2. Normalize datetime
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    aq_df["datetime"]      = pd.to_datetime(aq_df["datetime"])

    print("\nExecuting synchronized datetime alignment join...")
    dataset = (
        pd.merge(weather_df, aq_df, on="datetime", how="inner")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    print(f"Merged shape (before gap-fill): {dataset.shape}")

    # 3. Short-gap CAUSAL forward-fill only (≤ MAX_GAP_FILL_HOURS)
    # FIX: pandas interpolate() uses both neighbours as anchors even with
    # limit_direction="forward", leaking future values. ffill() is causal.
    pollutant_cols  = ["pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index"]
    weather_numeric = [
        c for c in dataset.select_dtypes(include="number").columns
        if c not in pollutant_cols
    ]

    for col in pollutant_cols + weather_numeric:
        if col not in dataset.columns:
            continue
        n_before = dataset[col].isna().sum()
        if n_before == 0:
            continue
        dataset[col] = dataset[col].ffill(limit=MAX_GAP_FILL_HOURS)
        n_after = dataset[col].isna().sum()
        if n_before != n_after:
            print(f"  [gap-fill] {col}: {n_before} → {n_after} NaN "
                  f"(filled {n_before - n_after})")

    # 4. Hard drop rows with no valid PM2.5
    initial_len = len(dataset)
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    dropped = initial_len - len(dataset)
    print(f"\nDropped {dropped} rows with missing PM2.5 "
          f"({dropped / initial_len * 100:.2f}% of dataset).")

    # 5. Save back to MongoDB via bulk_write (FIX: was row-by-row update_one)
    dataset_upload = dataset.copy()
    dataset_upload["datetime"] = dataset_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = dataset_upload.to_dict(orient="records")

    if records:
        output_collection = db["karachi_aqi_dataset"]
        print(f"\nSaving {len(records):,} rows to 'karachi_aqi_dataset' "
              f"via bulk ops (batch={BULK_BATCH_SIZE})...")

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
            print(f"  Batch {batch_num}/{total_batches} — "
                  f"upserted: {result.upserted_count}, "
                  f"modified: {result.modified_count}")

    client.close()
    print("\nSuccessfully built dataset and updated your Cloud Feature Store!")


if __name__ == "__main__":
    main()