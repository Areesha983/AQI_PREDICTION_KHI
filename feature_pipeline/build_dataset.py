"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data directly from MongoDB, merges them,
handles short-gap interpolation, and stores the aligned records back to MongoDB.
"""

import os
import pymongo
import pandas as pd

MAX_GAP_FILL_HOURS = 3


def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]

    print("Extracting raw data from MongoDB cloud collections...")
    
    # 1. Download records from MongoDB collections
    raw_weather_cursor = db["raw_weather"].find()
    raw_aqi_cursor = db["raw_air_quality"].find()

    weather_df = pd.DataFrame(list(raw_weather_cursor)).drop(columns=["_id"], errors="ignore")
    aq_df = pd.DataFrame(list(raw_aqi_cursor)).drop(columns=["_id"], errors="ignore")

    if weather_df.empty or aq_df.empty:
        raise RuntimeError("One or both MongoDB raw data collections are empty! Run the fetch scripts first.")

    # 2. Normalize datetime representations
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    print("\nExecuting synchronized datetime alignment join...")
    dataset = (
        pd.merge(weather_df, aq_df, on="datetime", how="inner")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    print(f"Merged shape (before gap-fill): {dataset.shape}")

    # 3. Short-gap linear interpolation (≤ MAX_GAP_FILL_HOURS)
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index"]
    weather_numeric = [c for c in dataset.select_dtypes(include="number").columns if c not in pollutant_cols]

    for col in pollutant_cols + weather_numeric:
        if col not in dataset.columns:
            continue
        n_before = dataset[col].isna().sum()
        if n_before == 0:
            continue
        dataset[col] = dataset[col].interpolate(method="linear", limit=MAX_GAP_FILL_HOURS, limit_direction="forward")
        n_after = dataset[col].isna().sum()
        if n_before != n_after:
            print(f"  [gap-fill] {col}: {n_before} → {n_after} NaN (filled {n_before - n_after})")

    # 4. Hard drop rows with no valid PM2.5 target
    initial_shape = len(dataset)
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    dropped = initial_shape - len(dataset)
    print(f"\nDropped {dropped} rows with missing PM2.5 ({dropped / initial_shape * 100:.2f}% of dataset).")

    # 5. Save the output dataset BACK to MongoDB
    dataset_upload = dataset.copy()
    dataset_upload["datetime"] = dataset_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = dataset_upload.to_dict(orient="records")

    if records:
        output_collection = db["karachi_aqi_dataset"]
        print(f"Saving {len(records)} clean merged rows to collection 'karachi_aqi_dataset'...")
        for record in records:
            output_collection.update_one(
                {"datetime": record["datetime"]},
                {"$set": record},
                upsert=True
            )

    client.close()
    print("\nSuccessfully built dataset and updated your Cloud Feature Store!")


if __name__ == "__main__":
    main()