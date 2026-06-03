"""
fetch_air_quality.py
---------------------
Fetches hourly air-quality forecast/reanalysis data for Karachi from Open-Meteo.
Saves data directly into MongoDB to act as a cloud-native feature store.
"""

import os
import requests
import pymongo
import pandas as pd
from config import LATITUDE, LONGITUDE, START_DATE, END_DATE


def fetch_air_quality() -> pd.DataFrame:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&hourly="
        "pm2_5,"
        "pm10,"
        "carbon_monoxide,"
        "nitrogen_dioxide,"
        "sulphur_dioxide,"
        "ozone,"
        "dust,"                 # Karachi-specific dust channel
        "uv_index"              # UV drives photochemical O3 production
        "&timezone=Asia%2FKarachi"
    )

    print("\n" + "=" * 80)
    print("AIR QUALITY REQUEST URL")
    print("=" * 80)
    print(url)

    response = requests.get(url, timeout=90)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    data = response.json()
    hourly = data.get("hourly")

    if hourly is None:
        raise ValueError(f"'hourly' section missing.\nResponse:\n{data}")

    aq_df = pd.DataFrame({
        "datetime": hourly["time"],
        "pm25":     hourly["pm2_5"],
        "pm10":     hourly["pm10"],
        "co":       hourly["carbon_monoxide"],
        "no2":      hourly["nitrogen_dioxide"],
        "so2":      hourly["sulphur_dioxide"],
        "o3":       hourly["ozone"],
        "dust":     hourly.get("dust",     [None] * len(hourly["time"])),
        "uv_index": hourly.get("uv_index", [None] * len(hourly["time"])),
    })

    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    if aq_df["datetime"].dt.tz is not None:
        aq_df["datetime"] = aq_df["datetime"].dt.tz_localize(None)

    aq_df = (
        aq_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved: {len(aq_df):,}")
    print(f"Date Range    : {aq_df['datetime'].min()}  →  {aq_df['datetime'].max()}")
    
    # Check PM2.5 missing percentage
    pm25_nan_pct = aq_df["pm25"].isna().mean() * 100
    if pm25_nan_pct > 30:
        raise RuntimeError(f"FATAL: PM2.5 is {pm25_nan_pct:.1f}% missing.")

    return aq_df


def save_to_mongodb(df: pd.DataFrame):
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing from the environment!")

    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]
    collection = db["raw_air_quality"]

    # Convert Datetime objects to string formats so MongoDB can serialize them natively
    df_upload = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if records:
        print(f"Uploading {len(records)} air quality records to MongoDB...")
        for record in records:
            collection.update_one(
                {"datetime": record["datetime"]},
                {"$set": record},
                upsert=True
            )
    client.close()
    print("Air Quality upload complete!")


if __name__ == "__main__":
    aq_df = fetch_air_quality()
    save_to_mongodb(aq_df)