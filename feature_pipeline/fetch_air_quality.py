"""
fetch_air_quality.py
---------------------
Fetches hourly air-quality data for Karachi from Open-Meteo.

FIX: save_to_mongodb() now uses bulk_write() instead of row-by-row update_one().
For ~33k rows the old approach took ~50 minutes. Bulk writes complete in <30s.
"""

import os
import requests
import pymongo
from pymongo import UpdateOne
import pandas as pd
from config import LATITUDE, LONGITUDE, START_DATE, END_DATE

BULK_BATCH_SIZE = 1000


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
        "dust,"
        "uv_index"
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

    data    = response.json()
    hourly  = data.get("hourly")

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

    pm25_nan_pct = aq_df["pm25"].isna().mean() * 100
    if pm25_nan_pct > 30:
        raise RuntimeError(f"FATAL: PM2.5 is {pm25_nan_pct:.1f}% missing.")

    return aq_df


def save_to_mongodb(df: pd.DataFrame):
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client     = pymongo.MongoClient(mongo_uri)
    collection = client["karachi_aqi"]["raw_air_quality"]

    df_upload             = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if not records:
        client.close()
        return

    # FIX: was row-by-row update_one() — ~50 min for 33k rows
    operations    = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

    print(f"Uploading {len(records):,} air quality records via bulk_write "
          f"(batch={BULK_BATCH_SIZE})...")
    for i in range(0, len(operations), BULK_BATCH_SIZE):
        batch_num = i // BULK_BATCH_SIZE + 1
        result = collection.bulk_write(operations[i : i + BULK_BATCH_SIZE], ordered=False)
        print(f"  Batch {batch_num}/{total_batches} — "
              f"upserted: {result.upserted_count}, modified: {result.modified_count}")

    client.close()
    print("Air quality upload complete!")


if __name__ == "__main__":
    aq_df = fetch_air_quality()
    save_to_mongodb(aq_df)