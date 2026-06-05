"""
fetch_weather.py
----------------
Fetches hourly historical weather data for Karachi from Open-Meteo Archive API.

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


def fetch_weather() -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
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
    print("WEATHER REQUEST URL")
    print("=" * 80)
    print(url)

    response = requests.get(url, timeout=90)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    data   = response.json()
    hourly = data.get("hourly")

    if hourly is None:
        raise ValueError(f"'hourly' section missing.\nResponse:\n{data}")

    weather_df = pd.DataFrame({
        "datetime":        hourly["time"],
        "temperature":     hourly["temperature_2m"],
        "humidity":        hourly["relative_humidity_2m"],
        "pressure":        hourly["pressure_msl"],
        "wind_speed":      hourly["wind_speed_10m"],
        "wind_direction":  hourly["wind_direction_10m"],
        "wind_gusts":      hourly["wind_gusts_10m"],
        "precipitation":   hourly["precipitation"],
        "cloud_cover":     hourly["cloud_cover"],
        "dew_point":       hourly["dew_point_2m"],
        "surface_pressure": hourly["surface_pressure"],
    })

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    if weather_df["datetime"].dt.tz is not None:
        weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)

    weather_df = (
        weather_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved : {len(weather_df):,}")
    print(f"Date Range     : {weather_df['datetime'].min()}  →  {weather_df['datetime'].max()}")

    return weather_df


def save_to_mongodb(df: pd.DataFrame):
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    client     = pymongo.MongoClient(mongo_uri)
    collection = client["karachi_aqi"]["raw_weather"]

    df_upload             = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if not records:
        client.close()
        return

    # FIX: was row-by-row update_one() — ~50 min for 33k rows
    operations    = [UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True) for r in records]
    total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

    print(f"Uploading {len(records):,} weather records via bulk_write "
          f"(batch={BULK_BATCH_SIZE})...")
    for i in range(0, len(operations), BULK_BATCH_SIZE):
        batch_num = i // BULK_BATCH_SIZE + 1
        result = collection.bulk_write(operations[i : i + BULK_BATCH_SIZE], ordered=False)
        print(f"  Batch {batch_num}/{total_batches} — "
              f"upserted: {result.upserted_count}, modified: {result.modified_count}")

    client.close()
    print("Weather upload complete!")


if __name__ == "__main__":
    weather_df = fetch_weather()
    save_to_mongodb(weather_df)