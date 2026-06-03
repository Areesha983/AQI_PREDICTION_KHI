"""
fetch_weather.py
----------------
Fetches hourly historical weather data for Karachi from Open-Meteo Archive API.
Saves data directly into MongoDB to act as a cloud-native feature store.
"""

import os
import requests
import pymongo
import pandas as pd
from config import LATITUDE, LONGITUDE, START_DATE, END_DATE


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
        "dew_point_2m,"           # dew point for humidity-based stability proxy
        "surface_pressure"        # surface pressure (better than MSL for dispersion)
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

    data = response.json()
    hourly = data.get("hourly")

    if hourly is None:
        raise ValueError(f"'hourly' section missing.\nResponse:\n{data}")

    weather_df = pd.DataFrame({
        "datetime":         hourly["time"],
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
        raise ValueError("MONGODB_URI environment variable is missing from the environment!")

    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]
    collection = db["raw_weather"]

    df_upload = df.copy()
    df_upload["datetime"] = df_upload["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df_upload.to_dict(orient="records")

    if records:
        print(f"Uploading {len(records)} weather records to MongoDB...")
        for record in records:
            collection.update_one(
                {"datetime": record["datetime"]},
                {"$set": record},
                upsert=True
            )
    client.close()
    print("Weather upload complete!")


if __name__ == "__main__":
    weather_df = fetch_weather()
    save_to_mongodb(weather_df)