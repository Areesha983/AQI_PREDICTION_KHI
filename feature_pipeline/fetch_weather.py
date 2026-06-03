"""
fetch_weather.py
----------------
Fetches hourly historical weather data for Karachi from Open-Meteo Archive API.

Key fix: requests wind_speed_10m in km/h (Open-Meteo default) — no unit mismatch.
Adds `cloud_cover` explicitly so it is always present in the merged dataset.
"""

import requests
import pandas as pd
from pathlib import Path

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
        "dew_point_2m,"           # ← NEW: dew point for humidity-based stability proxy
        "surface_pressure"        # ← NEW: surface pressure (better than MSL for dispersion)
        "&wind_speed_unit=kmh"    # explicit — matches feature-engineering expectations
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

    # Remove tz info if present (Open-Meteo with timezone=auto may return offset strings)
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
    print("\nWeather Summary")
    print(weather_df.describe().round(2))

    return weather_df


if __name__ == "__main__":
    weather_df = fetch_weather()

    output_path = (
        Path(__file__).resolve().parent
        / "data" / "raw" / "weather_karachi.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")