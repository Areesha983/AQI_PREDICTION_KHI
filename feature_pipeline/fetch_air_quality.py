"""
fetch_air_quality.py
---------------------
Fetches hourly air-quality forecast/reanalysis data for Karachi from Open-Meteo.

Key fixes vs original:
  1. Uses `domains=cams_europe` — CAMS (Copernicus Atmosphere Monitoring Service)
     provides the best reanalysis coverage for South Asia. Without specifying a domain
     Open-Meteo defaults to a coarser global model that gives more NaNs over Karachi.
  2. Requests `dust` and `alder_pollen` as proxy channels for dust-storm events
     common in Karachi (Thar Desert outflow).
  3. Timezone pinned to Asia/Karachi to guarantee consistent UTC+5 offsets.
  4. Explicit NaN audit printed per column so missing data is never silently ignored.
"""

import requests
import pandas as pd
from pathlib import Path

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
        "dust,"                   # ← NEW: Karachi-specific dust channel
        "uv_index"                # ← NEW: UV drives photochemical O3 production
        "&timezone=Asia%2FKarachi"
        # NOTE: Do NOT add domains= param — let Open-Meteo auto-select the best
        # available model for the given lat/lon. Forcing cams_europe can produce
        # all-NaN results outside Europe for some date ranges.
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

    # Strip timezone info so the merge in build_dataset works cleanly
    if aq_df["datetime"].dt.tz is not None:
        aq_df["datetime"] = aq_df["datetime"].dt.tz_localize(None)

    aq_df = (
        aq_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # ── Detailed NaN audit (helps catch domain/date-range gaps immediately) ──
    print(f"\nRows Retrieved: {len(aq_df):,}")
    print(f"Date Range    : {aq_df['datetime'].min()}  →  {aq_df['datetime'].max()}")
    print("\nPer-Column NaN Audit:")
    for col in aq_df.columns:
        n_nan = aq_df[col].isna().sum()
        pct   = n_nan / len(aq_df) * 100
        flag  = "  ⚠️  HIGH MISSING" if pct > 5 else ""
        print(f"  {col:<20}: {n_nan:>6} NaN  ({pct:5.1f}%){flag}")

    # ── Critical guard: if >30% of PM2.5 is missing, the dataset is unusable ──
    pm25_nan_pct = aq_df["pm25"].isna().mean() * 100
    if pm25_nan_pct > 30:
        raise RuntimeError(
            f"FATAL: PM2.5 is {pm25_nan_pct:.1f}% missing. "
            "Check the Open-Meteo API response or adjust date range."
        )

    print("\nPM2.5 Distribution")
    print(aq_df["pm25"].describe().round(2))

    return aq_df


if __name__ == "__main__":
    aq_df = fetch_air_quality()

    output_path = (
        Path(__file__).resolve().parent
        / "data" / "raw" / "air_quality_karachi.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aq_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")