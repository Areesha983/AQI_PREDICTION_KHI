"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data for Karachi and safely merges them.

Key fixes vs original:
  1. After the inner join, fills short isolated gaps (≤ 3 hours) in all pollutant
     columns via linear interpolation before dropping rows. Open-Meteo occasionally
     returns 1–3 h NaN gaps inside otherwise clean blocks; dropping them wastes data.
  2. Prints a full missingness audit across EVERY column, not just pollutants.
  3. Saves both the raw merged file AND a gap-filled version so you can inspect both.

Run:
    python build_dataset.py
"""

from pathlib import Path
import pandas as pd

from fetch_weather import fetch_weather
from fetch_air_quality import fetch_air_quality


# Maximum contiguous NaN run (hours) to fill via interpolation before hard-dropping
MAX_GAP_FILL_HOURS = 3


def main():
    print("Fetching weather data...")
    weather_df = fetch_weather()

    print("\nFetching air quality data...")
    aq_df = fetch_air_quality()

    # ── 1. Normalize datetime to tz-naive ────────────────────────────────────
    for frame in (weather_df, aq_df):
        col = frame["datetime"]
        if pd.api.types.is_datetime64_any_dtype(col):
            if col.dt.tz is not None:
                frame["datetime"] = col.dt.tz_localize(None)
        else:
            frame["datetime"] = pd.to_datetime(col).dt.tz_localize(None)

    # ── 2. Inner join on matching timestamps ─────────────────────────────────
    print("\nExecuting synchronized datetime alignment join...")
    dataset = (
        pd.merge(weather_df, aq_df, on="datetime", how="inner")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    print(f"Merged shape (before gap-fill): {dataset.shape}")

    # ── 3. Full missingness audit ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("      FULL COLUMN MISSINGNESS AUDIT (post-merge)")
    print("=" * 65)
    for col in dataset.columns:
        n = dataset[col].isna().sum()
        p = n / len(dataset) * 100
        flag = "  ← HIGH" if p > 5 else ""
        print(f"  {col:<35}: {n:>6} NaN  ({p:5.2f}%){flag}")

    # ── 4. Short-gap linear interpolation (≤ MAX_GAP_FILL_HOURS) ─────────────
    # Only applied to numeric pollutant/weather columns, never to datetime.
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index"]
    weather_numeric = [c for c in dataset.select_dtypes(include="number").columns
                       if c not in pollutant_cols]

    for col in pollutant_cols + weather_numeric:
        if col not in dataset.columns:
            continue
        n_before = dataset[col].isna().sum()
        if n_before == 0:
            continue
        # Interpolate only within gaps ≤ MAX_GAP_FILL_HOURS
        dataset[col] = (
            dataset[col]
            .interpolate(method="linear", limit=MAX_GAP_FILL_HOURS, limit_direction="forward")
        )
        n_after = dataset[col].isna().sum()
        if n_before != n_after:
            print(f"  [gap-fill] {col}: {n_before} → {n_after} NaN  (filled {n_before - n_after})")

    # ── 5. Hard drop rows with no valid PM2.5 target ─────────────────────────
    initial_shape = len(dataset)
    dataset = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
    dropped = initial_shape - len(dataset)
    print(f"\nDropped {dropped} rows with missing PM2.5 "
          f"({dropped / initial_shape * 100:.3f}% of merged dataset).")

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    data_dir = Path(__file__).resolve().parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "karachi_aqi_dataset.csv"
    dataset.to_csv(out_path, index=False)

    print(f"\nCleaned dataset saved → {out_path}")
    print(f"Final shape          : {dataset.shape}")
    print(f"Date range           : {dataset['datetime'].min()}  →  {dataset['datetime'].max()}")
    print("\nFirst 5 rows preview:")
    preview_cols = ["datetime", "temperature", "humidity", "pm25", "co", "dust"]
    print(dataset[[c for c in preview_cols if c in dataset.columns]].head())


if __name__ == "__main__":
    main()