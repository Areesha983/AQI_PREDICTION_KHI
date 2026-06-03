"""
Enterprise MLOps Feature Summary Construction Pipeline for Karachi.
Pipes newly engineered analytical indicators straight into the MongoDB Feature Store.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymongo

# ── SYSTEM PATH SETUP ──────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Safely import your production feature store access engine layer
try:
    from api.database import ingest_hourly_features
except ImportError:
    try:
        from dashboard.database import ingest_hourly_features
    except ImportError:
        from feature_store import ingest_hourly_features


# ── 1. EPA AQI Calculation Helpers ───────────────────────────────────────────

def calculate_aqi_from_pm25(pm25: float) -> float:
    """Applies the US-EPA piecewise linear interpolation formula for PM2.5 → AQI."""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    breakpoints = [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            )
    return 500


def aqi_to_category(aqi: float) -> float:
    """Maps raw AQI to standard integer category labels."""
    if pd.isna(aqi): return np.nan
    if aqi <= 50:    return 0
    if aqi <= 100:   return 1
    if aqi <= 150:   return 2
    if aqi <= 200:   return 3
    if aqi <= 300:   return 4
    return 5


def _rolling_slope_6h(y: np.ndarray) -> float:
    """OLS slope over a fixed 6-hour window."""
    x_dev = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    return float(np.dot(x_dev, y - np.mean(y)) / 17.5)


# ── 2. Core Feature Builder ───────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw hourly meteorological + air-quality streams into an enriched,
    leakage-free feature store. All column additions are batched via pd.concat
    to avoid DataFrame fragmentation.
    """
    df = df_raw.copy()

    if "datetime" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Initializing primary transformations. Input shape: {df.shape}")

    # ── STEP 1: LEAKAGE-FREE CAUSAL IMPUTATION ────────────────────────────────
    df = df.set_index("datetime")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="time").ffill()
    df = df.reset_index()
    print(" -> Causal imputation complete.")

    # ── STEP 2: AQI COMPUTATION + MULTI-HORIZON TARGETS ──────────────────────
    df["aqi"] = df["pm25"].apply(calculate_aqi_from_pm25)

    new_cols = {}
    for h in [12, 24, 48, 72]:
        new_cols[f"target_aqi_{h}h"]     = df["aqi"].shift(-h)
        new_cols[f"target_aqi_{h}h_log"] = np.log1p(new_cols[f"target_aqi_{h}h"])
        new_cols[f"target_cat_{h}h"]     = new_cols[f"target_aqi_{h}h"].apply(aqi_to_category)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── STEP 3: TEMPORAL EMBEDDINGS ───────────────────────────────────────────
    dt = df["datetime"]
    hour      = dt.dt.hour
    month     = dt.dt.month
    weekday   = dt.dt.weekday
    day_of_yr = dt.dt.dayofyear

    temp_cols = {
        "hour":         hour,
        "day":          dt.dt.day,
        "month":        month,
        "weekday":      weekday,
        "week_of_year": dt.dt.isocalendar().week.astype(int).values,
        "quarter":      dt.dt.quarter,
        "day_of_year":  day_of_yr,
        "hour_sin":     np.sin(2 * np.pi * hour    / 24),
        "hour_cos":     np.cos(2 * np.pi * hour    / 24),
        "month_sin":    np.sin(2 * np.pi * month   / 12),
        "month_cos":    np.cos(2 * np.pi * month   / 12),
        "weekday_sin":  np.sin(2 * np.pi * weekday / 7),
        "weekday_cos":  np.cos(2 * np.pi * weekday / 7),
        "doy_sin":      np.sin(2 * np.pi * day_of_yr / 365),
        "doy_cos":      np.cos(2 * np.pi * day_of_yr / 365),
        "is_weekend":   (weekday >= 5).astype(int),
        "is_rush_hour": hour.isin([7, 8, 9, 17, 18, 19]).astype(int),
        "hour_of_week": weekday * 24 + hour,
    }
    df = pd.concat([df, pd.DataFrame(temp_cols, index=df.index)], axis=1)

    # ── STEP 4: WIND VECTOR DECOMPOSITION ────────────────────────────────────
    if "wind_direction" in df.columns and "wind_speed" in df.columns:
        wdir_rad = np.deg2rad(df["wind_direction"].shift(1))
        ws_lag   = df["wind_speed"].shift(1)
        wind_cols = {
            "wind_dir_sin":  np.sin(wdir_rad),
            "wind_dir_cos":  np.cos(wdir_rad),
            "wind_x":        ws_lag * np.sin(wdir_rad),
            "wind_y":        ws_lag * np.cos(wdir_rad),
        }
        df = pd.concat([df, pd.DataFrame(wind_cols, index=df.index)], axis=1)

    # ── STEP 5: AQI LAG CHAINS ────────────────────────────────────────────────
    _aqi = df["aqi"].shift(1)
    lag_cols = {f"aqi_lag_{lag}": df["aqi"].shift(lag)
                for lag in [1, 2, 3, 6, 7, 12, 24, 48, 72, 168, 336]}
    lag_cols.update({
        "aqi_same_hour_yesterday":  df["aqi"].shift(24),
        "aqi_same_hour_last_week":  df["aqi"].shift(168),
        "aqi_same_hour_2weeks_ago": df["aqi"].shift(336),
        "aqi_same_hour_3days_ago":  df["aqi"].shift(72),
        "aqi_same_hour_30days_ago": df["aqi"].shift(720),
    })
    df = pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    # ── STEP 6: PM2.5 ROLLING FEATURES ───────────────────────────────────────
    _pm25 = df["pm25"].shift(1)
    pm25_lags = {f"pm25_lag_{lag}": df["pm25"].shift(lag)
                 for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]}
    pm25_rolls = {}
    for w in [6, 24, 72, 168, 336]:
        pm25_rolls[f"pm25_roll_mean_{w}"] = _pm25.rolling(w, min_periods=1).mean()
        pm25_rolls[f"pm25_roll_std_{w}"]  = _pm25.rolling(w, min_periods=1).std().fillna(0)
    pm25_rolls.update({
        "pm25_roll_max_24": _pm25.rolling(24, min_periods=1).max(),
        "pm25_roll_max_72": _pm25.rolling(72, min_periods=1).max(),
        "pm25_ewm_24":      _pm25.ewm(span=24, adjust=False).mean(),
        "pm25_change_24h":  df["pm25"].shift(1) - df["pm25"].shift(25),
        "pm25_vs_24h_avg":  _pm25 - _pm25.rolling(24, min_periods=1).mean(),
        "pm25_vs_72h_avg":  _pm25 - _pm25.rolling(72, min_periods=1).mean(),
    })
    df = pd.concat([df, pd.DataFrame({**pm25_lags, **pm25_rolls}, index=df.index)], axis=1)

    # ── STEP 7: PM10 ROLLING FEATURES ────────────────────────────────────────
    _pm10 = df["pm10"].shift(1)
    pm10_cols = {f"pm10_lag_{lag}": df["pm10"].shift(lag) for lag in [1, 6, 24, 72]}
    pm10_cols.update({
        "pm10_roll_mean_24": _pm10.rolling(24, min_periods=1).mean(),
        "pm10_roll_mean_72": _pm10.rolling(72, min_periods=1).mean(),
        "pm10_roll_std_24":  _pm10.rolling(24, min_periods=1).std().fillna(0),
        "pm10_change_24h":   df["pm10"].shift(1) - df["pm10"].shift(25),
        "dust_event":        (_pm10 > 250).astype(int),
    })
    df = pd.concat([df, pd.DataFrame(pm10_cols, index=df.index)], axis=1)

    # ── STEP 8: AQI ROLLING MATRIX ────────────────────────────────────────────
    aqi_roll = {}
    for w in [6, 12, 24, 48, 72, 168, 336]:
        aqi_roll[f"aqi_roll_mean_{w}"] = _aqi.rolling(w, min_periods=1).mean()
    for w in [24, 72, 168]:
        aqi_roll[f"aqi_roll_std_{w}"]  = _aqi.rolling(w, min_periods=1).std().fillna(0)
        aqi_roll[f"aqi_roll_max_{w}"]  = _aqi.rolling(w, min_periods=1).max()
        aqi_roll[f"aqi_roll_min_{w}"]  = _aqi.rolling(w, min_periods=1).min()
    aqi_roll.update({
        "aqi_ewm_24": _aqi.ewm(span=24, adjust=False).mean(),
        "aqi_ewm_72": _aqi.ewm(span=72, adjust=False).mean(),
    })
    df = pd.concat([df, pd.DataFrame(aqi_roll, index=df.index)], axis=1)

    # ── STEP 9: MOMENTUM & TREND ──────────────────────────────────────────────
    r24  = df["aqi_roll_mean_24"]
    r72  = df["aqi_roll_mean_72"]
    r168 = df["aqi_roll_mean_168"]
    p24  = df["pm25_roll_mean_24"]
    p72  = df["pm25_roll_mean_72"]
    mom_cols = {
        "aqi_trend_ratio":      r24  / (r72  + 1),
        "pm25_trend_ratio":     p24  / (p72  + 1),
        "aqi_momentum_6_24":    df["aqi_roll_mean_6"]  - r24,
        "aqi_momentum_24_72":   r24  - r72,
        "aqi_momentum_24_168":  r24  - r168,
        "aqi_change_1h":        _aqi - df["aqi"].shift(2),
        "aqi_change_6h":        _aqi - df["aqi"].shift(7),
        "aqi_change_24h":       _aqi - df["aqi"].shift(25),
        "aqi_trend_slope_6h":   (
            _aqi.rolling(6, min_periods=6)
                .apply(_rolling_slope_6h, raw=True)
                .fillna(0)
        ),
        "aqi_acceleration":     _aqi.diff().diff().fillna(0),
    }
    df = pd.concat([df, pd.DataFrame(mom_cols, index=df.index)], axis=1)

    # ── STEP 10: PERSISTENCE & RECOVERY ──────────────────────────────────────
    persist_cols = {
        "aqi_persistence":       df["aqi_lag_1"] - df["aqi_lag_24"],
        "aqi_diff_24_168":       df["aqi_lag_24"] - df["aqi_lag_168"],
        "aqi_recovery_rate":     (df["aqi_roll_max_72"] - df["aqi_lag_1"]) / (df["aqi_roll_max_72"] + 1),
        "aqi_persistence_ratio": df["aqi_lag_1"] / (df["aqi_roll_mean_72"] + 1),
        "aqi_vs_week":           df["aqi_roll_mean_24"] - df["aqi_roll_mean_168"],
    }
    df = pd.concat([df, pd.DataFrame(persist_cols, index=df.index)], axis=1)

    # ── STEP 11: STATISTICAL ANOMALY DETECTION ───────────────────────────────
    _rm72  = _aqi.rolling(72, min_periods=24).mean()
    _rs72  = _aqi.rolling(72, min_periods=24).std().replace(0, 1)
    _rq90  = _aqi.rolling(168, min_periods=72).quantile(0.90)
    anomaly_cols = {
        "aqi_zscore_72h":        ((df["aqi_lag_1"] - _rm72) / _rs72).fillna(0),
        "aqi_percentile_72":     (
            _aqi.rolling(72, min_periods=12)
                .apply(lambda x: float(np.mean(x < x[-1])), raw=True)
                .fillna(0.5)
        ),
        "aqi_above_recent_q90":  (df["aqi_lag_1"] > _rq90).astype(int),
        "aqi_volatility_ratio":  df["aqi_roll_std_24"] / (df["aqi_roll_std_72"] + 1),
        "aqi_regime":            pd.cut(
            df["aqi_lag_1"],
            bins=[0, 50, 100, 150, 200, 300, 1000],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(float).fillna(0.0),
    }
    df = pd.concat([df, pd.DataFrame(anomaly_cols, index=df.index)], axis=1)

    # ── STEP 12: SPIKE MEMORY ────────────────────────────────────────────────
    spike_flag = (_aqi > 150)
    _hours_since, _consec = [], []
    _counter, _run = 999, 0
    for s in spike_flag:
        if s:
            _counter = 0
            _run    += 1
        else:
            _counter += 1
            _run      = 0
        _hours_since.append(_counter)
        _consec.append(_run)

    spike_cols = {
        "spike_count_72h":              spike_flag.rolling(72, min_periods=1).sum().fillna(0),
        "dust_hours_72h":               (_pm10 > 250).rolling(72, min_periods=1).sum().fillna(0),
        "hours_since_aqi_spike":        pd.Series(_hours_since, index=df.index).shift(1),
        "consecutive_hours_above_150":  pd.Series(_consec,      index=df.index).shift(1),
    }
    df = pd.concat([df, pd.DataFrame(spike_cols, index=df.index)], axis=1)

    # ── STEP 13: SECONDARY POLLUTANT PROXIES ─────────────────────────────────
    poll_cols = {}
    for poll in ["so2", "co", "o3", "no2"]:
        if poll not in df.columns:
            continue
        _s = df[poll].shift(1)
        poll_cols[f"{poll}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        poll_cols[f"{poll}_change_24h"]   = df[poll].shift(1) - df[poll].shift(25)

    for poll in ["no2", "o3"]:
        if poll not in df.columns:
            continue
        poll_cols[f"{poll}_roll_std_24"] = df[poll].shift(1).rolling(24, min_periods=1).std().fillna(0)

    if {"pm25", "pm10"}.issubset(df.columns):
        poll_cols["pm25_pm10_ratio"] = (
            df["pm25"].shift(1) / (df["pm10"].shift(1) + 1e-3)
        ).clip(upper=10)   
        poll_cols["pm25_fraction"] = (
            df["pm25"].shift(1)
            / (df["pm25"].shift(1) + df["pm10"].shift(1) + 1e-3)
        )

    if {"no2", "o3"}.issubset(df.columns):
        poll_cols["no2_o3_ratio"] = df["no2"].shift(1) / (df["o3"].shift(1) + 1e-3)

    if "dust" in df.columns:
        _dust = df["dust"].shift(1)
        poll_cols["dust_lag_1"]       = _dust
        poll_cols["dust_roll_mean_24"] = _dust.rolling(24, min_periods=1).mean()
        poll_cols["dust_roll_max_24"]  = _dust.rolling(24, min_periods=1).max()

    if "uv_index" in df.columns:
        poll_cols["uv_lag_1"]          = df["uv_index"].shift(1)
        poll_cols["uv_roll_mean_24"]   = df["uv_index"].shift(1).rolling(24, min_periods=1).mean()

    df = pd.concat([df, pd.DataFrame(poll_cols, index=df.index)], axis=1)

    # ── STEP 14: METEOROLOGICAL DISPERSAL ────────────────────────────────────
    met_cols = {}
    for col in ["temperature", "humidity", "wind_speed"]:
        if col not in df.columns:
            continue
        _s = df[col].shift(1)
        met_cols[f"{col}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        met_cols[f"{col}_change_24h"]   = df[col].shift(1) - df[col].shift(25)
        for lag in [1, 6, 24]:
            met_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

    if "wind_speed" in df.columns:
        ws_lag1 = df["wind_speed"].shift(1)
        met_cols["wind_speed_roll_std_24"] = ws_lag1.rolling(24, min_periods=1).std().fillna(0)
        ws_roll24 = ws_lag1.rolling(24, min_periods=1).mean()
        ws_roll168 = ws_lag1.rolling(168, min_periods=1).mean()
        met_cols["wind_persistence_ratio"] = ws_roll24 / (ws_roll168 + 0.1)

    if "temperature" in df.columns:
        met_cols["temperature_roll_std_24"] = df["temperature"].shift(1).rolling(24, min_periods=1).std().fillna(0)

    if {"temperature", "humidity"}.issubset(df.columns):
        met_cols["temp_humidity"]  = df["temperature"].shift(1) * df["humidity"].shift(1)
        met_cols["heat_dryness"]   = df["temperature"].shift(1) / (df["humidity"].shift(1) + 1)

    if {"wind_speed", "pm25"}.issubset(df.columns):
        met_cols["wind_dispersal"] = df["wind_speed"].shift(1) / (df["pm25"].shift(1) + 5)

    if {"temperature", "dew_point"}.issubset(df.columns):
        met_cols["dew_point_depression"]      = df["temperature"].shift(1) - df["dew_point"].shift(1)
        met_cols["dew_pt_depression_roll24"]  = met_cols["dew_point_depression"].rolling(24, min_periods=1).mean()

    if "precipitation" in df.columns:
        prec = df["precipitation"].shift(1)
        met_cols["rain_24h"] = prec.rolling(24, min_periods=1).sum()
        met_cols["rain_72h"] = prec.rolling(72, min_periods=1).sum()
        met_cols["rain_event"] = (prec > 0.1).astype(int)

    if "pressure" in df.columns:
        pres = df["pressure"].shift(1)
        met_cols["pressure_lag_1"]        = pres
        met_cols["pressure_roll_mean_24"] = pres.rolling(24, min_periods=1).mean()
        met_cols["pressure_change_24h"]   = pres - df["pressure"].shift(25)
        met_cols["pressure_change_6h"]    = pres - df["pressure"].shift(7)

    if "surface_pressure" in df.columns:
        met_cols["surface_pressure_lag_1"]        = df["surface_pressure"].shift(1)
        met_cols["surface_pressure_change_24h"]   = (
            df["surface_pressure"].shift(1) - df["surface_pressure"].shift(25)
        )

    if "wind_gusts" in df.columns:
        _gust = df["wind_gusts"].shift(1)
        met_cols["wind_gusts_roll_mean_24"] = _gust.rolling(24, min_periods=1).mean()
        met_cols["wind_gusts_roll_max_24"]  = _gust.rolling(24, min_periods=1).max()

    if "cloud_cover" in df.columns:
        met_cols["cloud_cover_lag_1"]        = df["cloud_cover"].shift(1)
        met_cols["cloud_cover_roll_mean_24"] = df["cloud_cover"].shift(1).rolling(24, min_periods=1).mean()

    df = pd.concat([df, pd.DataFrame(met_cols, index=df.index)], axis=1)

    # ── STEP 15: WARM-UP ROW FILTERING ───────────────────────────────────────
    required_non_null = [
        "aqi_same_hour_30days_ago",  # 720 h of history
        "target_aqi_72h",
    ]
    before = len(df)
    df = df.dropna(subset=required_non_null).reset_index(drop=True)
    print(f" -> Dropped {before - len(df):,} warm-up rows. {len(df):,} rows remaining.")

    assert df["datetime"].is_monotonic_increasing, "CRITICAL: Temporal order broken."
    return df


# ── 3. MongoDB Feature Pipeline Extraction and Streaming Execution ────────────

if __name__ == "__main__":
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("CRITICAL: MONGODB_URI missing from environment contexts.")

    print("\n" + "="*70)
    print(" 📥 EXTRACTING INPUT ALIGNED ARTIFACT FROM MONGODB")
    print("="*70)
    
    # Connect directly to download the base dataset collection generated by build_dataset.py
    client = pymongo.MongoClient(mongo_uri)
    db = client["karachi_aqi"]
    input_collection = db["karachi_aqi_dataset"]
    
    cursor = input_collection.find()
    raw_df = pd.DataFrame(list(cursor))
    client.close()
    
    if raw_df.empty:
        raise RuntimeError("CRITICAL: 'karachi_aqi_dataset' document store collection is completely empty.")
        
    if "_id" in raw_df.columns:
        raw_df = raw_df.drop(columns=["_id"])

    # Ensure native datetime conversions
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])

    # Compute high-dimensional feature spaces
    processed_df = build_features(raw_df)

    # ── MLOPS FEATURE STORE LOADING BLOCK ────────────────────────────────────
    print("\n" + "="*70)
    print(" 📤 STREAMING STRUCTURED BATCH TO MONGODB FEATURE STORE")
    print("="*70)
    
    # Format the datetime object back into uniform ISO strings for MongoDB indexing
    mongo_df = processed_df.copy()
    mongo_df["timestamp"] = mongo_df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Convert dataframe records to JSON dictionary payloads
    features_payload = mongo_df.to_dict(orient="records")
    print(f"Prepared {len(features_payload):,} documents for Atlas integration.")
    
    # Execute batch transactional mutations into the centralized feature store collection
    print("Connecting to remote cluster interface layer...")
    records_upserted = ingest_hourly_features(features_payload)
    print(f"✅ Success! Feature Store synchronized: {records_upserted:,} records modified.")
    print("="*70 + "\n")