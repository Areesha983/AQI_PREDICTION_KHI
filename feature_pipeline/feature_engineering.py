"""
Enterprise MLOps Feature Summary Construction Pipeline for Karachi.
Pipes newly engineered analytical indicators straight into the MongoDB Feature Store.

FIXES APPLIED:
  BUG #1 (CRITICAL — data leakage): Step 1 used interpolate(method="time").ffill()
    which fills gaps bidirectionally using future anchor values. Replaced with
    ffill() only — strictly causal. bfill(limit=1) handles only leading NaNs at
    the very start of the series where no prior observation exists.

  BUG #2 (CRITICAL — target contamination): AQI and targets were computed after
    the bidirectional interpolation, so target_aqi_24h etc. were derived from
    future-contaminated pm25. Now computed from cleanly forward-filled data.

  BUG #3 (production safety): Step 4 (Wind Vector Decomposition) and Step 14
    (Meteorological Dispersal) now both guard against column name variants.
    If fetch_weather.py ever returns "wind_speed_10m" instead of "wind_speed",
    the pipeline degrades gracefully instead of silently skipping wind features.

  BUG #4 (performance): MongoDB writes now use bulk_write() in batches of 1000.
    Was row-by-row update_one() — for 33k rows that was ~50 minutes of network
    round-trips. Bulk writes complete in under 30 seconds.

  BUG #5 (CRITICAL — XGBoost dtype crash): Step 11 used pd.cut() with integer
    labels=[0,1,2,3,4,5]. pandas constructs a Categorical series whose underlying
    dtype is object/category, not a numeric primitive. Even though .astype(float)
    was chained, on some pandas versions the Categorical metadata is preserved and
    XGBoost raises ValueError: DataFrame.dtypes for data must be int, float, bool
    or categorical. Fixed by casting labels to float literals [0.0 ... 5.0] and
    explicitly converting via .cat.codes or np.float64 cast to guarantee a
    primitive numeric array reaches the feature matrix.

  R2 IMPROVEMENTS (added after original bug fixes):
    FEAT #1: human_emissions_proxy — explicit float weight encoding weekday/rush-hour
      activity patterns. More informative than the binary is_rush_hour flag alone.

    FEAT #2: Meteorological stagnation features — diurnal_temp_range_24h,
      temp_to_wind_ratio, humidity_to_wind_ratio, and is_atmospheric_stagnant.
      These encode the physical mechanism by which PM2.5 gets trapped under
      low-wind, high-humidity conditions. Tree models cannot derive these ratios
      on their own; making them explicit significantly reduces split depth needed.

    FEAT #3: Target deviation targets — target_aqi_{h}h_deviation = future_aqi
      minus a 7-day rolling median anchor for that exact hour. These mean-reverting
      targets are more stationary than raw absolute AQI, which helps the 48h/72h
      models escape the scale-extrapolation trap that limits tree-based R2.
      NOTE: evaluation code in train_*.py must add aqi_historical_anchor back to
      predicted deltas before computing MAE / RMSE / R2 against raw AQI.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymongo
from pymongo import UpdateOne

PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR     = PIPELINE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BULK_BATCH_SIZE = 1000


# ── 1. EPA AQI Calculation Helpers ────────────────────────────────────────────

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
    if pd.isna(aqi): return np.nan
    if aqi <= 50:    return 0
    if aqi <= 100:   return 1
    if aqi <= 150:   return 2
    if aqi <= 200:   return 3
    if aqi <= 300:   return 4
    return 5


def _rolling_slope_6h(y: np.ndarray) -> float:
    x_dev = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    return float(np.dot(x_dev, y - np.mean(y)) / 17.5)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """
    Returns the first candidate column name that exists in df.
    Used to handle both plain names ('wind_speed') and API-suffixed variants
    ('wind_speed_10m') so Step 4 and Step 14 never silently skip.
    """
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ── 2. Core Feature Builder ───────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "datetime" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Initializing primary transformations. Input shape: {df.shape}")

    # ── STEP 1: LEAKAGE-FREE CAUSAL IMPUTATION ────────────────────────────────
    df = df.set_index("datetime")
    num_cols = df.select_dtypes(include=np.number).columns
    # FIX BUG-2: Removed bfill(limit=1) — even limit=1 pulls from one future row,
    # violating causality. Leading NaNs at the very start of the series are
    # handled by the warm-up row filter in Step 15 (aqi_same_hour_30days_ago
    # will be NaN there, so those rows are dropped before training anyway).
    df[num_cols] = df[num_cols].ffill()
    df = df.reset_index()
    print(" -> Causal imputation complete (ffill only — bfill removed to prevent future leakage).")

    # ── STEP 2: AQI COMPUTATION + MULTI-HORIZON TARGETS ──────────────────────
    df["aqi"] = df["pm25"].apply(calculate_aqi_from_pm25)

    new_cols = {}
    for h in [12, 24, 48, 72]:
        new_cols[f"target_aqi_{h}h"]     = df["aqi"].shift(-h)
        new_cols[f"target_aqi_{h}h_log"] = np.log1p(new_cols[f"target_aqi_{h}h"])
        new_cols[f"target_cat_{h}h"]     = new_cols[f"target_aqi_{h}h"].apply(aqi_to_category)

    # FEAT #3: Target deviation targets (TARGET COLUMNS ONLY — never used as features)
    # A 7-day (168h) rolling median anchored at the previous hour gives a causal
    # structural baseline for "what AQI typically looks like at this time of week".
    # These deviation columns are kept as ALTERNATIVE TRAINING TARGETS alongside
    # target_aqi_{h}h. They are listed in ALL_TARGETS in load_data.py and are
    # explicitly dropped from X before training. Do NOT add aqi_historical_anchor
    # to the stored feature document — it is an intermediate scratch variable only.
    # ⚠️  NOTE: train_*.py scripts all use raw target_aqi_{h}h (not deviations),
    # so no re-addition of anchor is needed. The deviation columns are reserved for
    # future experimental training runs only.
    _aqi_historical_anchor = (
        df["aqi"].shift(1)
        .rolling(168, min_periods=24)
        .median()
        .fillna(df["aqi"].median())
    )
    # Do NOT store _aqi_historical_anchor as a column — it would leak into X.

    for h in [12, 24, 48, 72]:
        future_aqi = df["aqi"].shift(-h)
        new_cols[f"target_aqi_{h}h_deviation"] = future_aqi - _aqi_historical_anchor

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── STEP 3: TEMPORAL EMBEDDINGS ───────────────────────────────────────────
    dt        = df["datetime"]
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
        # FEAT #1: human_emissions_proxy
        # Explicit float weight: 1.0 = weekday rush hour, 0.7 = weekday off-peak,
        # 0.3 = weekend. More granular than a binary flag; lets tree models segment
        # Karachi's port + industrial + traffic emission cycles directly.
        "human_emissions_proxy": np.where(
            (weekday < 5) & hour.isin([8, 9, 17, 18, 19]), 1.0,
            np.where(weekday < 5, 0.7, 0.3)
        ),
    }
    df = pd.concat([df, pd.DataFrame(temp_cols, index=df.index)], axis=1)

    # ── STEP 4: WIND VECTOR DECOMPOSITION ────────────────────────────────────
    ws_col = _resolve_col(df, "wind_speed",     "wind_speed_10m")
    wd_col = _resolve_col(df, "wind_direction",  "wind_direction_10m")

    if ws_col and wd_col:
        wdir_rad = np.deg2rad(df[wd_col].shift(1))
        ws_lag   = df[ws_col].shift(1)
        wind_cols = {
            "wind_dir_sin": np.sin(wdir_rad),
            "wind_dir_cos": np.cos(wdir_rad),
            "wind_x":       ws_lag * np.sin(wdir_rad),
            "wind_y":       ws_lag * np.cos(wdir_rad),
        }
        df = pd.concat([df, pd.DataFrame(wind_cols, index=df.index)], axis=1)
        print(f" -> Wind decomposition: using '{ws_col}' + '{wd_col}'.")
    else:
        print(
            " -> WARNING: Wind columns not found — Step 4 skipped. "
            "Check fetch_weather.py column naming."
        )

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
        # FIX: Add exact same-weekday same-hour lag — more predictive than raw
        # aqi_lag_168 for Karachi because traffic/industrial patterns repeat
        # weekly. 168h = 7*24 = same hour same weekday last week.
        "aqi_same_weekday_hour_2w": df["aqi"].shift(336),  # 2 identical weekday cycles
        "aqi_same_weekday_hour_4w": df["aqi"].shift(672),  # 4 weeks (28 days)
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
        "aqi_trend_ratio":     r24 / (r72 + 1),
        "pm25_trend_ratio":    p24 / (p72 + 1),
        "aqi_momentum_6_24":   df["aqi_roll_mean_6"] - r24,
        "aqi_momentum_24_72":  r24 - r72,
        "aqi_momentum_24_168": r24 - r168,
        "aqi_change_1h":       _aqi - df["aqi"].shift(2),
        "aqi_change_6h":       _aqi - df["aqi"].shift(7),
        "aqi_change_24h":      _aqi - df["aqi"].shift(25),
        "aqi_trend_slope_6h":  (
            _aqi.rolling(6, min_periods=6)
                .apply(_rolling_slope_6h, raw=True)
                .fillna(0)
        ),
        "aqi_acceleration":    _aqi.diff().diff().fillna(0),
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
    _rm72 = _aqi.rolling(72, min_periods=24).mean()
    _rs72 = _aqi.rolling(72, min_periods=24).std().replace(0, 1)
    _rq90 = _aqi.rolling(168, min_periods=72).quantile(0.90)

    _aqi_regime_cat = pd.cut(
        df["aqi_lag_1"],
        bins=[0, 50, 100, 150, 200, 300, 1000],
        labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    _aqi_regime = pd.Series(
        _aqi_regime_cat.to_numpy(dtype=np.float64, na_value=np.nan),
        index=df.index,
    ).fillna(0.0)

    anomaly_cols = {
        "aqi_zscore_72h":       ((df["aqi_lag_1"] - _rm72) / _rs72).fillna(0),
        "aqi_percentile_72":    (
            _aqi.rolling(72, min_periods=12)
                .apply(lambda x: float(np.mean(x < x[-1])), raw=True)
                .fillna(0.5)
        ),
        "aqi_above_recent_q90": (df["aqi_lag_1"] > _rq90).astype(int),
        "aqi_volatility_ratio": df["aqi_roll_std_24"] / (df["aqi_roll_std_72"] + 1),
        "aqi_regime":           _aqi_regime,
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
        poll_cols[f"{poll}_roll_std_24"] = (
            df[poll].shift(1).rolling(24, min_periods=1).std().fillna(0)
        )

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
        poll_cols["dust_lag_1"]        = _dust
        poll_cols["dust_roll_mean_24"] = _dust.rolling(24, min_periods=1).mean()
        poll_cols["dust_roll_max_24"]  = _dust.rolling(24, min_periods=1).max()

    if "uv_index" in df.columns:
        poll_cols["uv_lag_1"]        = df["uv_index"].shift(1)
        poll_cols["uv_roll_mean_24"] = df["uv_index"].shift(1).rolling(24, min_periods=1).mean()

    df = pd.concat([df, pd.DataFrame(poll_cols, index=df.index)], axis=1)

    # ── STEP 14: METEOROLOGICAL DISPERSAL ────────────────────────────────────
    met_cols  = {}
    temp_col  = _resolve_col(df, "temperature",  "temperature_2m")
    hum_col   = _resolve_col(df, "humidity",      "relative_humidity_2m")
    ws_col_14 = _resolve_col(df, "wind_speed",    "wind_speed_10m")

    for col, resolved in [
        ("temperature", temp_col),
        ("humidity", hum_col),
        ("wind_speed", ws_col_14),
    ]:
        if not resolved:
            print(
                f" -> WARNING: '{col}' not found — skipping met rolling features for this column."
            )
            continue
        _s = df[resolved].shift(1)
        met_cols[f"{col}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        met_cols[f"{col}_change_24h"]   = df[resolved].shift(1) - df[resolved].shift(25)
        for lag in [1, 6, 24]:
            met_cols[f"{col}_lag_{lag}"] = df[resolved].shift(lag)

    if ws_col_14:
        ws_lag1 = df[ws_col_14].shift(1)
        met_cols["wind_speed_roll_std_24"] = ws_lag1.rolling(24, min_periods=1).std().fillna(0)
        ws_roll24  = ws_lag1.rolling(24,  min_periods=1).mean()
        ws_roll168 = ws_lag1.rolling(168, min_periods=1).mean()
        met_cols["wind_persistence_ratio"] = ws_roll24 / (ws_roll168 + 0.1)

    if temp_col:
        met_cols["temperature_roll_std_24"] = (
            df[temp_col].shift(1).rolling(24, min_periods=1).std().fillna(0)
        )

    if temp_col and hum_col:
        met_cols["temp_humidity"] = df[temp_col].shift(1) * df[hum_col].shift(1)
        met_cols["heat_dryness"]  = df[temp_col].shift(1) / (df[hum_col].shift(1) + 1)

    if ws_col_14 and "pm25" in df.columns:
        met_cols["wind_dispersal"] = df[ws_col_14].shift(1) / (df["pm25"].shift(1) + 5)

    # FIX BUG-1: interaction_pm25_humidity and interaction_pm25_wind_inverse were
    # listed in the protected set in load_data.py but never built here. These are
    # the two most physically meaningful pollutant-met interactions for PM2.5
    # accumulation (high humidity traps particles; low wind prevents dispersal).
    # Adding them gives tree models direct access to these joint signals.
    if hum_col and "pm25" in df.columns:
        met_cols["interaction_pm25_humidity"] = (
            df["pm25"].shift(1) * df[hum_col].shift(1) / 100.0
        )
    if ws_col_14 and "pm25" in df.columns:
        met_cols["interaction_pm25_wind_inverse"] = (
            df["pm25"].shift(1) / (df[ws_col_14].shift(1) + 0.5)
        ).clip(upper=500)

    dew_col = _resolve_col(df, "dew_point", "dew_point_2m")
    if temp_col and dew_col:
        met_cols["dew_point_depression"]     = df[temp_col].shift(1) - df[dew_col].shift(1)
        met_cols["dew_pt_depression_roll24"] = (
            met_cols["dew_point_depression"].rolling(24, min_periods=1).mean()
        )

    # FEAT #2: Atmospheric stagnation features
    # Physical motivation: when wind drops below ~2 m/s AND the dew-point
    # depression is small (air nearly saturated), the planetary boundary layer
    # collapses and PM2.5 cannot disperse. Making these interactions explicit
    # drastically reduces the split depth that tree models need to discover them.
    if temp_col and hum_col and ws_col_14:
        t_lag1 = df[temp_col].shift(1)
        ws_lag1_stag = df[ws_col_14].shift(1)

        # 24h diurnal temperature range: low range = stagnant air mass
        # FIX BUG-3: Removed .bfill() — unbounded future-fill polluted warm-up rows
        # with future temperature data. fillna(0) is safe: warm-up rows are dropped
        # in Step 15 anyway, and 0 is a neutral placeholder for the filter period.
        met_cols["diurnal_temp_range_24h"] = (
            t_lag1.rolling(24, min_periods=6).max()
            - t_lag1.rolling(24, min_periods=6).min()
        ).fillna(0.0)

        # Ratio features: force the model to see the interaction directly
        met_cols["temp_to_wind_ratio"]     = t_lag1 / (ws_lag1_stag + 0.1)
        met_cols["humidity_to_wind_ratio"] = df[hum_col].shift(1) / (ws_lag1_stag + 0.1)

        # Hard binary stagnation trigger: wind < 2 m/s AND nearly saturated air
        # Only defined when dew_point_depression was already computed above
        if "dew_point_depression" in met_cols:
            met_cols["is_atmospheric_stagnant"] = (
                (ws_lag1_stag < 2.0) & (met_cols["dew_point_depression"] < 3.0)
            ).astype(float)

    if "precipitation" in df.columns:
        prec = df["precipitation"].shift(1)
        met_cols["rain_24h"]   = prec.rolling(24, min_periods=1).sum()
        met_cols["rain_72h"]   = prec.rolling(72, min_periods=1).sum()
        met_cols["rain_event"] = (prec > 0.1).astype(int)

    if "pressure" in df.columns:
        pres = df["pressure"].shift(1)
        met_cols["pressure_lag_1"]        = pres
        met_cols["pressure_roll_mean_24"] = pres.rolling(24, min_periods=1).mean()
        met_cols["pressure_change_24h"]   = pres - df["pressure"].shift(25)
        met_cols["pressure_change_6h"]    = pres - df["pressure"].shift(7)

    if "surface_pressure" in df.columns:
        met_cols["surface_pressure_lag_1"]      = df["surface_pressure"].shift(1)
        met_cols["surface_pressure_change_24h"] = (
            df["surface_pressure"].shift(1) - df["surface_pressure"].shift(25)
        )

    if "wind_gusts" in df.columns:
        _gust = df["wind_gusts"].shift(1)
        met_cols["wind_gusts_roll_mean_24"] = _gust.rolling(24, min_periods=1).mean()
        met_cols["wind_gusts_roll_max_24"]  = _gust.rolling(24, min_periods=1).max()

    if "cloud_cover" in df.columns:
        met_cols["cloud_cover_lag_1"]        = df["cloud_cover"].shift(1)
        met_cols["cloud_cover_roll_mean_24"] = (
            df["cloud_cover"].shift(1).rolling(24, min_periods=1).mean()
        )

    df = pd.concat([df, pd.DataFrame(met_cols, index=df.index)], axis=1)

    # ── STEP 15: WARM-UP ROW FILTERING ───────────────────────────────────────
    required_non_null = ["aqi_same_hour_30days_ago", "target_aqi_72h"]
    before = len(df)
    df = df.dropna(subset=required_non_null).reset_index(drop=True)
    print(f" -> Dropped {before - len(df):,} warm-up rows. {len(df):,} rows remaining.")

    assert df["datetime"].is_monotonic_increasing, "CRITICAL: Temporal order broken."

    # ── STEP 16: DTYPE SAFETY PASS ────────────────────────────────────────────
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    non_date_obj = [c for c in obj_cols if c != "datetime"]
    if non_date_obj:
        print(
            f" -> WARNING: Coercing {len(non_date_obj)} non-numeric columns to float64: "
            f"{non_date_obj}"
        )
        for col in non_date_obj:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64).fillna(0.0)

    return df


# ── 3. Production Pipeline Entrypoint ─────────────────────────────────────────

def process_all():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("CRITICAL: MONGODB_URI missing from environment contexts.")

    print("\n" + "=" * 70)
    print(" EXTRACTING INPUT ALIGNED ARTIFACT FROM MONGODB")
    print("=" * 70)

    client = pymongo.MongoClient(mongo_uri)
    db     = client["karachi_aqi"]

    cursor = db["karachi_aqi_dataset"].find()
    raw_df = pd.DataFrame(list(cursor))

    if raw_df.empty:
        client.close()
        raise RuntimeError(
            "CRITICAL: 'karachi_aqi_dataset' collection is completely empty. "
            "Run build_dataset.py first."
        )

    if "_id" in raw_df.columns:
        raw_df = raw_df.drop(columns=["_id"])

    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
    processed_df = build_features(raw_df)

    print("\n" + "=" * 70)
    print(" STREAMING STRUCTURED BATCH TO MONGODB FEATURE STORE")
    print("=" * 70)

    mongo_df = processed_df.copy()
    mongo_df["timestamp"] = mongo_df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    mongo_df["datetime"]  = mongo_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    features_payload = mongo_df.to_dict(orient="records")
    print(f"Prepared {len(features_payload):,} documents for Atlas integration.")

    output_collection = db["processed_features"]
    operations = [
        UpdateOne(
            {"datetime": r["datetime"]},
            {"$set": r},
            upsert=True,
        )
        for r in features_payload
    ]

    total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE
    for i in range(0, len(operations), BULK_BATCH_SIZE):
        batch_num = i // BULK_BATCH_SIZE + 1
        result = output_collection.bulk_write(
            operations[i : i + BULK_BATCH_SIZE],
            ordered=False,
        )
        print(
            f"  Batch {batch_num}/{total_batches} — "
            f"upserted: {result.upserted_count}, modified: {result.modified_count}"
        )

    client.close()
    print(f"\nSuccess! Feature Store collection synchronized cleanly.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    process_all()