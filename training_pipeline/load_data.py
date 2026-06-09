"""
load_data.py  (STREAMLINED METRICS ENGINE)
------------
Location: training_pipeline/load_data.py

Production MLOps Data Loader Module. Bypasses and decouples feature calculation 
dependencies by treating MongoDB as a read-only, fully pre-computed Feature Store.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Directory anchor mappings
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name == "training_pipeline" else SCRIPT_DIR

load_dotenv(BASE_DIR / ".env")

MONGO_URI       = os.getenv("MONGODB_URI")
DB_NAME         = "karachi_aqi"
COLLECTION_NAME = "processed_features"

# ── Global Cache Objects ─────────────────────────────────────────────────────
_RAW_DF_CACHE: dict = {}      # Shared single-fetch memory frame
_DATA_CACHE: dict   = {}      # Sliced horizon arrays: (X, y_log, y_raw)
_CORR_FILTER_CACHE: dict = {}


def impute_for_linear(X_train: pd.DataFrame, X_cal: pd.DataFrame | None, X_test: pd.DataFrame) -> tuple:
    """Column-wise mean imputation fitted strictly on X_train for Ridge Regression."""
    train_means = X_train.mean()
    X_train_imp = X_train.fillna(train_means)
    X_test_imp  = X_test.fillna(train_means)
    if X_cal is not None:
        X_cal_imp = X_cal.fillna(train_means)
        return X_train_imp, X_cal_imp, X_test_imp
    return X_train_imp, X_test_imp


# Constants for operational tracking and target partitioning
REDUNDANT_TIME_COLS = ["hour", "day", "month", "weekday", "day_of_year", "week_of_year", "hour_of_week"]
ALL_TARGETS = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
    "target_aqi_12h_deviation", "target_aqi_24h_deviation", "target_aqi_48h_deviation", "target_aqi_72h_deviation"
]
CURRENT_TIMESTEP_COLS = [
    # These are raw current-hour sensor readings — genuine leakage if included.
    # NOTE: aqi_historical_anchor is a 168h *lagged* rolling median — NOT leakage.
    #       It was incorrectly placed here; it's now kept in the feature matrix.
    "aqi", "pm25", "pm10", "co", "no2", "so2", "o3",
    "dust", "uv_index", "temperature", "temperature_2m", "humidity",
    "relative_humidity_2m", "wind_speed", "wind_speed_10m", "wind_direction",
    "wind_direction_10m", "wind_gusts", "precipitation", "cloud_cover",
    "dew_point", "dew_point_2m", "pressure", "surface_pressure", "aqi_same_weekday_hour_2w"
]

BASE_DROP     = ["datetime", "timestamp"] + REDUNDANT_TIME_COLS + ALL_TARGETS + CURRENT_TIMESTEP_COLS
LEAKAGE_EXACT = frozenset(ALL_TARGETS + CURRENT_TIMESTEP_COLS)


def _fetch_from_feature_store() -> pd.DataFrame:
    """Streams documents from remote MongoDB cluster exactly once.

    Uses a projection to exclude columns that _build_X_y always drops:
      - CURRENT_TIMESTEP_COLS (leakage columns, excluded from X)
      - REDUNDANT_TIME_COLS (dropped in BASE_DROP)
    Target columns are kept so _build_X_y can construct y vectors.
    Temporal anchors (timestamp, datetime) are kept for chronological sorting.
    This cuts the network payload by ~60-70% on large collections, preventing
    socket timeouts on GitHub Actions runners with big processed_features collections.
    """
    global _RAW_DF_CACHE

    if "master_df" in _RAW_DF_CACHE:
        return _RAW_DF_CACHE["master_df"]

    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGODB_URI environment variable is missing or unset.")

    # Exclude leakage + redundant columns — safe to skip since _build_X_y drops
    # them anyway. Targets are NOT excluded (needed to build y vectors).
    _EXCLUDE_COLS = set(CURRENT_TIMESTEP_COLS) | set(REDUNDANT_TIME_COLS)
    projection = {"_id": 0}
    for col in _EXCLUDE_COLS:
        projection[col] = 0

    # Date filter: only fetch last 2 years of data.
    # ~33k rows exist since 2022 — fetching all is too slow on Atlas M0.
    # 2 years (~17,500 rows) is sufficient — data only goes back to Aug 2022 anyway.
    from datetime import datetime, timedelta
    cutoff_dt = datetime.utcnow() - timedelta(days=730)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    query = {"timestamp": {"$gte": cutoff_str}}

    print(f"Connecting to feature warehouse: {DB_NAME}.{COLLECTION_NAME}", flush=True)
    print(f"  Projection excludes {len(_EXCLUDE_COLS)} leakage/redundant columns to reduce payload.", flush=True)
    print(f"  Date filter: fetching rows >= {cutoff_str} (~2yr window, ~17k rows).", flush=True)
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            socketTimeoutMS=120000,
            tlsCAFile=certifi.where(),
        )
        db = client[DB_NAME]

        # batch_size(500) prevents a single oversized read that exceeds the
        # 120s socket timeout on large collections (>10k docs).
        cursor = db[COLLECTION_NAME].find(query, projection).batch_size(500)
        documents = list(cursor)
        client.close()
    except PyMongoError as e:
        print(f"CRITICAL network error connecting to MongoDB Atlas: {e}", flush=True)
        raise

    if not documents:
        raise RuntimeError(f"Feature store collection '{COLLECTION_NAME}' is currently empty.")

    print(f"  --> Successfully loaded {len(documents):,} rows. Mapping schema...", flush=True)
    df = pd.DataFrame(documents)

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise KeyError("Pulled collection is missing mandatory temporal reference anchors.")

    df = df.sort_values("datetime").reset_index(drop=True)
    
    _RAW_DF_CACHE["master_df"] = df
    return df


def _build_X_y(df: pd.DataFrame, horizon: int):
    """Slices pre-engineered historical feature matrices purely in RAM."""
    raw_target_col = f"target_aqi_{horizon}h"
    log_target_col = f"target_aqi_{horizon}h_log"

    if raw_target_col not in df.columns:
        raise ValueError(f"Target vector not found in database columns: '{raw_target_col}'")

    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)

    if log_target_col not in df.columns:
        df[log_target_col] = np.log1p(df[raw_target_col])

    y_log = df[log_target_col].copy()
    y_raw = df[raw_target_col].copy()

    X = df.drop(columns=[c for c in BASE_DROP if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.20].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)

    leaky = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky:
        raise ValueError(f"CRITICAL: Data leakage detected. Forbidden columns present: {leaky}")

    return X, y_log, y_raw


def load_xy_both(horizon: int) -> tuple:
    """
    Returns (X, y_log, y_raw).

    Resolution order:
      1. In-process RAM cache (_DATA_CACHE) — free if same process.
      2. Parquet snapshot written by run_training_pipeline.py Phase 0.
      3. Full MongoDB fetch (fallback, also populates RAM cache).
    """
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72."

    if horizon in _DATA_CACHE:
        return _DATA_CACHE[horizon]

    # ── Parquet cache intercept (written by Phase 0 across separate steps) ──
    PARQUET_DIR  = SCRIPT_DIR / "_parquet_cache"
    pq_X        = PARQUET_DIR / f"X_{horizon}h.parquet"
    pq_y_log    = PARQUET_DIR / f"y_log_{horizon}h.parquet"
    pq_y_raw    = PARQUET_DIR / f"y_raw_{horizon}h.parquet"

    if pq_X.exists() and pq_y_log.exists() and pq_y_raw.exists():
        print(f"  [Phase-0 cache] Loading Parquet snapshot for {horizon}h.")
        X     = pd.read_parquet(pq_X)
        y_log = pd.read_parquet(pq_y_log)["y_log"]
        y_raw = pd.read_parquet(pq_y_raw)["y_raw"]
        result = (X, y_log, y_raw)
        _DATA_CACHE[horizon] = result
        return result

    # ── Full MongoDB fetch ───────────────────────────────────────────────────
    df = _fetch_from_feature_store()
    result = _build_X_y(df, horizon)
    _DATA_CACHE[horizon] = result
    return result


def load_xy(horizon: int, use_log: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    X, y_log, y_raw = load_xy_both(horizon)
    y = y_log if use_log else y_raw
    return X, y


def get_chronological_splits(X: pd.DataFrame, y: pd.Series, horizon: int):
    """Chronological 75/10/15 data split partitions."""
    gap = min(horizon, 24)
    n         = len(X)
    train_end = int(n * 0.75)
    cal_end   = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end]

    X_cal   = X.iloc[train_end + gap : cal_end].copy()
    y_cal   = y.iloc[train_end + gap : cal_end]

    X_test  = X.iloc[cal_end + gap :].copy()
    y_test  = y.iloc[cal_end + gap :]

    return X_train, y_train, X_cal, y_cal, X_test, y_test


def get_spike_augmented_train(X_train: pd.DataFrame, y_train: pd.Series, y_train_raw: pd.Series = None, spike_threshold: float = 150.0, target_spike_fraction: float = 0.05) -> tuple[pd.DataFrame, pd.Series]:
    ref = y_train_raw if y_train_raw is not None else y_train
    spike_mask = ref > spike_threshold
    n_spike = spike_mask.sum()
    n_total = len(X_train)
    
    if (n_spike / n_total) >= target_spike_fraction or n_spike == 0:
        return X_train, y_train

    n_needed = int(target_spike_fraction * n_total / (1 - target_spike_fraction)) - n_spike
    if n_needed <= 0:
        return X_train, y_train

    rng = np.random.default_rng(42)
    spike_idx = np.where(spike_mask)[0]
    resample_idx = rng.choice(spike_idx, size=n_needed, replace=True)

    X_aug = pd.concat([X_train, X_train.iloc[resample_idx]], ignore_index=True)
    y_aug = pd.concat([y_train, y_train.iloc[resample_idx]], ignore_index=True)
    return X_aug, y_aug


def apply_leakage_free_correlation_filter(X_train: pd.DataFrame, X_test: pd.DataFrame, X_cal: pd.DataFrame = None, threshold: float = 0.97, horizon: int = 0) -> tuple:
    cache_key = (horizon, threshold, tuple(sorted(X_train.columns)))
    if cache_key in _CORR_FILTER_CACHE:
        to_drop = _CORR_FILTER_CACHE[cache_key]
    else:
        protected = {
            "aqi_lag_1", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
            "aqi_change_1h", "aqi_change_6h", "aqi_change_24h", "aqi_acceleration", "pm25_lag_1",
            "pm25_roll_mean_24", "pm25_roll_std_24", "pm10_lag_1", "interaction_pm25_humidity"
        }
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        train_var = X_train.var()

        to_drop = set()
        for col in upper.columns:
            if col in to_drop or col in protected:
                continue
            correlated_with = upper.index[upper[col] > threshold].tolist()
            for other in correlated_with:
                if other in to_drop or other in protected:
                    continue
                if train_var.get(col, 0) < train_var.get(other, 0):
                    to_drop.add(col)
                    break
                else:
                    to_drop.add(other)
        to_drop = list(to_drop)
        _CORR_FILTER_CACHE[cache_key] = to_drop

    X_train_c = X_train.drop(columns=to_drop)
    X_test_c  = X_test.drop(columns=to_drop)
    if X_cal is not None:
        return X_train_c, X_cal.drop(columns=to_drop), X_test_c, to_drop
    return X_train_c, X_test_c, to_drop


def calculate_conformal_margin(abs_residuals: np.ndarray, alpha: float = 0.05) -> float:
    n_cal = len(abs_residuals)
    if n_cal == 0: return 0.0
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(abs_residuals, q_level))


def compute_aqi_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: int) -> dict:
    true_ev = (y_true > threshold).astype(int)
    pred_ev = (y_pred > threshold).astype(int)
    tp = np.sum((true_ev == 1) & (pred_ev == 1))
    fp = np.sum((true_ev == 0) & (pred_ev == 1))
    fn = np.sum((true_ev == 1) & (pred_ev == 0))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {f"precision_gt{threshold}": p, f"recall_gt{threshold}": r, f"f1_gt{threshold}": 2*p*r/(p+r) if (p+r)>0 else 0.0}


def export_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, model_name: str, metrics_dir: Path, models_dir: Path) -> None:
    residuals = y_true - y_pred
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": y_true, "predicted": y_pred, "residual": residuals}).to_csv(
        metrics_dir / f"{model_name.lower()}_residuals_{horizon}h.csv", index=False
    )


def get_persistence_baseline_col(X_test: pd.DataFrame, horizon: int) -> str | None:
    if f"aqi_lag_{horizon}" in X_test.columns: return f"aqi_lag_{horizon}"
    return "aqi_lag_1" if "aqi_lag_1" in X_test.columns else None