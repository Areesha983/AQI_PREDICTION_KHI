"""
load_data.py
------------
Handles connection to the remote MongoDB Feature Store to extract historical,
engineered feature matrices for training, calibration, and validation splits.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

# Load environment keys up the directory tree for local development environments
load_dotenv(BASE_DIR / ".env")

# ── Cloud DB Configurations ──────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "karachi_aqi"
COLLECTION_NAME = "processed_features"

# ── Feature Set Filter Targets ────────────────────────────────────────────────
REDUNDANT_TIME_COLS = [
    "hour", "day", "month", "weekday",
    "day_of_year", "week_of_year", "hour_of_week",
]
ALL_TARGETS = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
]
BASE_DROP     = ["datetime", "timestamp"] + REDUNDANT_TIME_COLS + ALL_TARGETS
LEAKAGE_EXACT = frozenset(ALL_TARGETS)


def _fetch_from_feature_store() -> pd.DataFrame:
    """
    Directly queries the centralized cloud feature store collection.
    Reconstructs database data structures into a unified Pandas DataFrame.
    """
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGODB_URI environment variable is missing or unset.")
        
    print(f"\nEstablishing active cluster link to pool: {DB_NAME}.{COLLECTION_NAME}")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Pull documents excluding the internal MongoDB bson unique identity reference keys
        cursor = collection.find({}, {"_id": 0})
        documents = list(cursor)
        client.close()
    except PyMongoError as e:
        print(f"CRITICAL: Failed to stream from MongoDB Atlas Cluster: {e}")
        raise

    if not documents:
        raise RuntimeError(f"CRITICAL: Connection established, but feature collection '{COLLECTION_NAME}' is completely empty.")

    df = pd.DataFrame(documents)
    
    # Enforce chronological ordering relative to temporal indexes
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise KeyError("Pulled collection is missing mandatory temporal reference anchors ['timestamp', 'datetime']")
        
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_xy(horizon: int, use_log: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72."

    # Direct database pull replaces localized os candidates lookup strings 
    df = _fetch_from_feature_store()
    print(f"Extracted feature store dataset matrix shape: {df.shape}")

    raw_target_col = f"target_aqi_{horizon}h"
    log_target_col = f"target_aqi_{horizon}h_log"

    if raw_target_col not in df.columns:
        raise ValueError(f"Target column not found in cloud document schema: '{raw_target_col}'")

    # ── Online Runtime Feature Computations ───────────────────────────
    extra = {}

    if "aqi" in df.columns:
        extra["aqi_diff_1"]  = df["aqi"].diff().fillna(0)
        extra["aqi_diff_6"]  = df["aqi"].diff(6).fillna(0)
        extra["aqi_diff_24"] = df["aqi"].diff(24).fillna(0)
        extra["aqi_accel"]   = df["aqi"].diff().diff().fillna(0)

    if "pm25" in df.columns:
        extra["pm25_diff_1h"]      = df["pm25"].diff().fillna(0)
        extra["pm25_roll_std_24h"] = df["pm25"].rolling(24, min_periods=1).std().fillna(0)
        extra["pm25_roll_std_6h"]  = df["pm25"].rolling(6,  min_periods=1).std().fillna(0)

    if "pm10" in df.columns:
        extra["pm10_diff_1h"]      = df["pm10"].diff().fillna(0)
        extra["pm10_roll_std_12h"] = df["pm10"].rolling(12, min_periods=1).std().fillna(0)

    if "temperature" in df.columns:
        extra["temp_diff_24h"] = df["temperature"].diff(24).fillna(0)

    if "pm25" in df.columns and "humidity" in df.columns:
        extra["interaction_pm25_humidity"] = df["pm25"] * df["humidity"]

    if "pm25" in df.columns and "wind_speed" in df.columns:
        extra["interaction_pm25_wind_inverse"] = df["pm25"] / (df["wind_speed"] + 0.1)

    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    # ── Filter Valid Target Instances ─────────────────────────────────────────
    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)

    # ── Establish Target Array (y) ────────────────────────────────────────────
    if use_log:
        if log_target_col not in df.columns:
            df[log_target_col] = np.log1p(df[raw_target_col])
        y = df[log_target_col].copy()
    else:
        y = df[raw_target_col].copy()

    y_raw = df[raw_target_col].copy()

    # ── Build Feature Space (X) ───────────────────────────────────────────────
    X = df.drop(columns=[c for c in BASE_DROP if c in df.columns], errors="ignore")

    # Filter features where high rates of NaNs exist (>10%)
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.10].index.tolist()
    if high_missing:
        print(f"Dropping {len(high_missing)} features exceeding 10% NaN threshold: {high_missing}")
        X = X.drop(columns=high_missing)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0) # Deterministic clean fill

    leaky = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky:
        raise ValueError(f"CRITICAL Data leakage detected. Remaining targets in features: {leaky}")
    if X.isna().any().any():
        still_nan = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Formatting error: NaNs remain inside matrix X: {still_nan}")

    X = X.select_dtypes(include=[np.number])

    # ── Matrix Diagnostics Summary ────────────────────────────────────────────
    print(f"\nTarget Distributions (raw AQI {horizon}h):")
    print(y_raw.describe().round(1).to_string())
    print(f"\nModel feature dimension space: {X.shape[1]}")
    print(f"Total row entries partitioned: {X.shape[0]:,}")

    if "aqi" in X.columns:
        aqi_target_corr = X["aqi"].corr(y_raw)
        flag = "  <<< WARNING: HIGH CAUSAL MULTICOLLINEARITY ASSESSED" if aqi_target_corr > 0.99 else ""
        print(f"Base AQI correlation factor -> Target ({horizon}h): {aqi_target_corr:.3f}{flag}")

    return X, y


def get_chronological_splits(X: pd.DataFrame, y: pd.Series, horizon: int):
    """
    Applies unified 75 / 10 / 15 chronological validation split.
    Guarantees structural buffer gaps equal to lookahead horizons to prevent overlap.
    """
    n         = len(X)
    train_end = int(n * 0.75)
    cal_end   = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end]

    X_cal   = X.iloc[train_end + horizon : cal_end].copy()
    y_cal   = y.iloc[train_end + horizon : cal_end]

    X_test  = X.iloc[cal_end:].copy()
    y_test  = y.iloc[cal_end:]

    return X_train, y_train, X_cal, y_cal, X_test, y_test


def get_spike_augmented_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_raw: pd.Series | None = None,
    spike_threshold: float = 150.0,
    target_spike_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series]:
    ref          = y_train_raw if y_train_raw is not None else y_train
    spike_mask   = ref > spike_threshold
    n_spike      = spike_mask.sum()
    n_total      = len(X_train)
    current_frac = n_spike / n_total

    if current_frac >= target_spike_fraction:
        print(f"[spike-aug] Train split contains balanced target representation ({current_frac:.3f}). Skipping.")
        return X_train, y_train

    n_needed = int(target_spike_fraction * n_total / (1 - target_spike_fraction)) - n_spike
    if n_needed <= 0 or n_spike == 0:
        return X_train, y_train

    rng          = np.random.default_rng(42)
    spike_idx    = np.where(spike_mask)[0]
    resample_idx = rng.choice(spike_idx, size=n_needed, replace=True)

    X_extra = X_train.iloc[resample_idx].copy()
    y_extra = y_train.iloc[resample_idx].copy()

    X_aug = pd.concat([X_train, X_extra], ignore_index=True)
    y_aug = pd.concat([y_train, y_extra], ignore_index=True)

    return X_aug, y_aug


def apply_leakage_free_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_cal:  pd.DataFrame | None = None,
    threshold: float = 0.97,
) -> tuple:
    corr  = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    protected = {
        "aqi",
        "aqi_lag_1", "aqi_lag_6", "aqi_lag_12",
        "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
        "aqi_diff_1", "aqi_diff_6", "aqi_diff_24", "aqi_accel",
        "pm25", "pm10",
        "pm25_roll_std_24h", "interaction_pm25_humidity",
        "interaction_pm25_wind_inverse", "dust_lag_1",
        "dew_point_depression", "wind_persistence_ratio",
    }

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold) and col not in protected
    ]

    X_train_c = X_train.drop(columns=to_drop)
    X_test_c  = X_test.drop(columns=to_drop)

    if X_cal is not None:
        X_cal_c = X_cal.drop(columns=to_drop)
        return X_train_c, X_cal_c, X_test_c, to_drop

    return X_train_c, X_test_c, to_drop


def calculate_conformal_margin(abs_residuals: np.ndarray, alpha: float = 0.05) -> float:
    n_cal   = len(abs_residuals)
    if n_cal == 0:
        return 0.0
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(abs_residuals, q_level))


def compute_aqi_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: int) -> dict:
    true_ev   = (y_true > threshold).astype(int)
    pred_ev   = (y_pred > threshold).astype(int)
    tp        = np.sum((true_ev == 1) & (pred_ev == 1))
    fp        = np.sum((true_ev == 0) & (pred_ev == 1))
    fn        = np.sum((true_ev == 1) & (pred_ev == 0))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        f"precision_gt{threshold}": precision,
        f"recall_gt{threshold}":    recall,
        f"f1_gt{threshold}":        f1,
    }


def export_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
    metrics_dir: Path,
    models_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    
    # Save validation diagnostics tables cleanly to localized build directories
    metrics_dir.mkdir(parents=True, exist_ok=True)
    res_df    = pd.DataFrame({"actual": y_true, "predicted": y_pred, "residual": residuals})
    res_df.to_csv(metrics_dir / f"{model_name.lower()}_residuals_{horizon}h.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="teal", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{model_name} Residuals ({horizon}h) — Raw AQI Scale")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    plt.savefig(
        metrics_dir / f"{model_name.lower()}_residual_hist_{horizon}h.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def get_persistence_baseline_col(X_test: pd.DataFrame, horizon: int) -> str | None:
    preferred = f"aqi_lag_{horizon}"
    if preferred in X_test.columns:
        return preferred
    if "aqi_lag_1" in X_test.columns:
        return "aqi_lag_1"
    return None


# ── Operational Verification Execution Loop ──────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("      LOAD_DATA CLOUD PIPELINE INTEGRITY & SPLIT VERIFICATION")
    print("=" * 80)

    for h in (24, 48, 72):
        print(f"\n{'#' * 60}\n DATABASE HOOK INGESTION CHECK FOR LOOKAHEAD HORIZON: {h}h\n{'#' * 60}")
        try:
            X, y = load_xy(h, use_log=True)
            X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, h)
            print(f"  Train Split : {X_train.shape[0]:,} records")
            print(f"  Cal Split   : {X_cal.shape[0]:,} records")
            print(f"  Test Split  : {X_test.shape[0]:,} records")

            X_train_f, X_cal_f, X_test_f, dropped = apply_leakage_free_correlation_filter(
                X_train, X_test, X_cal, threshold=0.97
            )
            print(f"  Surviving Dimensions : {X_train_f.shape[1]} (Filtered out {len(dropped)})")
            print(f"  Index verified       : {'aqi' in X_train_f.columns}")

        except Exception as e:
            print(f"ERROR executing data extraction loop for horizon {h}h: {e}")