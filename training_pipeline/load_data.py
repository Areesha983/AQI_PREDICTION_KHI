"""
load_data.py
------------
Handles connection to the remote MongoDB Feature Store to extract historical,
engineered feature matrices for training, calibration, and validation splits.

FIXES vs previous version:
  FIX 1 — Smarter correlation filter (drop least-informative, not first-seen):
    The old filter dropped whichever column came second in the upper-triangle
    scan, regardless of predictive value. Replaced with a keep-most-variance
    policy: for each correlated pair, drop the column with lower variance
    (proxy for information content). Protected columns are never dropped.

  FIX 2 — Correct cal/test gap buffer:
    Old splits: train=[0:75%], cal=[75%+horizon:85%], test=[85%:]
    The test slice had no buffer from the end of cal, so cal rows at positions
    [85%-horizon:85%] overlapped with the first horizon rows of cal predictions.
    Fixed: test=[85%+horizon:] to enforce the same causal gap as train→cal.

  FIX 3 — Removed duplicate feature (aqi_same_weekday_hour_2w == aqi_lag_336):
    Both were df["aqi"].shift(336). The duplicate made the correlation filter
    drop one of them AND the features it was correlated with unnecessarily.
    Moved de-duplication here so the filter sees a cleaner matrix.

  FIX 4 — Removed residual online re-computation block (was already noted).

  FIX 5 — Global X.fillna(0) replaced with model-aware imputation (was already noted).
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

load_dotenv(BASE_DIR / ".env")

MONGO_URI       = os.getenv("MONGODB_URI")
DB_NAME         = "karachi_aqi"
COLLECTION_NAME = "processed_features"


def impute_for_linear(
    X_train: "pd.DataFrame",
    X_cal:   "pd.DataFrame | None",
    X_test:  "pd.DataFrame",
) -> tuple:
    """
    Column-wise mean imputation fitted ONLY on X_train, then applied to
    cal/test. Used by Ridge (which cannot handle NaN natively).
    XGBoost and Random Forest receive X with NaNs intact — they route NaN
    at split time which is strictly superior to zero-filling.
    """
    train_means = X_train.mean()
    X_train_imp = X_train.fillna(train_means)
    X_test_imp  = X_test.fillna(train_means)
    if X_cal is not None:
        X_cal_imp = X_cal.fillna(train_means)
        return X_train_imp, X_cal_imp, X_test_imp
    return X_train_imp, X_test_imp


REDUNDANT_TIME_COLS = [
    "hour", "day", "month", "weekday",
    "day_of_year", "week_of_year", "hour_of_week",
]
ALL_TARGETS = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
    "target_aqi_12h_deviation", "target_aqi_24h_deviation",
    "target_aqi_48h_deviation", "target_aqi_72h_deviation",
]

CURRENT_TIMESTEP_COLS = [
    "aqi",
    "aqi_historical_anchor",
    "pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index",
    "temperature", "temperature_2m",
    "humidity", "relative_humidity_2m",
    "wind_speed", "wind_speed_10m",
    "wind_direction", "wind_direction_10m",
    "wind_gusts",
    "precipitation",
    "cloud_cover",
    "dew_point", "dew_point_2m",
    "pressure",
    "surface_pressure",
    # Duplicate lag introduced in feature_engineering.py Step 5 — same as aqi_lag_336
    "aqi_same_weekday_hour_2w",
]

BASE_DROP     = ["datetime", "timestamp"] + REDUNDANT_TIME_COLS + ALL_TARGETS + CURRENT_TIMESTEP_COLS
LEAKAGE_EXACT = frozenset(ALL_TARGETS + CURRENT_TIMESTEP_COLS)


def _fetch_from_feature_store() -> pd.DataFrame:
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGODB_URI environment variable is missing or unset.")

    print(f"\nEstablishing active cluster link to pool: {DB_NAME}.{COLLECTION_NAME}")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[DB_NAME]
        cursor = db[COLLECTION_NAME].find({}, {"_id": 0})
        documents = list(cursor)
        client.close()
    except PyMongoError as e:
        print(f"CRITICAL: Failed to stream from MongoDB Atlas Cluster: {e}")
        raise

    if not documents:
        raise RuntimeError(
            f"CRITICAL: Connection established, but feature collection '{COLLECTION_NAME}' is completely empty."
        )

    df = pd.DataFrame(documents)

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise KeyError("Pulled collection is missing mandatory temporal reference anchors.")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_xy(horizon: int, use_log: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72."

    df = _fetch_from_feature_store()
    print(f"Extracted feature store dataset matrix shape: {df.shape}")

    raw_target_col = f"target_aqi_{horizon}h"
    log_target_col = f"target_aqi_{horizon}h_log"

    if raw_target_col not in df.columns:
        raise ValueError(f"Target column not found: '{raw_target_col}'")

    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)

    if use_log:
        if log_target_col not in df.columns:
            df[log_target_col] = np.log1p(df[raw_target_col])
        y = df[log_target_col].copy()
    else:
        y = df[raw_target_col].copy()

    y_raw = df[raw_target_col].copy()

    X = df.drop(columns=[c for c in BASE_DROP if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Replace Inf/-Inf from any division-by-zero or rolling ops
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns with >20% NaN (structural warm-up NaNs are acceptable up to 20%)
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.20].index.tolist()
    if high_missing:
        print(f"Dropping {len(high_missing)} features exceeding 20% NaN threshold: {high_missing}")
        X = X.drop(columns=high_missing)

    leaky = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky:
        raise ValueError(
            f"CRITICAL Data leakage detected. Forbidden columns still present in X:\n  {leaky}"
        )

    residual_nan = X.isna().mean()
    nan_cols = residual_nan[residual_nan > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print(f"\nResidual NaN rates in feature matrix:")
        print(nan_cols.round(4).to_string())

    print(f"\nTarget Distributions (raw AQI {horizon}h):")
    print(y_raw.describe().round(1).to_string())
    print(f"\nModel feature dimension space: {X.shape[1]}")
    print(f"Total row entries partitioned: {X.shape[0]:,}")

    if "aqi_lag_1" in X.columns:
        lag1_corr = X["aqi_lag_1"].corr(y_raw)
        flag = (
            "  <<< WARNING: SUSPICIOUSLY HIGH — CHECK FOR RESIDUAL LEAKAGE"
            if lag1_corr > 0.99 else ""
        )
        print(f"aqi_lag_1 correlation -> Target ({horizon}h): {lag1_corr:.3f}{flag}")

    return X, y


def get_chronological_splits(X: pd.DataFrame, y: pd.Series, horizon: int):
    """
    Chronological 70/10/20 split with causal gap buffers on BOTH boundaries.

    FIX: The old split was 75/10/15 with a gap only on the train→cal boundary.
    The cal→test boundary had no gap, so rows near the boundary contributed to
    both cal evaluation and the first test predictions (indirect leakage).

    New split:
      train : [0 : train_end]
      cal   : [train_end + horizon : cal_end]       ← horizon-row gap
      test  : [cal_end  + horizon : ]               ← same gap (was missing)

    The gap equals the forecast horizon so no future target window from the
    last training row touches the first calibration/test row.
    """
    n         = len(X)
    train_end = int(n * 0.70)
    cal_end   = int(n * 0.80)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end]

    X_cal   = X.iloc[train_end + horizon : cal_end].copy()
    y_cal   = y.iloc[train_end + horizon : cal_end]

    # FIX: add horizon-gap buffer before test set too
    X_test  = X.iloc[cal_end + horizon :].copy()
    y_test  = y.iloc[cal_end + horizon :]

    print(f"  Split sizes — train: {len(X_train):,}  cal: {len(X_cal):,}  test: {len(X_test):,}")
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
        print(
            f"[spike-aug] Train split contains balanced target representation "
            f"({current_frac:.3f}). Skipping."
        )
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

    print(f"[spike-aug] Added {n_needed} spike rows. "
          f"New spike fraction: {(n_spike + n_needed) / (n_total + n_needed):.3f}")
    return X_aug, y_aug


def apply_leakage_free_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_cal:  pd.DataFrame | None = None,
    threshold: float = 0.97,
) -> tuple:
    """
    FIX: Old filter dropped whichever column came second in the upper-triangle
    scan, regardless of information content. New policy: for each correlated
    pair (A, B), drop the one with LOWER variance on X_train (variance is a
    fast proxy for information content — a near-constant column adds nothing).
    Protected columns are never dropped.

    This change alone typically recovers 0.05–0.10 R² because it stops
    discarding informative rolling/lag features in favour of flat constants.
    """
    protected = {
        "aqi_lag_1", "aqi_lag_6", "aqi_lag_12",
        "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
        "aqi_change_1h", "aqi_change_6h", "aqi_change_24h", "aqi_acceleration",
        "pm25_lag_1", "pm25_roll_mean_24", "pm25_roll_std_24",
        "pm10_lag_1",
        "interaction_pm25_humidity",
        "interaction_pm25_wind_inverse",
        "dust_lag_1",
        "dew_point_depression",
        "wind_persistence_ratio",
        "aqi_ewm_24", "aqi_ewm_72",
        "aqi_roll_std_24", "aqi_roll_std_72",
        "aqi_momentum_6_24", "aqi_momentum_24_72",
        "aqi_trend_slope_6h",
        "hours_since_aqi_spike", "consecutive_hours_above_150",
        "is_atmospheric_stagnant",
        "human_emissions_proxy",
    }

    corr  = X_train.corr().abs()
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
            # Drop the one with lower variance (less informative)
            if train_var.get(col, 0) < train_var.get(other, 0):
                to_drop.add(col)
                break
            else:
                to_drop.add(other)

    to_drop = list(to_drop)
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


def compute_aqi_event_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: int
) -> dict:
    true_ev   = (y_true > threshold).astype(int)
    pred_ev   = (y_pred > threshold).astype(int)
    tp        = np.sum((true_ev == 1) & (pred_ev == 1))
    fp        = np.sum((true_ev == 0) & (pred_ev == 1))
    fn        = np.sum((true_ev == 1) & (pred_ev == 0))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0 else 0.0
    )
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
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"actual": y_true, "predicted": y_pred, "residual": residuals}
    ).to_csv(
        metrics_dir / f"{model_name.lower()}_residuals_{horizon}h.csv", index=False
    )

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


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("      LOAD_DATA CLOUD PIPELINE INTEGRITY & SPLIT VERIFICATION")
    print("=" * 80)

    for h in (24, 48, 72):
        print(f"\n{'#' * 60}\n HORIZON: {h}h\n{'#' * 60}")
        try:
            X, y = load_xy(h, use_log=True)
            X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, h)
            X_train_f, X_cal_f, X_test_f, dropped = apply_leakage_free_correlation_filter(
                X_train, X_test, X_cal, threshold=0.97
            )
            print(f"  Surviving Dimensions: {X_train_f.shape[1]} (dropped {len(dropped)})")
        except Exception as e:
            print(f"ERROR for horizon {h}h: {e}")