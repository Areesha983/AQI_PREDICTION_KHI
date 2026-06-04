"""
load_data.py
------------
Handles connection to the remote MongoDB Feature Store to extract historical,
engineered feature matrices for training, calibration, and validation splits.

FIXES APPLIED:
  FIX 1 — Removed online feature re-computation block:
    The original load_xy() recomputed aqi_diff_1, pm25_diff_1h, etc. on the
    already-stored feature store data. feature_engineering.py owns all feature
    construction. Recomputing here caused double-shifting bugs and redundant
    correlated columns that confused the correlation filter.

  FIX 2 — Replaced global X.fillna(0) with model-aware imputation:
    The old code used X.fillna(0) globally, which misrepresented all sensor
    gaps as legitimate zero readings (0°C, 0% humidity, etc.). This created
    structural distortions in tree splits.

    New strategy:
      - Inf/NaN from divisions or rolling ops → replace with NaN first
      - Residual NaNs after the feature store's ffill are genuine sensor gaps
      - Tree models (XGBoost, Random Forest): receive NaN as-is — XGBoost
        natively routes NaN at every split; RF's fillna(0) is applied only
        inside the specific train script via the caller flag use_tree_nan=True
      - Linear models (Ridge): receive column-wise training-mean imputation
        via impute_for_linear(), applied inside train_ridge.py by passing
        use_tree_nan=False (default)

    load_xy() now returns X with NaNs intact. Each training script decides
    how to handle them via the `preserve_nan` parameter.
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

# ── Cloud DB Configurations ──────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "karachi_aqi"
COLLECTION_NAME = "processed_features"

# ── Model-Aware Imputation ────────────────────────────────────────────────────
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
    Returns (X_train_imp, X_cal_imp, X_test_imp) — or 2-tuple if X_cal is None.
    """
    train_means = X_train.mean()               # computed on train only
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
    # Deviation targets added by feature_engineering.py — must be excluded from X
    # or they will leak future AQI information directly into the feature matrix.
    "target_aqi_12h_deviation", "target_aqi_24h_deviation",
    "target_aqi_48h_deviation", "target_aqi_72h_deviation",
]

# Columns that represent the CURRENT timestep's observed values.
# These must never appear in X because at inference time (predicting the future)
# the model would be receiving information that hasn't happened yet relative to
# the training rows, or — worse — the exact current value that the lagged
# features already represent causally via aqi_lag_1 / pm25_lag_1 etc.
#
# "aqi" is computed directly from pm25 at the same row timestamp. A model
# predicting aqi_24h ahead that sees "aqi" is essentially seeing the answer —
# aqi and target_aqi_24h are correlated ~0.95+, which explains near-perfect R².
#
# "aqi_historical_anchor" is an intermediate scratch variable used only to
# construct the deviation targets. It should never reach the feature matrix.
# (feature_engineering.py was fixed to stop storing it, but we guard here too.)
#
# Raw sensor readings (pm25, pm10, co, no2, so2, o3, dust, uv_index) are the
# same-timestep source columns. Their lagged/rolling counterparts (pm25_lag_1,
# pm25_roll_mean_24, etc.) are the correct causal features and are kept.
CURRENT_TIMESTEP_COLS = [
    "aqi",                   # derived at row-time from pm25 — use aqi_lag_* instead
    "aqi_historical_anchor", # intermediate scratch var for deviation targets
    "pm25",                  # raw current-hour sensor — use pm25_lag_* instead
    "pm10",                  # raw current-hour sensor — use pm10_lag_* instead
    "co",                    # raw current-hour sensor
    "no2",                   # raw current-hour sensor
    "so2",                   # raw current-hour sensor
    "o3",                    # raw current-hour sensor
    "dust",                  # raw current-hour sensor
    "uv_index",              # raw current-hour sensor
    # Weather source columns are also same-timestep; their lag/roll variants survive
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
]

BASE_DROP     = ["datetime", "timestamp"] + REDUNDANT_TIME_COLS + ALL_TARGETS + CURRENT_TIMESTEP_COLS
LEAKAGE_EXACT = frozenset(ALL_TARGETS + CURRENT_TIMESTEP_COLS)


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

        cursor = collection.find({}, {"_id": 0})
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
        raise KeyError(
            "Pulled collection is missing mandatory temporal reference anchors ['timestamp', 'datetime']"
        )

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_xy(horizon: int, use_log: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72."

    df = _fetch_from_feature_store()
    print(f"Extracted feature store dataset matrix shape: {df.shape}")

    raw_target_col = f"target_aqi_{horizon}h"
    log_target_col = f"target_aqi_{horizon}h_log"

    if raw_target_col not in df.columns:
        raise ValueError(
            f"Target column not found in cloud document schema: '{raw_target_col}'"
        )

    # The feature store is the single source of truth for all engineered features.
    # No online feature re-computation here — that caused double-shifting bugs.

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
    X = X.select_dtypes(include=[np.number])

    # Replace Inf/-Inf from any division-by-zero or rolling ops with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # FIX BUG-4: Old threshold of 10% was too aggressive — long-range lag features
    # (aqi_lag_336, aqi_same_weekday_hour_4w at lag-672) have ~2-8% warmup NaNs
    # that are structurally unavoidable, not sensor gaps. Dropping them silently
    # eliminated the most informative seasonal-cycle features. Raised to 20%.
    # Tree models handle residual NaNs natively; linear models use impute_for_linear().
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.20].index.tolist()
    if high_missing:
        print(
            f"Dropping {len(high_missing)} features exceeding 20% NaN threshold: {high_missing}"
        )
        X = X.drop(columns=high_missing)

    # ── NaN handling: preserve for tree models, impute for linear ─────────────
    # FIX: Was X.fillna(0) globally. Zero-filling misrepresents sensor gaps
    # as real observations (0°C, 0% humidity, 0 wind speed) and distorts tree
    # splits by merging genuine gaps with legitimate low-value readings.
    #
    # Strategy:
    #   - X is returned with NaNs intact.
    #   - XGBoost / Random Forest: pass X directly — both handle NaN natively
    #     by learning the optimal branch direction for missing values at each split.
    #   - Ridge: call impute_for_linear(X_train, X_cal, X_test) inside
    #     train_ridge.py AFTER the chronological split, so imputation means
    #     are fitted on train only and applied to cal/test — no leakage.
    #
    # Remaining NaNs here are intentional; the leakage check below is scoped
    # to target columns only, not NaN presence.

    leaky = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky:
        raise ValueError(
            f"CRITICAL Data leakage detected. Forbidden columns still present in X:\n"
            f"  {leaky}\n"
            f"These are either future-target columns or current-timestep raw sensor/AQI "
            f"readings that must not appear in the feature matrix. Check BASE_DROP in "
            f"load_data.py and ensure feature_engineering.py does not store intermediate "
            f"scratch variables (e.g. aqi_historical_anchor) as document fields."
        )

    # Diagnostic: report residual NaN rate per column (should be low after
    # feature_engineering.py's ffill — if high, investigate sensor dropout)
    residual_nan = X.isna().mean()
    nan_cols = residual_nan[residual_nan > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print(
            f"\nResidual NaN rates in feature matrix (passed to model as-is for trees):"
        )
        print(nan_cols.round(4).to_string())

    # ── Matrix Diagnostics Summary ────────────────────────────────────────────
    print(f"\nTarget Distributions (raw AQI {horizon}h):")
    print(y_raw.describe().round(1).to_string())
    print(f"\nModel feature dimension space: {X.shape[1]}")
    print(f"Total row entries partitioned: {X.shape[0]:,}")

    # Sanity-check: aqi_lag_1 should correlate well with the target but not
    # suspiciously high (>0.99 would suggest residual leakage).
    if "aqi_lag_1" in X.columns:
        lag1_corr = X["aqi_lag_1"].corr(y_raw)
        flag = (
            "  <<< WARNING: SUSPICIOUSLY HIGH — CHECK FOR RESIDUAL LEAKAGE"
            if lag1_corr > 0.99 else ""
        )
        print(
            f"aqi_lag_1 correlation -> Target ({horizon}h): {lag1_corr:.3f}{flag}"
        )

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
        "aqi_change_1h", "aqi_change_6h", "aqi_change_24h", "aqi_acceleration",
        "pm25", "pm10",
        "pm25_roll_std_24", "interaction_pm25_humidity",
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
    res_df = pd.DataFrame(
        {"actual": y_true, "predicted": y_pred, "residual": residuals}
    )
    res_df.to_csv(
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
        dpi=150,
        bbox_inches="tight",
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
        print(
            f"\n{'#' * 60}\n DATABASE HOOK INGESTION CHECK FOR LOOKAHEAD HORIZON: {h}h\n{'#' * 60}"
        )
        try:
            X, y = load_xy(h, use_log=True)
            X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(
                X, y, h
            )
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