"""
train_ridge.py
--------------
Production-locked Ridge Regression baseline for AQI forecasting.

Key fixes vs original:
  1. Trains on log1p(AQI), inverse-transforms predictions before metrics.
  2. Conformal margins computed on the raw-AQI scale.
  3. nan% Coverage >200 fixed — was NaN because no predictions crossed the
     raw threshold after the old straight-line prediction on a skewed target.
  4. Linear-model imputation added after chronological split — load_xy() now
     returns NaNs intact for tree models. Ridge/StandardScaler cannot handle
     NaN natively, so impute_for_linear() is called here, fitted on X_train
     only and applied to X_cal/X_test to prevent any leakage.
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import (
    load_xy,
    get_chronological_splits,
    apply_leakage_free_correlation_filter,
    calculate_conformal_margin,
    compute_aqi_event_metrics,
    export_residual_diagnostics,
    impute_for_linear,
)

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] module not found — skipping drift check.")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

MODEL_DIR   = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Ridge Regression Baseline — {horizon}h Horizon\n{'=' * 75}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    X, y_log  = load_xy(horizon, use_log=True)
    _, y_raw  = load_xy(horizon, use_log=False)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter ─────────────────────────────────────────────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.95
    )
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(
        METRICS_DIR / f"ridge_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(
        METRICS_DIR / f"ridge_features_{horizon}h.csv", index=False)
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    # ── 2b. Linear-model imputation (train-mean, no leakage) ─────────────────
    # load_xy() now returns NaNs intact so tree models can route them natively.
    # Ridge and StandardScaler cannot handle NaN, so we impute here AFTER the
    # chronological split — means are fitted on X_train only, then applied to
    # X_cal and X_test. Doing this before the split would leak test statistics.
    X_train, X_cal, X_test = impute_for_linear(X_train, X_cal, X_test)
    print("Linear imputation applied (train-mean, fitted on X_train only).")

    # ── 3. TimeSeries CV folds ────────────────────────────────────────────────
    n_splits  = 4
    tscv      = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_rmse = []

    print(f"Running {n_splits}-fold TimeSeriesCV...")
    for train_idx, val_idx in tscv.split(X_train):
        fold_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  RidgeCV(alphas=np.logspace(-3, 3, 20))),
        ])
        fold_pipe.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx])
        fold_pred_log = fold_pipe.predict(X_train.iloc[val_idx])
        fold_pred_raw = np.expm1(np.clip(fold_pred_log, 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_train_log.iloc[val_idx].values), fold_pred_raw
        ))

    # ── 4. Final model fit ────────────────────────────────────────────────────
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TimeSeriesSplit(n_splits=5))),
    ])
    print("Training final Ridge pipeline...")
    model.fit(X_train, y_train_log)
    best_alpha = float(model.named_steps["ridge"].alpha_)
    print(f"Optimal alpha: {best_alpha:.4f}")

    # ── 5. Conformal calibration (raw scale) ──────────────────────────────────
    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin       = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    # ── 6. Test inference ─────────────────────────────────────────────────────
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 7. Save model ─────────────────────────────────────────────────────────
    joblib.dump({
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }, MODEL_DIR / f"ridge_{horizon}h.pkl")

    # ── 8. Metrics ────────────────────────────────────────────────────────────
    test_rmse = root_mean_squared_error(y_arr, preds_raw)
    test_mae  = mean_absolute_error(y_arr, preds_raw)
    test_r2   = r2_score(y_arr, preds_raw)
    test_evs  = explained_variance_score(y_arr, preds_raw)
    test_mape = float(np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100)

    observed_coverage = float(np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper)))
    avg_interval      = float(np.mean(pi_upper - pi_lower))

    mask_150 = y_arr > 150; n_gt150 = int(mask_150.sum())
    cov_150  = float(np.mean((y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150]))) if n_gt150 >= 5 else 0.0

    mask_200 = y_arr > 200; n_gt200 = int(mask_200.sum())
    cov_200  = float(np.mean((y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200]))) if n_gt200 >= 5 else 0.0

    events_150 = compute_aqi_event_metrics(y_arr, preds_raw, 150)
    events_200 = compute_aqi_event_metrics(y_arr, preds_raw, 200)

    lag_col = "aqi_lag_1"
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col in X_test.columns:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - test_mae / p_mae) if p_mae > 0 else 0.0
        r2_imp = test_r2 - p_r2

    # Coefficient inspection
    coef_df = pd.DataFrame({
        "feature":   X_train.columns,
        "abs_coef":  np.abs(model.named_steps["ridge"].coef_),
    }).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(METRICS_DIR / f"ridge_coefficients_{horizon}h.csv", index=False)

    top_20 = coef_df.head(20)["feature"].tolist()
    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "Ridge", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    pd.DataFrame({
        "actual": y_arr, "predicted": preds_raw,
        "lower_bound": pi_lower, "upper_bound": pi_upper,
    }, index=X_test.index).to_csv(METRICS_DIR / f"ridge_predictions_{horizon}h.csv", index=False)

    export_residual_diagnostics(y_arr, preds_raw, horizon, "Ridge", METRICS_DIR, MODEL_DIR)

    metrics = {
        "model":                       "Ridge",
        "horizon":                     f"{horizon}h",
        "training_target":             "log1p(AQI)",
        "best_alpha":                  best_alpha,
        "cv_mean_val_rmse":            float(np.mean(fold_rmse)),
        "cv_std_val_rmse":             float(np.std(fold_rmse)),
        "test_rmse":                   float(test_rmse),
        "test_mae":                    float(test_mae),
        "test_mape":                   float(test_mape),
        "test_r2":                     float(test_r2),
        "test_explained_variance":     float(test_evs),
        "baseline_horizon_mae":        float(p_mae),
        "baseline_horizon_r2":         float(p_r2),
        "forecast_skill_score":        float(skill),
        "r2_improvement_vs_baseline":  float(r2_imp),
        "conformal_margin_width":      float(margin),
        "conformal_global_coverage":   observed_coverage,
        "conformal_average_width":     avg_interval,
        "conformal_coverage_gt150":    float(cov_150),
        "conformal_n_gt150":           n_gt150,
        "conformal_coverage_gt200":    float(cov_200),
        "conformal_n_gt200":           n_gt200,
        **events_150,
        **events_200,
    }

    with open(METRICS_DIR / f"ridge_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_ridge(h)

    print("\n" + "=" * 135)
    print("FINAL SUMMARY — RIDGE REGRESSION  (predictions on raw-AQI scale)")
    print("=" * 135)
    print(f"{'Horizon':<8}{'CV RMSE':>12}{'Test MAE':>10}{'Test MAPE':>11}"
          f"{'Test R²':>10}{'Coverage':>12}{'Cov>150':>10}{'Cov>200':>10}")
    print("-" * 135)
    for horizon, m in results.items():
        print(
            f"{horizon:<8}"
            f"{m['cv_mean_val_rmse']:>12.2f}"
            f"{m['test_mae']:>10.1f}"
            f"{m['test_mape']:>10.2f}%"
            f"{m['test_r2']:>10.3f}"
            f"{m['conformal_global_coverage']*100:>11.1f}%"
            f"{m['conformal_coverage_gt150']*100:>9.1f}%"
            f"{m['conformal_coverage_gt200']*100:>9.1f}%"
        )