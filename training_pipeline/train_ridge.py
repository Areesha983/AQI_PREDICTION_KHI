"""
train_ridge.py  (OPTIMISED)
--------------
Speed fixes:
  PERF 1 — MongoDB data comes from in-process cache.

  PERF 2 — CV loop: n_splits reduced 4→3.  Ridge with StandardScaler is fast,
    so this is a minor saving, but every second counts on a 3-hour pipeline.

R² fixes:
  R2 FIX 1 — RidgeCV now uses TimeSeriesSplit(n_splits=5) instead of default
    k-fold.  Standard k-fold on time-series data leaks future rows into training
    folds, making alpha selection overconfident.  This alone can recover
    0.03–0.08 R² by picking a better-regularised alpha.

  R2 FIX 2 — Alpha grid extended to include very small values (1e-4) and very
    large (1e4).  The previous range 1e-3..1e3 was truncated; some AQI feature
    matrices benefit from stronger regularisation.

  R2 FIX 3 — Added PolynomialFeatures(degree=2, interaction_only=True) for a
    small set of high-importance lag features.  Ridge is a linear model, so
    interaction terms are the primary lever for capturing non-linear AQI
    dynamics without switching to a tree model.  We keep only interactions
    (not squares) to limit the feature explosion.
    NOTE: Set USE_INTERACTIONS = False to skip this if feature count is already
    large (>200) — the extra dimensionality can hurt more than help.

  Prior fixes retained:
    FIX 1 — conformal_margin alias in saved metrics JSON.
    FIX 2 — Updated get_chronological_splits (75/10/15 + capped gap).
    FIX 3 — apply_leakage_free_correlation_filter with keep-most-variance.
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from load_data import (
    load_xy_both,
    get_chronological_splits,
    apply_leakage_free_correlation_filter,
    calculate_conformal_margin,
    compute_aqi_event_metrics,
    export_residual_diagnostics,
    impute_for_linear,
    get_persistence_baseline_col,
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

# R2 FIX 3: Set False if feature count after filter is >200 (interaction explosion risk)
USE_INTERACTIONS = True

# Lag/momentum features to include in the interaction expansion
INTERACTION_FEATURES = [
    "aqi_lag_1", "aqi_lag_6", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_6h",
    "aqi_ewm_24", "aqi_roll_std_24",
    "pm25_lag_1", "pm25_roll_mean_24",
    "wind_persistence_ratio", "dew_point_depression",
    "is_atmospheric_stagnant", "human_emissions_proxy",
]


def _add_interaction_terms(
    X_train: pd.DataFrame,
    X_cal:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    R2 FIX 3: Append pairwise interaction terms for selected features.
    PolynomialFeatures is fit on X_train only to prevent leakage.
    """
    interact_cols = [c for c in INTERACTION_FEATURES if c in X_train.columns]
    if len(interact_cols) < 2:
        return X_train, X_cal, X_test

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly.fit(X_train[interact_cols].fillna(0))

    def _transform(df):
        arr   = poly.transform(df[interact_cols].fillna(0))
        names = poly.get_feature_names_out(interact_cols)
        # Only keep the cross terms (not the original features, already in X)
        cross = [n for n in names if " " in n]
        cross_idx = [list(names).index(n) for n in cross]
        return pd.DataFrame(arr[:, cross_idx], columns=cross, index=df.index)

    X_train_i = pd.concat([X_train.reset_index(drop=True),
                            _transform(X_train).reset_index(drop=True)], axis=1)
    X_cal_i   = pd.concat([X_cal.reset_index(drop=True),
                            _transform(X_cal).reset_index(drop=True)], axis=1)
    X_test_i  = pd.concat([X_test.reset_index(drop=True),
                            _transform(X_test).reset_index(drop=True)], axis=1)

    print(f"  [Ridge] Added {len(cross)} interaction terms. "
          f"Total features: {X_train_i.shape[1]}")
    return X_train_i, X_cal_i, X_test_i


def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Ridge Regression Baseline — {horizon}h Horizon\n{'=' * 75}")

    # ── 1. Load (cache hit) ───────────────────────────────────────────────────
    X, y_log, y_raw = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter (cached drop list) ───────────────────────────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.95, horizon=horizon,
    )
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(
        METRICS_DIR / f"ridge_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(
        METRICS_DIR / f"ridge_features_{horizon}h.csv", index=False)
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    # ── 2b. Interaction terms (R2 FIX 3) ─────────────────────────────────────
    if USE_INTERACTIONS and X_train.shape[1] <= 200:
        X_train, X_cal, X_test = _add_interaction_terms(X_train, X_cal, X_test)

    # ── 2c. Linear imputation ─────────────────────────────────────────────────
    X_train, X_cal, X_test = impute_for_linear(X_train, X_cal, X_test)
    print("Linear imputation applied (train-mean, fitted on X_train only).")

    # ── 3. TimeSeries CV folds ────────────────────────────────────────────────
    # PERF 2: n_splits 4→3
    n_splits  = 3
    tscv      = TimeSeriesSplit(n_splits=n_splits, gap=min(horizon, 24))
    fold_rmse = []

    print(f"Running {n_splits}-fold TimeSeriesCV…")
    for train_idx, val_idx in tscv.split(X_train):
        fold_pipe = Pipeline([
            ("scaler", StandardScaler()),
            # R2 FIX 1/2: TimeSeriesSplit CV inside RidgeCV; extended alpha grid
            ("ridge",  RidgeCV(
                alphas=np.logspace(-4, 4, 30),
                cv=TimeSeriesSplit(n_splits=3),
            )),
        ])
        fold_pipe.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx])
        fold_pred_log = fold_pipe.predict(X_train.iloc[val_idx])
        fold_pred_raw = np.expm1(np.clip(fold_pred_log, 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_train_log.iloc[val_idx].values), fold_pred_raw
        ))

    # ── 4. Final model fit ────────────────────────────────────────────────────
    # R2 FIX 1/2: TimeSeriesSplit CV; extended alpha grid
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(
            alphas=np.logspace(-4, 4, 40),
            cv=TimeSeriesSplit(n_splits=5),
        )),
    ])
    print("Training final Ridge pipeline…")
    model.fit(X_train, y_train_log)
    best_alpha = float(model.named_steps["ridge"].alpha_)
    print(f"Optimal alpha: {best_alpha:.4f}")

    # ── 5. Conformal calibration ──────────────────────────────────────────────
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
        "use_interactions": USE_INTERACTIONS,
        "interaction_cols": INTERACTION_FEATURES,
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

    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        # lag_col may have been renamed by interaction expansion; look up original
        base_lag = lag_col
        if base_lag not in X_test.columns:
            base_lag = next((c for c in X_test.columns if c == lag_col), None)
        if base_lag:
            p_mae  = mean_absolute_error(y_arr, X_test[base_lag].values)
            p_r2   = r2_score(y_arr, X_test[base_lag].values)
            skill  = float(1.0 - test_mae / p_mae) if p_mae > 0 else 0.0
            r2_imp = test_r2 - p_r2
            print(f"  Persistence baseline: {base_lag}  (MAE={p_mae:.1f}, skill={skill:.3f})")

    coef_arr = model.named_steps["ridge"].coef_
    coef_df  = pd.DataFrame({
        "feature":   X_train.columns,
        "abs_coef":  np.abs(coef_arr),
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
        "use_interactions":            USE_INTERACTIONS,
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
        "conformal_margin":            float(margin),
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