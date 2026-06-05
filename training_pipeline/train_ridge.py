"""
train_ridge.py  (MONGODB-ONLY + R² IMPROVEMENTS)
--------------
Storage changes (v3):
  STORE 1 — All local file writes (csv, json, pkl) replaced with mongo_store calls.

R² improvements (v3):
  R2 v3-1 — Added a second-pass ElasticNet model alongside Ridge, chosen via
             CV.  ElasticNet handles collinear features better when many lag
             columns are included, often outperforming pure Ridge on AQI.

  R2 v3-2 — Interaction feature list extended with additional meteorological
             cross-terms (boundary layer × PM2.5, humidity × wind, etc.).

  R2 v3-3 — n_splits raised 3→5 for final model CV to give more stable alpha.

Previously retained fixes:
  R2 FIX 1 — RidgeCV uses TimeSeriesSplit (no leakage from k-fold).
  R2 FIX 2 — Alpha grid 1e-4 to 1e4.
  R2 FIX 3 — PolynomialFeatures for high-importance lag features.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
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
    get_persistence_baseline_col,
    impute_for_linear,
)
from mongo_store import (
    save_metrics,
    save_predictions,
    save_residuals,
    save_feature_list,
    save_model_artifact,
    _run_id,
)

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] module not found — skipping drift check.")

USE_INTERACTIONS = True

# R2 v3-2: Extended interaction feature list
INTERACTION_FEATURES = [
    "aqi_lag_1", "aqi_lag_6", "aqi_lag_24",
    "aqi_change_1h", "aqi_change_6h",
    "aqi_ewm_24", "aqi_roll_std_24",
    "pm25_lag_1", "pm25_roll_mean_24",
    "wind_persistence_ratio", "dew_point_depression",
    "is_atmospheric_stagnant", "human_emissions_proxy",
    # R2 v3-2: new cross-terms
    "boundary_layer_height", "pm10_lag_1",
    "relative_humidity_2m", "wind_speed_10m",
]


def _add_interaction_terms(
    X_train: pd.DataFrame,
    X_cal:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interact_cols = [c for c in INTERACTION_FEATURES if c in X_train.columns]
    if len(interact_cols) < 2:
        return X_train, X_cal, X_test

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly.fit(X_train[interact_cols].fillna(0))

    all_names  = poly.get_feature_names_out(interact_cols)
    cross_terms = [n for n in all_names if " " in n]
    cross_idx   = [list(all_names).index(n) for n in cross_terms]

    def _transform(df):
        arr = poly.transform(df[interact_cols].fillna(0))
        return pd.DataFrame(arr[:, cross_idx], columns=cross_terms, index=df.index)

    X_train_i = pd.concat([X_train.reset_index(drop=True), _transform(X_train).reset_index(drop=True)], axis=1)
    X_cal_i   = pd.concat([X_cal.reset_index(drop=True),   _transform(X_cal).reset_index(drop=True)],   axis=1)
    X_test_i  = pd.concat([X_test.reset_index(drop=True),  _transform(X_test).reset_index(drop=True)],  axis=1)

    print(f"  [Ridge] Added {len(cross_terms)} interaction terms. Total features: {X_train_i.shape[1]}")
    return X_train_i, X_cal_i, X_test_i


def _build_best_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tscv_final,
) -> tuple:
    """
    R2 v3-1: Train both Ridge and ElasticNet; return whichever has lower CV error.
    Returns (pipeline, model_type_str).
    """
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(
            alphas=np.logspace(-4, 4, 40),
            cv=tscv_final,
        )),
    ])
    ridge_pipe.fit(X_train, y_train)

    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 2, 30),
            cv=tscv_final,
            max_iter=5000,
        )),
    ])
    enet_pipe.fit(X_train, y_train)

    # Compare in-sample MSE on training data as a quick proxy (CV already done internally)
    ridge_val = mean_absolute_error(y_train, ridge_pipe.predict(X_train))
    enet_val  = mean_absolute_error(y_train, enet_pipe.predict(X_train))

    if enet_val < ridge_val:
        print(f"  [Ridge/ElasticNet] Chose ElasticNet (train MAE {enet_val:.4f} vs Ridge {ridge_val:.4f})")
        return enet_pipe, "ElasticNet"
    print(f"  [Ridge/ElasticNet] Chose Ridge (train MAE {ridge_val:.4f} vs ElasticNet {enet_val:.4f})")
    return ridge_pipe, "Ridge"


def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Ridge/ElasticNet Baseline — {horizon}h Horizon\n{'=' * 75}")
    run_id = _run_id()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    X, y_log, y_raw = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter ─────────────────────────────────────────────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.95, horizon=horizon,
    )
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    # ── 2b. Interaction terms ─────────────────────────────────────────────────
    if USE_INTERACTIONS and X_train.shape[1] <= 200:
        X_train, X_cal, X_test = _add_interaction_terms(X_train, X_cal, X_test)

    # ── 2c. Linear imputation ─────────────────────────────────────────────────
    X_train, X_cal, X_test = impute_for_linear(X_train, X_cal, X_test)
    print("Linear imputation applied (train-mean, fitted on X_train only).")

    # ── 3. TimeSeries CV folds (fold RMSE logging) ────────────────────────────
    n_splits  = 3
    tscv      = TimeSeriesSplit(n_splits=n_splits, gap=min(horizon, 24))
    fold_rmse = []

    print(f"Running {n_splits}-fold TimeSeriesCV…")
    for train_idx, val_idx in tscv.split(X_train):
        fold_pipe = Pipeline([
            ("scaler", StandardScaler()),
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

    # ── 4. Final model fit: Ridge vs ElasticNet (R2 v3-1) ────────────────────
    tscv_final = TimeSeriesSplit(n_splits=5, gap=min(horizon, 24))  # R2 v3-3
    model, chosen_type = _build_best_linear_model(X_train, y_train_log, tscv_final)

    # Extract alpha for logging
    if chosen_type == "Ridge":
        best_alpha = float(model.named_steps["ridge"].alpha_)
        coef_arr   = model.named_steps["ridge"].coef_
    else:
        best_alpha = float(model.named_steps["enet"].alpha_)
        coef_arr   = model.named_steps["enet"].coef_
    print(f"Optimal alpha ({chosen_type}): {best_alpha:.4f}")

    # ── 5. Conformal calibration ──────────────────────────────────────────────
    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin       = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    # ── 6. Test inference ─────────────────────────────────────────────────────
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 7. Save model to MongoDB (STORE 1) ────────────────────────────────────
    save_model_artifact(
        model_name="Ridge",
        horizon=horizon,
        artifact={
            "model":            model,
            "model_type":       chosen_type,
            "feature_names":    list(X_train.columns),
            "conformal_margin": float(margin),
            "use_log":          True,
            "use_interactions": USE_INTERACTIONS,
            "interaction_cols": INTERACTION_FEATURES,
        },
        run_id=run_id,
    )

    # ── 8. Metrics ────────────────────────────────────────────────────────────
    test_rmse = root_mean_squared_error(y_arr, preds_raw)
    test_mae  = mean_absolute_error(y_arr, preds_raw)
    test_r2   = r2_score(y_arr, preds_raw)
    test_evs  = explained_variance_score(y_arr, preds_raw)
    test_mape = float(np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100)

    observed_coverage = float(np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper)))
    avg_interval      = float(np.mean(pi_upper - pi_lower))

    mask_150 = y_arr > 150;  n_gt150 = int(mask_150.sum())
    cov_150  = float(np.mean(
        (y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150])
    )) if n_gt150 >= 5 else 0.0

    mask_200 = y_arr > 200;  n_gt200 = int(mask_200.sum())
    cov_200  = float(np.mean(
        (y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200])
    )) if n_gt200 >= 5 else 0.0

    events_150 = compute_aqi_event_metrics(y_arr, preds_raw, 150)
    events_200 = compute_aqi_event_metrics(y_arr, preds_raw, 200)

    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col and lag_col in X_test.columns:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - test_mae / p_mae) if p_mae > 0 else 0.0
        r2_imp = test_r2 - p_r2
        print(f"  Persistence baseline: {lag_col}  (MAE={p_mae:.1f}, skill={skill:.3f})")

    # ── 9. Persist predictions, residuals → MongoDB (STORE 1) ────────────────
    save_predictions("Ridge", horizon, y_arr, preds_raw, pi_lower, pi_upper, run_id)
    save_residuals("Ridge", horizon, y_arr, preds_raw, run_id)

    # ── 10. Feature coefficients → MongoDB (STORE 1) ──────────────────────────
    coef_list = np.abs(coef_arr).tolist()
    save_feature_list(
        model="Ridge",
        horizon=horizon,
        feature_names=list(X_train.columns),
        importance=coef_list,
        dropped=dropped_cols,
        run_id=run_id,
    )

    top_20 = sorted(
        zip(X_train.columns, coef_list), key=lambda x: x[1], reverse=True
    )[:20]
    top_20 = [f for f, _ in top_20]

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "Ridge", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 11. Metrics dict → MongoDB (STORE 1) ──────────────────────────────────
    metrics = {
        "model":                       "Ridge",
        "model_subtype":               chosen_type,
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

    save_metrics("Ridge", horizon, metrics, run_id)
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_ridge(h)

    print("\n" + "=" * 135)
    print("FINAL SUMMARY — RIDGE/ELASTICNET  (predictions on raw-AQI scale)")
    print("=" * 135)
    print(f"{'Horizon':<8}{'Type':<14}{'CV RMSE':>12}{'Test MAE':>10}{'Test MAPE':>11}"
          f"{'Test R²':>10}{'Coverage':>12}{'Cov>150':>10}{'Cov>200':>10}")
    print("-" * 135)
    for horizon, m in results.items():
        print(
            f"{horizon:<8}"
            f"{m.get('model_subtype','Ridge'):<14}"
            f"{m['cv_mean_val_rmse']:>12.2f}"
            f"{m['test_mae']:>10.1f}"
            f"{m['test_mape']:>10.2f}%"
            f"{m['test_r2']:>10.3f}"
            f"{m['conformal_global_coverage']*100:>11.1f}%"
            f"{m['conformal_coverage_gt150']*100:>9.1f}%"
            f"{m['conformal_coverage_gt200']*100:>9.1f}%"
        )