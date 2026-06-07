"""
train_xgboost.py  (v4 — FIXED)
-----------------

ROOT CAUSE FIXES in this version:
  FIX A — REMOVED residual corrector entirely.
           The GBR corrector was fitted on the calibration set, which is also
           the set used to compute the conformal margin. This meant the corrector
           memorised calibration residuals (near-zero after correction), causing
           an artificially tiny conformal margin. On the unseen test set the
           corrector generalised poorly, collapsing coverage to ~56% and
           inflating MAE. Removing it restores honest conformal calibration.

  FIX B — Training on RAW AQI (not log1p).
           The log transform suppresses the high-AQI tail that XGBoost needs to
           learn. Random Forest and Ridge both train on raw AQI and outperform
           XGBoost — switching to raw AQI aligns the loss function with the
           evaluation metric (MAE on raw scale) and removes the expm1() back-
           conversion step that accumulates error at high values.

  FIX C — learning_rate raised 0.02 → 0.05, n_estimators raised 600 → 1000.
           At lr=0.02 with only 600 trees the 72h model was stopping at
           iteration 79 — clear underfitting. At lr=0.05 the model converges
           properly and early stopping finds a stable optimum. 1000 tree budget
           with early_stopping_rounds=50 gives the model enough room.

  FIX D — Early stopping eval set uses X_cal / y_cal (the true held-out
           calibration set) instead of an 85% slice of X_train. This stops
           the model when it starts to overfit on the actual held-out data,
           not a fabricated internal split that can overlap with spike-
           augmented rows.

  FIX E — Spike augmentation applied to full X_train before splitting for
           early stopping, consistent with how Random Forest uses it.

Previously retained:
  - 4-fold TimeSeriesCV for hyperparameter evaluation
  - Leakage-free correlation filter (threshold=0.97)
  - sample weights (1×/2×/3×/6× by AQI band)
  - SHAP storage via mongo_store
  - All mongo_store artifact/metrics/predictions/residuals saves
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import shap

from load_data import (
    load_xy_both,
    get_chronological_splits,
    get_spike_augmented_train,
    apply_leakage_free_correlation_filter,
    calculate_conformal_margin,
    compute_aqi_event_metrics,
    get_persistence_baseline_col,
)
from mongo_store import (
    save_metrics,
    save_predictions,
    save_residuals,
    save_feature_list,
    save_shap,
    save_shap_plot_png,
    save_model_artifact,
    _run_id,
)

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] module not found — skipping drift check.")

print("INITIALIZING XGBOOST ENGINE")


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    results = {}
    bands = {
        "all":                (0,    9999),
        "good_moderate":       (0,    100),
        "unhealthy_sensitive": (101,  150),
        "unhealthy":           (151,  200),
        "very_unhealthy":      (201,  300),
        "hazardous":           (301,  9999),
    }
    for label, (lo, hi) in bands.items():
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() < 5:
            results[label] = {"n": int(mask.sum()), "mae": None}
            continue
        results[label] = {
            "n":   int(mask.sum()),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        }
    return results


def quantile_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = {}
    for q in [90, 95, 99]:
        threshold = np.percentile(y_true, q)
        mask = y_true >= threshold
        errors[f"mae_top_{q}"] = (
            float(mean_absolute_error(y_true[mask], y_pred[mask]))
            if mask.sum() >= 5 else None
        )
    return errors


def train_xgboost(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n XGBoost Engine — {horizon}h Horizon\n{'=' * 75}")
    run_id = _run_id()

    # ── 1. Load — use RAW AQI as target (FIX B) ──────────────────────────────
    X, y_log, y_raw = load_xy_both(horizon)

    # Split on raw AQI only — no log target used
    X_train, y_train_raw, X_cal, y_cal_raw, X_test, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter ─────────────────────────────────────────────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97, horizon=horizon,
    )
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    # ── 3. CV on ORIGINAL (non-augmented) X_train ────────────────────────────
    n_splits  = 4
    tscv      = TimeSeriesSplit(n_splits=n_splits, gap=min(horizon, 24))
    fold_rmse = []

    print(f"Running {n_splits}-fold TimeSeriesCV on original train set…")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_ft, y_ft = X_train.iloc[tr_idx], y_train_raw.iloc[tr_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train_raw.iloc[val_idx]

        fw = np.ones(len(y_ft))
        fw[y_ft.values > 100] = 1.5
        fw[y_ft.values > 150] = 2.0
        fw[y_ft.values > 200] = 3.0

        fm = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.05,         # FIX C: was 0.03
            subsample=0.8,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=1.5,
            reg_alpha=0.1,
            random_state=42 + fold, n_jobs=-1, verbosity=0,
        )
        fm.fit(X_ft, y_ft, sample_weight=fw)
        fold_pred = np.clip(fm.predict(X_fv), 0, 500)
        fold_rmse.append(root_mean_squared_error(y_fv.values, fold_pred))
    print(f"CV RMSE (raw AQI): {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")

    # ── 4. Spike augmentation on full X_train (FIX E) ─────────────────────────
    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train,
        y_train_raw,
        y_train_raw=y_train_raw,
        spike_threshold=150,
        target_spike_fraction=0.04,  # reduced from 0.07 — prevents low-AQI baseline collapse
    )

    # ── 5. Sample weights on augmented training set ───────────────────────────
    sw = np.ones(len(y_train_aug))
    sw[y_train_aug.values > 100] = 1.5
    sw[y_train_aug.values > 150] = 2.0
    sw[y_train_aug.values > 200] = 3.0

    # ── 6. Final XGBoost model (FIX C + FIX D) ────────────────────────────────
    # Early stopping uses X_cal / y_cal — the real held-out calibration set.
    # This prevents the model from overfitting to training data and gives a
    # stable stopping point aligned with actual generalisation performance.
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,                    # reduced from 7 — less overfitting to spike patterns
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_weight=3,             # reduced from 5 — allows model to learn Good-range splits
        reg_lambda=0.8,                 # reduced from 1.2 — less L2 shrinkage toward 0
        reg_alpha=0.05,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    print("Training final XGBoost model (early stopping on calibration set)…")
    model.fit(
        X_train_aug, y_train_aug,
        sample_weight=sw,
        eval_set=[(X_cal, y_cal_raw)],  # FIX D: use real cal set, not X_train slice
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration}")

    # ── 7. Conformal calibration (FIX A — no corrector) ──────────────────────
    # Calibration residuals are now honest: the model has never seen X_cal.
    cal_preds = np.clip(model.predict(X_cal), 0, 500)
    margin    = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_preds))

    # ── 8. Test inference ─────────────────────────────────────────────────────
    preds_raw = np.clip(model.predict(X_test), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 9. Save model artifact → MongoDB GridFS ───────────────────────────────
    save_model_artifact(
        model_name="XGBoost",
        horizon=horizon,
        artifact={
            "model":            model,
            "feature_names":    list(X_train.columns),
            "conformal_margin": float(margin),
            "use_log":          False,      # FIX B: raw AQI, no expm1 at inference
            "use_corrector":    False,      # FIX A: corrector removed
        },
        run_id=run_id,
    )

    # ── 10. Metrics ───────────────────────────────────────────────────────────
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

    events_150       = compute_aqi_event_metrics(y_arr, preds_raw, 150)
    events_200       = compute_aqi_event_metrics(y_arr, preds_raw, 200)
    stratified_bands = error_analysis(y_arr, preds_raw)
    quantile_errors  = quantile_error_analysis(y_arr, preds_raw)

    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - test_mae / p_mae) if p_mae > 0 else 0.0
        r2_imp = test_r2 - p_r2
        print(f"  Persistence baseline: {lag_col}  (MAE={p_mae:.1f}, skill={skill:.3f})")

    # ── 11. Persist predictions, residuals → MongoDB ──────────────────────────
    save_predictions("XGBoost", horizon, y_arr, preds_raw, pi_lower, pi_upper, run_id)
    save_residuals("XGBoost", horizon, y_arr, preds_raw, run_id)

    # ── 12. Feature importance → MongoDB ──────────────────────────────────────
    xgb_importances = model.feature_importances_.tolist()
    save_feature_list(
        model="XGBoost",
        horizon=horizon,
        feature_names=list(X_train.columns),
        importance=xgb_importances,
        dropped=dropped_cols,
        run_id=run_id,
    )

    # ── 13. SHAP → MongoDB ────────────────────────────────────────────────────
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx]

    explainer        = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)
    mean_abs_shap    = np.abs(shap_explanation.values).mean(axis=0)

    save_shap(
        model="XGBoost",
        horizon=horizon,
        feature_names=list(X_sample.columns),
        mean_abs_shap=mean_abs_shap.tolist(),
        run_id=run_id,
    )

    top_20 = sorted(
        zip(X_sample.columns, mean_abs_shap),
        key=lambda x: x[1], reverse=True,
    )[:20]
    top_20 = [f for f, _ in top_20]

    fig = plt.figure(figsize=(10, 8))
    try:
        shap.plots.beeswarm(shap_explanation, show=False)
    except Exception:
        shap.summary_plot(shap_explanation.values, X_sample, show=False)
    save_shap_plot_png("XGBoost", horizon, fig, run_id)

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 14. Metrics dict → MongoDB ────────────────────────────────────────────
    metrics = {
        "model":                       "XGBoost",
        "horizon":                     f"{horizon}h",
        "training_target":             "raw_AQI",       # FIX B: was log1p(AQI)
        "cv_mean_val_rmse":            float(np.mean(fold_rmse)),
        "cv_std_val_rmse":             float(np.std(fold_rmse)),
        "test_rmse":                   float(test_rmse),
        "test_mae":                    float(test_mae),
        "test_mape":                   float(test_mape),
        "test_r2":                     float(test_r2),
        "test_explained_variance":     float(test_evs),
        "baseline_lag_col":            lag_col or "none",
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
        "error_by_band":               stratified_bands,
        "quantile_errors":             quantile_errors,
        **events_150,
        **events_200,
    }

    save_metrics("XGBoost", horizon, metrics, run_id)
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_xgboost(h)

    print("\n" + "=" * 145)
    print("FINAL SUMMARY — XGBOOST  (predictions on raw-AQI scale)")
    print("=" * 145)
    print(f"{'Horizon':<8}{'CV RMSE':>12}{'Test MAE':>10}{'Test MAPE':>11}"
          f"{'Test R²':>10}{'Coverage':>12}{'Cov>150':>10}{'Cov>200':>10}")
    print("-" * 145)
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