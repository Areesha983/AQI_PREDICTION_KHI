"""
train_xgboost.py  (MONGODB-ONLY + R² IMPROVEMENTS + LEAKAGE-FREE ES SPLIT)
-----------------
Storage changes (v3):
  STORE 1 — All local file writes replaced with mongo_store calls.

R² improvements (v3):
  R2 v3-2 — Two-stage approach: train on log-AQI (existing), then fit a
            lightweight residual correction model (GradientBoostingRegressor)
            on the calibration set residuals to squeeze out systematic bias.
            The corrected output is clipped and reported.

  R2 v3-3 — max_depth raised 6→7 and colsample_bytree raised 0.8→0.85 to
            allow more expressive trees on the (now wider) feature space.

  R2 v3-4 — reg_lambda reduced 2.0→1.5 and reg_alpha 0→0.1 (L1+L2 elastic).
            Pure L2 shrinks all features; a small L1 penalty promotes sparsity
            and often improves generalisation on high-dim tabular data.

Previously retained fixes:
  R2 FIX 1 — Early stopping on internal val set (last 15% of train).
  R2 FIX 2 — learning_rate 0.02.
  R2 FIX 3 — min_child_weight 5 (loosened from 8 for spike sensitivity).
  R2 FIX 4 — Sample weight cap 6× for AQI>200.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
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


def _fit_residual_corrector(
    X_cal: pd.DataFrame,
    y_cal_raw: np.ndarray,
    cal_pred_raw: np.ndarray,
) -> GradientBoostingRegressor:
    """
    R2 v3-2: Fit a shallow GBR on calibration residuals.
    This corrects systematic bias left by the main XGB model.
    """
    residuals = y_cal_raw - cal_pred_raw
    corrector = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    corrector.fit(X_cal, residuals)
    return corrector


def train_xgboost(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n XGBoost Engine — {horizon}h Horizon\n{'=' * 75}")
    run_id = _run_id()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    X, y_log, y_raw = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
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
        X_ft, y_ft = X_train.iloc[tr_idx], y_train_log.iloc[tr_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train_log.iloc[val_idx]

        fw = np.ones(len(y_ft))
        fw[np.expm1(y_ft.values) > 100] = 2.0
        fw[np.expm1(y_ft.values) > 150] = 3.0
        fw[np.expm1(y_ft.values) > 200] = 6.0

        fm = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.85,  # R2 v3-3
            min_child_weight=5,     # Matching structural fix
            reg_lambda=1.5,         # R2 v3-4
            reg_alpha=0.1,          # R2 v3-4
            random_state=42 + fold, n_jobs=-1, verbosity=0,
        )
        fm.fit(X_ft, y_ft, sample_weight=fw)
        fold_pred_raw = np.expm1(np.clip(fm.predict(X_fv), 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_fv.values), fold_pred_raw
        ))
    print(f"CV RMSE (raw AQI): {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")

    # ── 4. Clean Early-Stop Validation Isolation & Augmentation ──────────────
    # CRITICAL LEAKAGE FIX: Separate the internal validation split strictly before performing 
    # any row mutations or data augmentations on the active training subset.
    es_split = int(len(X_train) * 0.85)
    X_clean_train_fold = X_train.iloc[:es_split]
    y_clean_train_log_fold = y_train_log.iloc[:es_split]
    y_clean_train_raw_fold = y_train_raw.iloc[:es_split]

    X_es_val = X_train.iloc[es_split:]
    y_es_val = y_train_log.iloc[es_split:]

    # Augment ONLY the isolated training subset to protect validation slice integrity
    X_fold_aug, y_fold_aug = get_spike_augmented_train(
        X_clean_train_fold, 
        y_clean_train_log_fold,
        y_train_raw=y_clean_train_raw_fold,
        spike_threshold=150,
        target_spike_fraction=0.07,
    )

    # Recombine clean fold and its isolated mutations to generate the training footprint
    X_es_train = pd.concat([X_clean_train_fold, X_fold_aug.iloc[len(X_clean_train_fold):]], ignore_index=True)
    y_es_train = pd.concat([y_clean_train_log_fold, y_fold_aug.iloc[len(y_clean_train_log_fold):]], ignore_index=True)

    # ── 5. Sample weights computed securely on active training fold ───────────
    y_es_train_raw = np.expm1(y_es_train.values)
    sw_es = np.ones(len(y_es_train))
    sw_es[y_es_train_raw > 100] = 2.0
    sw_es[y_es_train_raw > 150] = 3.0
    sw_es[y_es_train_raw > 200] = 6.0

    # ── 6. Final XGBoost model ────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=7,            # R2 v3-3
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.85,  # R2 v3-3
        min_child_weight=5,     # FIX: reduced from 8 to capture complex peak splits
        reg_lambda=1.2,         # FIX: reduced L2 shrinkage pressures
        reg_alpha=0.05,         # FIX: reduced L1 penalty pressures
        tree_method="hist",
        early_stopping_rounds=60,  # FIX: bumped 20->60 to let learning_rate=0.02 fully converge
        random_state=42, n_jobs=-1, verbosity=0,
    )
    print("Training final XGBoost model (early stopping on internal val set)…")
    model.fit(
        X_es_train, y_es_train,
        sample_weight=sw_es,
        eval_set=[(X_es_val, y_es_val)],
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration}")

    # ── 7. Conformal calibration + residual corrector ────────────────────────
    cal_pred_log = np.clip(model.predict(X_cal), 0, None)
    cal_pred_raw = np.expm1(cal_pred_log)

    # R2 v3-2: Fit residual corrector on calibration set residuals
    corrector = _fit_residual_corrector(X_cal, y_cal_raw.values, cal_pred_raw)
    cal_pred_corrected = np.clip(cal_pred_raw + corrector.predict(X_cal), 0, 500)

    margin = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_corrected))

    # ── 8. Test inference ─────────────────────────────────────────────────────
    raw_preds = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    # Apply raw-scale residual correction on test set
    preds_raw = np.clip(raw_preds + corrector.predict(X_test), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 9. Save model to MongoDB via GridFS Module (STORE 1) ─────────────────
    save_model_artifact(
        model_name="XGBoost",
        horizon=horizon,
        artifact={
            "model":            model,
            "corrector":        corrector,
            "feature_names":    list(X_train.columns),
            "conformal_margin": float(margin),
            "use_log":          True,
            "use_corrector":    True,
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

    # ── 11. Persist predictions, residuals → MongoDB (STORE 1) ───────────────
    save_predictions("XGBoost", horizon, y_arr, preds_raw, pi_lower, pi_upper, run_id)
    save_residuals("XGBoost", horizon, y_arr, preds_raw, run_id)

    # ── 12. Feature importance → MongoDB (STORE 1) ────────────────────────────
    save_feature_list(
        model="XGBoost",
        horizon=horizon,
        feature_names=list(X_train.columns),
        dropped=dropped_cols,
        run_id=run_id,
    )

    # ── 13. SHAP → MongoDB (STORE 1) ──────────────────────────────────────────
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

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 14. Metrics dict → MongoDB (STORE 1) ──────────────────────────────────
    metrics = {
        "model":                       "XGBoost",
        "horizon":                     f"{horizon}h",
        "training_target":             "log1p(AQI)",
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