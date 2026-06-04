"""
train_xgboost.py  (FIXED)
--------------------------
Key fixes vs previous version:
  1. Uses get_persistence_baseline_col() so 48h/72h skill scores don't silently
     zero out when aqi_lag_48/aqi_lag_72 are absent after the filter.
  2. XGBoost hyperparameters tuned for the larger (32 k row) dataset:
       - n_estimators: 800 → 1200 for final model (more trees = better R²)
       - max_depth: 7 → 6  (prevents overfitting on larger data)
       - learning_rate: 0.02 → 0.015 (slower learning with more trees)
       - min_child_weight: 3 → 5 (stronger regularisation)
       - subsample: 0.85 → 0.8
       - reg_lambda: added = 2.0 (L2 regularisation)
  3. CV folds raised from 4 → 5 for a more reliable RMSE estimate.
  4. Spike threshold for augmentation raised to target_spike_fraction=0.07
     (was 0.05) — the RF already shows ~8.6% spike rate; matching it helps XGB.
  5. apply_leakage_free_correlation_filter threshold uses 0.97 (aligned with
     the fixed load_data.py).
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
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
    load_xy,
    get_chronological_splits,
    get_spike_augmented_train,
    apply_leakage_free_correlation_filter,
    calculate_conformal_margin,
    compute_aqi_event_metrics,
    export_residual_diagnostics,
    get_persistence_baseline_col,   # FIX #1
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

print("INITIALIZING XGBOOST ENGINE")


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    results = {}
    bands = {
        "all":                 (0,    9999),
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

    # ── 1. Load ───────────────────────────────────────────────────────────────
    X, y_log  = load_xy(horizon, use_log=True)
    _, y_raw  = load_xy(horizon, use_log=False)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter (threshold 0.97) ────────────────────────────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97
    )
    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(
        METRICS_DIR / f"xgb_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(
        METRICS_DIR / f"xgb_features_{horizon}h.csv", index=False)
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    # ── 3. Spike augmentation (target 7% of training set) ────────────────────
    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train, y_train_log,
        y_train_raw=y_train_raw,
        spike_threshold=150,
        target_spike_fraction=0.07,   # FIX #4
    )

    # ── 4. Sample weights ─────────────────────────────────────────────────────
    y_aug_raw = np.expm1(y_train_aug.values)
    sample_weights = np.ones(len(y_train_aug))
    sample_weights[y_aug_raw > 100] = 2.0
    sample_weights[y_aug_raw > 150] = 4.0
    sample_weights[y_aug_raw > 200] = 8.0

    # ── 5. TimeSeries CV (5 folds) ────────────────────────────────────────────
    n_splits  = 5   # FIX #3
    tscv      = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_rmse = []

    print(f"Running {n_splits}-fold TimeSeriesCV...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_ft, y_ft = X_train.iloc[tr_idx], y_train_log.iloc[tr_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train_log.iloc[val_idx]

        # FIX BUG-6: CV fold weights now match final model weights exactly.
        # The old code was missing the >100 band (weight 2.0), so CV RMSE was
        # not representative of the final model's training objective, causing
        # hyperparameter selection to be misaligned with actual loss.
        fw = np.ones(len(y_ft))
        fw[np.expm1(y_ft.values) > 100] = 2.0
        fw[np.expm1(y_ft.values) > 150] = 4.0
        fw[np.expm1(y_ft.values) > 200] = 8.0

        fm = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=2.0,
            random_state=42 + fold, n_jobs=-1, verbosity=0,
        )
        fm.fit(X_ft, y_ft, sample_weight=fw)
        fold_pred_raw = np.expm1(np.clip(fm.predict(X_fv), 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_fv.values), fold_pred_raw
        ))
    print(f"CV RMSE (raw AQI): {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")

    # ── 6. Final model (FIX #2: tuned hyperparameters) ───────────────────────
    # FIX BUG-5: 1200 trees with no early stopping overfits on smaller datasets.
    # Use the calibration set as a held-out eval set for early stopping.
    # Also add tree_method='hist' for ~3x faster training on the 32k dataset.
    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.015,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        tree_method="hist",        # FIX: ~3x faster on tabular data
        early_stopping_rounds=50,  # FIX: stop before overfitting
        random_state=42, n_jobs=-1, verbosity=0,
    )
    print("Training final XGBoost model (log target, early stopping on cal set)...")
    model.fit(
        X_train_aug, y_train_aug,
        sample_weight=sample_weights,
        eval_set=[(X_cal, y_cal_log)],
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration}")
    print("Done.")

    # ── 7. Conformal calibration ──────────────────────────────────────────────
    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin       = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    # ── 8. Test inference ─────────────────────────────────────────────────────
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 9. Save model ─────────────────────────────────────────────────────────
    joblib.dump({
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }, MODEL_DIR / f"xgboost_{horizon}h.pkl")

    # ── 10. Metrics ───────────────────────────────────────────────────────────
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

    events_150      = compute_aqi_event_metrics(y_arr, preds_raw, 150)
    events_200      = compute_aqi_event_metrics(y_arr, preds_raw, 200)
    stratified_bands = error_analysis(y_arr, preds_raw)
    quantile_errors  = quantile_error_analysis(y_arr, preds_raw)

    # FIX #1: use robust persistence baseline
    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - test_mae / p_mae) if p_mae > 0 else 0.0
        r2_imp = test_r2 - p_r2
        print(f"  Persistence baseline: {lag_col}  (MAE={p_mae:.1f}, skill={skill:.3f})")

    pd.DataFrame({
        "actual": y_arr, "predicted": preds_raw,
        "lower_bound": pi_lower, "upper_bound": pi_upper,
    }, index=X_test.index).to_csv(METRICS_DIR / f"xgb_predictions_{horizon}h.csv", index=False)

    export_residual_diagnostics(y_arr, preds_raw, horizon, "XGB", METRICS_DIR, MODEL_DIR)

    # ── 11. SHAP ──────────────────────────────────────────────────────────────
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx]

    explainer        = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample)
    mean_abs_shap    = np.abs(shap_explanation.values).mean(axis=0)

    shap_imp = pd.DataFrame({
        "feature": X_sample.columns, "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(METRICS_DIR / f"xgb_top_features_{horizon}h.csv", index=False)

    top_20 = shap_imp.head(20)["feature"].tolist()
    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 12. Metrics dict ──────────────────────────────────────────────────────
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

    with open(METRICS_DIR / f"xgb_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)

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