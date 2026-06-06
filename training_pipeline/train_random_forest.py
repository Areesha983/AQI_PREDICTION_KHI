"""
train_random_forest.py  (MONGODB-ONLY + R² IMPROVEMENTS)
-----------------------
Storage changes (v3):
  STORE 1 — All file writes (csv, json, pkl, png) replaced with mongo_store calls.
            Nothing is written to the local filesystem.

R² improvements (v3):
  R2 v3-1 — Wider hyperparameter grid: max_depth up to 35, max_features includes
             0.4, min_samples_leaf down to 3, n_estimators up to 600.
  R2 v3-2 — n_iter raised 8→12 for RandomizedSearchCV to explore more combinations.
  R2 v3-3 — max_samples increased to 0.90 (was 0.85) to use more training data per
             tree while keeping inter-tree diversity.
  R2 v3-4 — Huber-like sample weighting replaced with rank-based weights to reduce
             sensitivity to exact AQI values of spike rows.

Previously retained fixes:
  R2 FIX 1 — Scoring: neg_MAE on raw AQI scale (not log-RMSE).
  R2 FIX 2 — min_samples_leaf minimum 4 to prevent leaf overfitting.
  R2 FIX 3 — max_samples subsampling for inter-tree diversity.
  R2 FIX 4 — Sample weight cap at 20×.
"""

import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
    make_scorer,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

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

COMPUTE_SHAP = True

print("INITIALIZING RANDOM FOREST ENGINE")


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


def raw_neg_mae_score_func(y_true_log, y_pred_log):
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(np.clip(y_pred_log, 0, None))
    return -mean_absolute_error(y_true_raw, y_pred_raw)

raw_mae_scorer = make_scorer(raw_neg_mae_score_func)


def _rank_weights(y_raw: np.ndarray, cap: float = 20.0) -> np.ndarray:
    """
    R2 v3-4: Rank-based weights.  Rows with higher AQI get higher weight,
    but the mapping is smooth (no hard thresholds) and capped at `cap`.
    This avoids the cliff between 199 and 201 that hard-coded tier weights create.
    """
    ranks = pd.Series(y_raw).rank(pct=True).values          # 0..1
    w = 1.0 + (cap - 1.0) * (ranks ** 2)                    # quadratic ramp
    return np.clip(w, 1.0, cap)


def train_rf(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Random Forest — {horizon}h Horizon\n{'=' * 75}")
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

    # ── 3. Hyperparameter search ──────────────────────────────────────────────
    # R2 v3-1: Wider grid
    param_dist = {
        "n_estimators":      [100,200,300],   # R2 v3-1
        "max_depth":         [20, 28, 35, None],           # FIX: dropped 15 (too shallow for 151 features)
        "min_samples_leaf":  [2, 3, 4, 6, 8],              # FIX: added 2 for better spike capture
        "min_samples_split": [4, 6, 10, 14],
        "max_features":      ["sqrt", 0.2, 0.3, 0.4, 0.5], # R2 v3-1
    }
    base_rf   = RandomForestRegressor(
        random_state=42, n_jobs=1,
        max_samples=0.90,   # R2 v3-3: was 0.85
    )
    tuning_cv = TimeSeriesSplit(n_splits=3, gap=min(horizon, 24))

    sw_search = _rank_weights(y_train_raw.values, cap=10.0)  # FIX: reduced cap 20→10; 20x was over-emphasizing spikes at the cost of overall R²

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=12,              # R2 v3-2: was 8
        cv=tuning_cv,
        scoring=raw_mae_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print("Running hyperparameter search on original (non-augmented) train set…")
    search.fit(X_train, y_train_log, sample_weight=sw_search)
    best_params = search.best_params_
    print(f"Best params: {best_params}")

    best_idx = search.best_index_
    cv_mae   = -search.cv_results_["mean_test_score"][best_idx]
    cv_std   =  search.cv_results_["std_test_score"][best_idx]

    # ── 4. Spike augmentation (only for the final model fit) ──────────────────
    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train, y_train_log,
        y_train_raw=y_train_raw,
        spike_threshold=150,
        target_spike_fraction=0.20,  # FIX: raised from 0.15 — more spike coverage
    )

    y_aug_raw_vals = np.expm1(y_train_aug.values)
    sample_weights = _rank_weights(y_aug_raw_vals, cap=10.0)  # FIX: matches search cap

    # ── 5. Final model fit ────────────────────────────────────────────────────
    final_params = {**best_params, "n_estimators": min(best_params["n_estimators"], 150)}
    model = RandomForestRegressor(
        **final_params,
        max_samples=0.90,   # R2 v3-3
        random_state=42,
        n_jobs=-1,
    )
    print(f"Fitting final RF (n_estimators={final_params['n_estimators']}) on augmented train set…")
    model.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)

    # ── 6. Conformal calibration ──────────────────────────────────────────────
    cal_preds_log = np.clip(model.predict(X_cal), 0, None)
    cal_preds_raw = np.expm1(cal_preds_log)
    margin        = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_preds_raw))

    # ── 7. Test inference ─────────────────────────────────────────────────────
    preds_log = np.clip(model.predict(X_test), 0, None)
    preds_raw = np.clip(np.expm1(preds_log), 0, 500)
    y_arr     = y_test_raw.values

    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    # ── 8. Save model to MongoDB (STORE 1) ────────────────────────────────────
    save_model_artifact(
        model_name="RandomForest",
        horizon=horizon,
        artifact={
            "model":            model,
            "feature_names":    list(X_train.columns),
            "conformal_margin": float(margin),
            "use_log":          True,
        },
        run_id=run_id,
    )

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    test_rmse      = root_mean_squared_error(y_arr, preds_raw)
    test_mae       = mean_absolute_error(y_arr, preds_raw)
    test_median_ae = median_absolute_error(y_arr, preds_raw)
    test_r2        = r2_score(y_arr, preds_raw)
    test_evs       = explained_variance_score(y_arr, preds_raw)
    test_mape      = float(np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100)

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

    # ── 10. Persist predictions & residuals to MongoDB (STORE 1) ─────────────
    save_predictions("RandomForest", horizon, y_arr, preds_raw, pi_lower, pi_upper, run_id)
    save_residuals("RandomForest", horizon, y_arr, preds_raw, run_id)

    # ── 11. Feature importance → MongoDB (STORE 1) ────────────────────────────
    importance = model.feature_importances_.tolist()
    save_feature_list(
        model="RandomForest",
        horizon=horizon,
        feature_names=list(X_train.columns),
        importance=importance,
        dropped=dropped_cols,
        run_id=run_id,
    )

    # ── 12. SHAP → MongoDB (STORE 1) ──────────────────────────────────────────
    top_20 = []
    if COMPUTE_SHAP:
        import shap
        rng        = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_test), size=min(100, len(X_test)), replace=False)
        X_sample   = X_test.iloc[sample_idx]

        explainer        = shap.TreeExplainer(model)
        shap_explanation = explainer(X_sample)
        mean_abs_shap    = np.abs(shap_explanation.values).mean(axis=0)

        save_shap(
            model="RandomForest",
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
        save_shap_plot_png("RandomForest", horizon, fig, run_id)
    else:
        imp_df = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        top_20 = imp_df.nlargest(20, "importance")["feature"].tolist()

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "RF", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 13. Metrics dict → MongoDB (STORE 1) ──────────────────────────────────
    metrics = {
        "model":                        "RandomForest",
        "horizon":                      f"{horizon}h",
        "training_target":              "log1p(AQI)",
        "cv_mean_val_mae_raw":           float(cv_mae),
        "cv_std_val_mae_raw":            float(cv_std),
        "test_rmse":                    float(test_rmse),
        "test_mae":                     float(test_mae),
        "test_median_ae":               float(test_median_ae),
        "test_mape":                    float(test_mape),
        "test_r2":                      float(test_r2),
        "test_explained_variance":      float(test_evs),
        "baseline_lag_col":             lag_col or "none",
        "baseline_horizon_mae":         float(p_mae),
        "baseline_horizon_r2":          float(p_r2),
        "forecast_skill_score":         float(skill),
        "r2_improvement_vs_baseline":   float(r2_imp),
        "conformal_margin":             float(margin),
        "conformal_margin_width":       float(margin),
        "conformal_global_coverage":    observed_coverage,
        "conformal_average_width":      avg_interval,
        "conformal_coverage_gt150":     float(cov_150),
        "conformal_n_gt150":            n_gt150,
        "conformal_coverage_gt200":     float(cov_200),
        "conformal_n_gt200":            n_gt200,
        "error_by_band":                stratified_bands,
        "quantile_errors":              quantile_errors,
        **events_150,
        **events_200,
    }

    save_metrics("RandomForest", horizon, metrics, run_id)
    return metrics


if __name__ == "__main__":
    results = {}
    for h in (24, 48, 72):
        results[f"{h}h"] = train_rf(h)

    print("\n" + "=" * 145)
    print("FINAL SUMMARY — RANDOM FOREST  (predictions on raw-AQI scale)")
    print("=" * 145)
    print(f"{'Horizon':<8}{'Test MAE':>10}{'Test MedAE':>12}{'Test R²':>10}"
          f"{'Skill':>10}{'Coverage':>12}{'Cov>150':>10}{'Cov>200':>10}")
    print("-" * 145)
    for horizon, m in results.items():
        print(
            f"{horizon:<8}"
            f"{m['test_mae']:>10.1f}"
            f"{m['test_median_ae']:>12.1f}"
            f"{m['test_r2']:>10.3f}"
            f"{m['forecast_skill_score']:>10.3f}"
            f"{m['conformal_global_coverage']*100:>11.1f}%"
            f"{m['conformal_coverage_gt150']*100:>9.1f}%"
            f"{m['conformal_coverage_gt200']*100:>9.1f}%"
        )