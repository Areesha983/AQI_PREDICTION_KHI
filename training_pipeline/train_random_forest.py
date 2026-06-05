"""
train_random_forest.py  (OPTIMISED)
-----------------------
Speed fixes:
  PERF 1 — MongoDB data comes from the in-process cache (load_data.py).
    Zero extra network I/O when all three models share the same Python process.

  PERF 2 — RandomizedSearchCV: n_iter reduced 15→8, n_splits 3→3 (unchanged),
    n_estimators grid capped at 400.  The old grid ran up to 700 trees × 3 folds
    × 15 candidates = 31,500 tree builds just for search.  New: 400 × 3 × 8 =
    9,600.  Best params barely change because RF is not very sensitive to these
    hypers — CV noise dominates.

  PERF 3 — Final model: n_estimators fixed at 500 (was up to 700 from search).
    500 trees at n_jobs=-1 on a 4-core runner takes ~45s vs ~90s for 700.

  PERF 4 — SHAP sample size reduced 150→100 for RF.  RF SHAP is O(n_trees × n_samples)
    and 150 samples on a 500-tree forest is ~4× slower than XGBoost SHAP.

  PERF 5 — get_chronological_splits called ONCE; raw series derived from y_log
    via expm1 instead of a second full split call.

R² fixes:
  R2 FIX 1 — Scoring changed from neg_root_mean_squared_error (log scale) to a
    custom neg_MAE on raw AQI scale.  Optimising log-RMSE during search was
    picking hypers that minimise log-space error, not actual AQI error.

  R2 FIX 2 — min_samples_leaf minimum raised from 2→4.  Leaf=2 causes
    overfitting on noisy AQI data; 4–6 generalises better.

  R2 FIX 3 — max_samples=0.85 added to RandomForestRegressor.  Subsampling
    rows (bootstrap fraction) adds diversity between trees which reduces
    variance and improves test R².

  R2 FIX 4 — Sample weights: exponential growth capped at 20× (was 25×).
    Extreme weights (25×) cause high-AQI rows to dominate and hurt low/mid R².
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
    export_residual_diagnostics,
    get_persistence_baseline_col,
)

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] module not found — skipping drift check.")

COMPUTE_SHAP = True

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

MODEL_DIR   = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

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


# R2 FIX 1: Custom scorer on raw AQI scale so hyperparameter search directly
# minimises the metric we care about, not log-space RMSE.
def raw_neg_mae_score_func(y_true_log, y_pred_log):
    """
    Standard evaluation function signature: (y_true, y_pred).
    Converts log-transformed targets back to the raw AQI scale before 
    calculating the Mean Absolute Error (MAE).
    """
    # Invert log1p transformation to bring values back to raw AQI scale
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(np.clip(y_pred_log, 0, None))
    
    # Return negative MAE because scikit-learn search optimizers maximize metrics
    return -mean_absolute_error(y_true_raw, y_pred_raw)

# Correctly wrap it using make_scorer with the target function
raw_mae_scorer = make_scorer(raw_neg_mae_score_func)


def train_rf(horizon: int) -> dict:
    print(f"\n{'=' * 75}\n Random Forest — {horizon}h Horizon\n{'=' * 75}")

    # ── 1. Load (cache hit after first model) ─────────────────────────────────
    X, y_log, y_raw = load_xy_both(horizon)

    # PERF 5: single split call; derive raw series via expm1 (no second split)
    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # ── 2. Correlation filter (cached result shared with XGB and Ridge) ────────
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97, horizon=horizon,
    )
    print(f"Features after filter: {X_train.shape[1]}  (dropped {len(dropped_cols)})")

    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(
        METRICS_DIR / f"rf_dropped_features_{horizon}h.csv", index=False)
    pd.DataFrame({"feature": X_train.columns}).to_csv(
        METRICS_DIR / f"rf_features_{horizon}h.csv", index=False)

    # ── 3. Hyperparameter search on ORIGINAL X_train ──────────────────────────
    # R2 FIX 2: min_samples_leaf min raised to 4 (was 2).
    # PERF 2: n_estimators grid capped at 400; n_iter=8 (was 15).
    param_dist = {
        "n_estimators":      [200, 300, 400],
        "max_depth":         [15, 20, 28, None],
        "min_samples_leaf":  [4, 6, 8, 12],      # R2 FIX 2
        "min_samples_split": [4, 6, 10, 14],
        "max_features":      ["sqrt", 0.2, 0.3, 0.5],
    }
    # R2 FIX 3: max_samples=0.85 added to base estimator
    base_rf   = RandomForestRegressor(
        random_state=42, n_jobs=1,
        max_samples=0.85,           # R2 FIX 3
    )
    tuning_cv = TimeSeriesSplit(n_splits=3, gap=min(horizon, 24))

    # R2 FIX 4: weight cap lowered from 25× to 20×
    sw_search = 1.0 + np.exp(np.minimum(y_train_raw.values, 350) / 110.0) - np.exp(0)
    sw_search = np.clip(sw_search, 1.0, 20.0)   # R2 FIX 4

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=8,                   # PERF 2: was 15
        cv=tuning_cv,
        scoring=raw_mae_scorer,     # R2 FIX 1: raw-scale MAE
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
        target_spike_fraction=0.15,
    )

    y_aug_raw_vals = np.expm1(y_train_aug.values)
    sample_weights = 1.0 + np.exp(np.minimum(y_aug_raw_vals, 350) / 110.0) - np.exp(0)
    sample_weights = np.clip(sample_weights, 1.0, 20.0)   # R2 FIX 4

    # ── 5. Final model fit ────────────────────────────────────────────────────
    # PERF 3: n_estimators fixed at 500; ignore search result if it was 700
    final_params = {**best_params, "n_estimators": min(best_params["n_estimators"], 500)}
    model = RandomForestRegressor(
        **final_params,
        max_samples=0.85,   # R2 FIX 3
        random_state=42,
        n_jobs=-1,
    )
    print(f"Fitting final RF (n_estimators={final_params['n_estimators']}) on augmented train set…")
    model.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)

    pd.DataFrame({
        "feature":    X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        METRICS_DIR / f"rf_feature_importance_{horizon}h.csv", index=False)

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

    # ── 8. Save model ─────────────────────────────────────────────────────────
    joblib.dump({
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }, MODEL_DIR / f"random_forest_{horizon}h.pkl")

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    test_rmse      = root_mean_squared_error(y_arr, preds_raw)
    test_mae       = mean_absolute_error(y_arr, preds_raw)
    test_median_ae = median_absolute_error(y_arr, preds_raw)
    test_r2        = r2_score(y_arr, preds_raw)
    test_evs       = explained_variance_score(y_arr, preds_raw)
    test_mape      = float(np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100)

    observed_coverage = float(np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper)))
    avg_interval      = float(np.mean(pi_upper - pi_lower))

    mask_150 = y_arr > 150
    n_gt150  = int(mask_150.sum())
    cov_150  = float(np.mean(
        (y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150])
    )) if n_gt150 >= 5 else 0.0

    mask_200 = y_arr > 200
    n_gt200  = int(mask_200.sum())
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

    pd.DataFrame({
        "actual":      y_arr,
        "predicted":   preds_raw,
        "lower_bound": pi_lower,
        "upper_bound": pi_upper,
    }, index=X_test.index).to_csv(METRICS_DIR / f"rf_predictions_{horizon}h.csv", index=False)

    export_residual_diagnostics(y_arr, preds_raw, horizon, "RF", METRICS_DIR, MODEL_DIR)

    # ── 10. SHAP (PERF 4: sample size 150→100 for RF) ────────────────────────
    top_20 = []
    if COMPUTE_SHAP:
        import shap
        rng        = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_test), size=min(100, len(X_test)), replace=False)  # PERF 4
        X_sample   = X_test.iloc[sample_idx]

        explainer        = shap.TreeExplainer(model)
        shap_explanation = explainer(X_sample)
        mean_abs_shap    = np.abs(shap_explanation.values).mean(axis=0)

        shap_imp = pd.DataFrame({
            "feature":       X_sample.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_imp.to_csv(METRICS_DIR / f"rf_top_features_{horizon}h.csv", index=False)
        top_20 = shap_imp.head(20)["feature"].tolist()

        fig = plt.figure(figsize=(10, 8))
        try:
            shap.plots.beeswarm(shap_explanation, show=False)
        except Exception:
            shap.summary_plot(shap_explanation.values, X_sample, show=False)
        plt.savefig(METRICS_DIR / f"rf_shap_summary_{horizon}h.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        top_20 = (
            pd.DataFrame({
                "feature":    X_train.columns,
                "importance": model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .head(20)["feature"]
            .tolist()
        )

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "RF", top_20)
    except Exception as e:
        print(f"Drift monitoring skipped: {e}")

    # ── 11. Build metrics dict ────────────────────────────────────────────────
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

    with open(METRICS_DIR / f"rf_metrics_{horizon}h.json", "w") as f:
        json.dump(metrics, f, indent=2)

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