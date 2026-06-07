"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.
api/app.py

KEY FIXES IN THIS VERSION
──────────────────────────
FIX 1 — METRICS PATH SIMPLIFIED
FIX 2 — MODEL-AWARE FEATURE VECTOR (_build_feature_vector)
FIX 3 — PREDICT ROUTE WIRED TO FIXED BUILDER
FIX 4 — NAMING CONSISTENCY
FIX 5 — RANDOM FOREST GUNICORN DEADLOCK
FIX 6 — XGBOOST RESIDUAL CORRECTOR APPLIED AT INFERENCE
FIX 7 — FULL TRACEBACK LOGGED ON PREDICT FAILURE
FIX 8 — RIDGE FEATURE IMPORTANCE ROUTE (NEW)
  Ridge does not compute SHAP values (no TreeExplainer). It stores coefficient
  magnitudes in model_features via save_feature_list(). The /features/<model>/<horizon>
  route reads that collection and returns records in the same format as /shap so
  the dashboard chart works without any changes.
FIX 9 — /features/ ROUTE NULL-GUARD + INDEX MAP CORRECTED (NEW)
  Importance values from MongoDB could theoretically be None in a corrupt/partial
  doc. The sort key and record builder now guard against None to prevent
  TypeError crashes. The root endpoint's endpoint map is also corrected to
  accurately reflect that /features/ works for all models, not only Ridge.
FIX 10 — DEAD _get_latest_metrics_doc STUB REMOVED (NEW)
  The function always returned None and was never called from any route.
  Keeping it created false impression of a second metrics code path; removed.
"""

import os
import sys
import time
import traceback
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from itertools import combinations

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# ── PATH & ENV SETUP ──────────────────────────────────────────────────────────
API_DIR  = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent

load_dotenv(BASE_DIR / ".env")
MONGO_URI   = os.getenv("MONGODB_URI")
DB_NAME     = "karachi_aqi"
FEAT_COL    = "processed_features"
METRICS_COL = "model_metrics"

_PIPELINE_DIR = BASE_DIR / "training_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_CACHE    : dict = {}
VALID_MODELS   = ["random_forest", "xgboost", "ridge"]
VALID_HORIZONS = {24, 48, 72}

_API_TO_MONGO_NAME: dict[str, str] = {
    "random_forest": "RandomForest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# ── In-memory caches (5-min TTL) ─────────────────────────────────────────────
_FEATURE_CACHE: dict = {"doc": None, "ts": 0.0}
_METRICS_CACHE: dict = {"doc": None, "ts": 0.0}
_CACHE_TTL = 300

# ── Metric key aliases ────────────────────────────────────────────────────────
_METRIC_ALIASES: dict[str, list[str]] = {
    "r2":       ["R2", "r2", "test_r2", "R²", "r2_score"],
    "rmse":     ["RMSE", "rmse", "test_rmse", "root_mean_squared_error"],
    "mae":      ["MAE", "mae", "test_mae", "mean_absolute_error"],
    "mape":     ["MAPE", "mape", "test_mape"],
    "coverage": ["Coverage", "coverage", "conformal_global_coverage", "global_coverage"],
    "margin":   ["margin", "Margin", "conformal_margin_width", "conformal_margin"],
}


# ── Utility helpers ───────────────────────────────────────────────────────────
def _mongo_client() -> MongoClient:
    if not MONGO_URI:
        raise ValueError("MONGODB_URI is not set in environment / .env")
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000, socketTimeoutMS=120_000)


def _extract_metric(doc: dict, key: str) -> float:
    for alias in _METRIC_ALIASES.get(key, [key]):
        v = doc.get(alias)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Feature helpers ───────────────────────────────────────────────────────────
def _get_latest_feature_doc() -> dict | None:
    now = time.time()
    if _FEATURE_CACHE["doc"] is not None and (now - _FEATURE_CACHE["ts"]) < _CACHE_TTL:
        return _FEATURE_CACHE["doc"]
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        col    = db[FEAT_COL]
        doc    = col.find_one({}, sort=[("timestamp", -1)])
        if doc is None:
            doc = col.find_one({}, sort=[("datetime", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            _FEATURE_CACHE["doc"] = doc
            _FEATURE_CACHE["ts"]  = now
            print(f"[feature-cache] Loaded doc — sample keys: {list(doc.keys())[:10]} ...")
        else:
            print("[feature-cache] WARNING: processed_features collection returned no documents.")
        return doc
    except PyMongoError as e:
        print(f"[feature-cache] ERROR: MongoDB fetch failed — {e}")
        return None


def _reconstruct_interaction_terms(base: dict, interaction_cols: list[str]) -> dict:
    extra: dict = {}
    present = [c for c in interaction_cols if c in base]
    for a, b in combinations(present, 2):
        col_name = f"{a} {b}"
        extra[col_name] = base.get(a, 0.0) * base.get(b, 0.0)
    return extra


def _build_feature_vector(artifact: dict, slider_overrides: dict) -> pd.DataFrame:
    expected_features : list[str] = artifact["feature_names"]
    use_interactions  : bool      = artifact.get("use_interactions", False)
    interaction_cols  : list[str] = artifact.get("interaction_cols", [])

    latest_doc = _get_latest_feature_doc()

    base: dict = (
        {f: _safe_float(latest_doc.get(f), 0.0) for f in expected_features}
        if latest_doc
        else {f: 0.0 for f in expected_features}
    )

    def _aqi_from_pm25(pm25: float) -> float:
        if np.isnan(pm25) or pm25 < 0:
            return float("nan")
        bps = [
            (0.0,   12.0,  0,   50),   (12.1,  35.4,  51,  100),
            (35.5,  55.4,  101, 150),  (55.5,  150.4, 151, 200),
            (150.5, 250.4, 201, 300),  (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_lo, c_hi, a_lo, a_hi in bps:
            if c_lo <= pm25 <= c_hi:
                return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
        return 500.0

    pm25 = _safe_float(slider_overrides.get("pm25",        base.get("pm25_lag_1")),   75.0)
    pm10 = _safe_float(slider_overrides.get("pm10",        base.get("pm10_lag_1")),  140.0)
    temp = _safe_float(slider_overrides.get("temperature", base.get("temperature_lag_1")), 32.0)
    hum  = _safe_float(slider_overrides.get("humidity",    base.get("humidity_lag_1")),    65.0)
    wind = _safe_float(slider_overrides.get("wind_speed",  base.get("wind_speed_lag_1")),  10.0)

    aqi_now = _aqi_from_pm25(pm25)
    if np.isnan(aqi_now):
        aqi_now = _safe_float(base.get("aqi_lag_1"), 75.0)

    lag1_overrides: dict = {
        "aqi_lag_1":         aqi_now,
        "pm25_lag_1":        pm25,
        "pm10_lag_1":        pm10,
        "temperature_lag_1": temp,
        "humidity_lag_1":    hum,
        "wind_speed_lag_1":  wind,
        "interaction_pm25_humidity":     pm25 * hum / 100.0,
        "interaction_pm25_wind_inverse": pm25 / (wind + 0.5),
        "temp_humidity":                 temp * hum,
        "heat_dryness":                  temp / (hum + 1.0),
        "wind_dispersal":                wind / (pm25 + 5.0),
        "temp_to_wind_ratio":            temp / (wind + 0.1),
        "humidity_to_wind_ratio":        hum  / (wind + 0.1),
    }
    for k, v in lag1_overrides.items():
        if k in base:
            base[k] = v

    if use_interactions and interaction_cols:
        full_lookup = {**base, **lag1_overrides}
        cross_terms = _reconstruct_interaction_terms(full_lookup, interaction_cols)
        base.update(cross_terms)

    row = {f: _safe_float(base.get(f), 0.0) for f in expected_features}
    df  = pd.DataFrame([row])[expected_features]
    df  = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ── Model artifact loader ─────────────────────────────────────────────────────
def load_prediction_artifacts(model_type: str, horizon: int) -> dict:
    cache_key = f"{model_type}_{horizon}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    store_name = _API_TO_MONGO_NAME.get(model_type)
    if not store_name:
        raise ValueError(f"Unknown model type: {model_type!r}")

    from mongo_store import load_model_artifact

    artifact = load_model_artifact(model_name=store_name, horizon=horizon)
    if artifact is None:
        raise FileNotFoundError(
            f"No artifact found in MongoDB for model='{store_name}', horizon={horizon}h. "
            f"Run the training script for '{model_type}' first."
        )

    model_obj = artifact.get("model")
    if model_obj is not None and hasattr(model_obj, "n_jobs"):
        model_obj.n_jobs = 1
        print(f"[model-cache] Patched n_jobs=1 on {store_name} {horizon}h (gunicorn safety)")

    MODEL_CACHE[cache_key] = artifact
    n_features = len(artifact.get("feature_names", []))
    print(f"[model-cache] Loaded '{store_name}' {horizon}h from MongoDB ✓  ({n_features} features)")
    return artifact


# ── Metrics helpers ───────────────────────────────────────────────────────────
def _metrics_from_mongo(model_type: str) -> dict:
    store_name = _API_TO_MONGO_NAME.get(model_type)
    if not store_name:
        return {str(h): {"error": f"Unknown model type: {model_type!r}"} for h in sorted(VALID_HORIZONS)}

    report: dict = {}
    for h in sorted(VALID_HORIZONS):
        h_key = f"{h}h"
        try:
            client = _mongo_client()
            db     = client[DB_NAME]
            m_doc  = db[METRICS_COL].find_one(
                {"model": store_name, "horizon_h": h},
                {"_id": 0},
            )
            client.close()

            if m_doc:
                report[str(h)] = {
                    "r2":       _extract_metric(m_doc, "r2"),
                    "rmse":     _extract_metric(m_doc, "rmse"),
                    "mae":      _extract_metric(m_doc, "mae"),
                    "mape":     _extract_metric(m_doc, "mape"),
                    "coverage": _extract_metric(m_doc, "coverage"),
                    "margin":   _extract_metric(m_doc, "margin"),
                }
                print(f"[metrics] {store_name}/{h_key} loaded from model_metrics ✓")
            else:
                report[str(h)] = {"error": (
                    f"No metrics found in MongoDB for model='{store_name}', horizon={h_key}. "
                    "Run the training script first."
                )}
                print(f"[metrics] {store_name}/{h_key} — MongoDB miss.")

        except PyMongoError as e:
            print(f"[metrics] {store_name}/{h_key} query failed: {e}")
            report[str(h)] = {"error": f"MongoDB query failed: {e}"}

    return report


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "AQI Prediction API is active.",
        "service": "Karachi AQI Forecasting API",
        "status":  "running",
        "endpoints": {
            # ── health / data ──────────────────────────────────────────────
            "health":              "/health",
            "latest_features":     "/latest_features",
            "latest_realtime":     "/latest_realtime",
            # ── predict ───────────────────────────────────────────────────
            "predict_24h_rf":      "/predict/random_forest/24",
            "predict_24h_xgb":     "/predict/xgboost/24",
            "predict_24h_ridge":   "/predict/ridge/24",
            # ── metrics ───────────────────────────────────────────────────
            "metrics_rf":          "/metrics/random_forest",
            "metrics_xgb":         "/metrics/xgboost",
            "metrics_ridge":       "/metrics/ridge",
            "all_metrics":         "/metrics/all",
            # ── SHAP (RF + XGBoost) ───────────────────────────────────────
            "shap_rf_24h":         "/shap/random_forest/24",
            "shap_xgb_24h":        "/shap/xgboost/24",
            # ── feature importance (all models, all horizons) ─────────────
            # FIX 9: corrected — this route works for rf/xgboost/ridge at
            # 24/48/72h. Ridge uses |coef| magnitudes; RF/XGB use gain
            # importances. Returns {"feature": str, "mean_abs_shap": float}
            # records for drop-in dashboard compatibility.
            "features_24h":        "/features/<model>/24",
            "features_48h":        "/features/<model>/48",
            "features_72h":        "/features/<model>/72",
            # ── debug ─────────────────────────────────────────────────────
            "debug_artifacts":     "/debug/artifacts",
            "debug_features":      "/debug/features_raw",
            "debug_metrics":       "/debug/metrics_raw",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    feat_doc = _get_latest_feature_doc()
    return jsonify({
        "status":                  "healthy",
        "service":                 "aqi-multi-model-forecasting-api",
        "storage_backend":         "MongoDB Atlas (no local files)",
        "warmed_models_in_cache":  list(MODEL_CACHE.keys()),
        "mongo_connected":         MONGO_URI is not None,
        "features_in_mongo":       feat_doc is not None,
        "feature_doc_sample_keys": list(feat_doc.keys())[:8] if feat_doc else [],
    }), 200


@app.route("/latest_features", methods=["GET"])
def latest_features():
    doc = _get_latest_feature_doc()
    if doc is None:
        return jsonify({"error": "No feature data in processed_features collection."}), 503
    safe_keys = [
        "timestamp", "datetime",
        "aqi_lag_1", "pm25_lag_1", "pm10_lag_1",
        "temperature_lag_1", "humidity_lag_1", "wind_speed_lag_1",
    ]
    return jsonify({k: doc.get(k) for k in safe_keys if k in doc}), 200


@app.route("/latest_realtime", methods=["GET"])
def latest_realtime():
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        col    = db["realtime_observations"]
        doc    = col.find_one({}, sort=[("datetime", -1)])
        client.close()

        if doc is None:
            return jsonify({
                "error": "realtime_observations is empty. update_realtime.py has not run yet."
            }), 503

        doc.pop("_id", None)
        doc.pop("fetched_at", None)
        return jsonify(doc), 200

    except PyMongoError as e:
        return jsonify({"error": f"realtime_observations query failed: {str(e)}"}), 500


@app.route("/predict/<string:model_type>/<int:horizon>", methods=["POST"])
def predict_aqi(model_type: str, horizon: int):
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {VALID_MODELS}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {sorted(VALID_HORIZONS)}"}), 400

    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Body must contain a 'features' key."}), 400

    try:
        artifact         = load_prediction_artifacts(model_type, horizon)
        model            = artifact["model"]
        conformal_margin = artifact["conformal_margin"]
        use_log          = artifact.get("use_log", False)

        input_df       = _build_feature_vector(artifact, payload["features"])
        raw_prediction = float(model.predict(input_df)[0])

        if use_log:
            raw_prediction = float(np.expm1(max(0.0, raw_prediction)))

        corrector = artifact.get("corrector")
        if corrector is not None:
            try:
                correction     = float(corrector.predict(input_df)[0])
                raw_prediction = float(np.clip(raw_prediction + correction, 0.0, 500.0))
                print(f"[predict] {model_type}/{horizon}h corrector applied: Δ={correction:+.2f}")
            except Exception as corr_err:
                print(f"[predict] {model_type}/{horizon}h corrector failed (skipped): {corr_err}")

        raw_prediction = float(np.clip(raw_prediction, 0.0, 500.0))
        lower = float(np.clip(raw_prediction - conformal_margin, 0.0, 500.0))
        upper = float(np.clip(raw_prediction + conformal_margin, 0.0, 500.0))

        if   raw_prediction <= 50:  category = "Good"
        elif raw_prediction <= 100: category = "Moderate"
        elif raw_prediction <= 150: category = "Unhealthy for Sensitive Groups"
        elif raw_prediction <= 200: category = "Unhealthy"
        elif raw_prediction <= 300: category = "Very Unhealthy"
        else:                       category = "Hazardous"

        return jsonify({
            "model_engine":             model_type,
            "horizon_hours":            horizon,
            "aqi_prediction":           round(raw_prediction, 1),
            "lower_bound_95ci":         round(lower, 1),
            "upper_bound_95ci":         round(upper, 1),
            "conformal_margin_applied": round(conformal_margin, 1),
            "epa_category":             category,
            "status":                   "success",
        }), 200

    except FileNotFoundError as fnf:
        return jsonify({
            "error": str(fnf),
            "hint":  f"Train '{model_type}' and ensure mongo_store.save_model_artifact() ran.",
        }), 500
    except Exception as e:
        print(f"[predict] EXCEPTION for {model_type}/{horizon}h:\n{traceback.format_exc()}")
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500


@app.route("/metrics/<string:model_type>", methods=["GET"])
def get_model_metrics(model_type: str):
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {VALID_MODELS}"}), 400
    return jsonify(_metrics_from_mongo(model_type))


@app.route("/metrics/all", methods=["GET"])
def get_all_metrics():
    return jsonify({m: _metrics_from_mongo(m) for m in VALID_MODELS})


@app.route("/shap/<string:model_type>/<int:horizon>", methods=["GET"])
def get_shap(model_type: str, horizon: int):
    """Returns top-10 SHAP records from model_shap collection (RF + XGBoost).
    Ridge has no TreeExplainer so model_shap is never populated for it;
    the dashboard should fall back to /features/ridge/<horizon> in that case.
    """
    model_type = model_type.lower()
    store_name = _API_TO_MONGO_NAME.get(model_type)
    if not store_name:
        return jsonify({"error": f"Unknown model type: {model_type!r}. Choose from: {VALID_MODELS}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {sorted(VALID_HORIZONS)}"}), 400
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db["model_shap"].find_one(
            {"model": store_name, "horizon_h": horizon},
            {"_id": 0, "records": 1},
        )
        client.close()
        if not doc:
            return jsonify({
                "records": [],
                "message": f"No SHAP data found for {store_name} {horizon}h. Run training pipeline first.",
            }), 200
        records = doc.get("records", [])[:10]
        return jsonify({"model": store_name, "horizon_h": horizon, "records": records}), 200
    except Exception as e:
        return jsonify({"error": f"SHAP fetch failed: {str(e)}"}), 500


@app.route("/features/<string:model_type>/<int:horizon>", methods=["GET"])
def get_feature_importance(model_type: str, horizon: int):
    """
    Feature importance endpoint — works for ALL three models.

    - Ridge: returns |coefficient| magnitudes stored by train_ridge.py via
      save_feature_list(importance=[...]) — never has SHAP.
    - XGBoost: returns gain-based importances stored by train_xgboost.py via
      save_feature_list(importance=model.feature_importances_.tolist()).
      (FIX 9: previously importance was omitted, resulting in [] being stored.)
    - RandomForest: returns gain-based importances stored similarly.

    All three return records in the same format as /shap for drop-in dashboard
    compatibility:
        {"feature": str, "mean_abs_shap": float}

    The field is named mean_abs_shap for schema consistency; its value is
    |coef| for Ridge and feature_importances_ gain for RF/XGBoost.

    FIX 9 — None guard: importance values are guarded against None before
    abs() and float() conversion to prevent TypeError on corrupt/partial docs.
    """
    model_type = model_type.lower()
    store_name = _API_TO_MONGO_NAME.get(model_type)
    if not store_name:
        return jsonify({"error": f"Unknown model type: {model_type!r}. Choose from: {VALID_MODELS}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {sorted(VALID_HORIZONS)}"}), 400
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db["model_features"].find_one(
            {"model": store_name, "horizon_h": horizon},
            {"_id": 0, "feature_names": 1, "importance": 1},
        )
        client.close()

        if not doc:
            return jsonify({
                "records": [],
                "message": (
                    f"No feature data found for {store_name} {horizon}h. "
                    "Run the training pipeline first."
                ),
            }), 200

        feature_names = doc.get("feature_names", [])
        importance    = doc.get("importance", [])

        if not feature_names or not importance:
            return jsonify({
                "records": [],
                "message": (
                    f"Feature list or importance array is empty for {store_name} {horizon}h. "
                    "Retrain the model to populate importance values."
                ),
            }), 200

        # FIX 9: Guard against None values in importance (corrupt/partial doc).
        # Without this guard, abs(None) raises TypeError and the route crashes.
        pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: abs(x[1]) if x[1] is not None else 0.0,
            reverse=True,
        )[:10]

        records = [
            {
                "feature":       f,
                "mean_abs_shap": float(abs(v)) if v is not None else 0.0,
            }
            for f, v in pairs
        ]

        return jsonify({
            "model":      store_name,
            "horizon_h":  horizon,
            "value_type": "coefficient_magnitude" if model_type == "ridge" else "gain_importance",
            "records":    records,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Feature fetch failed: {str(e)}"}), 500


@app.route("/debug/metrics_raw", methods=["GET"])
def debug_metrics_raw():
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        docs   = list(db[METRICS_COL].find({}, {"_id": 0}).sort("updated_at", -1).limit(5))
        client.close()
        return jsonify({"found": bool(docs), "count": len(docs), "docs": docs}), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/features_raw", methods=["GET"])
def debug_features_raw():
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db[FEAT_COL].find_one({}, sort=[("timestamp", -1)])
        if doc is None:
            doc = db[FEAT_COL].find_one({}, sort=[("datetime", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            return jsonify({"found": True, "doc": doc}), 200
        return jsonify({
            "found": False,
            "hint":  "processed_features is empty. Run the feature pipeline first.",
        }), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/artifacts", methods=["GET"])
def debug_artifacts():
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        docs   = list(db["model_artifacts"].find(
            {},
            {"_id": 0, "model": 1, "horizon_h": 1, "updated_at": 1, "feature_names": 1},
        ))
        client.close()
        for d in docs:
            if "feature_names" in d:
                d["feature_count"] = len(d.pop("feature_names"))
        return jsonify({"found": bool(docs), "count": len(docs), "artifacts": docs}), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)