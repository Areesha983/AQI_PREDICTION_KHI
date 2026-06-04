"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.

FIXES APPLIED:
  FIX 1 — Feature vector zero-fill caused Ridge to always predict 500.
    Builds feature vector from latest MongoDB processed_features document,
    then overrides only the slider fields the user changed.

  FIX 2 — Metrics endpoint now reads from MongoDB model_metrics collection.
    The old code read local JSON files that don't exist on this machine.
    evaluate.py pushes to model_metrics with this structure:
      doc.performance_summary["{h}h"]["models"]["Random Forest"] = {
          "MAE": float, "R2": float, "MAPE": float, "Coverage": float
      }
    The endpoint fetches the most recent document and normalises it into
    the shape the dashboard charts expect.

  FIX 3 — MongoDB-backed latest_features endpoint for Streamlit UI.
"""

import os
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# ── PATH & ENV SETUP ─────────────────────────────────────────────────────────
API_DIR     = Path(__file__).resolve().parent
BASE_DIR    = API_DIR.parent
MODELS_DIR  = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"   # kept as fallback only

load_dotenv(BASE_DIR / ".env")
MONGO_URI  = os.getenv("MONGODB_URI")
DB_NAME    = "karachi_aqi"
FEAT_COL   = "processed_features"
METRICS_COL = "model_metrics"

app = Flask(__name__)
MODEL_CACHE = {}
VALID_MODELS   = {"random_forest", "xgboost", "ridge"}
VALID_HORIZONS = {24, 48, 72}

# ── MongoDB name → evaluate.py model_name mapping ────────────────────────────
# evaluate.py stores keys as "Random Forest", "XGBoost", "Ridge"
_API_TO_EVAL_NAME = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# ── In-memory caches (5-min TTL each) ────────────────────────────────────────
_FEATURE_CACHE: dict = {"doc": None, "ts": 0.0}
_METRICS_CACHE: dict = {"doc": None, "ts": 0.0}
_CACHE_TTL = 300


def _mongo_client():
    if not MONGO_URI:
        raise ValueError("MONGODB_URI is not set in environment / .env")
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)


# ── Feature context helpers ───────────────────────────────────────────────────
def _get_latest_feature_doc() -> dict | None:
    now = time.time()
    if _FEATURE_CACHE["doc"] is not None and (now - _FEATURE_CACHE["ts"]) < _CACHE_TTL:
        return _FEATURE_CACHE["doc"]
    try:
        client = _mongo_client()
        db  = client[DB_NAME]
        doc = db[FEAT_COL].find_one({}, sort=[("timestamp", -1)])
        if doc is None:
            doc = db[FEAT_COL].find_one({}, sort=[("datetime", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            _FEATURE_CACHE["doc"] = doc
            _FEATURE_CACHE["ts"]  = now
        return doc
    except PyMongoError as e:
        print(f"WARNING: feature store fetch failed: {e}")
        return None


def _build_feature_vector(expected_features: list, slider_overrides: dict) -> pd.DataFrame:
    """Use latest MongoDB doc as base; overlay slider values on lag-1 features."""
    latest_doc = _get_latest_feature_doc()
    base = {f: float(latest_doc.get(f, 0.0)) for f in expected_features} if latest_doc \
           else {f: 0.0 for f in expected_features}

    # Inline AQI calc — avoids importing feature_engineering at module level
    def _aqi_from_pm25(pm25: float) -> float:
        if np.isnan(pm25) or pm25 < 0:
            return float("nan")
        bps = [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
               (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
        for c_lo, c_hi, a_lo, a_hi in bps:
            if c_lo <= pm25 <= c_hi:
                return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
        return 500.0

    pm25 = float(slider_overrides.get("pm25",        base.get("pm25_lag_1",  75.0)))
    pm10 = float(slider_overrides.get("pm10",        base.get("pm10_lag_1", 140.0)))
    temp = float(slider_overrides.get("temperature", base.get("temperature_lag_1", 32.0)))
    hum  = float(slider_overrides.get("humidity",    base.get("humidity_lag_1",    65.0)))
    wind = float(slider_overrides.get("wind_speed",  base.get("wind_speed_lag_1",  10.0)))

    aqi_now = _aqi_from_pm25(pm25)
    if np.isnan(aqi_now):
        aqi_now = base.get("aqi_lag_1", 75.0)

    overrides = {
        "aqi_lag_1": aqi_now, "pm25_lag_1": pm25, "pm10_lag_1": pm10,
        "temperature_lag_1": temp, "humidity_lag_1": hum, "wind_speed_lag_1": wind,
        "interaction_pm25_humidity":     pm25 * hum,
        "interaction_pm25_wind_inverse": pm25 / (wind + 0.1),
        "temp_humidity":                 temp * hum,
        "heat_dryness":                  temp / (hum + 1),
        "wind_dispersal":                wind / (pm25 + 5),
        "temp_to_wind_ratio":            temp / (wind + 0.1),
        "humidity_to_wind_ratio":        hum  / (wind + 0.1),
    }
    for k, v in overrides.items():
        if k in base:
            base[k] = v

    return pd.DataFrame([base])[expected_features]


# ── Metrics helpers ───────────────────────────────────────────────────────────
def _get_latest_metrics_doc() -> dict | None:
    """
    Fetches the most recent model_metrics document from MongoDB.
    evaluate.py inserts one document per pipeline run with shape:
      { performance_summary: { "24h": { models: {...} }, "48h": ..., "72h": ... } }
    Returns the performance_summary dict, or None if unavailable.
    """
    now = time.time()
    if _METRICS_CACHE["doc"] is not None and (now - _METRICS_CACHE["ts"]) < _CACHE_TTL:
        return _METRICS_CACHE["doc"]
    try:
        client = _mongo_client()
        db  = client[DB_NAME]
        doc = db[METRICS_COL].find_one(
            {"pipeline_run_status": "SUCCESS"},
            sort=[("timestamp", -1)]
        )
        client.close()
        if doc and "performance_summary" in doc:
            summary = doc["performance_summary"]
            _METRICS_CACHE["doc"] = summary
            _METRICS_CACHE["ts"]  = now
            return summary
        return None
    except PyMongoError as e:
        print(f"WARNING: metrics fetch failed: {e}")
        return None


def _metrics_from_mongo(model_type: str) -> dict:
    """
    Builds per-horizon metrics dict for one model from the MongoDB summary.
    Falls back to local JSON files if MongoDB is unavailable (backwards compat).
    """
    eval_name   = _API_TO_EVAL_NAME.get(model_type, model_type.title())
    mongo_summary = _get_latest_metrics_doc()

    report = {}
    for h in VALID_HORIZONS:
        h_key = f"{h}h"
        mongo_ok = False

        if mongo_summary and h_key in mongo_summary:
            models_block = mongo_summary[h_key].get("models", {})
            m = models_block.get(eval_name, {})
            if m:
                report[str(h)] = {
                    "r2":       m.get("R2")       or 0.0,
                    "rmse":     m.get("RMSE")      or 0.0,
                    "mae":      m.get("MAE")       or 0.0,
                    "mape":     m.get("MAPE")      or 0.0,
                    "coverage": m.get("Coverage")  or 0.0,
                    "margin":   m.get("margin")    or 0.0,
                }
                mongo_ok = True

        if not mongo_ok:
            # Fallback: try local JSON file
            if model_type == "random_forest":
                prefix = "rf"
            elif model_type == "xgboost":
                prefix = "xgb"
            else:
                prefix = model_type
            local_path = METRICS_DIR / f"{prefix}_metrics_{h}h.json"
            try:
                with open(local_path) as f:
                    raw = json.load(f)
                report[str(h)] = {
                    "r2":       raw.get("test_r2",  0.0),
                    "rmse":     raw.get("test_rmse", 0.0),
                    "mae":      raw.get("test_mae",  0.0),
                    "mape":     raw.get("test_mape", 0.0),
                    "coverage": raw.get("conformal_global_coverage", 0.0),
                    "margin":   raw.get("conformal_margin_width",    0.0),
                }
            except Exception:
                report[str(h)] = {
                    "r2": 0.0, "rmse": 0.0, "mae": 0.0,
                    "mape": 0.0, "coverage": 0.0, "margin": 0.0,
                }
    return report


# ── Model artifact loader ─────────────────────────────────────────────────────
def load_prediction_artifacts(model_type: str, horizon: int):
    key = f"{model_type}_{horizon}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    path = MODELS_DIR / f"{model_type}_{horizon}h.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing: {path.name}")
    artifacts = joblib.load(path)
    MODEL_CACHE[key] = artifacts
    return artifacts


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health_check():
    found = {}
    for m in VALID_MODELS:
        found[m] = [f"{h}h" for h in VALID_HORIZONS
                    if (MODELS_DIR / f"{m}_{h}h.pkl").exists()]
    mongo_summary = _get_latest_metrics_doc()
    return jsonify({
        "status":                       "healthy",
        "service":                      "aqi-multi-model-forecasting-api",
        "warmed_models_in_cache":       list(MODEL_CACHE.keys()),
        "discovered_artifacts_on_disk": found,
        "mongo_connected":              MONGO_URI is not None,
        "metrics_in_mongo":             mongo_summary is not None,
    }), 200


@app.route("/latest_features", methods=["GET"])
def latest_features():
    doc = _get_latest_feature_doc()
    if doc is None:
        return jsonify({"error": "No feature data available from MongoDB."}), 503
    safe = ["timestamp", "datetime", "aqi_lag_1", "pm25_lag_1", "pm10_lag_1",
            "temperature_lag_1", "humidity_lag_1", "wind_speed_lag_1"]
    return jsonify({k: doc.get(k) for k in safe if k in doc}), 200


@app.route("/predict/<string:model_type>/<int:horizon>", methods=["POST"])
def predict_aqi(model_type: str, horizon: int):
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {list(VALID_MODELS)}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {list(VALID_HORIZONS)}"}), 400

    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Body must contain a 'features' key."}), 400

    try:
        artifacts         = load_prediction_artifacts(model_type, horizon)
        model             = artifacts["model"]
        expected_features = artifacts["feature_names"]
        conformal_margin  = artifacts["conformal_margin"]
        use_log           = artifacts.get("use_log", False)

        input_data     = _build_feature_vector(expected_features, payload["features"])
        raw_prediction = float(model.predict(input_data)[0])
        if use_log:
            raw_prediction = float(np.expm1(max(0.0, raw_prediction)))
        raw_prediction = float(np.clip(raw_prediction, 0.0, 500.0))

        lower = float(np.clip(raw_prediction - conformal_margin, 0.0, 500.0))
        upper = float(np.clip(raw_prediction + conformal_margin, 0.0, 500.0))

        if raw_prediction <= 50:    category = "Good"
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
        return jsonify({"error": str(fnf),
                        "hint": f"Run the training script for '{model_type}'."}), 500
    except Exception as e:
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500


@app.route("/metrics/<string:model_type>", methods=["GET"])
def get_model_metrics(model_type):
    """
    Returns per-horizon metrics for one model.
    Source priority: MongoDB model_metrics collection → local JSON fallback.
    """
    return jsonify(_metrics_from_mongo(model_type.lower()))


@app.route("/metrics/all", methods=["GET"])
def get_all_metrics():
    """
    Returns metrics for all three models in one call — lets the dashboard
    populate both charts with a single request instead of three.
    """
    return jsonify({m: _metrics_from_mongo(m) for m in VALID_MODELS})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)