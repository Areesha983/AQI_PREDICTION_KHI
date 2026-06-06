"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.

CHANGES vs previous version:
  MONGO FIX — load_prediction_artifacts() was reading from local .pkl files
    (MODELS_DIR / f"{model_type}_{horizon}h.pkl"). All model artifacts are now
    stored in MongoDB Atlas (model_artifacts collection) by the training scripts
    via mongo_store.save_model_artifact(). This version uses
    mongo_store.load_model_artifact() exclusively; no local filesystem reads.

  The artifact dict stored in MongoDB has this shape (written by every trainer):
    {
      "model":            <fitted estimator>,
      "feature_names":    [...],
      "conformal_margin": float,
      "use_log":          bool,
      ...trainer-specific keys...
    }
  This is identical to what the old joblib.load() returned, so no other
  call-site changes were needed.

RENDER FIXES:
  - Added flask-cors: CORS(app) so Streamlit frontend can call this API
  - Removed duplicate home() route that conflicted with index()
  - Set debug=False for production safety
"""

import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS                          # ← RENDER FIX 1: import CORS
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# ── PATH & ENV SETUP ─────────────────────────────────────────────────────────
API_DIR  = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent

load_dotenv(BASE_DIR / ".env")
MONGO_URI   = os.getenv("MONGODB_URI")
DB_NAME     = "karachi_aqi"
FEAT_COL    = "processed_features"
METRICS_COL = "model_metrics"

app = Flask(__name__)
CORS(app)                                            # ← RENDER FIX 2: enable CORS

MODEL_CACHE    = {}   # key: "random_forest_24" → artifact dict
VALID_MODELS   = ["random_forest", "xgboost", "ridge"]   # list: deterministic order
VALID_HORIZONS = {24, 48, 72}

# Map API names → evaluate.py / mongo_store names
_API_TO_EVAL_NAME = {
    "random_forest": "RandomForest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# Map API names → mongo_store model_name strings (as saved by train_*.py)
_API_TO_STORE_NAME = {
    "random_forest": "RandomForest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# ── In-memory caches (5-min TTL) ─────────────────────────────────────────────
_FEATURE_CACHE: dict = {"doc": None, "ts": 0.0}
_METRICS_CACHE: dict = {"doc": None, "ts": 0.0}
_CACHE_TTL = 300

# Sentinel: distinguishes 'tried MongoDB, got nothing' from 'never tried'
_MONGO_MISS = "__MONGO_MISS__"

# ── Metric key aliases ────────────────────────────────────────────────────────
_METRIC_ALIASES = {
    "r2":       ["R2", "r2", "test_r2", "R²", "r2_score"],
    "rmse":     ["RMSE", "rmse", "test_rmse", "root_mean_squared_error"],
    "mae":      ["MAE", "mae", "test_mae", "mean_absolute_error"],
    "mape":     ["MAPE", "mape", "test_mape"],
    "coverage": ["Coverage", "coverage", "conformal_global_coverage", "global_coverage"],
    "margin":   ["margin", "Margin", "conformal_margin_width", "conformal_margin"],
}

# ← RENDER FIX 3: removed duplicate home() route — index() below is the real root


def _mongo_client():
    if not MONGO_URI:
        raise ValueError("MONGODB_URI is not set in environment / .env")
    # socketTimeoutMS raised to 120 s to match mongo_store.py — GridFS reads
    # large model artifacts in chunks and can exceed the old 5 s default.
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


def _build_feature_vector(expected_features: list, slider_overrides: dict) -> pd.DataFrame:
    """Use latest MongoDB doc as base; overlay slider values on lag-1 features."""
    latest_doc = _get_latest_feature_doc()

    base = (
        {f: _safe_float(latest_doc.get(f), 0.0) for f in expected_features}
        if latest_doc
        else {f: 0.0 for f in expected_features}
    )

    def _aqi_from_pm25(pm25: float) -> float:
        if np.isnan(pm25) or pm25 < 0:
            return float("nan")
        bps = [
            (0.0,   12.0,  0,   50),  (12.1,  35.4,  51,  100),
            (35.5,  55.4,  101, 150), (55.5,  150.4, 151, 200),
            (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_lo, c_hi, a_lo, a_hi in bps:
            if c_lo <= pm25 <= c_hi:
                return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
        return 500.0

    pm25 = _safe_float(slider_overrides.get("pm25",        base.get("pm25_lag_1")),  75.0)
    pm10 = _safe_float(slider_overrides.get("pm10",        base.get("pm10_lag_1")), 140.0)
    temp = _safe_float(slider_overrides.get("temperature", base.get("temperature_lag_1")), 32.0)
    hum  = _safe_float(slider_overrides.get("humidity",    base.get("humidity_lag_1")),    65.0)
    wind = _safe_float(slider_overrides.get("wind_speed",  base.get("wind_speed_lag_1")),  10.0)

    aqi_now = _aqi_from_pm25(pm25)
    if np.isnan(aqi_now):
        aqi_now = _safe_float(base.get("aqi_lag_1"), 75.0)

    overrides = {
        "aqi_lag_1":                     aqi_now,
        "pm25_lag_1":                    pm25,
        "pm10_lag_1":                    pm10,
        "temperature_lag_1":             temp,
        "humidity_lag_1":                hum,
        "wind_speed_lag_1":              wind,
        "interaction_pm25_humidity":     pm25 * hum / 100.0,
        "interaction_pm25_wind_inverse": pm25 / (wind + 0.5),
        "temp_humidity":                 temp * hum,
        "heat_dryness":                  temp / (hum + 1),
        "wind_dispersal":                wind / (pm25 + 5),
        "temp_to_wind_ratio":            temp / (wind + 0.1),
        "humidity_to_wind_ratio":        hum  / (wind + 0.1),
    }
    for k, v in overrides.items():
        if k in base:
            base[k] = v

    df = pd.DataFrame([base])[expected_features]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ── Model artifact loader (MONGO-ONLY) ───────────────────────────────────────
def load_prediction_artifacts(model_type: str, horizon: int) -> dict:
    """
    Loads a trained model artifact from MongoDB (model_artifacts collection).

    Previously this read from local .pkl files on disk. All training scripts
    now call mongo_store.save_model_artifact() instead of joblib.dump(), so
    the local files no longer exist.  This function uses
    mongo_store.load_model_artifact() and caches the result in MODEL_CACHE.

    The artifact dict returned has at minimum:
      {
        "model":            <fitted estimator>,
        "feature_names":    [...],
        "conformal_margin": float,
        "use_log":          bool,
      }
    """
    cache_key = f"{model_type}_{horizon}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    store_name = _API_TO_STORE_NAME.get(model_type)
    if not store_name:
        raise ValueError(f"Unknown model type: {model_type!r}")

    # Import here to avoid circular dependency issues at module load time
    from mongo_store import load_model_artifact

    artifact = load_model_artifact(model_name=store_name, horizon=horizon)
    if artifact is None:
        raise FileNotFoundError(
            f"No artifact found in MongoDB for model='{store_name}', horizon={horizon}h. "
            f"Run the training script for '{model_type}' first."
        )

    MODEL_CACHE[cache_key] = artifact
    print(f"[model-cache] Loaded '{store_name}' {horizon}h from MongoDB ✓")
    return artifact


# ── Metrics helpers ───────────────────────────────────────────────────────────
def _get_latest_metrics_doc() -> dict | None:
    now    = time.time()
    cached = _METRICS_CACHE["doc"]
    if cached is not None and (now - _METRICS_CACHE["ts"]) < _CACHE_TTL:
        return None if cached == _MONGO_MISS else cached
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        col    = db[METRICS_COL]

        doc = col.find_one({"pipeline_run_status": "SUCCESS"}, sort=[("timestamp", -1)])
        if doc is None:
            doc = col.find_one({}, sort=[("timestamp", -1)])
        if doc is None:
            doc = col.find_one({})

        client.close()

        if doc is None:
            print("[metrics-cache] WARNING: model_metrics collection is empty.")
            _METRICS_CACHE["doc"] = _MONGO_MISS
            _METRICS_CACHE["ts"]  = now
            return None

        doc.pop("_id", None)
        print(f"[metrics-cache] Raw doc top-level keys: {list(doc.keys())}")

        summary = _normalise_metrics_doc(doc)
        if summary:
            _METRICS_CACHE["doc"] = summary
            _METRICS_CACHE["ts"]  = now
            return summary

        print(
            "[metrics-cache] WARNING: could not extract performance_summary. "
            f"Full top-level keys: {sorted(doc.keys())}"
        )
        _METRICS_CACHE["doc"] = _MONGO_MISS
        _METRICS_CACHE["ts"]  = now
        return None

    except PyMongoError as e:
        print(f"[metrics-cache] ERROR: MongoDB fetch failed — {e}")
        _METRICS_CACHE["doc"] = _MONGO_MISS
        _METRICS_CACHE["ts"]  = now
        return None


def _normalise_metrics_doc(doc: dict) -> dict | None:
    """
    Handles all known shapes the model_metrics document might have.

    Shape A (evaluate.py standard):
      { performance_summary: { "24h": { models: { "RandomForest": {...} } } } }

    Shape B (flat per-horizon):
      { "24h": { "RandomForest": {...} }, "48h": {...}, ... }

    Shape C (flat with model prefix):
      { "RandomForest_24h": {...}, "XGBoost_48h": {...}, ... }

    Shape D (individual per-model-per-horizon flat docs from save_metrics):
      Individual documents stored per model per horizon in model_metrics.
      Handled separately by _metrics_from_mongo() via direct per-doc queries.

    Returns normalised dict keyed by horizon string ("24h", "48h", "72h").
    """
    horizons = ["24h", "48h", "72h"]
    models   = ["RandomForest", "XGBoost", "Ridge"]

    # Shape A
    if "performance_summary" in doc:
        ps = doc["performance_summary"]
        if isinstance(ps, dict) and any(h in ps for h in horizons):
            return ps

    # Shape B
    if any(h in doc for h in horizons):
        result = {}
        for h in horizons:
            if h not in doc:
                continue
            h_block = doc[h]
            if not isinstance(h_block, dict):
                continue
            result[h] = h_block if "models" in h_block else {"models": h_block}
        if result:
            return result

    # Shape C
    result: dict = {h: {"models": {}} for h in horizons}
    found_any = False
    for model in models:
        for h in horizons:
            for key in [f"{model}_{h}", f"{model.lower()}_{h}"]:
                if key in doc and isinstance(doc[key], dict):
                    result[h]["models"][model] = doc[key]
                    found_any = True
    if found_any:
        return result

    print(
        f"[metrics-cache] _normalise_metrics_doc: no schema matched. "
        f"Full keys: {sorted(doc.keys())}"
    )
    return None


def _metrics_from_mongo(model_type: str) -> dict:
    """
    Primary path: tries the pipeline_runs-style summary doc (evaluate.py output).
    Fallback path: queries model_metrics directly per model+horizon (save_metrics
    output from train_*.py). This covers the common case where evaluate.py has
    not run yet but individual training runs have already written their metrics.
    """
    eval_name     = _API_TO_EVAL_NAME.get(model_type, model_type.title())
    store_name    = _API_TO_STORE_NAME.get(model_type, eval_name)
    mongo_summary = _get_latest_metrics_doc()
    report        = {}

    for h in sorted(VALID_HORIZONS):
        h_key    = f"{h}h"
        mongo_ok = False

        # ── Primary path: evaluate.py summary doc ────────────────────────────
        if mongo_summary and h_key in mongo_summary:
            horizon_block = mongo_summary[h_key]
            models_block  = {}
            if isinstance(horizon_block, dict):
                models_block = (
                    horizon_block["models"]
                    if "models" in horizon_block and isinstance(horizon_block["models"], dict)
                    else horizon_block
                )
            m = models_block.get(eval_name, {}) if isinstance(models_block, dict) else {}
            if m and isinstance(m, dict):
                report[str(h)] = {
                    "r2":       _extract_metric(m, "r2"),
                    "rmse":     _extract_metric(m, "rmse"),
                    "mae":      _extract_metric(m, "mae"),
                    "mape":     _extract_metric(m, "mape"),
                    "coverage": _extract_metric(m, "coverage"),
                    "margin":   _extract_metric(m, "margin"),
                }
                if report[str(h)]["r2"] != 0.0 or report[str(h)]["mae"] != 0.0:
                    mongo_ok = True
                    print(f"[metrics] {store_name}/{h_key} loaded from summary doc ✓")

        # ── Fallback path: individual model_metrics documents ────────────────
        # train_*.py writes one flat doc per (model, horizon) via save_metrics().
        # Query it directly when the summary doc is missing or doesn't cover this
        # model/horizon.
        if not mongo_ok:
            try:
                client = _mongo_client()
                db     = client[DB_NAME]
                col    = db[METRICS_COL]
                # save_metrics() stores the document with "model" and "horizon_h" fields
                m_doc = col.find_one(
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
                    mongo_ok = True
                    print(f"[metrics] {store_name}/{h_key} loaded from model_metrics doc ✓")
            except PyMongoError as e:
                print(f"[metrics] {store_name}/{h_key} fallback query failed: {e}")

        if not mongo_ok:
            reason = (
                f"No metrics found in MongoDB for model='{store_name}', horizon={h_key}. "
                "Run the training script first."
            )
            report[str(h)] = {"error": reason}
            print(f"[metrics] {store_name}/{h_key} — MongoDB miss: {reason}")

    return report


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Root endpoint — confirms the API is online (required by Render health checks)."""
    return jsonify({
        "message": "AQI Prediction API is active.",
        "service": "Karachi AQI Forecasting API",
        "status":  "running",
        "endpoints": {
            "health":           "/health",
            "latest_features":  "/latest_features",
            "predict_24h_rf":   "/predict/random_forest/24",
            "predict_24h_xgb":  "/predict/xgboost/24",
            "predict_24h_ridge":"/predict/ridge/24",
            "metrics_rf":       "/metrics/random_forest",
            "metrics_xgb":      "/metrics/xgboost",
            "metrics_ridge":    "/metrics/ridge",
            "all_metrics":      "/metrics/all",
            "debug_artifacts":  "/debug/artifacts",
            "debug_features":   "/debug/features_raw",
            "debug_metrics":    "/debug/metrics_raw",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    mongo_summary = _get_latest_metrics_doc()
    feat_doc      = _get_latest_feature_doc()
    return jsonify({
        "status":                  "healthy",
        "service":                 "aqi-multi-model-forecasting-api",
        "storage_backend":         "MongoDB Atlas (no local files)",
        "warmed_models_in_cache":  list(MODEL_CACHE.keys()),
        "mongo_connected":         MONGO_URI is not None,
        "metrics_in_mongo":        mongo_summary is not None,
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
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500


@app.route("/metrics/<string:model_type>", methods=["GET"])
def get_model_metrics(model_type):
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {VALID_MODELS}"}), 400
    return jsonify(_metrics_from_mongo(model_type))


@app.route("/metrics/all", methods=["GET"])
def get_all_metrics():
    return jsonify({m: _metrics_from_mongo(m) for m in VALID_MODELS})


@app.route("/debug/metrics_raw", methods=["GET"])
def debug_metrics_raw():
    """Returns raw documents from model_metrics for schema inspection."""
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
    """Returns the latest processed_features document for field-name inspection."""
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
    """Lists all model artifacts stored in MongoDB (model_name, horizon, updated_at)."""
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
    app.run(host="0.0.0.0", port=port, debug=False)   # ← RENDER FIX 4: debug=False