"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.

FIXES IN THIS VERSION:
  FIX 1 — _get_latest_metrics_doc() was filtering by pipeline_run_status="SUCCESS"
    which fails silently if the stored document uses a different field name or value
    (e.g. "success", "COMPLETE", or the field simply doesn't exist). The query now
    fetches ANY document sorted by timestamp desc, then tries both field structures.

  FIX 2 — metrics document schema is normalised defensively. The stored document
    may have keys in any casing ("r2" / "R2" / "test_r2"). _extract_metric() tries
    all known aliases so the endpoint never returns zeros for real data.

  FIX 3 — _get_latest_feature_doc() sorts by "timestamp" first, then falls back
    to "datetime". The processed_features collection stores timestamp as a string
    ("2026-06-01T..."), so PyMongo string-sorts it correctly. No change needed
    there — but the sort field must match what's actually in the document.
    Added a debug log so connection failures are visible in Flask logs.

  FIX 4 — /metrics/all route was building a new dict comprehension that iterated
    VALID_MODELS (a set), producing non-deterministic key order. Made it a list.

  FIX 5 — _build_feature_vector: values coming from MongoDB may be stored as
    strings (MongoDB round-trips via JSON serialisation). Added explicit float()
    cast with fallback 0.0 for every field pulled from latest_doc.

  FIX 6 — NaN/Inf in feature vector crashes joblib models. Added np.nan_to_num()
    before predict() to replace any surviving NaN/Inf with column means from the
    training distribution (approximated by 0 for normalised features).

  FIX 7 — model_metrics query: also tries fetching the document without any
    filter in case pipeline_run_status field is absent entirely.

  FIX 8 (NEW) — _metrics_from_mongo() previously returned {"r2": 0.0, ...} on
    any MongoDB failure or unrecognised schema, causing the Streamlit charts to
    render with silent zero data and no visible error. It now returns an explicit
    {"error": "<reason>"} dict for every horizon that has no real data.
    The /metrics/all and /metrics/<model> routes preserve this error key so the
    Streamlit frontend can detect it and surface a real error message instead of
    plotting zeros.

  FIX 9 (NEW) — _get_latest_metrics_doc() now caches a sentinel value on MongoDB
    failure so repeated requests within the TTL window do not all hammer the DB.
    Previously a None return was never cached, hammering Atlas on every page load.

  FIX 10 (NEW) — Added /debug/features_raw endpoint (mirrors /debug/metrics_raw)
    so the processed_features document can be inspected without a Mongo shell.

  FIX 11 (NEW) — _normalise_metrics_doc() now logs the full top-level key list
    when it cannot match any known shape, making future schema debugging trivial.
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
API_DIR      = Path(__file__).resolve().parent
BASE_DIR     = API_DIR.parent
MODELS_DIR   = BASE_DIR / "models"
METRICS_DIR  = BASE_DIR / "metrics"   # local JSON fallback only

load_dotenv(BASE_DIR / ".env")
MONGO_URI    = os.getenv("MONGODB_URI")
DB_NAME      = "karachi_aqi"
FEAT_COL     = "processed_features"
METRICS_COL  = "model_metrics"

app = Flask(__name__)
MODEL_CACHE     = {}
VALID_MODELS    = ["random_forest", "xgboost", "ridge"]   # FIX 4: list not set
VALID_HORIZONS  = {24, 48, 72}

_API_TO_EVAL_NAME = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# ── In-memory caches (5-min TTL) ─────────────────────────────────────────────
_FEATURE_CACHE: dict = {"doc": None, "ts": 0.0}
_METRICS_CACHE: dict = {"doc": None, "ts": 0.0}
_CACHE_TTL = 300

# FIX 9: Sentinel to distinguish 'already tried MongoDB, got nothing' from
# 'never tried'. Prevents hammering Atlas on every Streamlit page load when
# the collection is empty or MONGODB_URI is wrong.
_MONGO_MISS = '__MONGO_MISS__'


def _mongo_client():
    if not MONGO_URI:
        raise ValueError("MONGODB_URI is not set in environment / .env")
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


# ── Metric key aliases ────────────────────────────────────────────────────────
# FIX 2: The stored document may use any of these casings. _extract_metric()
# tries all aliases and returns the first non-None, non-zero value found.
_METRIC_ALIASES = {
    "r2":       ["R2", "r2", "test_r2", "R²", "r2_score"],
    "rmse":     ["RMSE", "rmse", "test_rmse", "root_mean_squared_error"],
    "mae":      ["MAE", "mae", "test_mae", "mean_absolute_error"],
    "mape":     ["MAPE", "mape", "test_mape"],
    "coverage": ["Coverage", "coverage", "conformal_global_coverage", "global_coverage"],
    "margin":   ["margin", "Margin", "conformal_margin_width", "conformal_margin"],
}

def _extract_metric(doc: dict, key: str) -> float:
    for alias in _METRIC_ALIASES.get(key, [key]):
        v = doc.get(alias)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


# ── Feature helpers ───────────────────────────────────────────────────────────
def _get_latest_feature_doc() -> dict | None:
    now = time.time()
    if _FEATURE_CACHE["doc"] is not None and (now - _FEATURE_CACHE["ts"]) < _CACHE_TTL:
        return _FEATURE_CACHE["doc"]
    try:
        client = _mongo_client()
        db  = client[DB_NAME]
        col = db[FEAT_COL]
        # Try timestamp field first (ISO string), then datetime string field
        doc = col.find_one({}, sort=[("timestamp", -1)])
        if doc is None:
            doc = col.find_one({}, sort=[("datetime", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            _FEATURE_CACHE["doc"] = doc
            _FEATURE_CACHE["ts"]  = now
            print(f"[feature-cache] Loaded doc with keys: {list(doc.keys())[:10]} ...")
        else:
            print("[feature-cache] WARNING: processed_features collection returned no documents.")
        return doc
    except PyMongoError as e:
        print(f"[feature-cache] ERROR: MongoDB fetch failed — {e}")
        return None


def _safe_float(val, default: float = 0.0) -> float:
    """FIX 5: MongoDB may store numeric fields as strings. Cast defensively."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _build_feature_vector(expected_features: list, slider_overrides: dict) -> pd.DataFrame:
    """Use latest MongoDB doc as base; overlay slider values on lag-1 features."""
    latest_doc = _get_latest_feature_doc()

    # FIX 5: cast every value through _safe_float — MongoDB may return strings
    if latest_doc:
        base = {f: _safe_float(latest_doc.get(f), 0.0) for f in expected_features}
    else:
        base = {f: 0.0 for f in expected_features}

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
        "aqi_lag_1":                    aqi_now,
        "pm25_lag_1":                   pm25,
        "pm10_lag_1":                   pm10,
        "temperature_lag_1":            temp,
        "humidity_lag_1":               hum,
        "wind_speed_lag_1":             wind,
        "interaction_pm25_humidity":    pm25 * hum / 100.0,
        "interaction_pm25_wind_inverse":pm25 / (wind + 0.5),
        "temp_humidity":                temp * hum,
        "heat_dryness":                 temp / (hum + 1),
        "wind_dispersal":               wind / (pm25 + 5),
        "temp_to_wind_ratio":           temp / (wind + 0.1),
        "humidity_to_wind_ratio":       hum  / (wind + 0.1),
    }
    for k, v in overrides.items():
        if k in base:
            base[k] = v

    df = pd.DataFrame([base])[expected_features]

    # FIX 6: Replace any NaN/Inf that survived so model.predict() never crashes
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ── Metrics helpers ───────────────────────────────────────────────────────────
def _get_latest_metrics_doc() -> dict | None:
    """
    FIX 1 + FIX 7: Fetch metrics without hard-filtering on pipeline_run_status.
    The stored document may not have that field at all. Try:
      1. Filter by pipeline_run_status="SUCCESS" (original intent)
      2. Any document sorted by timestamp desc
      3. Truly any document (no sort — last resort)
    Then normalise whatever shape we get.
    """
    now = time.time()
    # FIX 9: treat sentinel as a cached miss (return None, don't re-query)
    cached = _METRICS_CACHE["doc"]
    if cached is not None and (now - _METRICS_CACHE["ts"]) < _CACHE_TTL:
        return None if cached == _MONGO_MISS else cached
    try:
        client = _mongo_client()
        db  = client[DB_NAME]
        col = db[METRICS_COL]

        # Attempt 1: original strict filter
        doc = col.find_one({"pipeline_run_status": "SUCCESS"}, sort=[("timestamp", -1)])

        # Attempt 2: any document with a timestamp field
        if doc is None:
            doc = col.find_one({}, sort=[("timestamp", -1)])

        # Attempt 3: truly any document
        if doc is None:
            doc = col.find_one({})

        client.close()

        if doc is None:
            print("[metrics-cache] WARNING: model_metrics collection is empty.")
            # FIX 9: cache the miss so we don't hammer Atlas on every request
            _METRICS_CACHE["doc"] = _MONGO_MISS
            _METRICS_CACHE["ts"]  = now
            return None

        doc.pop("_id", None)
        print(f"[metrics-cache] Raw doc top-level keys: {list(doc.keys())}")

        # Normalise: dig out the performance_summary block regardless of nesting
        summary = _normalise_metrics_doc(doc)
        if summary:
            _METRICS_CACHE["doc"] = summary
            _METRICS_CACHE["ts"]  = now
            return summary

        # FIX 9 + FIX 11: log full key list so schema mismatches are obvious
        print(
            "[metrics-cache] WARNING: could not extract performance_summary from doc. "
            f"Full top-level keys: {list(doc.keys())}"
        )
        _METRICS_CACHE["doc"] = _MONGO_MISS
        _METRICS_CACHE["ts"]  = now
        return None

    except PyMongoError as e:
        print(f"[metrics-cache] ERROR: MongoDB fetch failed — {e}")
        # FIX 9: cache the miss to avoid hammering Atlas
        _METRICS_CACHE["doc"] = _MONGO_MISS
        _METRICS_CACHE["ts"]  = now
        return None


def _normalise_metrics_doc(doc: dict) -> dict | None:
    """
    Handles all known shapes the model_metrics document might have:

    Shape A (evaluate.py standard):
      { performance_summary: { "24h": { models: { "Random Forest": {...} } } } }

    Shape B (flat per-horizon keys):
      { "24h": { "Random Forest": { MAE: ..., R2: ... } } }

    Shape C (flat with model prefix):
      { "Random Forest_24h": { MAE: ..., R2: ... } }

    Returns a dict normalised to Shape A's performance_summary value, i.e.:
      { "24h": { "models": { "Random Forest": { R2, RMSE, MAE, ... } } }, ... }
    """
    horizons = ["24h", "48h", "72h"]
    models   = ["Random Forest", "XGBoost", "Ridge"]

    # Shape A — already in the right format
    if "performance_summary" in doc:
        ps = doc["performance_summary"]
        if isinstance(ps, dict) and any(h in ps for h in horizons):
            return ps

    # Shape B — top-level horizon keys
    if any(h in doc for h in horizons):
        result = {}
        for h in horizons:
            if h not in doc:
                continue
            h_block = doc[h]
            if not isinstance(h_block, dict):
                continue
            # h_block might be { "Random Forest": {...} } or { "models": { "Random Forest": {...} } }
            if "models" in h_block:
                result[h] = h_block
            else:
                result[h] = {"models": h_block}
        if result:
            return result

    # Shape C — flat keys like "Random Forest_24h" or "RandomForest_24h"
    result: dict = {h: {"models": {}} for h in horizons}
    found_any = False
    for model in models:
        model_slug = model.replace(" ", "")
        for h in horizons:
            for key in [f"{model}_{h}", f"{model_slug}_{h}", f"{model.lower()}_{h}"]:
                if key in doc and isinstance(doc[key], dict):
                    result[h]["models"][model] = doc[key]
                    found_any = True
    if found_any:
        return result

    # Shape D — completely flat, single-model single-horizon document
    # e.g. the document IS the metrics for one model/horizon
    if any(alias in doc for alias in _METRIC_ALIASES["r2"]):
        print("[metrics-cache] Doc appears to be a flat single-record — cannot determine model/horizon.")

    # FIX 11: Always log the full key list on a schema miss so debugging is trivial
    print(
        f"[metrics-cache] _normalise_metrics_doc: could not match any known schema. "
        f"Full top-level keys in document: {sorted(doc.keys())}"
    )
    return None


def _metrics_from_mongo(model_type: str) -> dict:
    eval_name     = _API_TO_EVAL_NAME.get(model_type, model_type.title())
    mongo_summary = _get_latest_metrics_doc()
    report        = {}

    for h in sorted(VALID_HORIZONS):
        h_key    = f"{h}h"
        mongo_ok = False

        if mongo_summary and h_key in mongo_summary:
            # Safely grab the dictionary under "models"
            horizon_block = mongo_summary[h_key]
            models_block = {}
            
            if isinstance(horizon_block, dict):
                # If "models" is nested inside (Shape A / evaluate.py standard)
                if "models" in horizon_block and isinstance(horizon_block["models"], dict):
                    models_block = horizon_block["models"]
                else:
                    models_block = horizon_block

            # Extract the metrics for our specific model engine
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
                # If we actually got real data (not all zeros), mark as successful
                if report[str(h)]["r2"] != 0.0 or report[str(h)]["mae"] != 0.0:
                    mongo_ok = True
                    print(f"[metrics] {model_type} / {h_key} successfully loaded from MongoDB ✓")

        if not mongo_ok:
            # FIX 8: Do NOT silently return zeros — that makes Streamlit charts
            # show flat zero lines with no visible error. Instead return an
            # explicit error key so the frontend can display a real message.
            # We no longer fall back to local JSON; all data must come from MongoDB.
            reason = (
                "model_metrics collection is empty or schema unrecognised"
                if _get_latest_metrics_doc() is None
                else f"horizon {h_key} not found in metrics document for model {eval_name}"
            )
            report[str(h)] = {"error": reason}
            print(f"[metrics] {model_type} / {h_key} — MongoDB miss: {reason}")
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
        found[m] = [f"{h}h" for h in sorted(VALID_HORIZONS)
                    if (MODELS_DIR / f"{m}_{h}h.pkl").exists()]
    mongo_summary = _get_latest_metrics_doc()
    feat_doc      = _get_latest_feature_doc()
    return jsonify({
        "status":                       "healthy",
        "service":                      "aqi-multi-model-forecasting-api",
        "warmed_models_in_cache":       list(MODEL_CACHE.keys()),
        "discovered_artifacts_on_disk": found,
        "mongo_connected":              MONGO_URI is not None,
        "metrics_in_mongo":             mongo_summary is not None,
        "features_in_mongo":            feat_doc is not None,
        "feature_doc_keys":             list(feat_doc.keys())[:8] if feat_doc else [],
    }), 200


@app.route("/latest_features", methods=["GET"])
def latest_features():
    doc = _get_latest_feature_doc()
    if doc is None:
        return jsonify({"error": "No feature data available from MongoDB. "
                                 "Check processed_features collection."}), 503
    safe = [
        "timestamp", "datetime",
        "aqi_lag_1", "pm25_lag_1", "pm10_lag_1",
        "temperature_lag_1", "humidity_lag_1", "wind_speed_lag_1",
    ]
    return jsonify({k: doc.get(k) for k in safe if k in doc}), 200


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
        return jsonify({"error": str(fnf),
                        "hint": f"Run the training script for '{model_type}'."}), 500
    except Exception as e:
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500


@app.route("/metrics/<string:model_type>", methods=["GET"])
def get_model_metrics(model_type):
    return jsonify(_metrics_from_mongo(model_type.lower()))


@app.route("/metrics/all", methods=["GET"])
def get_all_metrics():
    # FIX 4: VALID_MODELS is now a list — order is deterministic
    return jsonify({m: _metrics_from_mongo(m) for m in VALID_MODELS})


@app.route("/debug/metrics_raw", methods=["GET"])
def debug_metrics_raw():
    """
    Debug endpoint — returns the raw document from model_metrics exactly as
    stored, so you can see the real field names and nesting structure.
    Hit GET /debug/metrics_raw to inspect what's actually in MongoDB.
    """
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db[METRICS_COL].find_one({}, sort=[("timestamp", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            return jsonify({"found": True, "doc": doc}), 200
        return jsonify({"found": False, "doc": None}), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/features_raw", methods=["GET"])
def debug_features_raw():
    """
    FIX 10: Debug endpoint — returns the raw document from processed_features
    exactly as stored, so you can verify field names match what the feature
    vector builder expects. Hit GET /debug/features_raw to inspect.
    """
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
        return jsonify({"found": False, "doc": None,
                        "hint": "processed_features collection is empty. "
                                "Run the feature pipeline first."}), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)