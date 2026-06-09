"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.
api/app.py

CONSOLIDATED CHANGE LOG
────────────────────────
FIX 1  — METRICS PATH SIMPLIFIED
FIX 2  — MODEL-AWARE FEATURE VECTOR (_build_feature_vector)
FIX 3  — PREDICT ROUTE WIRED TO FIXED BUILDER
FIX 4  — NAMING CONSISTENCY
FIX 5  — RANDOM FOREST GUNICORN DEADLOCK (n_jobs patched at load time)
FIX 6  — XGBOOST RESIDUAL CORRECTOR APPLIED AT INFERENCE (corrector removed in
          train_xgboost.py v4 — inference path now skips gracefully when absent)
FIX 7  — FULL TRACEBACK LOGGED ON PREDICT FAILURE
FIX 8  — RIDGE FEATURE IMPORTANCE ROUTE (/features/<model>/<horizon>)
FIX 9  — /features/ ROUTE NULL-GUARD + INDEX MAP CORRECTED
FIX 10 — DEAD _get_latest_metrics_doc STUB REMOVED
FIX 11 — MODEL_CACHE TTL (1 hour) so stale artifacts auto-expire after retraining
FIX 12 — _build_feature_vector always applies lag1_overrides via base.update()
          instead of only updating keys already present in base. Prevents 0-valued
          features when the feature doc is stale or missing a lag column.
FIX 13 — /debug/feature_mismatch/<model>/<horizon> route for diagnostics.
FIX 14 — Background warmup thread pre-loads Ridge + XGBoost (6 artifacts) at
          startup. Random Forest intentionally excluded — its GridFS artifact is
          large enough that warming it in a background thread during gunicorn
          --preload caused WORKER TIMEOUT → SIGKILL restart loops on Render free
          tier. RF loads lazily on the first /predict/random_forest/<h> call.
FIX 15 — socketTimeoutMS raised to 300_000 ms (5 min) in _mongo_client() to
          match gunicorn --timeout 300, so large RF GridFS transfers complete
          without hitting a mid-transfer socket timeout.
FIX 16 — _metrics_from_mongo opens a SINGLE MongoClient per call (not one per
          horizon). Replaced per-horizon open/close loop with a single
          try/finally block; previously opened 3 connections per model,
          9 per /metrics/all call.

ARTIFACT SCHEMAS (confirmed from training scripts)
───────────────────────────────────────────────────
RandomForest:
  { model, feature_names, conformal_margin, use_log=True }
  (n_jobs patched to 1 at training time, re-patched defensively at load time)

XGBoost:
  { model, feature_names, conformal_margin, use_log=False, use_corrector=False }
  (trains on raw AQI — no expm1 back-conversion needed at inference)

Ridge:
  { model, model_type, feature_names, conformal_margin, use_log=True,
    use_interactions, interaction_cols }
  (Pipeline wrapping StandardScaler + Ridge/ElasticNet; predict() returns log1p(AQI))

mongo_store.get_db() uses socketTimeoutMS=300_000 — this module creates its own
MongoClient in _mongo_client() with the same value to stay consistent.
"""

import os
import sys
import time
import threading
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
MODEL_CACHE     : dict = {}
MODEL_CACHE_TS  : dict = {}
MODEL_CACHE_TTL : int  = 3600        # 1-hour TTL — auto-expires after retraining

VALID_MODELS   = ["random_forest", "xgboost", "ridge"]
VALID_HORIZONS = {24, 48, 72}

_API_TO_MONGO_NAME: dict[str, str] = {
    "random_forest": "RandomForest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}

# ── In-memory feature-doc cache (5-min TTL) ───────────────────────────────────
_FEATURE_CACHE: dict = {"doc": None, "ts": 0.0}
_CACHE_TTL = 300

# ── Metric key aliases (handles inconsistent field names across training runs) ─
_METRIC_ALIASES: dict[str, list[str]] = {
    "r2":       ["R2", "r2", "test_r2", "R²", "r2_score"],
    "rmse":     ["RMSE", "rmse", "test_rmse", "root_mean_squared_error"],
    "mae":      ["MAE", "mae", "test_mae", "mean_absolute_error"],
    "mape":     ["MAPE", "mape", "test_mape"],
    "coverage": ["Coverage", "coverage", "conformal_global_coverage", "global_coverage"],
    "margin":   ["margin", "Margin", "conformal_margin_width", "conformal_margin"],
}


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _mongo_client() -> MongoClient:
    """
    Creates a new MongoClient.

    FIX 15: socketTimeoutMS=300_000 (5 min) matches gunicorn --timeout 300.
    The previous 120_000 ms limit caused large Random Forest GridFS downloads
    to raise "The read operation timed out" mid-transfer, which stalled the
    background warmup thread past the worker timeout → SIGKILL restart loop
    on Render free tier.

    mongo_store.get_db() already uses socketTimeoutMS=300_000 (confirmed in
    mongo_store.py line 61); this value keeps both clients consistent.
    """
    if not MONGO_URI:
        raise ValueError("MONGODB_URI is not set in environment / .env")
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10_000,
        socketTimeoutMS=300_000,
    )


def _extract_metric(doc: dict, key: str) -> float:
    """Walk the alias list for `key` and return the first numeric value found."""
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


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_latest_feature_doc() -> dict | None:
    """
    Fetch the most recent document from processed_features.
    Result is cached in-process for _CACHE_TTL seconds (5 min) to avoid
    a MongoDB round-trip on every /predict call.
    """
    now = time.time()
    if _FEATURE_CACHE["doc"] is not None and (now - _FEATURE_CACHE["ts"]) < _CACHE_TTL:
        return _FEATURE_CACHE["doc"]
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        col    = db[FEAT_COL]
        # Try datetime first (feature_engineering output), fall back to timestamp
        doc = col.find_one({}, sort=[("datetime", -1)])
        if doc is None:
            doc = col.find_one({}, sort=[("timestamp", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            _FEATURE_CACHE["doc"] = doc
            _FEATURE_CACHE["ts"]  = now
            print(f"[feature-cache] Loaded doc — sample keys: {list(doc.keys())[:10]} ...")
        else:
            print("[feature-cache] WARNING: processed_features returned no documents.")
        return doc
    except PyMongoError as e:
        print(f"[feature-cache] ERROR: MongoDB fetch failed — {e}")
        return None


def _reconstruct_interaction_terms(base: dict, interaction_cols: list[str]) -> dict:
    """
    Regenerate pairwise interaction features for Ridge artifacts that were
    trained with USE_INTERACTIONS=True.  Only columns present in `base` are
    crossed (same guard used in train_ridge.py _add_interaction_terms).
    """
    extra: dict = {}
    present = [c for c in interaction_cols if c in base]
    for a, b in combinations(present, 2):
        extra[f"{a} {b}"] = base.get(a, 0.0) * base.get(b, 0.0)
    return extra


def _build_feature_vector(artifact: dict, slider_overrides: dict) -> pd.DataFrame:
    """
    Build the inference feature vector for a single prediction request.

    Resolution order for each feature value:
      1. Derived from slider_overrides (user-supplied values from the dashboard)
      2. Corresponding lag column from the latest processed_features doc
      3. Hard-coded fallback default (Karachi typical values)

    FIX 12: base.update(lag1_overrides) always writes the overrides regardless
    of whether the key already exists in base.  The old `if k in base` guard
    caused 0-valued features when the feature doc was stale or missing a lag
    column, which made XGBoost predict ~0 on near-zero inputs.

    Artifact keys used here (confirmed from training scripts):
      feature_names     — ordered list matching the fitted model's input columns
      use_interactions  — bool (Ridge only); whether interaction terms were added
      interaction_cols  — list (Ridge only); source columns for interaction terms
      use_log           — bool; True for RF and Ridge (expm1 at inference),
                          False for XGBoost (trains on raw AQI, no back-conversion)
    """
    expected_features : list[str] = artifact["feature_names"]
    use_interactions  : bool      = artifact.get("use_interactions", False)
    interaction_cols  : list[str] = artifact.get("interaction_cols", [])

    latest_doc = _get_latest_feature_doc()

    # Seed base from the feature doc; fall back to zeros if doc is unavailable
    base: dict = (
        {f: _safe_float(latest_doc.get(f), 0.0) for f in expected_features}
        if latest_doc
        else {f: 0.0 for f in expected_features}
    )

    def _aqi_from_pm25(pm25: float) -> float:
        """EPA AQI linear interpolation from PM2.5 (µg/m³)."""
        if np.isnan(pm25) or pm25 < 0:
            return float("nan")
        breakpoints = [
            (0.0,   12.0,  0,   50),
            (12.1,  35.4,  51,  100),
            (35.5,  55.4,  101, 150),
            (55.5,  150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_lo, c_hi, a_lo, a_hi in breakpoints:
            if c_lo <= pm25 <= c_hi:
                return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
        return 500.0

    # Resolve the five primary sensor values from overrides → lag doc → default
    pm25 = _safe_float(slider_overrides.get("pm25",        base.get("pm25_lag_1")),        75.0)
    pm10 = _safe_float(slider_overrides.get("pm10",        base.get("pm10_lag_1")),        140.0)
    temp = _safe_float(slider_overrides.get("temperature", base.get("temperature_lag_1")), 32.0)
    hum  = _safe_float(slider_overrides.get("humidity",    base.get("humidity_lag_1")),    65.0)
    wind = _safe_float(slider_overrides.get("wind_speed",  base.get("wind_speed_lag_1")),  10.0)

    # Derive current-step AQI; fall back to stored lag value if PM2.5 is out-of-range
    aqi_now = _aqi_from_pm25(pm25)
    if np.isnan(aqi_now):
        aqi_now = _safe_float(base.get("aqi_lag_1"), 75.0)

    # All lag-1 overrides plus the engineered interaction features that both RF
    # and XGBoost were trained with (confirmed in feature_engineering.py output)
    lag1_overrides: dict = {
        "aqi_lag_1":         aqi_now,
        "pm25_lag_1":        pm25,
        "pm10_lag_1":        pm10,
        "temperature_lag_1": temp,
        "humidity_lag_1":    hum,
        "wind_speed_lag_1":  wind,
        # Engineered interaction columns (always overridden to stay consistent
        # with the values just derived — avoids stale doc values)
        "interaction_pm25_humidity":     pm25 * hum / 100.0,
        "interaction_pm25_wind_inverse": pm25 / (wind + 0.5),
        "temp_humidity":                 temp * hum,
        "heat_dryness":                  temp / (hum + 1.0),
        "wind_dispersal":                wind / (pm25 + 5.0),
        "temp_to_wind_ratio":            temp / (wind + 0.1),
        "humidity_to_wind_ratio":        hum  / (wind + 0.1),
    }

    # FIX 12: unconditional update — always write, even if key not yet in base
    base.update(lag1_overrides)

    # Ridge only: reconstruct pairwise interaction terms added during training
    if use_interactions and interaction_cols:
        base.update(_reconstruct_interaction_terms(base, interaction_cols))

    # Build the final ordered row; any still-missing feature defaults to 0.0
    row = {f: _safe_float(base.get(f), 0.0) for f in expected_features}

    # Warn when a large fraction of features are zero — likely a stale feature doc
    zero_count = sum(1 for v in row.values() if v == 0.0)
    if zero_count > len(expected_features) * 0.25:
        missing = [f for f in expected_features if base.get(f) is None or base.get(f) == 0.0]
        print(
            f"[WARNING] _build_feature_vector: {zero_count}/{len(expected_features)} "
            f"features are 0.0 — feature doc may be stale. "
            f"Sample missing: {missing[:8]}"
        )

    df = pd.DataFrame([row])[expected_features]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL ARTIFACT LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_prediction_artifacts(model_type: str, horizon: int) -> dict:
    """
    Load a model artifact from MongoDB GridFS, with a 1-hour in-process cache.

    GridFS notes (from mongo_store.py):
      - Artifacts are zlib-compressed before upload; load_model_artifact()
        handles decompression transparently.
      - Filename format: artifact_{ModelName}_{horizon}h.pkl
      - Pointer documents in model_artifacts allow existence checks without
        downloading the full binary.

    FIX 5: n_jobs is patched to 1 on any loaded estimator that exposes the
    attribute. RF serialised with n_jobs=1 at training time (train_random_forest.py
    FIX 5), but we re-patch defensively in case an older artifact is loaded.
    """
    cache_key = f"{model_type}_{horizon}"
    now = time.time()

    if cache_key in MODEL_CACHE and (now - MODEL_CACHE_TS.get(cache_key, 0)) < MODEL_CACHE_TTL:
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

    # FIX 5: defensive n_jobs patch (belt-and-suspenders; training already sets it)
    model_obj = artifact.get("model")
    if model_obj is not None and hasattr(model_obj, "n_jobs"):
        model_obj.n_jobs = 1
        print(f"[model-cache] Patched n_jobs=1 on {store_name} {horizon}h (gunicorn safety)")

    MODEL_CACHE[cache_key]    = artifact
    MODEL_CACHE_TS[cache_key] = now
    n_features = len(artifact.get("feature_names", []))
    print(f"[model-cache] Loaded '{store_name}' {horizon}h from MongoDB ✓  ({n_features} features)")
    return artifact


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 14: BACKGROUND MODEL WARMUP (Ridge + XGBoost only)
# ═════════════════════════════════════════════════════════════════════════════

def _warmup_models() -> None:
    """
    Pre-loads Ridge and XGBoost artifacts (6 total) from GridFS at startup.
    Runs as a daemon thread so the gunicorn worker becomes available immediately
    and can serve /health, /metrics, /latest_realtime while loading proceeds.

    Random Forest is intentionally EXCLUDED (FIX 14).
    The RF GridFS artifact is large enough that downloading it in a background
    thread during gunicorn --preload triggered:

      Step 1 — thread calls load_model_artifact("RandomForest", 24)
      Step 2 — GridFS read starts; transfer takes > old socketTimeoutMS (120 s)
      Step 3 — "The read operation timed out" raised mid-transfer
      Step 4 — thread stalled; worker stayed busy past --timeout 300
      Step 5 — gunicorn SIGKILL → restart loop on Render free tier

    FIX 15 (socketTimeoutMS=300_000) resolves the socket timeout, but RF is
    still excluded from warmup as belt-and-suspenders: a sync worker must never
    be kept busy by a background I/O thread.  RF is loaded lazily on the first
    /predict/random_forest/<h> call; subsequent calls hit MODEL_CACHE instantly.
    """
    def _load_fast_models() -> None:
        print("[warmup] Starting background pre-load (Ridge + XGBoost only) ...", flush=True)
        load_order = [
            ("ridge",   24), ("ridge",   48), ("ridge",   72),
            ("xgboost", 24), ("xgboost", 48), ("xgboost", 72),
        ]
        for model_type, horizon in load_order:
            try:
                load_prediction_artifacts(model_type, horizon)
                print(f"[warmup] ✓ {model_type}/{horizon}h cached", flush=True)
            except Exception as e:
                print(f"[warmup] ✗ {model_type}/{horizon}h failed: {e}", flush=True)
        print(
            "[warmup] Pre-load complete (Ridge + XGBoost). "
            "Random Forest will load lazily on first /predict call.",
            flush=True,
        )

    threading.Thread(target=_load_fast_models, daemon=True, name="model-warmup").start()


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _metrics_from_mongo(model_type: str) -> dict:
    """
    Query model_metrics for all three horizons.

    FIX 16: Uses a single MongoClient for all three horizon queries instead of
    opening and closing a new connection inside each loop iteration (which
    created 3 connections per model, 9 per /metrics/all call).
    """
    store_name = _API_TO_MONGO_NAME.get(model_type)
    if not store_name:
        return {
            str(h): {"error": f"Unknown model type: {model_type!r}"}
            for h in sorted(VALID_HORIZONS)
        }

    report: dict = {}
    client = None
    try:
        client = _mongo_client()
        db = client[DB_NAME]
        for h in sorted(VALID_HORIZONS):
            h_key = f"{h}h"
            try:
                m_doc = db[METRICS_COL].find_one(
                    {"model": store_name, "horizon_h": h},
                    {"_id": 0},
                )
                if m_doc:
                    report[str(h)] = {
                        "r2":       _extract_metric(m_doc, "r2"),
                        "rmse":     _extract_metric(m_doc, "rmse"),
                        "mae":      _extract_metric(m_doc, "mae"),
                        "mape":     _extract_metric(m_doc, "mape"),
                        "coverage": _extract_metric(m_doc, "coverage"),
                        "margin":   _extract_metric(m_doc, "margin"),
                    }
                    print(f"[metrics] {store_name}/{h_key} loaded ✓")
                else:
                    report[str(h)] = {"error": (
                        f"No metrics found for model='{store_name}', horizon={h_key}. "
                        "Run the training script first."
                    )}
                    print(f"[metrics] {store_name}/{h_key} — MongoDB miss.")
            except PyMongoError as e:
                print(f"[metrics] {store_name}/{h_key} query failed: {e}")
                report[str(h)] = {"error": f"MongoDB query failed: {e}"}
    finally:
        if client is not None:
            client.close()

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
            "health":          "/health",
            "latest_features": "/latest_features",
            "latest_realtime": "/latest_realtime",
            "predict":         "/predict/<model>/<horizon>  [POST]",
            "metrics_model":   "/metrics/<model>",
            "metrics_all":     "/metrics/all",
            "shap":            "/shap/<model>/<horizon>",
            "features":        "/features/<model>/<horizon>",
            "debug_artifacts": "/debug/artifacts",
            "debug_features":  "/debug/features_raw",
            "debug_metrics":   "/debug/metrics_raw",
            "debug_mismatch":  "/debug/feature_mismatch/<model>/<horizon>",
        },
        "valid_models":   VALID_MODELS,
        "valid_horizons": sorted(VALID_HORIZONS),
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    feat_doc = _get_latest_feature_doc()
    return jsonify({
        "status":                  "healthy",
        "service":                 "aqi-multi-model-forecasting-api",
        "storage_backend":         "MongoDB Atlas + GridFS (no local files)",
        "warmed_models_in_cache":  list(MODEL_CACHE.keys()),
        "mongo_connected":         MONGO_URI is not None,
        "features_in_mongo":       feat_doc is not None,
        "feature_doc_sample_keys": list(feat_doc.keys())[:8] if feat_doc else [],
    }), 200


@app.route("/latest_features", methods=["GET"])
def latest_features():
    """Return a safe subset of the most recent processed_features document."""
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
    """Return the most recent document from realtime_observations (TTL collection)."""
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db["realtime_observations"].find_one({}, sort=[("datetime", -1)])
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
    """
    Run inference for a given model / horizon combination.

    Request body (JSON):
      { "features": { "pm25": float, "pm10": float, "temperature": float,
                      "humidity": float, "wind_speed": float, ... } }

    Notes on use_log flag (confirmed from training scripts):
      - RandomForest: use_log=True  → model predicts log1p(AQI); expm1 applied here
      - XGBoost:      use_log=False → model predicts raw AQI directly (FIX B in v4)
      - Ridge:        use_log=True  → Pipeline.predict returns log1p(AQI); expm1 applied
    """
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {VALID_MODELS}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {sorted(VALID_HORIZONS)}"}), 400

    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Request body must contain a 'features' key."}), 400

    try:
        artifact         = load_prediction_artifacts(model_type, horizon)
        model            = artifact["model"]
        conformal_margin = artifact["conformal_margin"]
        use_log          = artifact.get("use_log", False)

        input_df       = _build_feature_vector(artifact, payload["features"])
        raw_prediction = float(model.predict(input_df)[0])

        # RF and Ridge train on log1p(AQI) — back-convert to raw AQI scale
        if use_log:
            raw_prediction = float(np.expm1(max(0.0, raw_prediction)))

        # XGBoost v4 removed the residual corrector (train_xgboost.py FIX A).
        # This block is kept for backward-compatibility with any older artifact
        # that may still carry a corrector key.
        corrector = artifact.get("corrector")
        if corrector is not None:
            try:
                correction     = float(corrector.predict(input_df)[0])
                raw_prediction = float(np.clip(raw_prediction + correction, 0.0, 500.0))
                print(f"[predict] {model_type}/{horizon}h legacy corrector applied: Δ={correction:+.2f}")
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
    except Exception:
        # FIX 7: log full traceback server-side; return a clean message to client
        print(f"[predict] EXCEPTION for {model_type}/{horizon}h:\n{traceback.format_exc()}")
        return jsonify({"error": "Inference failure — check server logs for traceback."}), 500


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
    """
    Return top-10 mean |SHAP| records from model_shap collection.
    Populated by save_shap() in train_random_forest.py and train_xgboost.py.
    Ridge does NOT call save_shap() — use /features/<model>/<horizon> instead,
    which returns coefficient magnitudes in the same record format.
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
                "message": (
                    f"No SHAP data for {store_name} {horizon}h. "
                    "Run the training pipeline first.  "
                    "For Ridge, use /features/<model>/<horizon> instead."
                ),
            }), 200
        return jsonify({
            "model":     store_name,
            "horizon_h": horizon,
            "records":   doc.get("records", [])[:10],
        }), 200
    except Exception as e:
        return jsonify({"error": f"SHAP fetch failed: {str(e)}"}), 500


@app.route("/features/<string:model_type>/<int:horizon>", methods=["GET"])
def get_feature_importance(model_type: str, horizon: int):
    """
    Return top-10 features by importance from model_features collection.

    For Random Forest and XGBoost: importance = gain-based feature importance
    stored by save_feature_list() during training.

    For Ridge: importance = |coefficient| stored by save_feature_list() with
    the coefficient array as the `importance` field.  This endpoint is the
    correct one for Ridge because Ridge does not use TreeExplainer / SHAP.

    Records are returned in the same schema as /shap/ for Streamlit compatibility:
      { "feature": str, "mean_abs_shap": float }
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
                    f"No feature data for {store_name} {horizon}h. "
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
                    "Retrain the model to populate these values."
                ),
            }), 200

        pairs = sorted(
            zip(feature_names, importance),
            key=lambda x: abs(x[1]) if x[1] is not None else 0.0,
            reverse=True,
        )[:10]

        records = [
            {"feature": f, "mean_abs_shap": float(abs(v)) if v is not None else 0.0}
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


# ── Debug routes ──────────────────────────────────────────────────────────────

@app.route("/debug/metrics_raw", methods=["GET"])
def debug_metrics_raw():
    """Return the 5 most recent model_metrics documents (includes error_by_band)."""
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
    """Return the most recent processed_features document in full."""
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        doc    = db[FEAT_COL].find_one({}, sort=[("datetime", -1)])
        if doc is None:
            doc = db[FEAT_COL].find_one({}, sort=[("timestamp", -1)])
        client.close()
        if doc:
            doc.pop("_id", None)
            return jsonify({"found": True, "doc": doc}), 200
        return jsonify({
            "found": False,
            "hint":  "processed_features is empty — run the feature pipeline first.",
        }), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/artifacts", methods=["GET"])
def debug_artifacts():
    """List all model_artifacts pointer documents (no GridFS binary download)."""
    try:
        client = _mongo_client()
        db     = client[DB_NAME]
        docs   = list(db["model_artifacts"].find(
            {},
            {"_id": 0, "model": 1, "horizon_h": 1, "updated_at": 1,
             "feature_names": 1, "size_bytes": 1, "compressed": 1},
        ))
        client.close()
        for d in docs:
            if "feature_names" in d:
                d["feature_count"] = len(d.pop("feature_names"))
        return jsonify({"found": bool(docs), "count": len(docs), "artifacts": docs}), 200
    except PyMongoError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/feature_mismatch/<string:model_type>/<int:horizon>", methods=["GET"])
def debug_feature_mismatch(model_type: str, horizon: int):
    """
    Show which features the artifact expects but are absent from the latest
    processed_features document.  Useful for diagnosing high 0-feature warnings
    in _build_feature_vector after a schema change in feature_engineering.py.
    """
    model_type = model_type.lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {VALID_MODELS}"}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Invalid horizon. Choose from: {sorted(VALID_HORIZONS)}"}), 400
    try:
        artifact   = load_prediction_artifacts(model_type, horizon)
        latest_doc = _get_latest_feature_doc()
        expected   = set(artifact.get("feature_names", []))
        present    = set(latest_doc.keys()) if latest_doc else set()
        missing    = sorted(expected - present)
        extra      = sorted(present - expected)
        return jsonify({
            "model":                   _API_TO_MONGO_NAME[model_type],
            "horizon_h":               horizon,
            "artifact_feature_count":  len(expected),
            "feature_doc_key_count":   len(present),
            "missing_from_doc":        missing,
            "missing_count":           len(missing),
            "extra_in_doc_not_needed": extra[:20],
            "use_log":                 artifact.get("use_log"),
            "use_interactions":        artifact.get("use_interactions"),
            "use_corrector":           artifact.get("use_corrector"),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Kick off background warmup when gunicorn imports this module ──────────────
_warmup_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)