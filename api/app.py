"""
Enterprise MLOps Prediction API Service Layer.
Exposes real-time endpoints for Multi-Model 24h, 48h, and 72h AQI forecasting.
"""

import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

# ── SYSTEM CONSTANTS & LOCAL PATH LAYOUT ───────────────────────────────────
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent 
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

app = Flask(__name__)
MODEL_CACHE = {}
VALID_MODELS = {"random_forest", "xgboost", "ridge"}
VALID_HORIZONS = {24, 48, 72}


def load_prediction_artifacts(model_type: str, horizon: int):
    """Retrieves serialized model instances and conformal parameters from disk."""
    cache_key = f"{model_type}_{horizon}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    # Map API model_type names to the file prefixes used by the training scripts:
    #   train_random_forest.py → random_forest_{h}h.pkl
    #   train_xgboost.py       → xgboost_{h}h.pkl
    #   train_ridge.py         → ridge_{h}h.pkl
    FILE_PREFIX_MAP = {
        "random_forest": "random_forest",
        "xgboost":       "xgboost",
        "ridge":         "ridge",
    }
    file_prefix = FILE_PREFIX_MAP[model_type]
    model_path = MODELS_DIR / f"{file_prefix}_{horizon}h.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Requested model artifact file asset is missing on server disk: {model_path.name}"
        )

    artifacts = joblib.load(model_path)
    MODEL_CACHE[cache_key] = artifacts
    return artifacts


@app.route("/health", methods=["GET"])
def health_check():
    """System liveness and readiness probe for cloud monitoring gateways."""
    discovered_artifacts = {}
    
    # Audit model inventory footprint across your local storage layer
    for m in VALID_MODELS:
        discovered_artifacts[m] = []
        for h in VALID_HORIZONS:
            if (MODELS_DIR / f"{m}_{h}h.pkl").exists():
                discovered_artifacts[m].append(f"{h}h")

    return jsonify({
        "status": "healthy",
        "service": "aqi-multi-model-forecasting-api",
        "warmed_models_in_cache": list(MODEL_CACHE.keys()),
        "discovered_artifacts_on_disk": discovered_artifacts
    }), 200


@app.route("/predict/<string:model_type>/<int:horizon>", methods=["POST"])
def predict_aqi(model_type: str, horizon: int):
    """
    Main inference routing portal.
    Accepts a model algorithm name, a forecasting hour horizon, 
    and a JSON feature vector array payload to output bounded predictions.
    """
    # Normalize path strings to prevent registry casing issues
    model_type = model_type.lower()

    if model_type not in VALID_MODELS:
        return jsonify({
            "error": f"Invalid model engine path request. Choose from: {list(VALID_MODELS)}"
        }), 400

    if horizon not in VALID_HORIZONS:
        return jsonify({
            "error": f"Invalid forecast horizon request route. Choose from: {list(VALID_HORIZONS)}"
        }), 400

    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({
            "error": "Malformed request. JSON payload must contain a valid 'features' key dictionary."
        }), 400

    try:
        # Load cached or on-disk targeted machine learning engine components
        artifacts = load_prediction_artifacts(model_type, horizon)
        model = artifacts["model"]
        expected_features = artifacts["feature_names"]
        conformal_margin = artifacts["conformal_margin"]

        # Convert incoming JSON vector array directly into a Pandas DataFrame
        input_data = pd.DataFrame([payload["features"]])

        # Schema Alignment: Create template and update
        df_template = pd.DataFrame(0.0, index=[0], columns=expected_features)
        df_template.update(input_data)
        
        # Assign back to input_data
        input_data = df_template

        # Execute model inference
        raw_prediction = float(model.predict(input_data)[0])

        # All three models train on log1p(AQI) — inverse-transform before use
        use_log = artifacts.get("use_log", False)
        if use_log:
            raw_prediction = float(np.expm1(max(0.0, raw_prediction)))
        raw_prediction = min(max(raw_prediction, 0.0), 500.0)

        # Apply Conformal Calibration Safety Boundaries
        lower_bound = max(0.0, raw_prediction - conformal_margin)
        upper_bound = min(500.0, raw_prediction + conformal_margin)

        # Categorize the prediction (Cleaned up the syntax error here)
        if raw_prediction <= 50:
            category = "Good"
        elif raw_prediction <= 100:
            category = "Moderate"
        elif raw_prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif raw_prediction <= 200:
            category = "Unhealthy"
        elif raw_prediction <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        return jsonify({
            "model_engine": model_type,
            "horizon_hours": horizon,
            "aqi_prediction": round(raw_prediction, 1),
            "lower_bound_95ci": round(lower_bound, 1),
            "upper_bound_95ci": round(upper_bound, 1),
            "conformal_margin_applied": round(conformal_margin, 1),
            "epa_category": category,
            "status": "success"
        }), 200
    
    except FileNotFoundError as fnf:
        return jsonify({
            "error": str(fnf), 
            "hint": f"Run training script for '{model_type}' to generate binary weights before calling this endpoint."
        }), 500
    except Exception as e:
        return jsonify({
            "error": f"Internal inference execution pipeline failure: {str(e)}"
        }), 500
@app.route("/metrics/<string:model_type>", methods=["GET"])
def get_model_metrics(model_type):
    """Exposes the training artifacts' performance metrics to the dashboard."""
    metrics_report = {}
    model_type = model_type.lower()
    if model_type == "random_forest":
        file_prefix = "rf"
    elif model_type == "xgboost":
        file_prefix = "xgb"
    else:
        file_prefix = model_type
    for h in VALID_HORIZONS:
        try:
            # Assumes your training script saves a metrics_{model}_{h}h.json file
            m_path = METRICS_DIR / f"{file_prefix}_metrics_{h}h.json"
            with open(m_path, "r") as f:
                metrics_report[str(h)] = json.load(f)
        except:
            metrics_report[str(h)] = {"r2": 0.0, "rmse": 0.0}
    return jsonify(metrics_report)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)