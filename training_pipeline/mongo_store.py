"""
mongo_store.py
--------------
Centralised MongoDB I/O layer for the Karachi AQI MLOps pipeline.

Replaces ALL local file writes (json, csv, pkl, png) with Atlas-backed
collections.  Every write is an upsert keyed on (model, horizon, run_id)
so repeated pipeline runs overwrite the previous run's records instead of
accumulating duplicates.

Collections
-----------
  model_metrics      — scalar KPI dicts from each training run
  model_predictions  — actual / predicted / PI arrays (stored as records)
  model_residuals    — residual diagnostics arrays
  model_features     — feature lists, importance scores, SHAP values
  model_artifacts    — joblib model blobs encoded as base64 GridFS-lite
  model_shap_plots   — SHAP beeswarm PNG blobs (base64)
"""

from __future__ import annotations

import base64
import io
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ── Connection singleton ──────────────────────────────────────────────────────
_CLIENT: MongoClient | None = None
_DB = None


def get_db():
    global _CLIENT, _DB
    if _DB is not None:
        return _DB
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set.")
    _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=10_000, socketTimeoutMS=60_000)
    _DB = _CLIENT["karachi_aqi"]
    return _DB


def _upsert(collection_name: str, filter_doc: dict, update_doc: dict) -> None:
    db = get_db()
    db[collection_name].update_one(filter_doc, {"$set": update_doc}, upsert=True)


def _run_id() -> str:
    """UTC timestamp string used to tag each pipeline run."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── Public API ────────────────────────────────────────────────────────────────

def save_metrics(model: str, horizon: int, metrics: dict, run_id: str | None = None) -> None:
    """Upsert scalar metrics dict into model_metrics collection."""
    rid = run_id or _run_id()
    doc = {
        **metrics,
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     rid,
        "updated_at": datetime.now(tz=timezone.utc),
    }
    _upsert(
        "model_metrics",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ Metrics saved → model_metrics ({model} {horizon}h)")


def save_predictions(
    model: str,
    horizon: int,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    run_id: str | None = None,
) -> None:
    """Store prediction arrays as a list of compact records."""
    rid = run_id or _run_id()
    records = [
        {
            "i":     int(i),
            "actual":  float(y_actual[i]),
            "pred":    float(y_pred[i]),
            "lower":   float(pi_lower[i]),
            "upper":   float(pi_upper[i]),
        }
        for i in range(len(y_actual))
    ]
    doc = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     rid,
        "updated_at": datetime.now(tz=timezone.utc),
        "records":    records,
    }
    _upsert(
        "model_predictions",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ Predictions saved → model_predictions ({model} {horizon}h, n={len(records)})")


def save_residuals(
    model: str,
    horizon: int,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    run_id: str | None = None,
) -> None:
    residuals = y_actual - y_pred
    records = [
        {"i": int(i), "actual": float(y_actual[i]), "predicted": float(y_pred[i]), "residual": float(residuals[i])}
        for i in range(len(y_actual))
    ]
    doc = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     run_id or _run_id(),
        "updated_at": datetime.now(tz=timezone.utc),
        "records":    records,
    }
    _upsert(
        "model_residuals",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ Residuals saved → model_residuals ({model} {horizon}h)")


def save_feature_list(
    model: str,
    horizon: int,
    feature_names: list[str],
    importance: list[float] | None = None,
    dropped: list[str] | None = None,
    run_id: str | None = None,
) -> None:
    doc = {
        "model":         model,
        "horizon_h":     horizon,
        "run_id":        run_id or _run_id(),
        "updated_at":    datetime.now(tz=timezone.utc),
        "feature_names": feature_names,
        "importance":    importance or [],
        "dropped":       dropped or [],
    }
    _upsert(
        "model_features",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ Features saved → model_features ({model} {horizon}h, n={len(feature_names)})")


def save_shap(
    model: str,
    horizon: int,
    feature_names: list[str],
    mean_abs_shap: list[float],
    run_id: str | None = None,
) -> None:
    records = [
        {"feature": f, "mean_abs_shap": float(v)}
        for f, v in zip(feature_names, mean_abs_shap)
    ]
    records.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    doc = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     run_id or _run_id(),
        "updated_at": datetime.now(tz=timezone.utc),
        "records":    records,
    }
    _upsert(
        "model_shap",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ SHAP saved → model_shap ({model} {horizon}h)")


def save_shap_plot_png(model: str, horizon: int, fig, run_id: str | None = None) -> None:
    """Encode a matplotlib figure as base64 PNG and store it in Atlas."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    doc = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     run_id or _run_id(),
        "updated_at": datetime.now(tz=timezone.utc),
        "png_b64":    b64,
    }
    _upsert(
        "model_shap_plots",
        {"model": model, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ SHAP plot saved → model_shap_plots ({model} {horizon}h)")


def save_model_artifact(model_name: str, horizon: int, artifact: dict, run_id: str | None = None) -> None:
    """
    Serialise the joblib artifact dict to bytes, base64-encode it, and
    store in model_artifacts.  The artifact dict should contain at minimum:
      {"model": <fitted estimator>, "feature_names": [...], ...}
    """
    buf = io.BytesIO()
    pickle.dump(artifact, buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    doc = {
        "model":       model_name,
        "horizon_h":   horizon,
        "run_id":      run_id or _run_id(),
        "updated_at":  datetime.now(tz=timezone.utc),
        "artifact_b64": b64,
        # Store metadata separately for quick reads (avoid deserialising blob)
        "feature_names": artifact.get("feature_names", []),
        "conformal_margin": artifact.get("conformal_margin"),
        "use_log": artifact.get("use_log", True),
    }
    _upsert(
        "model_artifacts",
        {"model": model_name, "horizon_h": horizon},
        doc,
    )
    print(f"  [mongo_store] ✅ Model artifact saved → model_artifacts ({model_name} {horizon}h)")


def load_model_artifact(model_name: str, horizon: int) -> dict | None:
    """Retrieve and deserialise a model artifact from Atlas."""
    db = get_db()
    doc = db["model_artifacts"].find_one(
        {"model": model_name, "horizon_h": horizon},
        {"artifact_b64": 1},
    )
    if not doc:
        return None
    raw = base64.b64decode(doc["artifact_b64"])
    return pickle.loads(raw)


def load_latest_metrics(model: str, horizon: int) -> dict | None:
    db = get_db()
    return db["model_metrics"].find_one(
        {"model": model, "horizon_h": horizon},
        {"_id": 0},
    )