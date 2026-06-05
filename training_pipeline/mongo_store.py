"""
mongo_store.py
--------------
Centralised MongoDB I/O layer for the Karachi AQI MLOps pipeline.

Model artifacts (large sklearn/xgboost blobs) are stored via GridFS which
handles files of any size by chunking them internally — no 16 MB BSON limit.
All other data (metrics, predictions, residuals, features, SHAP) stays in
regular collections as small documents.

Collections
-----------
  model_metrics      — scalar KPI dicts from each training run
  model_predictions  — actual / predicted / PI arrays (stored as records)
  model_residuals    — residual diagnostics arrays
  model_features     — feature lists, importance scores, SHAP values
  model_shap         — per-feature mean |SHAP| values
  model_shap_plots   — SHAP beeswarm PNG blobs stored via GridFS
  model_artifacts    — metadata doc + binary stored via GridFS (no size limit)
"""

from __future__ import annotations

import io
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs

# ── Connection singleton ──────────────────────────────────────────────────────
_CLIENT: MongoClient | None = None
_DB = None
_FS = None   # GridFS bucket


def get_db():
    global _CLIENT, _DB, _FS
    if _DB is not None:
        return _DB
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set.")
    _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=10_000, socketTimeoutMS=120_000)
    _DB = _CLIENT["karachi_aqi"]
    _FS = gridfs.GridFS(_DB, collection="model_fs")
    return _DB


def _get_fs() -> gridfs.GridFS:
    get_db()   # ensures _FS is initialised
    return _FS


def _upsert(collection_name: str, filter_doc: dict, update_doc: dict) -> None:
    db = get_db()
    db[collection_name].update_one(filter_doc, {"$set": update_doc}, upsert=True)


def _run_id() -> str:
    """UTC timestamp string used to tag each pipeline run."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── GridFS helpers ────────────────────────────────────────────────────────────

def _gridfs_upsert(filename: str, data: bytes, metadata: dict) -> str:
    """
    Store binary data in GridFS under `filename`.  If a file with the same
    filename already exists it is deleted first so we always have one current
    version per (model, horizon).  Returns the new file _id as a string.
    """
    fs = _get_fs()
    # Remove any existing version
    for existing in fs.find({"filename": filename}):
        fs.delete(existing._id)
    file_id = fs.put(data, filename=filename, metadata=metadata)
    return str(file_id)


def _gridfs_get(filename: str) -> bytes | None:
    fs = _get_fs()
    if not fs.exists({"filename": filename}):
        return None
    grid_out = fs.get_last_version(filename)
    return grid_out.read()


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
    _upsert("model_metrics", {"model": model, "horizon_h": horizon}, doc)
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
            "i":      int(i),
            "actual": float(y_actual[i]),
            "pred":   float(y_pred[i]),
            "lower":  float(pi_lower[i]),
            "upper":  float(pi_upper[i]),
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
    _upsert("model_predictions", {"model": model, "horizon_h": horizon}, doc)
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
        {
            "i":         int(i),
            "actual":    float(y_actual[i]),
            "predicted": float(y_pred[i]),
            "residual":  float(residuals[i]),
        }
        for i in range(len(y_actual))
    ]
    doc = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     run_id or _run_id(),
        "updated_at": datetime.now(tz=timezone.utc),
        "records":    records,
    }
    _upsert("model_residuals", {"model": model, "horizon_h": horizon}, doc)
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
    _upsert("model_features", {"model": model, "horizon_h": horizon}, doc)
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
    _upsert("model_shap", {"model": model, "horizon_h": horizon}, doc)
    print(f"  [mongo_store] ✅ SHAP saved → model_shap ({model} {horizon}h)")


def save_shap_plot_png(model: str, horizon: int, fig, run_id: str | None = None) -> None:
    """Store a matplotlib figure as a PNG in GridFS (no size limit)."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    rid      = run_id or _run_id()
    filename = f"shap_plot_{model}_{model}_{horizon}h.png"
    metadata = {
        "model":      model,
        "horizon_h":  horizon,
        "run_id":     rid,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    file_id = _gridfs_upsert(filename, png_bytes, metadata)

    # Keep a lightweight pointer doc in a regular collection for easy listing
    _upsert(
        "model_shap_plots",
        {"model": model, "horizon_h": horizon},
        {
            "model":      model,
            "horizon_h":  horizon,
            "run_id":     rid,
            "updated_at": datetime.now(tz=timezone.utc),
            "gridfs_filename": filename,
            "gridfs_id":       file_id,
        },
    )
    print(f"  [mongo_store] ✅ SHAP plot saved → GridFS ({model} {horizon}h, {len(png_bytes)//1024}KB)")


def save_model_artifact(
    model_name: str,
    horizon: int,
    artifact: dict,
    run_id: str | None = None,
) -> None:
    """
    Serialise the artifact dict with pickle and store via GridFS.
    GridFS chunks the binary into 255 KB pieces internally, so there is no
    document-size limit regardless of how large the fitted model is.

    A lightweight metadata document is upserted into model_artifacts so the
    dashboard / inference code can query which models exist without loading
    the full binary.
    """
    buf = io.BytesIO()
    pickle.dump(artifact, buf)
    buf.seek(0)
    model_bytes = buf.read()

    rid      = run_id or _run_id()
    filename = f"artifact_{model_name}_{horizon}h.pkl"
    metadata = {
        "model":      model_name,
        "horizon_h":  horizon,
        "run_id":     rid,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    file_id = _gridfs_upsert(filename, model_bytes, metadata)

    # Lightweight pointer document — never hits the 16 MB limit
    _upsert(
        "model_artifacts",
        {"model": model_name, "horizon_h": horizon},
        {
            "model":            model_name,
            "horizon_h":        horizon,
            "run_id":           rid,
            "updated_at":       datetime.now(tz=timezone.utc),
            "gridfs_filename":  filename,
            "gridfs_id":        file_id,
            "size_bytes":       len(model_bytes),
            # Metadata duplicated here for fast reads without touching GridFS
            "feature_names":    artifact.get("feature_names", []),
            "conformal_margin": artifact.get("conformal_margin"),
            "use_log":          artifact.get("use_log", True),
        },
    )
    print(
        f"  [mongo_store] ✅ Model artifact saved → GridFS ({model_name} {horizon}h, "
        f"{len(model_bytes) / 1_048_576:.1f} MB)"
    )


def load_model_artifact(model_name: str, horizon: int) -> dict | None:
    """
    Retrieve and deserialise a model artifact from GridFS.
    Returns the full artifact dict (including the fitted estimator) or None.
    """
    db = get_db()
    # Look up the GridFS filename from the pointer document
    pointer = db["model_artifacts"].find_one(
        {"model": model_name, "horizon_h": horizon},
        {"gridfs_filename": 1},
    )
    if not pointer:
        return None

    raw = _gridfs_get(pointer["gridfs_filename"])
    if raw is None:
        return None
    return pickle.loads(raw)


def load_latest_metrics(model: str, horizon: int) -> dict | None:
    db = get_db()
    return db["model_metrics"].find_one(
        {"model": model, "horizon_h": horizon},
        {"_id": 0},
    )