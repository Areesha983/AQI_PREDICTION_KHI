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

BUG FIX — _gridfs_upsert prune scope (root-cause of missing RF artifacts)
--------------------------------------------------------------------------
Previously the prune query inside _gridfs_upsert filtered only on
  { "metadata.model": ..., "metadata.horizon_h": ... }
Both artifact_RandomForest_24h.pkl AND shap_plot_RandomForest_24h.png share
those two metadata fields.  When save_shap_plot_png ran after save_model_artifact
(steps 12 vs 8 in train_random_forest.py), the SHAP-plot upsert found BOTH
files in the prune query, kept KEEP_N=1 (the freshly written SHAP plot at
index 0), and deleted the artifact at index 1.  This is why all RF /predict
routes returned HTTP 500 — load_model_artifact found no GridFS file.

Fix: scope the prune query to the exact filename so each logical file-type
only ever prunes its own prior versions.
"""

from __future__ import annotations

import io
import os
import pickle
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
from pymongo import MongoClient
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
    _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=10_000, socketTimeoutMS=300_000)
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
    Store binary data in GridFS under `filename`, then prune older versions
    of the SAME filename (keeping KEEP_N most recent).

    CRITICAL FIX: The prune query is now scoped to the exact `filename`.
    Previously it filtered only on (metadata.model, metadata.horizon_h), which
    caused artifact_RandomForest_24h.pkl to be deleted when
    shap_plot_RandomForest_24h.png was written — they share the same model +
    horizon_h metadata but are different logical files.  Scoping to filename
    ensures each file-type only prunes its own prior versions.

    Write-before-delete order is preserved: we write the new file first, then
    prune, so a crash during pruning leaves the new file intact.
    """
    fs = _get_fs()

    # 1. Write the new file first — always succeeds before any deletion
    file_id = fs.put(data, filename=filename, metadata=metadata)

    # 2. Find all GridFS versions of THIS EXACT filename, newest first.
    #    Scoping to filename (not just model+horizon_h) prevents cross-file
    #    deletions between artifacts and SHAP plots.
    existing_files = list(fs.find({"filename": filename}).sort("uploadDate", -1))

    # 3. Keep only the newest KEEP_N; delete the rest (all older than the
    #    file we just wrote, since GridFS appends by upload date)
    KEEP_N = 1
    for old_file in existing_files[KEEP_N:]:
        try:
            print(
                f"  [mongo_store] 🗑  Pruning old GridFS version: "
                f"{old_file.filename} ({old_file._id})"
            )
            fs.delete(old_file._id)
        except Exception as e:
            print(
                f"  [mongo_store] ⚠️  Failed to delete GridFS artifact "
                f"{old_file._id}: {e}"
            )

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
    """Store prediction arrays as a list of compact records.

    Retention: documents older than 30 days for the same (model, horizon)
    are pruned to prevent unbounded collection growth across daily retraining.
    """
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

    # 30-day retention — remove stale prediction docs for this (model, horizon)
    db = get_db()
    db["model_predictions"].delete_many(
        {
            "model":      model,
            "horizon_h":  horizon,
            "updated_at": {"$lt": datetime.now(timezone.utc) - timedelta(days=30)},
        }
    )


def save_residuals(
    model: str,
    horizon: int,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    run_id: str | None = None,
) -> None:
    """Store residual diagnostics arrays.

    Retention: documents older than 30 days for the same (model, horizon)
    are pruned to prevent unbounded collection growth across daily retraining.
    """
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

    # 30-day retention — remove stale residual docs for this (model, horizon)
    db = get_db()
    db["model_residuals"].delete_many(
        {
            "model":      model,
            "horizon_h":  horizon,
            "updated_at": {"$lt": datetime.now(timezone.utc) - timedelta(days=30)},
        }
    )


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
    filename = f"shap_plot_{model}_{horizon}h.png"
    metadata = {
        "model":      model,
        "horizon_h":  horizon,
        "file_type":  "shap_plot",   # extra discriminator for clarity
        "run_id":     rid,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    file_id = _gridfs_upsert(filename, png_bytes, metadata)

    # Keep a lightweight pointer doc in a regular collection for easy listing
    _upsert(
        "model_shap_plots",
        {"model": model, "horizon_h": horizon},
        {
            "model":             model,
            "horizon_h":         horizon,
            "run_id":            rid,
            "updated_at":        datetime.now(tz=timezone.utc),
            "gridfs_filename":   filename,
            "gridfs_id":         file_id,
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
    raw_bytes   = buf.read()
    model_bytes = zlib.compress(raw_bytes, level=6)  # FIX: compress before GridFS upload (3-5x smaller)
    compressed  = True

    rid      = run_id or _run_id()
    filename = f"artifact_{model_name}_{horizon}h.pkl"
    metadata = {
        "model":      model_name,
        "horizon_h":  horizon,
        "file_type":  "model_artifact",   # extra discriminator for clarity
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
            "size_bytes":       len(raw_bytes),
            "compressed":       compressed,
            # Metadata duplicated here for fast reads without touching GridFS
            "feature_names":    artifact.get("feature_names", []),
            "conformal_margin": artifact.get("conformal_margin"),
            "use_log":          artifact.get("use_log", True),
        },
    )
    print(
        f"  [mongo_store] ✅ Model artifact saved → GridFS ({model_name} {horizon}h, "
        f"{len(raw_bytes) / 1_048_576:.1f} MB raw → {len(model_bytes) / 1_048_576:.1f} MB compressed)"
    )


def load_model_artifact(model_name: str, horizon: int) -> dict | None:
    """
    Retrieve and deserialise a model artifact from GridFS.
    Returns the full artifact dict (including the fitted estimator) or None.
    Supports both compressed (zlib) and legacy uncompressed artifacts.
    """
    db = get_db()
    pointer = db["model_artifacts"].find_one(
        {"model": model_name, "horizon_h": horizon},
        {"gridfs_filename": 1, "compressed": 1},
    )
    if not pointer:
        return None

    raw = _gridfs_get(pointer["gridfs_filename"])
    if raw is None:
        return None

    # FIX: decompress if saved with zlib compression; fall back for old artifacts
    if pointer.get("compressed", False):
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass  # already uncompressed legacy artifact — load as-is

    return pickle.loads(raw)


def load_latest_metrics(model: str, horizon: int) -> dict | None:
    db = get_db()
    return db["model_metrics"].find_one(
        {"model": model, "horizon_h": horizon},
        {"_id": 0},
    )


def prune_old_artifacts(model_name: str, horizon: int, keep_last: int = 1) -> int:
    """
    Removes older GridFS model artifacts for a given (model_name, horizon) pair,
    keeping only the `keep_last` most recent versions.

    Scoped to the exact artifact filename so SHAP plots are never touched.

    Returns the number of files deleted.
    """
    fs = _get_fs()
    filename = f"artifact_{model_name}_{horizon}h.pkl"
    cursor = fs.find({"filename": filename}).sort("uploadDate", -1)

    all_files = list(cursor)
    to_delete = all_files[keep_last:]
    deleted = 0
    for doc in to_delete:
        try:
            fs.delete(doc._id)
            deleted += 1
            print(f"  [mongo_store] 🗑  prune_old_artifacts: deleted {doc._id} ({model_name} {horizon}h)")
        except Exception as e:
            print(f"  [mongo_store] ⚠️  Could not delete {doc._id}: {e}")
    return deleted


def prune_all_artifacts(keep_last: int = 1) -> None:
    """
    Convenience wrapper: prunes GridFS artifacts for every known
    (model, horizon) combination in one call.

    Run this once from a Python shell to immediately recover Atlas M0 storage
    after a write-block, then rely on _gridfs_upsert's filename-scoped
    single-version enforcement going forward:

        python -c "from mongo_store import prune_all_artifacts; prune_all_artifacts()"
    """
    models   = ["RandomForest", "XGBoost", "Ridge"]
    horizons = [24, 48, 72]
    total    = 0
    for m in models:
        for h in horizons:
            total += prune_old_artifacts(m, h, keep_last=keep_last)
    print(f"  [mongo_store] ✅ prune_all_artifacts complete — {total} file(s) deleted.")