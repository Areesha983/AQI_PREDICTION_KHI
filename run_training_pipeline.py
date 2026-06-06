"""
run_training_pipeline.py  (MONGODB-ONLY)
-------------------------
Location: ROOT DIRECTORY

Phase 0: Fetches the feature matrix from MongoDB exactly once and warms
the in-process load_data cache.  Because all training scripts now run in
the same GitHub Actions step (or can share the same Python process), the
in-memory _DATA_CACHE in load_data.py is sufficient — no pickle files are
written to disk.

If each training script is run in a SEPARATE step (separate process), the
cached data cannot be shared in memory.  In that case this script writes
lightweight per-horizon Parquet snapshots to a temp dir that is shared
across steps via the actions/cache mechanism.  This avoids the MongoDB
round-trip cost without cluttering the repo.
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR     = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "training_pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# FIX: was "from training_pipeline import load_data as _ld" which requires
# an __init__.py package. Since PIPELINE_DIR is already on sys.path, import directly.
import load_data as _ld

# Temporary Parquet cache dir (only used across separate process steps)
CACHE_DIR = PIPELINE_DIR / "_parquet_cache"


def run():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("🚀 PIPELINE PHASE 0: FEATURE STORE WARM-UP (MONGODB → PARQUET CACHE)", flush=True)
    print("=" * 80, flush=True)

    print("Connecting to MongoDB Atlas feature store…", flush=True)
    df = _ld._fetch_from_feature_store()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (24, 48, 72):
        print(f"\n📦 Building horizon {horizon}h feature matrix…", flush=True)
        X, y_log, y_raw = _ld._build_X_y(df, horizon)

        # Write compact Parquet files — much smaller than pickle, no format lock-in
        X.to_parquet(CACHE_DIR / f"X_{horizon}h.parquet",      index=True)
        y_log.to_frame("y_log").to_parquet(CACHE_DIR / f"y_log_{horizon}h.parquet")
        y_raw.to_frame("y_raw").to_parquet(CACHE_DIR / f"y_raw_{horizon}h.parquet")

        print(f"   Cached → {CACHE_DIR / f'X_{horizon}h.parquet'} ({len(X):,} rows × {X.shape[1]} cols)", flush=True)

    elapsed = time.time() - t0
    print(f"\n✅ Phase 0 complete in {elapsed:.2f}s. Parquet cache written to {CACHE_DIR}", flush=True)


if __name__ == "__main__":
    run()