"""
run_training_pipeline.py (CI/CD DEPLOYMENT ENGINE)
-------------------------------------------------
Location: ROOT DIRECTORY

Acts as Phase 0 Data Warm-up. Fetches features matrix from MongoDB exactly ONCE,
and saves individual horizon snapshots to disk inside 'training_pipeline/' so 
subsequent training scripts can load them instantly off the runner's local SSD.
"""

import os
import sys
import time
from pathlib import Path

# Resolve absolute path locations cleanly
ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "training_pipeline"

# Inject training_pipeline directly to head of system path so all sub-imports match
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Clear IDE analysis warnings by using explicit namespace path sourcing
from training_pipeline import load_data as _ld


def run():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("🚀 GITHUB ACTIONS PIPELINE: CACHE WARMING PHASE (SINGLE MONGODB FETCH)", flush=True)
    print("=" * 80, flush=True)

    # Core single-pass fetch operation from Atlas Cluster
    print("Establishing connection to cloud database pool to cache workspace...", flush=True)
    df = _ld._fetch_from_feature_store()
    
    # Materialize individual split matrix frames on runner high-speed virtual SSD
    for horizon in (24, 48, 72):
        print(f"\n📦 Materializing local workspace cache for horizon {horizon}h...", flush=True)
        X, y_log, y_raw = _ld._build_X_y(df, horizon)
        
        # Absolute serialization dump points inside the training_pipeline folder
        X.to_pickle(PIPELINE_DIR / f"X_cache_{horizon}h.pkl")
        y_log.to_pickle(PIPELINE_DIR / f"y_log_cache_{horizon}h.pkl")
        y_raw.to_pickle(PIPELINE_DIR / f"y_raw_cache_{horizon}h.pkl")
        print(f"   Saved local frames to: {PIPELINE_DIR / f'X_cache_{horizon}h.pkl'}", flush=True)

    print(f"\n✅ Phase 0 Warmup completed in {time.time() - t0:.2f}s. Feature workspace is primed!", flush=True)


if __name__ == "__main__":
    run()