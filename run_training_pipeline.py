"""
run_training_pipeline.py
------------------------
PERF FIX: MongoDB is fetched ONCE per horizon (inside load_xy_both_cached),
shared across all three models via an in-process cache, and never re-fetched.

Old code: 3 models × 3 horizons × 1 fetch each = 9 full MongoDB round-trips.
New code: 3 horizons × 1 fetch each = 3 round-trips total.  (~3× faster I/O)
"""
import os
import sys
import time

current_dir   = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = os.path.join(current_dir, "training_pipeline")
sys.path.insert(0, subfolder_dir)

import train_random_forest
import train_xgboost
import train_ridge

# ── Pre-warm the fetch cache for every horizon before training starts ─────────
# This guarantees all three model files share a single in-memory copy of the
# feature matrix instead of each pulling from MongoDB independently.
import load_data as _ld

def _prime_cache():
    for h in (24, 48, 72):
        if h not in _ld._DATA_CACHE:
            print(f"  [cache] Pre-fetching data for horizon {h}h …")
            _ld.load_xy_both(h)   # populates _ld._DATA_CACHE[h]

def run():
    t0 = time.time()
    print("🚀 Starting Master Retraining Loop across Forecasting Horizons…")
    print("\n── Phase 0: Pre-loading feature store (single fetch per horizon) ──")
    _prime_cache()

    for horizon in [24, 48, 72]:
        print(f"\n{'=' * 65}")
        print(f"🤖 Training all engines for {horizon}h horizon …")
        print(f"{'=' * 65}")
        train_random_forest.train_rf(horizon=horizon)
        train_xgboost.train_xgboost(horizon=horizon)
        train_ridge.train_ridge(horizon=horizon)

    elapsed = time.time() - t0
    print(f"\n✅ RETRAINING PIPELINE FINISHED  [{elapsed/60:.1f} min]")

if __name__ == "__main__":
    run()