import os
import sys

# 1. Force the subfolder to the front of Python's path so subfiles resolve imports flawlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = os.path.join(current_dir, "training_pipeline")
sys.path.insert(0, subfolder_dir)

# 2. Import your modules cleanly
import train_random_forest
import train_xgboost
import train_ridge

def run():
    print("🚀 Starting Master Retraining Loop across Forecasting Horizons...")
    
    # Loop across your operational lookahead windows
    for horizon in [24, 48, 72]:
        print(f"\n============================================================")
        print(f"🤖 Triggering specialized training engines for {horizon}h Horizon...")
        
        # Call the exact entry-point functions that exist inside your scripts
        # Pass only the horizon integer since the scripts manage their own data pipeline logic internally!
        train_random_forest.train_rf(horizon=horizon)
        train_xgboost.train_xgboost(horizon=horizon)
        train_ridge.train_ridge(horizon=horizon)
        
    print("\n✅ RETRAINING PIPELINE SYSTEM COMPLETELY FINISHED.")

if __name__ == "__main__":
    run()