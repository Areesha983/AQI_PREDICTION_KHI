import os
import sys

# 1. Force the subfolder to the front of Python's path so subfiles resolve imports flawlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = os.path.join(current_dir, "training_pipeline")
sys.path.insert(0, subfolder_dir)

# 2. Import your modules cleanly
import load_data
import train_random_forest
import train_xgboost
import train_ridge

def run():
    print("🚀 Starting Master Retraining Loop across Forecasting Horizons...")
    
    # Loop across your operational lookahead windows
    for horizon in [24, 48, 72]:
        print(f"\n============================================================")
        # load_xy returns (X, y) with NaNs preserved for tree-based native routing
        X, y = load_data.load_xy(horizon=horizon, use_log=True)
        
        # Chronologically partition the matrices into Train, Calibration, and Test splits
        X_train, y_train, X_cal, y_cal, X_test, y_test = load_data.get_chronological_splits(X, y, horizon)
        
        # Package splits into a clean tuple data container for the algorithm runners
        data_splits = (X_train, y_train, X_cal, y_cal, X_test, y_test)
        
        print(f"🤖 Retraining models for {horizon}h Horizon ({X_train.shape[0]:,} training rows)...")
        
        # Train and save the respective models for this specific forecasting horizon
        train_random_forest.train(data_splits, horizon=horizon)
        train_xgboost.train(data_splits, horizon=horizon)
        train_ridge.train(data_splits, horizon=horizon)
        
    print("\n✅ RETRAINING PIPELINE SYSTEM COMPLETELY FINISHED.")

if __name__ == "__main__":
    run()