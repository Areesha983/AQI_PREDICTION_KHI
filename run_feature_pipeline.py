"""
run_feature_pipeline.py
------------------------
The master execution controller. Launches data synchronization and
high-dimensional engineering sequences sequentially.
"""
import os
import sys

# Get the absolute path of the directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the feature_pipeline folder
FEATURE_PIPE_DIR = os.path.join(BASE_DIR, "feature_pipeline")

# Tell Python to look inside the feature_pipeline folder for imports
if FEATURE_PIPE_DIR not in sys.path:
    sys.path.append(FEATURE_PIPE_DIR)

# Now safely import the modules directly
import build_dataset
import feature_engineering

def run():
    print("Starting optimized incremental fetch and feature engineering sequence...")
    
    # 1. Runs build_dataset (which triggers dynamic weather & AQI catches, merges them, and saves)
    build_dataset.main() 
    
    # 2. Runs the fast vectorized high-dimensional feature pipeline
    feature_engineering.process_all()
    
    print("Feature pipeline finalized successfully.")

if __name__ == "__main__":
    run()