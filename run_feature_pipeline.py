import os
import sys

# Ensure current folder and subfolders are properly tracked in python module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_pipeline import build_dataset, feature_engineering

def run():
    print("Starting data fetch and feature generation...")
    
    # 1. Fetch, merge raw data, and save to 'karachi_aqi_dataset'
    # FIX: Changed from .run() to .main() to match build_dataset.py
    build_dataset.main() 
    
    # 2. Run high-dimensional transformations and update 'processed_features'
    feature_engineering.process_all()
    
    print("Feature pipeline complete.")

if __name__ == "__main__":
    run()