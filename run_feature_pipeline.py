# run_feature_pipeline.py
from feature_pipeline import build_dataset, feature_engineering

def run():
    print("Starting data fetch and feature generation...")
    # Add the functions you usually call to run the pipeline
    build_dataset.run() 
    feature_engineering.process_all()
    print("Feature pipeline complete.")

if __name__ == "__main__":
    run()