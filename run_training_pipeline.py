import os
import sys

# 1. GET THE ABSOLUTE PATH TO YOUR SUBFOLDER
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = os.path.join(current_dir, "training_pipeline")

# 2. FORCE IT TO THE ABSOLUTE FRONT OF PYTHON'S SEARCH PATH
# This ensures that ALL subfiles like train_random_forest.py can find load_data natively!
sys.path.insert(0, subfolder_dir)

# 3. RUN COMFORTABLE FLAT IMPORTS
import load_data
import train_random_forest
import train_xgboost
import train_ridge

def run():
    print("Loading data and starting training...")
    data = load_data.get_data()
    train_random_forest.train(data)
    train_xgboost.train(data)
    train_ridge.train(data)
    print("Training complete.")

if __name__ == "__main__":
    run()