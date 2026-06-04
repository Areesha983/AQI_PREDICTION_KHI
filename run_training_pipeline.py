import os
import sys

# Dynamically find the absolute path of the training_pipeline folder and add it to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = os.path.join(current_dir, "training_pipeline")
sys.path.append(subfolder_dir)

# Now these imports will resolve flawlessly regardless of where the script is executed from
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