# run_training_pipeline.py
from training_pipeline import load_data, train_random_forest, train_xgboost, train_ridge

def run():
    print("Loading data and starting training...")
    data = load_data.get_data()
    train_random_forest.train(data)
    train_xgboost.train(data)
    train_ridge.train(data)
    print("Training complete.")

if __name__ == "__main__":
    run()