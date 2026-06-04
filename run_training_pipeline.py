# No system path manipulation or hacks needed!
from training_pipeline import load_data
from training_pipeline import train_random_forest
from training_pipeline import train_xgboost
from training_pipeline import train_ridge

def run():
    print("Loading data and starting training...")
    data = load_data.get_data()
    train_random_forest.train(data)
    train_xgboost.train(data)
    train_ridge.train(data)
    print("Training complete.")

if __name__ == "__main__":
    run()