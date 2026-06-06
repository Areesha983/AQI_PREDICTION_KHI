import os
import json
from pymongo import MongoClient
from pathlib import Path
from dotenv import load_dotenv  # Add this

# Load environment variables
load_dotenv() 

# Connect to MongoDB
# Ensure MONGODB_URI is in your .env file
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["karachi_aqi"]

# Use the 'outputs/' folder so it is ignored by Git
OUTPUT_DIR = Path("outputs") 
OUTPUT_DIR.mkdir(exist_ok=True)

def export_report_data():
    # 1. Export Metrics
    metrics_cursor = db["model_metrics"].find({"model": "RandomForest"}, {"_id": 0})
    metrics_data = list(metrics_cursor)
    
    with open(OUTPUT_DIR / "metrics_report.json", "w") as f:
        json.dump(metrics_data, f, default=str, indent=4)
    print(f"✅ Metrics exported to {OUTPUT_DIR}/metrics_report.json")

    # 2. Export SHAP metadata
    shap_cursor = db["model_shap_plots"].find({}, {"_id": 0})
    shap_metadata = list(shap_cursor)
    
    with open(OUTPUT_DIR / "shap_metadata.json", "w") as f:
        json.dump(shap_metadata, f, default=str, indent=4)
    print(f"✅ SHAP metadata exported to {OUTPUT_DIR}/shap_metadata.json")

if __name__ == "__main__":
    export_report_data()