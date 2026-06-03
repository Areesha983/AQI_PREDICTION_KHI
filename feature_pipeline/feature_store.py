"""
Enterprise MLOps Feature Store Access Object Core Interface Layer.
Handles authenticated stream ingestion and historical analytics matrix extractions.
"""
import os
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Search for environment keys up the directory stack
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "KarachiAirQualityFeatureStore"
COLLECTION_NAME = "hourly_features"

def get_feature_store_client():
    """Establishes an authenticated connection pool with the remote Atlas Cluster."""
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGO_CONNECTION_STRING is missing from environment variables.")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force validation ping to catch network blocks early
        client.admin.command('ping')
        return client
    except PyMongoError as e:
        print(f"CRITICAL: Failed to establish secure connection pool to Atlas: {e}")
        raise

def ingest_hourly_features(features_payload: list[dict]) -> int:
    """
    Ingests fresh, engineered hourly tracking observations into the store.
    Ensures document integrity with strict upsert logic to prevent duplicate timestamps.
    """
    if not features_payload:
        return 0
        
    client = get_feature_store_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Create a unique index on timestamp to prevent duplicate records
    collection.create_index("timestamp", unique=True)
    
    inserted_count = 0
    for doc in features_payload:
        if "timestamp" not in doc:
            print("WARNING: Skipped document missing index anchor field: 'timestamp'")
            continue
        try:
            # Upsert operations update existing matching keys or insert if new
            result = collection.update_one(
                {"timestamp": doc["timestamp"]},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                inserted_count += 1
        except PyMongoError as e:
            print(f"Database error executing document record mutation: {e}")
            
    client.close()
    return inserted_count

def extract_historical_feature_matrix() -> pd.DataFrame:
    """
    Queries the remote cluster to extract historical records for training.
    Reconstructs data types and returns a clean, sorted Pandas DataFrame.
    """
    client = get_feature_store_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    try:
        cursor = collection.find({}, {"_id": 0})
        documents = list(cursor)
    except PyMongoError as e:
        print(f"Failed to query database feature records: {e}")
        raise
    finally:
        client.close()
        
    if not documents:
        print("WARNING: Feature store collection is completely empty. Returning fallback frame empty state.")
        return pd.DataFrame()
        
    df = pd.DataFrame(documents)
    # Enforce strict chronological order across the dataframe index
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
    return df

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" FEATURE STORE ACCESS ENGINE — DATA LAYER INTEGRITY TESTING")
    print("="*80)
    
    # Generate mock data payload to verify your remote connection string setup
    mock_payload = [
        {
            "timestamp": "2026-06-01T12:00:00Z",
            "aqi": 142.0,
            "pm25": 55.4,
            "pm10": 112.1,
            "temperature": 34.5,
            "humidity": 62.0,
            "wind_speed": 14.2
        },
        {
            "timestamp": "2026-06-01T13:00:00Z",
            "aqi": 155.0,
            "pm25": 64.1,
            "pm10": 128.5,
            "temperature": 35.1,
            "humidity": 59.0,
            "wind_speed": 12.8
        }
    ]
    
    print("\n[Step 1] Initializing secure cluster validation testing...")
    if not os.getenv("MONGO_CONNECTION_STRING"):
        print("-> Action Required: Create a '.env' file in the root directory and add:")
        print("   MONGO_CONNECTION_STRING=\"mongodb+srv://<user>:<pwd>@cluster...\"")
    else:
        try:
            write_count = ingest_hourly_features(mock_payload)
            print(f"-> Ingestion Complete! Successfully wrote or updated {write_count} records.")
            
            print("\n[Step 2] Executing historical extraction test...")
            historical_df = extract_historical_feature_matrix()
            print(f"-> Extraction Success! Retrieved DataFrame dimensions: {historical_df.shape}")
            print(historical_df.head().to_string())
        except Exception as err:
            print(f"-> Test Execution Failure: {err}")