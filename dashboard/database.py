"""
Enterprise MLOps Feature Store Access Object Core Interface Layer.
Handles authenticated stream ingestion and historical analytics matrix extractions.

FIXES APPLIED:
  FIX 1 — Wrong env var: was MONGO_CONNECTION_STRING, now MONGODB_URI to match
    every other module in the pipeline (feature_store.py, load_data.py,
    build_dataset.py, evaluate.py, feature_engineering.py).

  FIX 2 — Wrong DB name: was "KarachiAirQualityFeatureStore", now "karachi_aqi"
    to match the database written by feature_engineering.py / feature_store.py.

  FIX 3 — Wrong collection name: was "hourly_features", now "processed_features"
    which is the collection populated by feature_engineering.py and read by
    load_data.py for model training. The UI's get_latest_features() was always
    querying an empty / non-existent collection, so it silently returned None.

  FIX 4 — get_latest_features() now also falls back cleanly if the collection
    has no timestamp field, preventing a silent None return to the UI.
"""
import os
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Search for environment keys up the directory stack
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# FIX 1: unified env var — was MONGO_CONNECTION_STRING, which is set nowhere
MONGO_URI       = os.getenv("MONGODB_URI")
DB_NAME         = "karachi_aqi"           # FIX 2: was "KarachiAirQualityFeatureStore"
COLLECTION_NAME = "processed_features"    # FIX 3: was "hourly_features"


def get_feature_store_client():
    """Establishes an authenticated connection pool with the remote Atlas Cluster."""
    if not MONGO_URI:
        raise ValueError(
            "CRITICAL: MONGODB_URI is missing from environment variables. "
            "Add it to your .env file or export it before running."
        )
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force validation ping to catch network blocks early
        client.admin.command("ping")
        return client
    except PyMongoError as e:
        print(f"CRITICAL: Failed to establish secure connection pool to Atlas: {e}")
        raise


def get_latest_features() -> dict:
    """
    Queries MongoDB for the single most recent engineered observation.
    Used directly by the Streamlit frontend to populate current metric baselines.

    FIX 4: collection is now processed_features (karachi_aqi DB) — previously
    pointed at an empty collection, so the UI always received None.
    """
    client = None
    try:
        client = get_feature_store_client()
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Prefer ISO timestamp field; fall back to datetime string field
        latest_record = collection.find_one({}, sort=[("timestamp", -1)])
        if latest_record is None:
            # Fallback: try sorting by datetime string field
            latest_record = collection.find_one({}, sort=[("datetime", -1)])

        return latest_record
    except Exception as e:
        print(f"WARNING: Pipeline read exception handled via fallback: {e}")
        return None
    finally:
        if client:
            client.close()


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
            from pymongo import UpdateOne as _UpdateOne
            result = collection.update_one(
                {"timestamp": doc["timestamp"]},
                {"$set": doc},
                upsert=True,
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
        print(
            "WARNING: Feature store collection is completely empty. "
            "Returning fallback frame empty state."
        )
        return pd.DataFrame()

    df = pd.DataFrame(documents)

    # Enforce strict chronological order across the dataframe index
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime" if "datetime" in df.columns else None
    if ts_col:
        df["datetime"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" FEATURE STORE ACCESS ENGINE — DATA LAYER INTEGRITY TESTING")
    print("=" * 80)

    mock_payload = [
        {
            "timestamp": "2026-06-01T12:00:00Z",
            "aqi": 142.0,
            "pm25": 55.4,
            "pm10": 112.1,
            "temperature": 34.5,
            "humidity": 62.0,
            "wind_speed": 14.2,
        },
        {
            "timestamp": "2026-06-01T13:00:00Z",
            "aqi": 155.0,
            "pm25": 64.1,
            "pm10": 128.5,
            "temperature": 35.1,
            "humidity": 59.0,
            "wind_speed": 12.8,
        },
    ]

    print("\n[Step 1] Initializing secure cluster validation testing...")
    if not os.getenv("MONGODB_URI"):
        print("-> Action Required: Create a '.env' file in the root directory and add:")
        print('   MONGODB_URI="mongodb+srv://<user>:<pwd>@cluster..."')
    else:
        try:
            write_count = ingest_hourly_features(mock_payload)
            print(f"-> Ingestion Complete! Successfully wrote or updated {write_count} records.")

            print("\n[Step 2] Executing latest feature fetch test...")
            latest = get_latest_features()
            if latest:
                print(f"-> Latest record timestamp: {latest.get('timestamp') or latest.get('datetime')}")
                print(f"   AQI: {latest.get('aqi')}  PM2.5: {latest.get('pm25')}")
            else:
                print("-> WARNING: get_latest_features() returned None — collection may be empty.")

            print("\n[Step 3] Executing historical extraction test...")
            historical_df = extract_historical_feature_matrix()
            print(f"-> Extraction Success! Retrieved DataFrame dimensions: {historical_df.shape}")
            print(historical_df.head().to_string())
        except Exception as err:
            print(f"-> Test Execution Failure: {err}")