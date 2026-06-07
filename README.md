# AQI Prediction KHI

A production-style MLOps project for forecasting Karachi air quality using historical pollution, weather, engineered time-series features, machine learning models, a Flask API, and a Streamlit dashboard.

The project is designed around a MongoDB Atlas feature store. It fetches raw air-quality and weather data, builds a processed feature table, trains multiple forecasting models, stores model outputs in MongoDB and GridFS, and serves predictions through a dashboard.

BACKEND: https://aqi-prediction-khi.onrender.com
FRONTEND: https://aqipredictionkhi-jtgqduyvkm4vt5aqvfdjal.streamlit.app/

## What this project does

- Forecasts Karachi AQI for multiple horizons, mainly 24, 48, and 72 hours.
- Uses weather, pollutant, lag, rolling, trend, spike, and meteorological dispersion features.
- Trains Random Forest, XGBoost, and Ridge or ElasticNet style models.
- Stores features, metrics, predictions, residuals, SHAP outputs, plots, and model artifacts in MongoDB.
- Serves predictions and metrics through a Flask API.
- Displays forecasts, real-time observations, model metrics, AQI guidance, and SHAP insights in a Streamlit dashboard.
- Automates feature updates hourly and model training daily through GitHub Actions.

## Repository structure

```text
AQI_PREDICTION_KHI/
├── api/
│   ├── __init__.py
│   └── app.py
├── dashboard/
│   ├── app.py
│   ├── alerts.py
│   ├── visualizations.py
│   └── requirements.txt
├── feature_pipeline/
│   ├── build_dataset.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── feature_store.py
│   ├── fetch_air_quality.py
│   └── fetch_weather.py
├── training_pipeline/
│   ├── evaluate.py
│   ├── load_data.py
│   ├── mongo_store.py
│   ├── train_random_forest.py
│   ├── train_ridge.py
│   └── train_xgboost.py
├── .github/workflows/
│   ├── daily_training_pipeline.yml
│   └── feature_pipeline.yml
├── .devcontainer/
│   └── devcontainer.json
├── .vscode/
│   └── settings.json
├── outputs/
│   ├── metrics_report.json
│   └── shap_metadata.json
├── run_feature_pipeline.py
├── run_training_pipeline.py
├── update_realtime.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## System architecture

```text
Open-Meteo air quality data
        +
Open-Meteo weather data
        |
        v
feature_pipeline/
fetch_air_quality.py + fetch_weather.py
        |
        v
feature_pipeline/build_dataset.py
Merged hourly Karachi dataset
        |
        v
feature_pipeline/feature_engineering.py
Processed feature store in MongoDB
        |
        v
training_pipeline/
Random Forest + XGBoost + Ridge
        |
        v
MongoDB collections + GridFS artifacts
        |
        v
api/app.py Flask backend
        |
        v
dashboard/app.py Streamlit dashboard
```

## Data sources

The feature pipeline pulls Karachi data using the coordinates defined in `feature_pipeline/config.py`:

```python
LATITUDE = 24.8607
LONGITUDE = 67.0011
```

The air quality fetcher collects pollutant variables including PM2.5, PM10, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, dust, and UV index.

The weather fetcher collects variables including temperature, humidity, pressure, surface pressure, wind speed, wind direction, wind gusts, precipitation, cloud cover, and dew point.

## MongoDB collections

The project uses MongoDB Atlas as the main data and artifact layer.

Common collections include:

| Collection | Purpose |
|---|---|
| `raw_air_quality` | Raw hourly air quality data |
| `raw_weather` | Raw hourly weather data |
| `karachi_aqi_dataset` | Merged air quality and weather records |
| `processed_features` | Final engineered feature matrix |
| `realtime_observations` | Recent live dashboard data with TTL cleanup |
| `model_metrics` | Model scores and evaluation metrics |
| `model_predictions` | Forecast outputs |
| `model_residuals` | Residual tracking |
| `model_features` | Feature importance data |
| `model_shap` | SHAP values and metadata |
| `model_shap_plots` | SHAP plot artifacts in GridFS |
| `model_artifacts` | Serialized model artifacts in GridFS |
| `pipeline_runs` | Cross-model evaluation summaries |

## Feature pipeline

The feature pipeline is controlled by:

```bash
python run_feature_pipeline.py
```

It runs two major steps:

1. `feature_pipeline/build_dataset.py`
   - Fetches air quality data.
   - Fetches weather data.
   - Merges both sources by timestamp.
   - Forward-fills short gaps.
   - Saves the merged dataset to MongoDB.

2. `feature_pipeline/feature_engineering.py`
   - Computes AQI from pollutant data.
   - Creates future target columns.
   - Creates temporal features.
   - Creates lag and rolling features.
   - Creates trend, momentum, spike, anomaly, recovery, and persistence features.
   - Writes processed records to the `processed_features` collection.

The realtime updater is separate:

```bash
python update_realtime.py
```

It fetches recent air quality and weather data for dashboard freshness and writes it to `realtime_observations`.

## Training pipeline

The training workflow uses MongoDB as the source of truth.

Phase 0 cache preparation:

```bash
python run_training_pipeline.py
```

This creates local Parquet snapshots from MongoDB for faster training steps.

Train the models:

```bash
python training_pipeline/train_random_forest.py
python training_pipeline/train_xgboost.py
python training_pipeline/train_ridge.py
```

Evaluate all models:

```bash
python training_pipeline/evaluate.py
```

### Models used

| Model | Purpose |
|---|---|
| Random Forest | Strong tree-based baseline with nonlinear feature handling |
| XGBoost | Gradient boosting model with residual correction and SHAP support |
| Ridge or ElasticNet style model | Linear benchmark with interaction features and coefficient interpretability |

### Training features

The training layer includes:

- Chronological train, calibration, and test splits.
- Gap buffers to reduce time-series leakage.
- Correlation filtering.
- Median imputation.
- Spike-aware augmentation and sample weighting.
- Conformal prediction intervals.
- Persistence baseline comparison.
- Error metrics by AQI severity band.
- SHAP or feature importance outputs.

## API backend

The Flask backend is located at:

```text
api/app.py
```

Run locally with:

```bash
gunicorn api.app:app
```

For development, you can also run Flask directly if your environment is configured for it.

### Main API routes

| Route | Purpose |
|---|---|
| `/` | API root and service info |
| `/health` | Health check |
| `/latest_features` | Latest processed feature row |
| `/latest_realtime` | Latest realtime observation |
| `/predict/<model_type>/<horizon>` | Forecast using a selected model and horizon |
| `/metrics/<model_type>` | Metrics for one model |
| `/metrics/all` | Metrics for all models |
| `/shap/<model_type>/<horizon>` | SHAP data for a model and horizon |

Example prediction route:

```text
/predict/xgboost/24
```

## Streamlit dashboard

The dashboard is located at:

```text
dashboard/app.py
```

Run locally with:

```bash
streamlit run dashboard/app.py
```

Dashboard capabilities include:

- Current AQI display.
- Realtime and processed feature fallback loading.
- AQI gauge visualization.
- Forecast charts.
- Model metrics.
- SHAP feature importance.
- EPA-style AQI health bands and guidance.
- Dark visual interface with Plotly charts.

The dashboard helper files are:

| File | Purpose |
|---|---|
| `dashboard/alerts.py` | AQI category labels, colors, and health guidance |
| `dashboard/visualizations.py` | Plotly chart helpers |
| `dashboard/requirements.txt` | Lightweight dashboard-specific dependencies |

## GitHub Actions automation

The repo includes two workflow files.

### Hourly feature pipeline

```text
.github/workflows/feature_pipeline.yml
```

Runs at the top of every hour and performs:

1. Historical or training feature pipeline update.
2. Realtime dashboard update.

### Daily training pipeline

```text
.github/workflows/daily_training_pipeline.yml
```

Runs daily at midnight UTC and performs:

1. Phase 0 MongoDB to Parquet cache creation.
2. Random Forest training.
3. XGBoost training.
4. Ridge or ElasticNet training.
5. Cross-model evaluation.

Both workflows use the MongoDB connection string from GitHub secrets.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/Areesha983/AQI_PREDICTION_KHI.git
cd AQI_PREDICTION_KHI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root:

```text
MONGODB_URI=your_mongodb_connection_string
```

For GitHub Actions, add the same value as a repository secret:

```text
MONGO_CONNECTION_STRING
```

### 5. Build the feature store

```bash
python run_feature_pipeline.py
```

### 6. Update realtime dashboard data

```bash
python update_realtime.py
```

### 7. Prepare the training cache

```bash
python run_training_pipeline.py
```

### 8. Train models

```bash
python training_pipeline/train_random_forest.py
python training_pipeline/train_xgboost.py
python training_pipeline/train_ridge.py
```

### 9. Evaluate models

```bash
python training_pipeline/evaluate.py
```

### 10. Start the API

```bash
gunicorn api.app:app
```

### 11. Start the dashboard

```bash
streamlit run dashboard/app.py
```

## Deployment notes

### API deployment

A typical Render setup can use:

```bash
gunicorn api.app:app
```

Required environment variable:

```text
MONGODB_URI
```

### Dashboard deployment

A typical Streamlit deployment should point to:

```text
dashboard/app.py
```

Make sure the dashboard can reach the deployed Flask API and that any required secrets are available in the deployment environment.

## Runtime

The repository specifies:

```text
python-3.11.9
```

The GitHub Actions workflows also use Python 3.11.

## Troubleshooting

### API returns a MongoDB error

Check that `MONGODB_URI` is set correctly and that the MongoDB Atlas network settings allow the deployment environment to connect.

### Dashboard shows no realtime data

Run:

```bash
python update_realtime.py
```

Then check the `realtime_observations` collection.

### Dashboard shows no predictions

Run the training scripts and confirm that model predictions and artifacts were written to MongoDB.

### SHAP section is empty

Train the model again and check the `model_shap`, `model_features`, and `model_shap_plots` collections.

### Model artifacts are missing

Check the `model_artifacts` GridFS collection and confirm that training completed successfully.

### Feature pipeline runs but training has no data

Check these collections in order:

1. `raw_air_quality`
2. `raw_weather`
3. `karachi_aqi_dataset`
4. `processed_features`

## Security notes

Do not commit secrets, `.env` files, model binaries, raw data exports, or generated artifacts to GitHub.

The `.gitignore` already excludes common generated folders and files such as:

- `models/`
- `data/`
- `metrics/`
- `venv/`
- `__pycache__/`
- image, pickle, Keras, and CSV outputs

## Suggested future improvements

- Add Docker deployment files for API and dashboard.
- Add API request examples in a Postman collection.
- Add automated tests for feature engineering and API routes.
- Add dashboard environment configuration for production API URLs.
- Add model registry version history and rollback instructions.
- Add monitoring alerts for failed hourly feature updates.

## License

No license file was visible in the repository at the time this README was created. Add a license file if this project will be shared, reused, or deployed publicly.
