👑 AQI_PREDICTION_KARACHI
This repository hosts an Enterprise MLOps Analytical UI Dashboard designed to forecast air quality in Karachi. By leveraging Random Forest regression, conformal prediction intervals, and SHAP explainability, this tool provides urban planners and environmental health researchers with actionable insights into pollution trends.

🛠 Project Structure
The repository is organized to separate inference logic from visualization and storage:

/api/: Flask backend for model inference and real-time alerts.

/dashboard/: Streamlit interface for interactive analysis and SHAP visualization.

/models/: Serialized model artifacts (.pkl) and SHAP summary plots.

/metrics/: Training logs and model cross-validation performance datasets.

🚀 Getting Started
Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:

Bash
pip install -r requirements.txt
Running the System
Initialize the Backend:
Navigate to the root and run the Flask API:

Bash
python api/main.py
Launch the Dashboard:
In a separate terminal, launch the Streamlit interface:

Bash
streamlit run dashboard/app.py
🧠 Core Features
Multi-Horizon Forecasting: Bounded AQI predictions for 24h, 48h, and 72h windows.

Conformal Calibration: 95% certainty boundary brackets to handle forecast uncertainty.

Explainability Matrices: Global SHAP feature attribution maps to understand local pollution drivers (humidity, wind speed, etc.).

Adaptive Alerts: Automatic EPA-standard hazard flagging for immediate urban response.

📈 Model Performance Baseline
The engine is currently tracking the following metrics across validation splits:

Target Metric: RMSE Optimization

Baseline Accuracy: Verified via Time-Series Cross-Validation.

Developed for the Karachi Urban Data Initiatives.
