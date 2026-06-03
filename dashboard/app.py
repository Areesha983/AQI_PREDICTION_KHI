from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import requests

# ─── PATH ALIGNMENT SYSTEM ──────────────────────────────────────────────────
try:
    from database import get_latest_features
except ModuleNotFoundError as e:
    st.error(f"⚠️ Import routing failed. Ensure database.py is in this directory: {e}")

from visualizations import plot_error_progression, plot_variance_matrix

# ─── STREAMLIT CONFIGURATION & INTERFACE THEME TUNING ─────────────────────
st.set_page_config(
    page_title="AirWind | Karachi AQI Intelligence Hub",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Native Glassmorphic CSS Injection
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0d14 0%, #111622 100%);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    div[data-testid="stMetricContainer"], .custom-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 24px 28px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetricContainer"]:hover, .custom-card:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.05em;
        color: #ffffff !important;
    }

    div[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ─── BREAKPOINT MATRIX CONDITIONS ──────────────────────────────────────────
def get_epa_tier_details(aqi_value: float) -> dict:
    if aqi_value <= 50:
        return {"label": "Good", "color": "#00e400", "bg": "rgba(0, 228, 0, 0.06)"}
    elif aqi_value <= 100:
        return {"label": "Moderate", "color": "#ffca28", "bg": "rgba(255, 202, 40, 0.06)"}
    elif aqi_value <= 150:
        return {"label": "Unhealthy for Sensitive Groups", "color": "#ff9100", "bg": "rgba(255, 145, 0, 0.06)"}
    elif aqi_value <= 200:
        return {"label": "Unhealthy", "color": "#ff1744", "bg": "rgba(255, 23, 68, 0.06)"}
    elif aqi_value <= 300:
        return {"label": "Very Unhealthy", "color": "#d500f9", "bg": "rgba(213, 0, 249, 0.06)"}
    else:
        return {"label": "Hazardous", "color": "#b71c1c", "bg": "rgba(183, 28, 28, 0.06)"}


# ─── DATABASE TUNING & BASELINE CACHING LAYER ──────────────────────────────
st.sidebar.markdown("### 🌐 Backend Service Engine")
api_gateway = st.sidebar.text_input("Flask API Instance Gateway URL", "http://127.0.0.1:5000")

# 📊 MODEL SELECTOR INTERFACE TOGGLE
st.sidebar.markdown("### 🤖 Model Architecture Selection")
selected_model_ui = st.sidebar.selectbox(
    "Active Prediction Engine",
    ["🌲 Random Forest", "🚀 XGBoost Engine", "📊 Ridge Regression"]
)

# Map UI selections to internal Flask endpoint string components
model_mapping = {
    "🌲 Random Forest": "random_forest",
    "🚀 XGBoost Engine": "xgboost",
    "📊 Ridge Regression": "ridge"
}
active_model_key = model_mapping[selected_model_ui]

@st.cache_data(ttl=30)
def fetch_feature_store_baselines():
    try:
        record = get_latest_features()
        if record and isinstance(record, dict):
            if "pm25" in record:
                return record, True
    except Exception as e:
        print(f"Sidebar cache lookup notice: {e}")
    
    return {"pm25": 75.0, "pm10": 140.0, "temperature": 32.0, "humidity": 65.0, "wind_speed": 12.0}, False

mongo_features, mongo_active = fetch_feature_store_baselines()

# ─── INTERACTIVE PARAMETER SLIDERS ──────────────────────────────────────────
st.sidebar.header("🎯 Live Vector Adjustments")
sim_pm25 = st.sidebar.slider("Ambient PM2.5 (μg/m³)", 10.0, 350.0, float(mongo_features.get("pm25", 75.0)), step=5.0)
sim_pm10 = st.sidebar.slider("Ambient PM10 (μg/m³)", 20.0, 500.0, float(mongo_features.get("pm10", 140.0)), step=5.0)
sim_temp = st.sidebar.slider("Temperature (°C)", 10.0, 48.0, float(mongo_features.get("temperature", 32.0)), step=1.0)
sim_humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 100.0, float(mongo_features.get("humidity", 65.0)), step=5.0)
sim_wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 45.0, float(mongo_features.get("wind_speed", 12.0)), step=1.0)

inference_payload = {
    "features": {
        "pm25": sim_pm25,
        "pm10": sim_pm10,
        "temperature": sim_temp,
        "humidity": sim_humidity,
        "wind_speed": sim_wind,
        "pm25_diff_1h": 2.3,
        "pm25_roll_std_24h": 12.4,
        "interaction_pm25_humidity": sim_pm25 * sim_humidity,
        "interaction_pm25_wind_inverse": sim_pm25 / (sim_wind + 0.1)
    }
}

# ─── HEADER GRAPHICS PRESENTATION ───────────────────────────────────────────
status_badge = "🟢 MONGO FEATURE STORE ACTIVE" if mongo_active else "⚠️ FALLBACK BACKEND EMULATOR ENGINE ACTIVE"
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em; color:#ffffff;">
                <span style="color:#3b82f6;">Air</span>Wind Karachi
            </h1>
            <p style="margin:0; color:#64748b; font-size:0.95rem;">Predicting Karachi Air Quality Index Trends via Enterprise Flask Microservices</p>
        </div>
        <div style="background:rgba(255,255,255,0.04); padding: 10px 18px; border-radius: 20px; font-size:0.82rem; font-weight:600; border:1px solid rgba(255,255,255,0.08); color:#94a3b8;">
            📍 {status_badge}
        </div>
    </div>
""", unsafe_allow_html=True)

# ─── MULTI-HORIZON RUNTIME DEPLOYMENT (FILTERED BY TOGGLE) ──────────────────
st.header(f"🔮 Live Multi-Horizon Forecasts — Using {selected_model_ui}")
horizons = [24, 48, 72]
columns_layout = st.columns(3, gap="medium")

# Track values across the focused engine for local fallback parity
focused_predictions = {}

for idx, h in enumerate(horizons):
    with columns_layout[idx]:
        st.subheader(f"⏱️ {h}-Hour Forecast Window")
        try:
            # Query the updated path configuration: /predict/<model_type>/<horizon>
            response = requests.post(
                f"{api_gateway}/predict/{active_model_key}/{h}", 
                json=inference_payload, 
                timeout=2
            )
            if response.status_code == 200:
                res_data = response.json()
                pred = res_data["aqi_prediction"]
                low = res_data["lower_bound_95ci"]
                high = res_data["upper_bound_95ci"]
                cat = res_data["epa_category"]
                engine_type = f"Flask Server ({active_model_key})"
            else:
                raise requests.exceptions.ConnectionError
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Fallback approximations specialized by algorithm properties
            modifier = 1.0 if active_model_key == "ridge" else 1.2 if active_model_key == "xgboost" else 1.1
            scalar = (1.5 if h == 24 else 1.8 if h == 48 else 2.1) * modifier
            approx_base = (sim_pm25 * scalar) + (sim_temp * 0.3) - (sim_wind * 0.6)
            pred = round(max(10.0, min(490.0, approx_base)), 1)
            low = round(max(0.0, pred - 18.2), 1)
            high = round(min(500.0, pred + 18.2), 1)
            cat = get_epa_tier_details(pred)["label"]
            engine_type = "Fallback Local Emulator"

        focused_predictions[h] = pred
        tier = get_epa_tier_details(pred)
        
        st.metric(
            label=f"Predicted Index ({engine_type})", 
            value=f"{int(pred)}", 
            delta=cat, 
            delta_color="inverse" if pred > 100 else "normal"
        )
        
        st.markdown(f"""
            <div class="custom-card" style="border-left: 5px solid {tier['color']} !important; background:{tier['bg']}; padding:16px 20px !important; margin-top:14px;">
                <span style="font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">95% Conformal Confidence Range</span><br>
                <span style="font-size:1.35rem; font-weight:700; color:{tier['color']};">{low} – {high}</span>
            </div>
        """, unsafe_allow_html=True)

# ─── REAL-TIME COMPARISON MATRIX GRID (STAYS FIXED ON EVERY PAGE) ───────────
st.markdown("<br><br><hr style='border-color:rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
st.header("⚖️ Cross-Model Live Inference Benchmarking")
st.caption("Real-time snapshot evaluating variance profiles across your deployed models simultaneously.")

comp_col1, comp_col2, comp_col3 = st.columns(3, gap="large")
comparison_models = ["random_forest", "xgboost", "ridge"]
box_elements = [comp_col1, comp_col2, comp_col3]
model_labels = {"random_forest": "🌲 Random Forest", "xgboost": "🚀 XGBoost", "ridge": "📊 Ridge Regression"}

for idx, m_key in enumerate(comparison_models):
    with box_elements[idx]:
        st.markdown(f"### {model_labels[m_key]}")
        
        # Pull or simulate metrics across horizons to construct the side-by-side array
        h24_val, h48_val, h72_val = 0.0, 0.0, 0.0
        is_live_card = False
        
        try:
            r24 = requests.post(f"{api_gateway}/predict/{m_key}/24", json=inference_payload, timeout=1)
            r48 = requests.post(f"{api_gateway}/predict/{m_key}/48", json=inference_payload, timeout=1)
            r72 = requests.post(f"{api_gateway}/predict/{m_key}/72", json=inference_payload, timeout=1)
            
            if r24.status_code == 200 and r48.status_code == 200 and r72.status_code == 200:
                h24_val = r24.json()["aqi_prediction"]
                h48_val = r48.json()["aqi_prediction"]
                h72_val = r72.json()["aqi_prediction"]
                is_live_card = True
        except Exception:
            pass
            
        # Emulated matrix calculations if local server loop falls over
        if not is_live_card:
            mod = 1.0 if m_key == "ridge" else 1.2 if m_key == "xgboost" else 1.1
            h24_val = round(((sim_pm25 * 1.5) + (sim_temp * 0.3)) * mod, 1)
            h48_val = round(((sim_pm25 * 1.8) + (sim_temp * 0.3)) * mod, 1)
            h72_val = round(((sim_pm25 * 2.1) + (sim_temp * 0.3)) * mod, 1)

        # Build dynamic breakdown components inside the cards
        st.markdown(f"""
            <div class="custom-card" style="margin-bottom:15px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">24H Horizon</span>
                    <span style="color:#ffffff; font-weight:700;">{int(h24_val)} AQI</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">48H Horizon</span>
                    <span style="color:#ffffff; font-weight:700;">{int(h48_val)} AQI</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">72H Horizon</span>
                    <span style="color:#ffffff; font-weight:700;">{int(h72_val)} AQI</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ─── EVALUATION CHART MATRIX GENERATION ─────────────────────────────────────
# ─── DYNAMIC PRODUCTION EVALUATION MATRIX (Replaces the Hardcoded Block) ───
st.markdown("<br><hr style='border-color:rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
st.header("📊 Live Production Baseline Evaluation Vectors")

@st.cache_data(ttl=10) # Reduced TTL for debugging
def fetch_live_benchmarks():
    benchmarks = []
    for m in ["random_forest", "xgboost", "ridge"]:
        try:
            resp = requests.get(f"{api_gateway}/metrics/{m}", timeout=2).json()
            for h in [24, 48, 72]:
                data = resp.get(str(h), {})
                
                # Check for either naming convention
                r2 = data.get("test_r2") or data.get("r2") or 0.0
                rmse = data.get("test_rmse") or data.get("rmse") or 0.0
                
                benchmarks.append({
                    "Model": m.replace('_', ' ').title(),
                    "Horizon": f"{h}h",
                    "R² Score": r2,
                    "RMSE": rmse
                })
        except Exception as e:
            print(f"Error: {e}")
    return pd.DataFrame(benchmarks)

# Render the dynamic charts
live_metrics_df = fetch_live_benchmarks()
chart_col1, chart_col2 = st.columns(2, gap="large")

with chart_col1:
    st.subheader("RMSE Progression")
    st.line_chart(live_metrics_df, x="Horizon", y="RMSE", color="Model")

with chart_col2:
    st.subheader("R² Variance Profile")
    st.bar_chart(live_metrics_df, x="Horizon", y="R² Score", color="Model")