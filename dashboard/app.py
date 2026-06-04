"""
AirLyst Karachi — AQI Intelligence Dashboard
Streamlit frontend consuming a Flask prediction microservice.
"""

import streamlit as st
import pandas as pd
import requests
import pymongo  # Added to prevent NameError on sorting parameters

from alerts import get_epa_tier_details
from visualizations import plot_error_progression, plot_variance_matrix, plot_aqi_gauge

try:
    from database import get_latest_features
    _DB_AVAILABLE = True
except ModuleNotFoundError:
    _DB_AVAILABLE = False

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWind | Karachi AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg-0:        #07090f;
    --bg-1:        #0d1117;
    --bg-2:        #111827;
    --surface:     rgba(255,255,255,0.03);
    --border:      rgba(255,255,255,0.07);
    --border-hover:rgba(59,130,246,0.45);
    --accent:      #3b82f6;
    --accent-glow: rgba(59,130,246,0.18);
    --text-primary:   #f1f5f9;
    --text-secondary: #64748b;
    --text-muted:     #334155;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── App shell ── */
.stApp {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,130,246,0.08) 0%, transparent 70%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    color: var(--text-primary);
    font-family: var(--font-mono);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-mono) !important; }
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px;
    backdrop-filter: blur(16px);
    transition: border-color .25s, box-shadow .25s, transform .2s;
}
.card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 0 28px var(--accent-glow);
    transform: translateY(-2px);
}

/* ── Metric containers ── */
div[data-testid="stMetricContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: blur(16px);
}
div[data-testid="stMetricContainer"]:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 0 28px var(--accent-glow);
}
div[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    font-family: var(--font-mono) !important;
    color: #ffffff !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-family: var(--font-mono) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary) !important;
}

/* ── Headings ── */
h1, h2, h3 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
.stMarkdown h3 { font-size: 1rem !important; font-weight: 600 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }

/* ── Plotly charts transparent bg ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Hide default chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: block !important; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; display: block !important; }
section[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom:1.5rem;'>
            <span style='font-family:"Syne",sans-serif; font-size:1.3rem; font-weight:800; color:#f1f5f9;'>
                Air<span style='color:#3b82f6;'>Wind</span>
            </span>
            <div style='font-size:0.68rem; color:#334155; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;'>
                Karachi AQI Intelligence
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🔗 Backend")
    api_gateway = st.text_input("Flask API URL", "http://127.0.0.1:5000", label_visibility="collapsed")

    st.markdown("##### 🤖 Model")
    selected_model_ui = st.selectbox(
        "Model",
        ["🌲 Random Forest", "🚀 XGBoost", "📊 Ridge Regression"],
        label_visibility="collapsed",
    )
    model_mapping = {
        "🌲 Random Forest": "random_forest",
        "🚀 XGBoost":       "xgboost",
        "📊 Ridge Regression": "ridge",
    }
    active_model_key = model_mapping[selected_model_ui]

    st.markdown("---")
    st.markdown("##### 🎯 Input Vectors")

    # Seed sliders from MongoDB if available
    @st.cache_data(ttl=30)
    def _load_mongo_features():
        if not _DB_AVAILABLE:
            return {}, False
        try:
            rec = get_latest_features()
            if rec and isinstance(rec, dict) and "pm25" in rec:
                return rec, True
        except Exception:
            pass
        return {}, False

    mongo_features, mongo_active = _load_mongo_features()
    _d = lambda k, v: float(mongo_features.get(k, v))

    sim_pm25     = st.slider("PM2.5 (μg/m³)",    10.0, 350.0, _d("pm25", 75.0),      5.0)
    sim_pm10     = st.slider("PM10 (μg/m³)",      20.0, 500.0, _d("pm10", 140.0),     5.0)
    sim_temp     = st.slider("Temperature (°C)",  10.0,  48.0, _d("temperature", 32.0), 1.0)
    sim_humidity = st.slider("Humidity (%)",       10.0, 100.0, _d("humidity", 65.0),   5.0)
    sim_wind     = st.slider("Wind Speed (km/h)",   0.0,  45.0, _d("wind_speed", 12.0), 1.0)

    st.markdown("---")
    db_dot = "🟢" if mongo_active else "🔴"
    st.markdown(f"""
        <div style='font-size:0.72rem; color:#475569;'>
            {db_dot} {'Mongo feature store' if mongo_active else 'Fallback defaults'}
        </div>
    """, unsafe_allow_html=True)


# ─── INFERENCE PAYLOAD ───────────────────────────────────────────────────────
inference_payload = {
    "features": {
        "pm25":                         sim_pm25,
        "pm10":                         sim_pm10,
        "temperature":                  sim_temp,
        "humidity":                     sim_humidity,
        "wind_speed":                   sim_wind,
        "pm25_diff_1h":                 2.3,
        "pm25_roll_std_24h":            12.4,
        "interaction_pm25_humidity":    sim_pm25 * sim_humidity,
        "interaction_pm25_wind_inverse": sim_pm25 / (sim_wind + 0.1),
    }
}


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:2rem;">
    <div>
        <h1 style="margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.03em; font-family:'Syne',sans-serif; color:#f1f5f9;">
            Air<span style="color:#3b82f6;">Wind</span> Karachi
        </h1>
        <p style="margin:4px 0 0; color:#475569; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">
            Multi-horizon AQI forecasting · Random Forest · XGBoost · Ridge
        </p>
    </div>
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                padding:8px 16px; border-radius:12px; font-size:0.72rem;
                color:#475569; font-family:'JetBrains Mono',monospace; white-space:nowrap;">
        Using &nbsp;<span style="color:#3b82f6;">{selected_model_ui}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── HELPER: single-horizon request ─────────────────────────────────────────
def _fetch_prediction(model_key: str, horizon: int) -> dict | None:
    try:
        r = requests.post(
            f"{api_gateway}/predict/{model_key}/{horizon}",
            json=inference_payload,
            timeout=2,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _fallback_pred(model_key: str, horizon: int) -> dict:
    mod    = 1.0 if model_key == "ridge" else 1.2 if model_key == "xgboost" else 1.1
    scalar = {24: 1.5, 48: 1.8, 72: 2.1}[horizon] * mod
    pred   = round(max(10.0, min(490.0, (sim_pm25 * scalar) + (sim_temp * 0.3) - (sim_wind * 0.6))), 1)
    tier   = get_epa_tier_details(pred)
    return {
        "aqi_prediction":   pred,
        "lower_bound_95ci": round(max(0.0, pred - 18.2), 1),
        "upper_bound_95ci": round(min(500.0, pred + 18.2), 1),
        "epa_category":     tier["label"],
        "_source":          "fallback",
    }


# ─── SECTION 1: MULTI-HORIZON FORECASTS ─────────────────────────────────────
st.markdown("### 🔮 Multi-Horizon Forecast")
st.caption(f"Real-time predictions via **{selected_model_ui}** — 24h · 48h · 72h windows")

gauge_cols   = st.columns(3, gap="medium")
detail_cols  = st.columns(3, gap="medium")
horizons     = [24, 48, 72]

for idx, h in enumerate(horizons):
    raw  = _fetch_prediction(active_model_key, h)
    data = raw if raw else _fallback_pred(active_model_key, h)
    pred = data["aqi_prediction"]
    low  = data["lower_bound_95ci"]
    high = data["upper_bound_95ci"]
    tier = get_epa_tier_details(pred)
    src  = "live" if raw else "fallback"

    # Gauge
    with gauge_cols[idx]:
        fig = plot_aqi_gauge(pred, tier, f"{h}h Forecast")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Detail card
    with detail_cols[idx]:
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid {tier['color']}; background:{tier['bg']};">
            <div style="font-size:0.68rem; color:#475569; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
                95% CI &nbsp;·&nbsp;
                <span style="color:{'#10b981' if src == 'live' else '#f59e0b'};">
                    {'● live' if src == 'live' else '◌ est.'}
                </span>
            </div>
            <div style="font-size:1.5rem; font-weight:700; color:{tier['color']};
                        font-family:'JetBrains Mono',monospace; letter-spacing:-0.02em;">
                {low} – {high}
            </div>
            <div style="font-size:0.73rem; color:#64748b; margin-top:10px; line-height:1.5;">
                {tier['advice']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── SECTION 2: CROSS-MODEL BENCHMARKING ─────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### ⚖️ Cross-Model Live Benchmarking")
st.caption("All three models evaluated simultaneously on the current input vector.")

bench_cols    = st.columns(3, gap="large")
all_models    = ["random_forest", "xgboost", "ridge"]
model_labels  = {"random_forest": "🌲 Random Forest", "xgboost": "🚀 XGBoost", "ridge": "📊 Ridge"}

for idx, m_key in enumerate(all_models):
    with bench_cols[idx]:
        st.markdown(f"**{model_labels[m_key]}**")
        rows = []
        for h in horizons:
            raw  = _fetch_prediction(m_key, h)
            data = raw if raw else _fallback_pred(m_key, h)
            pred = data["aqi_prediction"]
            tier = get_epa_tier_details(pred)
            rows.append((h, pred, tier))

        cards_html = ""
        for h, pred, tier in rows:
            cards_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:10px 14px; margin-bottom:8px; border-radius:10px;
                        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);">
                <span style="font-size:0.78rem; color:#64748b; font-family:'JetBrains Mono',monospace;">{h}h</span>
                <span style="font-size:0.78rem; font-weight:700; color:{tier['color']};
                             font-family:'JetBrains Mono',monospace;">{int(pred)}</span>
                <span style="font-size:0.65rem; color:{tier['color']}; opacity:0.7;">{tier['label']}</span>
            </div>"""

        st.markdown(f'<div style="margin-top:8px;">{cards_html}</div>', unsafe_allow_html=True)


# ─── SECTION 3: MODEL EVALUATION CHARTS (UPDATED FOR MONGODB) ─────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Live Model Evaluation Metrics")
st.caption("Dynamic performance telemetry ($R^2$, RMSE, Coverage) synchronized directly from your MongoDB Atlas MLOps history store.")

@st.cache_data(ttl=30)  # Cache for 30 seconds to prevent hammering your DB on slider moves
def _fetch_metrics_from_mongodb() -> pd.DataFrame:
    rows = []
    default_models = ["Random Forest", "XGBoost", "Ridge"]
    default_horizons = [24, 48, 72]
    
    # 1. Check if the database module is active and connected
    if not _DB_AVAILABLE:
        # Graceful fallback to zeros if database connection file is missing
        for m in default_models:
            for h in default_horizons:
                rows.append({"Model": m, "Horizon": f"{h}h", "R² Score": 0.0, "RMSE": 0.0, "Coverage": 0.0})
        return pd.DataFrame(rows)
        
    try:
        # Import your existing client/connection details 
        from database import db 
        
        # Query the latest evaluation record pushed by evaluate.py into 'model_metrics'
        latest_record = db["model_metrics"].find_one(
            {"type": "automated_pipeline_evaluation"},
            sort=[("timestamp", pymongo.DESCENDING)]
        )
        
        if latest_record and "performance_summary" in latest_record:
            summary = latest_record["performance_summary"]
            
            # Map structural components out to the plotting dataframe
            for h in default_horizons:
                horizon_key = f"{h}h"
                horizon_data = summary.get(horizon_key, {}).get("models", {})
                
                for m in default_models:
                    m_stats = horizon_data.get(m, {})
                    
                    # Safely map metrics with fallback parameters
                    r2 = m_stats.get("R2") or m_stats.get("test_r2") or 0.0
                    
                    # If your training pipeline saves MAE/MAPE instead of RMSE, fallback to MAE
                    rmse = m_stats.get("RMSE") or m_stats.get("MAE") or 0.0
                    coverage = m_stats.get("Coverage") or 0.0
                    
                    rows.append({
                        "Model": m, 
                        "Horizon": f"{h}h", 
                        "R² Score": float(r2), 
                        "RMSE": float(rmse),
                        "Coverage": float(coverage)
                    })
        else:
            raise ValueError("No automated pipeline evaluation record discovered in collection.")
            
    except Exception as e:
        # Fallback tracking printout if connection times out or fails
        for m in default_models:
            for h in default_horizons:
                rows.append({"Model": m, "Horizon": f"{h}h", "R² Score": 0.0, "RMSE": 0.0, "Coverage": 0.0})
                
    return pd.DataFrame(rows)

# Execute query to build dataframes
metrics_df = _fetch_metrics_from_mongodb()

chart_col1, chart_col2 = st.columns(2, gap="large")

with chart_col1:
    st.plotly_chart(plot_error_progression(metrics_df), use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.plotly_chart(plot_variance_matrix(metrics_df), use_container_width=True, config={"displayModeBar": False})