"""
AirLyst Karachi — AQI Intelligence Dashboard
Streamlit frontend consuming a Flask prediction microservice.

DEPLOYMENT FIXES (on top of previous FIX A–D):

  FIX DEPLOY-1 — api_gateway default was "http://127.0.0.1:5000" which
    breaks on Streamlit Cloud. Now reads from st.secrets["API_BASE_URL"]
    with the Render URL as fallback.

  FIX DEPLOY-2 — _load_mongo_features() called database.get_latest_features()
    directly (MongoDB). On Streamlit Cloud there is no MONGODB_URI so this
    always failed. Replaced with a call to the Flask /latest_features endpoint
    which already proxies MongoDB correctly.

  FIX DEPLOY-3 — _d() and mongo_features were defined inside `with st.sidebar:`
    but used outside it (inference_payload). Moved both above the sidebar block
    so they are always in scope.
"""

import streamlit as st
import pandas as pd
import requests

from alerts import get_epa_tier_details
from visualizations import plot_error_progression, plot_variance_matrix, plot_aqi_gauge

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


# ─── FIX DEPLOY-2: fetch latest features from Flask API, not MongoDB directly ─
@st.cache_data(ttl=30)
def _load_mongo_features(gateway: str) -> tuple[dict, bool, str | None]:
    """
    Returns (features_dict, is_active, error_message).
    Calls Flask /latest_features instead of hitting MongoDB directly,
    so this works on Streamlit Cloud without a MONGODB_URI secret.
    """
    try:
        resp = requests.get(f"{gateway}/latest_features", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, dict):
                return data, True, None
            return {}, False, "API returned empty feature document."
        return {}, False, f"Flask /latest_features returned HTTP {resp.status_code}."
    except requests.exceptions.ConnectionError:
        return {}, False, f"Cannot reach Flask API at {gateway}."
    except requests.exceptions.Timeout:
        return {}, False, "Flask /latest_features timed out."
    except Exception as exc:
        return {}, False, str(exc)


# ─── FIX DEPLOY-3: _d() and mongo_features defined at module level ────────────
# Previously inside `with st.sidebar:` but used outside it in inference_payload.
mongo_features: dict = {}


def _d(raw_key: str, lag_key: str, default: float) -> float:
    """Try lag-1 field name first (how processed_features stores it),
    then the raw field name, then fall back to the supplied default."""
    for k in (lag_key, raw_key):
        v = mongo_features.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


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
    # FIX DEPLOY-1: default to Render URL from secrets, not localhost
    _default_url = st.secrets.get("API_BASE_URL", "https://aqi-prediction-khi.onrender.com")
    api_gateway = st.text_input("Flask API URL", _default_url, label_visibility="collapsed")

    st.markdown("##### 🤖 Model")
    selected_model_ui = st.selectbox(
        "Model",
        ["🌲 Random Forest", "🚀 XGBoost", "📊 Ridge Regression"],
        label_visibility="collapsed",
    )
    model_mapping = {
        "🌲 Random Forest":    "random_forest",
        "🚀 XGBoost":          "xgboost",
        "📊 Ridge Regression": "ridge",
    }
    active_model_key = model_mapping[selected_model_ui]

    st.markdown("---")
    st.markdown("##### 🎯 Input Vectors")

    # FIX DEPLOY-2: call updated function with gateway argument
    _fetched, mongo_active, mongo_error = _load_mongo_features(api_gateway)
    # Update the module-level dict so _d() picks up live values
    mongo_features.update(_fetched)

    sim_pm25     = st.slider("PM2.5 (μg/m³)",    10.0, 350.0, _d("pm25",        "pm25_lag_1",        75.0), 5.0)
    sim_pm10     = st.slider("PM10 (μg/m³)",      20.0, 500.0, _d("pm10",        "pm10_lag_1",       140.0), 5.0)
    sim_temp     = st.slider("Temperature (°C)",  10.0,  48.0, _d("temperature", "temperature_lag_1",  32.0), 1.0)
    sim_humidity = st.slider("Humidity (%)",       10.0, 100.0, _d("humidity",    "humidity_lag_1",     65.0), 5.0)
    sim_wind     = st.slider("Wind Speed (km/h)",   0.0,  45.0, _d("wind_speed",  "wind_speed_lag_1",   12.0), 1.0)

    st.markdown("---")
    db_dot = "🟢" if mongo_active else "🔴"
    st.markdown(f"""
        <div style='font-size:0.72rem; color:#475569;'>
            {db_dot} {'Live feature store' if mongo_active else 'Fallback defaults'}
        </div>
    """, unsafe_allow_html=True)

    if not mongo_active and mongo_error:
        st.caption(f"⚠️ {mongo_error}")


# ─── INFERENCE PAYLOAD ───────────────────────────────────────────────────────
inference_payload = {
    "features": {
        "pm25":                          sim_pm25,
        "pm10":                          sim_pm10,
        "temperature":                   sim_temp,
        "humidity":                      sim_humidity,
        "wind_speed":                    sim_wind,
        "pm25_diff_1h":                  _d("pm25_diff_1h",      "pm25_diff_1h",      0.0),
        "pm25_roll_std_24h":             _d("pm25_roll_std_24h", "pm25_roll_std_24h", 0.0),
        "interaction_pm25_humidity":     sim_pm25 * sim_humidity,
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


# ─── HELPER: single-horizon prediction request ───────────────────────────────
def _fetch_prediction(model_key: str, horizon: int) -> dict | None:
    """
    FIX D: timeout raised from 2s → 15s.
    Flask deserialises the model artifact from MongoDB base64 on the first
    call which can take 3–5 s on a cold runner.
    """
    try:
        r = requests.post(
            f"{api_gateway}/predict/{model_key}/{horizon}",
            json=inference_payload,
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
        st.warning(
            f"Flask returned HTTP {r.status_code} for {model_key}/{horizon}h. "
            f"Response: {r.text[:200]}"
        )
    except requests.exceptions.ConnectionError:
        pass   # handled at call site — Flask unreachable message shown once
    except requests.exceptions.Timeout:
        st.warning(
            f"Flask timed out for {model_key}/{horizon}h. "
            "The model artifact may still be loading from MongoDB — try refreshing."
        )
    except Exception as exc:
        st.warning(f"Unexpected error calling Flask ({model_key}/{horizon}h): {exc}")
    return None


# ─── SECTION 1: MULTI-HORIZON FORECASTS ─────────────────────────────────────
st.markdown("### 🔮 Multi-Horizon Forecast")
st.caption(f"Real-time predictions via **{selected_model_ui}** — 24h · 48h · 72h windows")

gauge_cols  = st.columns(3, gap="medium")
detail_cols = st.columns(3, gap="medium")
horizons    = [24, 48, 72]

# Check Flask reachability once before looping to avoid repeating the same
# connection-error message three times (once per horizon).
_flask_reachable = True
try:
    requests.get(f"{api_gateway}/health", timeout=5)
except requests.exceptions.ConnectionError:
    _flask_reachable = False
    st.error(
        f"**Cannot reach Flask API at `{api_gateway}`.**\n\n"
        "Predictions require a live connection — no fabricated fallback values are shown."
    )
except Exception:
    pass  # non-connection errors will surface per-prediction below

for idx, h in enumerate(horizons):
    data = None
    if _flask_reachable:
        data = _fetch_prediction(active_model_key, h)

    with gauge_cols[idx]:
        if data:
            pred = data["aqi_prediction"]
            tier = get_epa_tier_details(pred)
            fig  = plot_aqi_gauge(pred, tier, f"{h}h Forecast")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                f"""<div class="card" style="text-align:center; padding:40px 20px;">
                    <div style="font-size:0.75rem; color:#475569; font-family:'JetBrains Mono',monospace;">
                        {h}h Forecast
                    </div>
                    <div style="font-size:1.1rem; color:#334155; margin-top:12px;">
                        — unavailable —
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    with detail_cols[idx]:
        if data:
            pred = data["aqi_prediction"]
            low  = data["lower_bound_95ci"]
            high = data["upper_bound_95ci"]
            tier = get_epa_tier_details(pred)
            st.markdown(f"""
            <div class="card" style="border-left: 3px solid {tier['color']}; background:{tier['bg']};">
                <div style="font-size:0.68rem; color:#475569; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
                    95% CI &nbsp;·&nbsp;
                    <span style="color:#10b981;">● live</span>
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
        else:
            st.markdown(
                f"""<div class="card" style="border-left:3px solid #334155;">
                    <div style="font-size:0.68rem; color:#475569; font-family:'JetBrains Mono',monospace;">
                        95% CI &nbsp;·&nbsp; <span style="color:#f59e0b;">◌ unavailable</span>
                    </div>
                    <div style="font-size:0.78rem; color:#334155; margin-top:10px;">
                        Flask API unreachable or model not trained yet.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─── SECTION 2: CROSS-MODEL BENCHMARKING ────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### ⚖️ Cross-Model Live Benchmarking")
st.caption("All three models evaluated simultaneously on the current input vector.")

bench_cols   = st.columns(3, gap="large")
all_models   = ["random_forest", "xgboost", "ridge"]
model_labels = {"random_forest": "🌲 Random Forest", "xgboost": "🚀 XGBoost", "ridge": "📊 Ridge"}

for idx, m_key in enumerate(all_models):
    with bench_cols[idx]:
        st.markdown(f"**{model_labels[m_key]}**")
        rows = []
        for h in horizons:
            data = _fetch_prediction(m_key, h) if _flask_reachable else None
            if data:
                rows.append((h, data["aqi_prediction"], get_epa_tier_details(data["aqi_prediction"])))

        if rows:
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
        else:
            st.markdown(
                """<div style="padding:16px; border-radius:10px; background:rgba(255,255,255,0.02);
                              border:1px solid rgba(255,255,255,0.05); font-size:0.78rem;
                              color:#334155; font-family:'JetBrains Mono',monospace;">
                    — unavailable —
                </div>""",
                unsafe_allow_html=True,
            )


# ─── SECTION 3: MODEL EVALUATION CHARTS ─────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Live Model Evaluation Metrics")
st.caption(
    "Dynamic performance telemetry ($R^2$, RMSE, Coverage) "
    "fetched from MongoDB via the Flask metrics API."
)


@st.cache_data(ttl=300)
def _fetch_metrics_from_api(gateway: str) -> tuple[pd.DataFrame, str | None]:
    """
    Calls GET /metrics/all on the Flask API and unpacks the response into a
    DataFrame with columns [Model, Horizon, R² Score, RMSE, Coverage].
    """
    model_name_map = {
        "random_forest": "Random Forest",
        "xgboost":       "XGBoost",
        "ridge":         "Ridge",
    }

    try:
        resp = requests.get(f"{gateway}/metrics/all", timeout=5)
    except requests.exceptions.ConnectionError:
        return pd.DataFrame(), (
            f"Cannot reach Flask API at **{gateway}**."
        )
    except requests.exceptions.Timeout:
        return pd.DataFrame(), "Flask API timed out while fetching metrics."
    except Exception as exc:
        return pd.DataFrame(), f"Unexpected error contacting Flask API: {exc}"

    if resp.status_code != 200:
        return pd.DataFrame(), (
            f"Flask /metrics/all returned HTTP {resp.status_code}. "
            f"Response: {resp.text[:200]}"
        )

    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame(), "Flask /metrics/all returned non-JSON response."

    rows   = []
    errors = []

    for api_key, display_name in model_name_map.items():
        model_data = payload.get(api_key, {})
        if not model_data:
            errors.append(f"No data for model '{api_key}' in /metrics/all response.")
            continue

        for h in [24, 48, 72]:
            horizon_key = str(h)
            h_data = model_data.get(horizon_key, {})

            if "error" in h_data:
                errors.append(f"{display_name} / {h}h: {h_data['error']}")
                continue

            r2       = float(h_data.get("r2",       0.0))
            rmse     = float(h_data.get("rmse",     0.0))
            mae      = float(h_data.get("mae",      0.0))
            coverage = float(h_data.get("coverage", 0.0))

            error_value = rmse if rmse != 0.0 else mae

            if r2 == 0.0 and error_value == 0.0:
                errors.append(
                    f"{display_name} / {h}h: metrics are all zero — "
                    "model_metrics collection may be empty or evaluate.py "
                    "has not run yet."
                )
                continue

            rows.append({
                "Model":    display_name,
                "Horizon":  f"{h}h",
                "R² Score": r2,
                "RMSE":     error_value,
                "Coverage": coverage,
            })

    if not rows:
        detail = " | ".join(errors) if errors else "No metric rows returned by Flask."
        return pd.DataFrame(), detail

    return pd.DataFrame(rows), "\n".join(errors) if errors else None


# ── Fetch and render ──────────────────────────────────────────────────────────
metrics_df, metrics_error = _fetch_metrics_from_api(api_gateway)

if metrics_error and metrics_df.empty:
    st.error(
        f"**Model metrics unavailable.**\n\n{metrics_error}\n\n"
        "Run the training + evaluation pipeline, or check that "
        "`MONGODB_URI` is set and the `model_metrics` collection has data. "
        "Use `GET /debug/metrics_raw` on the Flask API to inspect what is stored."
    )
elif metrics_error:
    st.warning(f"Some metric horizons are missing from MongoDB:\n\n{metrics_error}")

if not metrics_df.empty:
    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.plotly_chart(
            plot_error_progression(metrics_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with chart_col2:
        st.plotly_chart(
            plot_variance_matrix(metrics_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )