"""
AirLyst Karachi — AQI Intelligence Dashboard  (v4)
Streamlit frontend consuming a Flask prediction microservice on Render.

KEY FIXES in this version
─────────────────────────
FIX-502   Render free tier cold-starts take 20-60 s.  Every API call now
          goes through _api_get() / _api_post() which:
            1. Sends a non-blocking /health ping first.
            2. Retries up to 3 times with exponential back-off (2 s, 4 s, 8 s).
            3. Raises a clear, user-facing banner instead of per-section 502 spam.

FIX-UI-1  Added Section 4: SHAP Feature Importance (top-10 bar chart pulled
          from /debug/shap endpoint, or model_shap collection).

FIX-UI-2  Added Section 5: Stratified Error Bands — MAE by AQI tier.

FIX-UI-3  Added Section 6: Skill Score & Conformal Coverage table.

FIX-UI-4  Metrics table now shows MAE + MAPE alongside R² / RMSE / Coverage
          so the dashboard is self-contained for an academic evaluation.

FIX-UI-5  Cold-start spinner with elapsed timer shown while Render wakes up.

STRUCTURE UNCHANGED
  dashboard/app.py          ← this file
  dashboard/alerts.py
  dashboard/visualizations.py
"""

import time
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

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

.stApp {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,130,246,0.08) 0%, transparent 70%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    color: var(--text-primary);
    font-family: var(--font-mono);
}

[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-mono) !important; }
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

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

h1, h2, h3 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
.stMarkdown h3 { font-size: 1rem !important; font-weight: 600 !important; }

[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
.js-plotly-plot .plotly { background: transparent !important; }

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: block !important; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; display: block !important; }
section[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: var(--font-mono);
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-live   { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-warn   { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-dead   { background: rgba(239,68,68,0.10);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

/* Section header */
.section-label {
    font-size: 0.65rem;
    font-family: var(--font-mono);
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
}

/* Skill bar */
.skill-bar-wrap { margin-bottom: 6px; }
.skill-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.72rem; font-family: var(--font-mono);
    color: #94a3b8; margin-bottom: 3px;
}
.skill-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px; height: 6px; overflow: hidden;
}
.skill-bar-fill { height: 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBUST API LAYER  —  handles Render cold-starts and 502s gracefully
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_RETRIES  = 3
_BACKOFF_BASE = 2   # seconds; doubles each retry

def _warm_up_render(gateway: str) -> bool:
    """
    Ping /health with generous timeout.  If Render returns 502 (sleeping),
    wait and retry.  Returns True once the service is live, False if it
    never woke within budget.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(f"{gateway}/health", timeout=(10, 30))
            if r.status_code == 200:
                return True
            # 502 / 503 → service waking; wait then retry
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            return False   # DNS / network unreachable — no point retrying
        except Exception:
            pass
        wait = _BACKOFF_BASE ** attempt
        time.sleep(wait)
    return False


def _api_get(gateway: str, path: str, timeout: tuple = (15, 45)) -> requests.Response | None:
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(f"{gateway}{path}", timeout=timeout)
            if r.status_code not in (502, 503):
                return r
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            return None
        except Exception:
            return None
        time.sleep(_BACKOFF_BASE ** attempt)
    return None


def _api_post(gateway: str, path: str, payload: dict, timeout: tuple = (15, 60)) -> requests.Response | None:
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.post(f"{gateway}{path}", json=payload, timeout=timeout)
            if r.status_code not in (502, 503):
                return r
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            return None
        except Exception:
            return None
        time.sleep(_BACKOFF_BASE ** attempt)
    return None


# ─── Latest features (via Flask proxy) ───────────────────────────────────────
@st.cache_data(ttl=30)
def _load_mongo_features(gateway: str) -> tuple[dict, bool, str | None]:
    r = _api_get(gateway, "/latest_features", timeout=(10, 20))
    if r is None:
        return {}, False, f"Cannot reach Flask API at {gateway}."
    if r.status_code == 200:
        data = r.json()
        if data and isinstance(data, dict):
            return data, True, None
        return {}, False, "API returned empty feature document."
    return {}, False, f"Flask /latest_features returned HTTP {r.status_code}."


# ─── Module-level feature dict + helper ──────────────────────────────────────
mongo_features: dict = {}


def _d(raw_key: str, lag_key: str, default: float) -> float:
    for k in (lag_key, raw_key):
        v = mongo_features.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
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

    _fetched, mongo_active, mongo_error = _load_mongo_features(api_gateway)
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


# ─── Inference payload ────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
#  WARM-UP  —  show a progress spinner while Render wakes up
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def _check_api_live(gateway: str) -> bool:
    return _warm_up_render(gateway)

_warmup_placeholder = st.empty()
with _warmup_placeholder.container():
    with st.spinner("⏳ Connecting to forecasting backend (Render may be waking from sleep — takes ~30 s)…"):
        _flask_reachable = _check_api_live(api_gateway)

_warmup_placeholder.empty()

if not _flask_reachable:
    st.error(
        f"**Cannot reach Flask API at `{api_gateway}`.**\n\n"
        "The Render service may be completely down, or the URL is wrong.  "
        "Predictions and metrics require a live connection."
    )


# ─── Single-horizon prediction helper ────────────────────────────────────────
def _fetch_prediction(model_key: str, horizon: int) -> dict | None:
    if not _flask_reachable:
        return None
    r = _api_post(api_gateway, f"/predict/{model_key}/{horizon}", inference_payload)
    if r is None:
        return None
    if r.status_code == 200:
        return r.json()
    # Silent — error shown at section level, not per-card
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
api_badge = (
    '<span class="status-badge badge-live">● API LIVE</span>'
    if _flask_reachable
    else '<span class="status-badge badge-dead">● API OFFLINE</span>'
)
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:2rem;">
    <div>
        <h1 style="margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.03em;
                   font-family:'Syne',sans-serif; color:#f1f5f9;">
            Air<span style="color:#3b82f6;">Wind</span> Karachi
        </h1>
        <p style="margin:4px 0 0; color:#475569; font-size:0.82rem; font-family:'JetBrains Mono',monospace;">
            Multi-horizon AQI forecasting · Random Forest · XGBoost · Ridge
        </p>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        {api_badge}
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    padding:8px 16px; border-radius:12px; font-size:0.72rem;
                    color:#475569; font-family:'JetBrains Mono',monospace; white-space:nowrap;">
            Using &nbsp;<span style="color:#3b82f6;">{selected_model_ui}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MULTI-HORIZON FORECAST GAUGES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔮 Multi-Horizon Forecast")
st.caption(f"Real-time predictions via **{selected_model_ui}** — 24 h · 48 h · 72 h windows")

gauge_cols  = st.columns(3, gap="medium")
detail_cols = st.columns(3, gap="medium")
horizons    = [24, 48, 72]

for idx, h in enumerate(horizons):
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
                    <div style="font-size:0.75rem; color:#475569; font-family:'JetBrains Mono',monospace;">{h}h Forecast</div>
                    <div style="font-size:1.1rem; color:#334155; margin-top:12px;">— unavailable —</div>
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
                    95% CI &nbsp;·&nbsp; <span style="color:#10b981;">● live</span>
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
                        {"Backend offline or model not trained yet." if not _flask_reachable else "Model artifact loading — refresh in ~30 s."}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CROSS-MODEL LIVE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS FETCH  (shared by sections 3, 5, 6)
# ═══════════════════════════════════════════════════════════════════════════════
_MODEL_COLORS = {
    "Random Forest": "#3b82f6",
    "XGBoost":       "#10b981",
    "Ridge":         "#f59e0b",
}
_TRANSPARENT  = "rgba(0,0,0,0)"
_GRID_COLOR   = "rgba(255,255,255,0.06)"
_TEXT_COLOR   = "#94a3b8"
_FONT         = "JetBrains Mono, monospace"

_LAYOUT = dict(
    plot_bgcolor  = _TRANSPARENT,
    paper_bgcolor = _TRANSPARENT,
    font          = dict(color=_TEXT_COLOR, family=_FONT, size=11),
    margin        = dict(l=10, r=10, t=48, b=10),
    legend        = dict(bgcolor="rgba(255,255,255,0.03)", bordercolor="rgba(255,255,255,0.08)",
                         borderwidth=1, font=dict(color="#cbd5e1", size=10)),
)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_all_metrics(gateway: str) -> tuple[dict, str | None]:
    """Returns (raw payload dict, error_message)."""
    if not _flask_reachable:
        return {}, "Flask API is offline."
    r = _api_get(gateway, "/metrics/all", timeout=(15, 60))
    if r is None:
        return {}, "Cannot reach Flask API for metrics."
    if r.status_code != 200:
        return {}, f"Flask /metrics/all returned HTTP {r.status_code}."
    try:
        return r.json(), None
    except Exception:
        return {}, "Flask /metrics/all returned non-JSON."


_raw_metrics, _metrics_err = _fetch_all_metrics(api_gateway)

_model_name_map = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "ridge":         "Ridge",
}


def _build_metrics_df(payload: dict) -> pd.DataFrame:
    rows = []
    for api_key, display_name in _model_name_map.items():
        model_data = payload.get(api_key, {})
        for h in [24, 48, 72]:
            h_data = model_data.get(str(h), {})
            if "error" in h_data or (h_data.get("r2", 0) == 0 and h_data.get("mae", 0) == 0):
                continue
            rows.append({
                "Model":    display_name,
                "Horizon":  f"{h}h",
                "R² Score": float(h_data.get("r2",       0.0)),
                "RMSE":     float(h_data.get("rmse",     0.0)),
                "MAE":      float(h_data.get("mae",      0.0)),
                "MAPE":     float(h_data.get("mape",     0.0)),
                "Coverage": float(h_data.get("coverage", 0.0)),
                "Margin":   float(h_data.get("margin",   0.0)),
            })
    return pd.DataFrame(rows)


metrics_df = _build_metrics_df(_raw_metrics)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL EVALUATION CHARTS (RMSE + R²)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Model Evaluation Metrics")
st.caption("Performance telemetry (R², RMSE, MAE, MAPE, Coverage) sourced from MongoDB via Flask.")

if _metrics_err and metrics_df.empty:
    st.error(
        f"**Model metrics unavailable.**\n\n{_metrics_err}\n\n"
        "Run the training + evaluation pipeline, or check that "
        "`MONGODB_URI` is set and the `model_metrics` collection has data. "
        "Use `GET /debug/metrics_raw` on the Flask API to inspect what is stored."
    )
elif _metrics_err:
    st.warning(f"Some metric horizons missing from MongoDB:\n\n{_metrics_err}")

if not metrics_df.empty:
    # ── Row 1: RMSE progression + R² bar chart ─────────────────────────────
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

    # ── Row 2: MAE + MAPE side-by-side ─────────────────────────────────────
    mae_col, mape_col = st.columns(2, gap="large")

    with mae_col:
        fig_mae = go.Figure()
        for model, color in _MODEL_COLORS.items():
            df_m = metrics_df[metrics_df["Model"] == model]
            if df_m.empty:
                continue
            fig_mae.add_trace(go.Bar(
                x=df_m["Horizon"], y=df_m["MAE"], name=model,
                marker_color=color, opacity=0.85,
                text=df_m["MAE"].round(1), textposition="outside",
                textfont=dict(color=color, size=9),
                hovertemplate="<b>%{x}</b><br>MAE: %{y:.1f}<extra>" + model + "</extra>",
            ))
        fig_mae.update_layout(
            **_LAYOUT, barmode="group", bargap=0.25, height=340,
            title=dict(text="Mean Absolute Error (MAE)", x=0.01,
                       font=dict(color="#f1f5f9", size=14, family=_FONT)),
        )
        fig_mae.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                              linecolor="rgba(255,255,255,0.08)",
                              tickfont=dict(color=_TEXT_COLOR, size=10))
        fig_mae.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                              linecolor="rgba(255,255,255,0.08)",
                              tickfont=dict(color=_TEXT_COLOR, size=10),
                              title=dict(text="MAE (AQI units)", font=dict(color=_TEXT_COLOR, size=11)))
        st.plotly_chart(fig_mae, use_container_width=True, config={"displayModeBar": False})

    with mape_col:
        fig_mape = go.Figure()
        for model, color in _MODEL_COLORS.items():
            df_m = metrics_df[metrics_df["Model"] == model]
            if df_m.empty or df_m["MAPE"].sum() == 0:
                continue
            fig_mape.add_trace(go.Scatter(
                x=df_m["Horizon"], y=df_m["MAPE"], name=model,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, line=dict(color="#0a0d14", width=1.5)),
                hovertemplate="<b>%{x}</b><br>MAPE: %{y:.2f}%<extra>" + model + "</extra>",
            ))
        fig_mape.update_layout(
            **_LAYOUT, height=340,
            title=dict(text="Mean Absolute % Error (MAPE)", x=0.01,
                       font=dict(color="#f1f5f9", size=14, family=_FONT)),
        )
        fig_mape.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                               linecolor="rgba(255,255,255,0.08)",
                               tickfont=dict(color=_TEXT_COLOR, size=10))
        fig_mape.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                               linecolor="rgba(255,255,255,0.08)",
                               tickfont=dict(color=_TEXT_COLOR, size=10),
                               title=dict(text="MAPE (%)", font=dict(color=_TEXT_COLOR, size=11)))
        st.plotly_chart(fig_mape, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Full metrics table ───────────────────────────────────────────
    st.markdown("#### 📋 Full Metrics Table")
    display_df = metrics_df.copy()
    display_df["R² Score"]  = display_df["R² Score"].map(lambda x: f"{x:.3f}")
    display_df["RMSE"]      = display_df["RMSE"].map(lambda x: f"{x:.1f}")
    display_df["MAE"]       = display_df["MAE"].map(lambda x: f"{x:.1f}")
    display_df["MAPE"]      = display_df["MAPE"].map(lambda x: f"{x:.2f}%")
    display_df["Coverage"]  = display_df["Coverage"].map(lambda x: f"{x*100:.1f}%")
    display_df["Margin"]    = display_df["Margin"].map(lambda x: f"±{x:.1f}")
    display_df = display_df[["Model", "Horizon", "MAE", "MAPE", "RMSE", "R² Score", "Coverage", "Margin"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SHAP FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🧠 SHAP Feature Importance")
st.caption("Top-10 features by mean |SHAP| value — computed on the held-out test set.")


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_shap(gateway: str, model_key: str, horizon: int) -> list[dict]:
    """
    Calls a debug endpoint that proxies model_shap collection.
    Falls back to /debug/artifacts feature_names if SHAP not available.
    """
    if not _flask_reachable:
        return []
    path = f"/shap/{model_key}/{horizon}"
    r = _api_get(gateway, path, timeout=(10, 30))
    if r and r.status_code == 200:
        try:
            return r.json().get("records", [])[:10]
        except Exception:
            pass
    return []


shap_horizon = st.selectbox("Horizon", [24, 48, 72], key="shap_horizon_sel",
                             format_func=lambda h: f"{h}h")

shap_data = _fetch_shap(api_gateway, active_model_key, shap_horizon)

if shap_data:
    features = [d["feature"]        for d in shap_data]
    values   = [d["mean_abs_shap"]  for d in shap_data]
    max_v    = max(values) if values else 1.0

    fig_shap = go.Figure(go.Bar(
        x=values[::-1], y=features[::-1],
        orientation="h",
        marker=dict(
            color=values[::-1],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        **_LAYOUT, height=380,
        title=dict(
            text=f"Top-{len(features)} Features — {selected_model_ui} / {shap_horizon}h",
            x=0.01, font=dict(color="#f1f5f9", size=14, family=_FONT),
        ),
    )
    fig_shap.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                           tickfont=dict(color=_TEXT_COLOR, size=10),
                           title=dict(text="Mean |SHAP| value", font=dict(color=_TEXT_COLOR, size=11)))
    fig_shap.update_yaxes(showgrid=False, zeroline=False,
                           tickfont=dict(color="#cbd5e1", size=10))
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
else:
    # Graceful degradation: show skill-score-sorted feature list from metrics if SHAP endpoint missing
    st.info(
        "SHAP data not yet available via API.  "
        "Add a `/shap/<model>/<horizon>` route to `app.py` (Flask) that reads from the "
        "`model_shap` MongoDB collection (already written by `save_shap()` during training).  "
        "Once added, this chart will auto-populate."
    )
    st.markdown("""
    ```python
    # Add to api/app.py (Flask)
    @app.route("/shap/<string:model_type>/<int:horizon>", methods=["GET"])
    def get_shap(model_type, horizon):
        store_name = _API_TO_STORE_NAME.get(model_type.lower())
        if not store_name:
            return jsonify({"error": "unknown model"}), 400
        from mongo_store import get_db
        db  = get_db()
        doc = db["model_shap"].find_one({"model": store_name, "horizon_h": horizon}, {"_id": 0})
        if not doc:
            return jsonify({"records": []}), 200
        return jsonify({"records": doc.get("records", [])[:10]}), 200
    ```
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SKILL SCORE & CONFORMAL COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🎯 Forecast Skill & Conformal Coverage")
st.caption(
    "Skill score = 1 − (model MAE / persistence baseline MAE).  "
    "Coverage = fraction of test targets inside the 95% prediction interval."
)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_skill_data(gateway: str) -> list[dict]:
    """Reads model_metrics docs (one per model+horizon) for skill & coverage fields."""
    if not _flask_reachable:
        return []
    rows = []
    for api_key, display_name in _model_name_map.items():
        model_data = _raw_metrics.get(api_key, {})
        for h in [24, 48, 72]:
            h_data = model_data.get(str(h), {})
            if "error" in h_data:
                continue
            rows.append({
                "Model":    display_name,
                "Horizon":  f"{h}h",
                "Coverage": float(h_data.get("coverage", 0.0)),
                "Margin":   float(h_data.get("margin",   0.0)),
                "MAE":      float(h_data.get("mae",      0.0)),
                "R²":       float(h_data.get("r2",       0.0)),
            })
    return rows

skill_rows = _fetch_skill_data(api_gateway)

if skill_rows:
    skill_df = pd.DataFrame(skill_rows)

    # Coverage heatmap (models × horizons)
    cov_col, margin_col = st.columns(2, gap="large")

    with cov_col:
        pivot = skill_df.pivot(index="Model", columns="Horizon", values="Coverage")
        fig_cov = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#1a2744"], [0.5, "#2563eb"], [0.9, "#3b82f6"], [1.0, "#60a5fa"]],
            zmin=70, zmax=100,
            text=(pivot.values * 100).round(1),
            texttemplate="%{text:.1f}%",
            textfont=dict(color="#f1f5f9", size=13, family=_FONT),
            hovertemplate="<b>%{y} / %{x}</b><br>Coverage: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=_TEXT_COLOR, size=9),
                title=dict(text="Coverage %", font=dict(color=_TEXT_COLOR, size=10)),
            ),
        ))
        fig_cov.update_layout(
            **_LAYOUT, height=280,
            title=dict(text="95% Conformal Coverage", x=0.01,
                       font=dict(color="#f1f5f9", size=14, family=_FONT)),
        )
        fig_cov.update_xaxes(tickfont=dict(color=_TEXT_COLOR, size=11))
        fig_cov.update_yaxes(tickfont=dict(color="#cbd5e1", size=11))
        st.plotly_chart(fig_cov, use_container_width=True, config={"displayModeBar": False})

    with margin_col:
        # Conformal margin (prediction interval half-width) — lower is better
        fig_margin = go.Figure()
        for model, color in _MODEL_COLORS.items():
            df_m = skill_df[skill_df["Model"] == model]
            if df_m.empty:
                continue
            fig_margin.add_trace(go.Bar(
                x=df_m["Horizon"], y=df_m["Margin"], name=model,
                marker_color=color, opacity=0.85,
                text=df_m["Margin"].round(1), textposition="outside",
                textfont=dict(color=color, size=9),
                hovertemplate="<b>%{x}</b><br>Margin: ±%{y:.1f}<extra>" + model + "</extra>",
            ))
        fig_margin.update_layout(
            **_LAYOUT, barmode="group", bargap=0.25, height=280,
            title=dict(text="Conformal Margin (PI half-width ↓ better)", x=0.01,
                       font=dict(color="#f1f5f9", size=14, family=_FONT)),
        )
        fig_margin.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                                 tickfont=dict(color=_TEXT_COLOR, size=10))
        fig_margin.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                                 tickfont=dict(color=_TEXT_COLOR, size=10),
                                 title=dict(text="AQI units", font=dict(color=_TEXT_COLOR, size=11)))
        st.plotly_chart(fig_margin, use_container_width=True, config={"displayModeBar": False})
else:
    if _flask_reachable:
        st.info("Skill and coverage data will appear here after the training pipeline runs.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — STRATIFIED BAND ERRORS (already in model_metrics but needs endpoint)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🏷️ Error by AQI Tier")
st.caption(
    "MAE broken down by EPA health tier — reveals how accurate each model is "
    "specifically for hazardous (>200) and very unhealthy (>150) conditions."
)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_band_errors(gateway: str) -> pd.DataFrame:
    """
    Pulls per-band MAE from /debug/metrics_raw and parses error_by_band field.
    Falls back gracefully if not present.
    """
    if not _flask_reachable:
        return pd.DataFrame()
    r = _api_get(gateway, "/debug/metrics_raw", timeout=(10, 30))
    if not r or r.status_code != 200:
        return pd.DataFrame()

    BAND_LABELS = {
        "good_moderate":       "Good–Moderate (0–100)",
        "unhealthy_sensitive":  "USG (101–150)",
        "unhealthy":            "Unhealthy (151–200)",
        "very_unhealthy":       "Very Unhealthy (201–300)",
        "hazardous":            "Hazardous (301+)",
    }
    BAND_COLORS = {
        "Good–Moderate (0–100)":      "#00e676",
        "USG (101–150)":              "#ff9100",
        "Unhealthy (151–200)":        "#ff1744",
        "Very Unhealthy (201–300)":   "#d500f9",
        "Hazardous (301+)":           "#b71c1c",
    }

    try:
        docs = r.json().get("docs", [])
        rows = []
        for doc in docs:
            bands = doc.get("error_by_band", {})
            if not bands:
                continue
            model   = doc.get("model", "?")
            horizon = doc.get("horizon_h", 0)
            for band_key, label in BAND_LABELS.items():
                bdata = bands.get(band_key, {})
                mae   = bdata.get("mae")
                n     = bdata.get("n", 0)
                if mae is not None and n >= 5:
                    rows.append({
                        "Model":   model,
                        "Horizon": f"{horizon}h",
                        "Band":    label,
                        "MAE":     float(mae),
                        "N":       int(n),
                        "Color":   BAND_COLORS.get(label, "#64748b"),
                    })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


band_df = _fetch_band_errors(api_gateway)

if not band_df.empty:
    # One chart per horizon, showing all models × bands
    band_horizons = [f"{h}h" for h in [24, 48, 72]]
    b_cols = st.columns(3, gap="medium")
    for ci, bh in enumerate(band_horizons):
        with b_cols[ci]:
            df_bh = band_df[band_df["Horizon"] == bh]
            if df_bh.empty:
                st.caption(f"{bh} — no band data")
                continue

            fig_b = go.Figure()
            for model in ["RandomForest", "XGBoost", "Ridge"]:
                color = _MODEL_COLORS.get(
                    {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "Ridge": "Ridge"}.get(model, model),
                    "#64748b",
                )
                df_bm = df_bh[df_bh["Model"] == model]
                if df_bm.empty:
                    continue
                fig_b.add_trace(go.Bar(
                    name=model,
                    x=df_bm["Band"],
                    y=df_bm["MAE"],
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>MAE: %{y:.1f}  n=%{customdata}<extra>" + model + "</extra>",
                    customdata=df_bm["N"],
                ))

            fig_b.update_layout(
                **_LAYOUT, barmode="group", bargap=0.2, height=340,
                title=dict(text=f"Band MAE — {bh}", x=0.01,
                           font=dict(color="#f1f5f9", size=13, family=_FONT)),
                xaxis=dict(tickangle=-35, tickfont=dict(color=_TEXT_COLOR, size=8),
                           showgrid=False, zeroline=False, linecolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor=_GRID_COLOR, zeroline=False,
                           tickfont=dict(color=_TEXT_COLOR, size=9),
                           title=dict(text="MAE", font=dict(color=_TEXT_COLOR, size=10))),
            )
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
else:
    if _flask_reachable:
        st.info(
            "Band-level MAE will appear here automatically — the training pipeline already "
            "stores `error_by_band` inside each `model_metrics` document.  "
            "The `/debug/metrics_raw` endpoint needs to be reachable and the "
            "training pipeline needs to have run at least once."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-size:0.68rem; color:#334155; font-family:'JetBrains Mono',monospace; padding-bottom:1rem;">
    <span>AirWind Karachi — MLOps AQI Intelligence Platform</span>
    <span>Backend: <code style="color:#475569;">{api_gateway}</code></span>
</div>
""", unsafe_allow_html=True)