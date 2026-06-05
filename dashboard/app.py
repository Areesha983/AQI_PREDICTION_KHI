"""
AirLyst Karachi — AQI Intelligence Dashboard
Streamlit frontend consuming a Flask prediction microservice.

FIXES IN THIS VERSION:
  BUG 1 — WRONG IMPORT `from database import db, COLLECTION_NAME`
    database.py has no module-level `db` object. This raised ImportError
    on every cold start, fell to the bare except, and returned all-zero
    DataFrames which the charts rendered silently as flat lines.
    FIX: removed the broken import entirely. Metrics now come from Flask.

  BUG 2 — HARDCODED METRIC SEEDS
    The original fallback block fabricated numbers
    ({"Random Forest": {"R2": 0.84, ...}}) when model_metrics had no
    matching document. User requirement: "all data from mongodb only".
    FIX: hardcoded seeds removed. If the Flask API returns no real data
    an st.error() is shown so the cause is immediately visible.

  BUG 3 — WRONG METRICS QUERY FILTER
    db["model_metrics"].find_one({"type": "automated_pipeline_evaluation"})
    This filter matches nothing unless evaluate.py writes that exact field,
    so latest_record was always None, triggering the seed fallback.
    FIX: metrics are now fetched from Flask's /metrics/all endpoint which
    already has the correct multi-attempt query + schema normalisation logic.

  BUG 4 — METRICS BYPASSED FLASK ENTIRELY
    Section 3 opened its own direct PyMongo connection instead of calling
    Flask's /metrics/all route. This duplicated (and broke) the schema
    parsing that api/app.py already handles correctly.
    FIX: _fetch_metrics_from_api() calls GET /metrics/all via requests,
    exactly the same pattern used for /predict.

  BUG 5 — BARE EXCEPT SWALLOWED ALL ERRORS AND CACHED ZEROS
    A single try/except around the whole function caught BUGs 1-4 silently
    and returned zeros. @st.cache_data(ttl=15) then froze those zeros for
    15 s per page load with no user-visible indication of failure.
    FIX: errors are surfaced via st.error()/st.warning() so the user
    knows whether the Flask API is unreachable or the collection is empty.

  BUG 6 — SIDEBAR SLIDER SEEDS USED WRONG FIELD NAMES
    _d("pm25", 75.0) looked up "pm25" in the processed_features document,
    but that collection stores lag features as "pm25_lag_1". The lookup
    always missed, mongo_active showed green, and sliders stayed at the
    hardcoded defaults.
    FIX: _d() now tries the lag-1 key first ("pm25_lag_1"), then falls
    back to the raw key ("pm25"), then to the supplied default.
"""

import streamlit as st
import pandas as pd
import requests

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
        "🌲 Random Forest":    "random_forest",
        "🚀 XGBoost":          "xgboost",
        "📊 Ridge Regression": "ridge",
    }
    active_model_key = model_mapping[selected_model_ui]

    st.markdown("---")
    st.markdown("##### 🎯 Input Vectors")

    # ── BUG 6 FIX: seed sliders from MongoDB using correct lag-1 field names ──
    # processed_features stores "pm25_lag_1", not "pm25". The original _d()
    # only checked the raw key and always missed, keeping sliders at hardcoded
    # defaults while incorrectly showing the green "Mongo feature store" dot.
    @st.cache_data(ttl=30)
    def _load_mongo_features():
        if not _DB_AVAILABLE:
            return {}, False
        try:
            rec = get_latest_features()
            if rec and isinstance(rec, dict):
                return rec, True
        except Exception:
            pass
        return {}, False

    mongo_features, mongo_active = _load_mongo_features()

    def _d(raw_key: str, lag_key: str, default: float) -> float:
        """
        BUG 6 FIX: Try lag-1 field name first (how processed_features stores
        it), then the raw field name, then fall back to the supplied default.
        """
        for k in (lag_key, raw_key):
            v = mongo_features.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    sim_pm25     = st.slider("PM2.5 (μg/m³)",    10.0, 350.0, _d("pm25",        "pm25_lag_1",        75.0), 5.0)
    sim_pm10     = st.slider("PM10 (μg/m³)",      20.0, 500.0, _d("pm10",        "pm10_lag_1",       140.0), 5.0)
    sim_temp     = st.slider("Temperature (°C)",  10.0,  48.0, _d("temperature", "temperature_lag_1",  32.0), 1.0)
    sim_humidity = st.slider("Humidity (%)",       10.0, 100.0, _d("humidity",    "humidity_lag_1",     65.0), 5.0)
    sim_wind     = st.slider("Wind Speed (km/h)",   0.0,  45.0, _d("wind_speed",  "wind_speed_lag_1",   12.0), 1.0)

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
        "pm25":                          sim_pm25,
        "pm10":                          sim_pm10,
        "temperature":                   sim_temp,
        "humidity":                      sim_humidity,
        "wind_speed":                    sim_wind,
        "pm25_diff_1h":                  2.3,
        "pm25_roll_std_24h":             12.4,
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

gauge_cols  = st.columns(3, gap="medium")
detail_cols = st.columns(3, gap="medium")
horizons    = [24, 48, 72]

for idx, h in enumerate(horizons):
    raw  = _fetch_prediction(active_model_key, h)
    data = raw if raw else _fallback_pred(active_model_key, h)
    pred = data["aqi_prediction"]
    low  = data["lower_bound_95ci"]
    high = data["upper_bound_95ci"]
    tier = get_epa_tier_details(pred)
    src  = "live" if raw else "fallback"

    with gauge_cols[idx]:
        fig = plot_aqi_gauge(pred, tier, f"{h}h Forecast")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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


# ─── SECTION 3: MODEL EVALUATION CHARTS ─────────────────────────────────────
# BUG 1-5 FIX: This section previously attempted a direct PyMongo connection
# using `from database import db` (which doesn't exist), fell to a bare except,
# and returned hardcoded zeros. It now calls the Flask /metrics/all endpoint —
# the same API the rest of the dashboard already uses — which has full schema
# normalisation and multi-attempt querying built in.
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📊 Live Model Evaluation Metrics")
st.caption(
    "Dynamic performance telemetry ($R^2$, RMSE, Coverage) "
    "fetched from MongoDB via the Flask metrics API."
)


@st.cache_data(ttl=300)  # 5-min TTL — matches Flask's own _METRICS_CACHE TTL
def _fetch_metrics_from_api(gateway: str) -> tuple[pd.DataFrame, str | None]:
    """
    Calls GET /metrics/all on the Flask API and unpacks the response into a
    DataFrame with columns [Model, Horizon, R² Score, RMSE, Coverage].

    Returns (DataFrame, error_message). If error_message is not None the
    DataFrame will be empty and the caller should surface the error.

    BUG 1-5 FIX: replaces the broken direct-PyMongo approach with a clean
    HTTP call to Flask, which already handles all MongoDB querying and schema
    normalisation correctly.
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
            f"Cannot reach Flask API at **{gateway}**. "
            "Make sure `python api/app.py` is running."
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

            # BUG 5 FIX: surface the error key that api/app.py now emits
            # instead of zeros — so the user sees a real message, not a flat chart.
            if "error" in h_data:
                errors.append(f"{display_name} / {h}h: {h_data['error']}")
                continue

            r2       = float(h_data.get("r2",       0.0))
            rmse     = float(h_data.get("rmse",     0.0))
            mae      = float(h_data.get("mae",      0.0))
            coverage = float(h_data.get("coverage", 0.0))

            # Use RMSE if available, fall back to MAE (evaluate.py may write either)
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
    # Fatal — nothing to plot
    st.error(
        f"**Model metrics unavailable.**\n\n{metrics_error}\n\n"
        "Run the training + evaluation pipeline, or check that "
        "`MONGODB_URI` is set and the `model_metrics` collection has data. "
        "Use `GET /debug/metrics_raw` on the Flask API to inspect what is stored."
    )
elif metrics_error:
    # Partial data — plot what we have and warn about the gaps
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