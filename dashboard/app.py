"""
AirLyst Karachi — AQI Intelligence Dashboard  (v6)
Streamlit frontend consuming a Flask prediction microservice on Render.

FIXES vs v5
───────────
FIX-SHAP-RIDGE  Ridge never calls save_shap() — it has no TreeExplainer and
                stores coefficient magnitudes via save_feature_list() into the
                model_features collection, NOT model_shap.  The previous
                _fetch_shap() only queried /shap/<model>/<horizon>, which reads
                model_shap → always empty for Ridge → "SHAP data not yet
                available" message forever.

                Fix (two-part):
                  1. New _fetch_shap_or_coef() helper: tries /shap first; if
                     records come back empty it calls the new
                     /features/<model>/<horizon> route (see api/app.py patch)
                     which reads model_features and returns the top-10 features
                     by |coefficient|, formatted identically to SHAP records
                     {"feature": ..., "mean_abs_shap": ...} so the bar-chart
                     code needs zero changes.
                  2. Chart title updated to show "Mean |Coeff|" label for Ridge
                     so the user knows what they're looking at.

FIX-TIMESTAMP   MongoDB stores all datetimes in UTC.  Karachi is UTC+5.
                The sidebar showed the raw UTC string from the document
                (e.g. "2026-06-07 23:00") which appeared 5 h ahead of wall
                clock.  Fix: parse the string and offset by +5 h before
                rendering.  Handles both ISO ("T") and space-separated formats.
                Falls back gracefully if parsing fails.

FIX-LIVE-DATA   The module-level mongo_features dict was populated once at
                import time from the @st.cache_data return value, but on
                subsequent Streamlit rerenders the dict object was not
                re-populated because Python module state persists across
                rerenders.  Fix: mongo_features is now rebuilt inside the
                sidebar block on every rerender from the (possibly fresh)
                cached fetch result, matching what the user actually sees in
                the "As of:" timestamp.
"""

import time
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

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

/* Read-only sensor card */
.sensor-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.sensor-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}
.sensor-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}
.sensor-unit {
    font-size: 0.7rem;
    color: #64748b;
    margin-left: 3px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBUST API LAYER  —  handles Render cold-starts and 502s gracefully
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_RETRIES  = 3
_BACKOFF_BASE = 2   # seconds; doubles each retry


def _warm_up_render(gateway: str) -> bool:
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(f"{gateway}/health", timeout=(10, 30))
            if r.status_code == 200:
                return True
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            pass
        time.sleep(_BACKOFF_BASE ** attempt)
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


# ─── Feature fetchers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_realtime_features(gateway: str) -> tuple[dict, bool, str | None]:
    """Fetches from realtime_observations (today's partial data)."""
    r = _api_get(gateway, "/latest_realtime", timeout=(10, 20))
    if r is None:
        return {}, False, f"Cannot reach Flask API at {gateway}."
    if r.status_code == 200:
        data = r.json()
        if data and isinstance(data, dict) and "error" not in data:
            return data, True, None
        return {}, False, data.get("error", "Empty realtime document.")
    return {}, False, f"/latest_realtime returned HTTP {r.status_code}."


@st.cache_data(ttl=300)
def _load_processed_features(gateway: str) -> tuple[dict, bool, str | None]:
    """Fetches from processed_features (yesterday's finalised data)."""
    r = _api_get(gateway, "/latest_features", timeout=(10, 20))
    if r is None:
        return {}, False, f"Cannot reach Flask API at {gateway}."
    if r.status_code == 200:
        data = r.json()
        if data and isinstance(data, dict) and "error" not in data:
            return data, True, None
        return {}, False, data.get("error", "Empty feature document.")
    return {}, False, f"/latest_features returned HTTP {r.status_code}."


def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _d(features: dict, raw_key: str, lag_key: str, default: float) -> float:
    """
    FIX-LIVE-DATA: now takes the features dict explicitly rather than reading
    from a stale module-level variable, so every Streamlit rerender uses the
    latest fetched values.
    """
    for k in (lag_key, raw_key):
        v = features.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


# ─── FIX-TIMESTAMP helper ─────────────────────────────────────────────────────
_KARACHI_OFFSET = timedelta(hours=5)

def _to_karachi_str(raw_dt, already_pkt: bool = False) -> str | None:
    """
    Convert a datetime value to a "YYYY-MM-DD HH:MM PKT" display string.

    Two sources feed this function and they use DIFFERENT timezone conventions:

    SOURCE A — realtime_observations (update_realtime.py)
      Open-Meteo is called with &timezone=Asia%2FKarachi so every timestamp
      in the API response is already PKT (UTC+5).  The datetime string is
      stored as-is into MongoDB, e.g. "2026-06-07 16:00:00" meaning 16:00 PKT.
      Pass already_pkt=True — NO offset should be added.

    SOURCE B — processed_features (feature_engineering.py / feature_store.py)
      The historical pipeline stores timestamps as UTC ISO strings.
      Pass already_pkt=False (default) — add +5 h to convert to PKT.

    Sanity check: if already_pkt=True and the parsed time is more than 2 h
    in the future relative to current PKT wall clock, it means a future
    forecast row slipped through the update_realtime.py filter (e.g. pipeline
    ran at the top of the hour before the fix was deployed).  In that case the
    timestamp is clamped and flagged so the user isn't misled.
    """
    if raw_dt is None:
        return None

    dt = None
    if isinstance(raw_dt, datetime):
        dt = raw_dt
    elif isinstance(raw_dt, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(raw_dt[:26], fmt)
                break
            except ValueError:
                continue
        if dt is None:
            # Unparseable — return truncated raw string rather than crashing
            return str(raw_dt)[:16]

    if already_pkt:
        # Sanity check: realtime timestamps should never be more than 2 h ahead
        # of current PKT wall clock.  If they are, a future forecast row slipped
        # through — flag it so the user knows the data may be stale.
        now_pkt = datetime.utcnow() + _KARACHI_OFFSET
        if dt > now_pkt + timedelta(hours=2):
            return dt.strftime("%Y-%m-%d %H:%M PKT") + " ⚠️ (future)"
        return dt.strftime("%Y-%m-%d %H:%M PKT")

    # Timestamp is UTC — add +5 h to convert to PKT
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    karachi_dt = dt.astimezone(timezone.utc) + _KARACHI_OFFSET
    return karachi_dt.strftime("%Y-%m-%d %H:%M PKT")


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

    # ── FIX-LIVE-DATA: rebuild mongo_features on every rerender ──────────────
    # The old code populated a module-level dict once; subsequent rerenders saw
    # stale values.  Now we always read from the (possibly fresh) cache result
    # and build a local dict used throughout this rerender.
    mongo_features: dict = {}
    # Track whether the active timestamp is already PKT (realtime_observations)
    # or UTC (processed_features) so _to_karachi_str applies the right conversion.
    _dt_already_pkt: bool = False

    realtime_data, realtime_ok, realtime_err = _load_realtime_features(api_gateway)
    processed_data, processed_ok, processed_err = _load_processed_features(api_gateway)

    def _parse_dt_pkt(doc: dict, already_pkt: bool) -> datetime | None:
        """Parse a document's datetime field into a naive PKT datetime for comparison."""
        raw = doc.get("datetime") or doc.get("timestamp")
        if not raw:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(str(raw)[:26], fmt)
                return dt if already_pkt else dt + _KARACHI_OFFSET
            except ValueError:
                pass
        return None

    _now_pkt = datetime.utcnow() + _KARACHI_OFFSET

    # Parse timestamps from both sources (both normalised to PKT for comparison)
    _rt_dt_pkt  = _parse_dt_pkt(realtime_data,  already_pkt=True)  if realtime_ok  else None
    _proc_dt_pkt = _parse_dt_pkt(processed_data, already_pkt=False) if processed_ok else None

    # Discard any realtime row that is in the future (forecast row, not observation)
    if _rt_dt_pkt and _rt_dt_pkt > _now_pkt + timedelta(minutes=5):
        _rt_dt_pkt  = None
        realtime_ok = False
        realtime_err = (
            f"Realtime timestamp is in the future "
            f"(now PKT: {_now_pkt.strftime('%H:%M')}) — comparing with processed_features."
        )

    # Choose the FRESHEST valid source.
    # A 2-hour-old realtime row beats a 3-day-old processed row.
    _use_realtime = (
        realtime_ok and _rt_dt_pkt is not None and (
            _proc_dt_pkt is None or _rt_dt_pkt >= _proc_dt_pkt
        )
    )

    if _use_realtime:
        mongo_features.update(realtime_data)
        data_source_label = "🟢 Live  <span style='color:#334155;font-size:0.65rem;'>(realtime_observations)</span>"
        mongo_active    = True
        mongo_error     = None
        _dt_already_pkt = True
    elif processed_ok:
        mongo_features.update(processed_data)
        data_source_label = "🟡 Cached  <span style='color:#334155;font-size:0.65rem;'>(processed_features · T-1)</span>"
        mongo_active    = True
        mongo_error     = realtime_err
        _dt_already_pkt = False
    elif realtime_ok:
        # processed_features is unavailable but realtime exists (even if slightly stale)
        mongo_features.update(realtime_data)
        data_source_label = "🟡 Cached  <span style='color:#334155;font-size:0.65rem;'>(realtime_observations · stale)</span>"
        mongo_active    = True
        mongo_error     = processed_err
        _dt_already_pkt = True
    else:
        data_source_label = "🔴 Offline  <span style='color:#334155;font-size:0.65rem;'>(defaults)</span>"
        mongo_active    = False
        mongo_error     = processed_err or realtime_err
        _dt_already_pkt = False

    # ── Resolve current sensor values ─────────────────────────────────────────
    sim_pm25     = _d(mongo_features, "pm25",               "pm25_lag_1",        75.0)
    sim_pm10     = _d(mongo_features, "pm10",               "pm10_lag_1",       140.0)
    sim_temp     = _d(mongo_features, "temperature_2m",     "temperature_lag_1",  32.0)
    sim_humidity = _d(mongo_features, "relative_humidity_2m", "humidity_lag_1",   65.0)
    sim_wind     = _d(mongo_features, "wind_speed_10m",     "wind_speed_lag_1",   12.0)

    # ── Compute current AQI from PM2.5 ────────────────────────────────────────
    def _pm25_to_aqi(pm25: float) -> int:
        bps = [
            (0.0,   12.0,  0,   50),
            (12.1,  35.4,  51,  100),
            (35.5,  55.4,  101, 150),
            (55.5,  150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_lo, c_hi, a_lo, a_hi in bps:
            if c_lo <= pm25 <= c_hi:
                return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
        return 500

    sim_aqi  = _pm25_to_aqi(sim_pm25)
    aqi_tier = get_epa_tier_details(sim_aqi)

    # ── Read-only sensor display ──────────────────────────────────────────────
    st.markdown("##### 📡 Current Conditions")

    # FIX-TIMESTAMP: pass already_pkt so realtime timestamps (already PKT from
    # Open-Meteo) are NOT double-shifted, while processed_features (UTC) are
    # correctly converted by +5 h.
    _dt_raw = mongo_features.get("datetime") or mongo_features.get("timestamp")
    _dt_local = _to_karachi_str(_dt_raw, already_pkt=_dt_already_pkt)
    if _dt_local:
        st.caption(f"As of: {_dt_local}")
    else:
        st.caption("Live values from feature store · read-only")

    # AQI highlight card
    st.markdown(
        f"""<div style="
                background: {aqi_tier['bg']};
                border: 1px solid {aqi_tier['color']}44;
                border-left: 3px solid {aqi_tier['color']};
                border-radius: 10px;
                padding: 12px 14px;
                margin-bottom: 10px;">
            <div class="sensor-label">Current AQI</div>
            <div style="display:flex; align-items:baseline; gap:8px; margin-top:3px;">
                <span style="font-size:2rem; font-weight:700; color:{aqi_tier['color']};
                             font-family:'JetBrains Mono',monospace; line-height:1;">
                    {sim_aqi}
                </span>
                <span style="font-size:0.72rem; color:{aqi_tier['color']}; opacity:0.85;
                             font-family:'JetBrains Mono',monospace;">
                    {aqi_tier['label']}
                </span>
            </div>
            <div style="font-size:0.65rem; color:#64748b; margin-top:5px;
                        font-family:'JetBrains Mono',monospace; line-height:1.4;">
                {aqi_tier['advice']}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    _sensor_rows = [
        ("PM2.5",       sim_pm25,     "μg/m³"),
        ("PM10",        sim_pm10,     "μg/m³"),
        ("Temperature", sim_temp,     "°C"),
        ("Humidity",    sim_humidity, "%"),
        ("Wind Speed",  sim_wind,     "km/h"),
    ]
    for _label, _val, _unit in _sensor_rows:
        st.markdown(
            f"""<div class="sensor-card">
                    <div class="sensor-label">{_label}</div>
                    <div class="sensor-value">{_val:.1f}<span class="sensor-unit">{_unit}</span></div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"""<div style='font-size:0.72rem; color:#475569;'>{data_source_label}</div>""",
        unsafe_allow_html=True,
    )
    if mongo_error:
        st.caption(f"⚠️ {mongo_error}")


# ─── Inference payload ────────────────────────────────────────────────────────
inference_payload = {
    "features": {
        "pm25":                          sim_pm25,
        "pm10":                          sim_pm10,
        "temperature":                   sim_temp,
        "humidity":                      sim_humidity,
        "wind_speed":                    sim_wind,
        "pm25_diff_1h":                  _d(mongo_features, "pm25_diff_1h",      "pm25_diff_1h",      0.0),
        "pm25_roll_std_24h":             _d(mongo_features, "pm25_roll_std_24h", "pm25_roll_std_24h", 0.0),
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
st.caption("Top-10 features by mean |SHAP| value (RF/XGBoost) or |coefficient| (Ridge) — test set.")


# FIX-SHAP-RIDGE: two-stage fetch
#   Stage 1 — try /shap/<model>/<horizon>  (model_shap collection, RF + XGBoost)
#   Stage 2 — if empty, try /features/<model>/<horizon>  (model_features, Ridge coef)
#   Both return identical record format: {"feature": str, "mean_abs_shap": float}
#   so the chart below needs zero changes.
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_shap_or_coef(gateway: str, model_key: str, horizon: int) -> tuple[list[dict], str]:
    """
    Returns (records, value_label) where value_label is used in the chart title.
    Falls back from SHAP → coefficient magnitudes for Ridge.
    """
    if not _flask_reachable:
        return [], "Mean |SHAP|"

    # Stage 1: SHAP (works for RF and XGBoost)
    r = _api_get(gateway, f"/shap/{model_key}/{horizon}", timeout=(10, 30))
    if r and r.status_code == 200:
        try:
            records = r.json().get("records", [])[:10]
            if records:
                return records, "Mean |SHAP|"
        except Exception:
            pass

    # Stage 2: feature coefficients (Ridge fallback)
    # Calls the new /features/<model>/<horizon> route added to api/app.py
    r2 = _api_get(gateway, f"/features/{model_key}/{horizon}", timeout=(10, 30))
    if r2 and r2.status_code == 200:
        try:
            records = r2.json().get("records", [])[:10]
            if records:
                return records, "Mean |Coefficient|"
        except Exception:
            pass

    return [], "Mean |SHAP|"


shap_horizon = st.selectbox(
    "Horizon", [24, 48, 72],
    key="shap_horizon_sel",
    format_func=lambda h: f"{h}h",
)

shap_data, shap_value_label = _fetch_shap_or_coef(api_gateway, active_model_key, shap_horizon)

if shap_data:
    features = [d["feature"]       for d in shap_data]
    values   = [d["mean_abs_shap"] for d in shap_data]

    fig_shap = go.Figure(go.Bar(
        x=values[::-1], y=features[::-1],
        orientation="h",
        marker=dict(
            color=values[::-1],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>" + shap_value_label + ": %{x:.4f}<extra></extra>",
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
                           title=dict(text=shap_value_label, font=dict(color=_TEXT_COLOR, size=11)))
    fig_shap.update_yaxes(showgrid=False, zeroline=False,
                           tickfont=dict(color="#cbd5e1", size=10))
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
else:
    st.info(
        "Feature importance data not yet available for this model/horizon. "
        "Once the training pipeline runs and writes to `model_shap` (RF/XGBoost) "
        "or `model_features` (Ridge) in MongoDB, this chart will populate automatically."
    )


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
#  SECTION 6 — STRATIFIED BAND ERRORS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🏷️ Error by AQI Tier")
st.caption(
    "MAE broken down by EPA health tier — reveals how accurate each model is "
    "specifically for hazardous (>200) and very unhealthy (>150) conditions."
)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_band_errors(gateway: str) -> pd.DataFrame:
    if not _flask_reachable:
        return pd.DataFrame()
    r = _api_get(gateway, "/debug/metrics_raw", timeout=(10, 30))
    if not r or r.status_code != 200:
        return pd.DataFrame()

    BAND_LABELS = {
        "good_moderate":       "Good–Moderate (0–100)",
        "unhealthy_sensitive": "USG (101–150)",
        "unhealthy":           "Unhealthy (151–200)",
        "very_unhealthy":      "Very Unhealthy (201–300)",
        "hazardous":           "Hazardous (301+)",
    }
    BAND_COLORS = {
        "Good–Moderate (0–100)":    "#00e676",
        "USG (101–150)":            "#ff9100",
        "Unhealthy (151–200)":      "#ff1744",
        "Very Unhealthy (201–300)": "#d500f9",
        "Hazardous (301+)":         "#b71c1c",
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