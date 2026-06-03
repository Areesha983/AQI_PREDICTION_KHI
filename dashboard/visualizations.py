"""
Dashboard visualization utilities for AirLyst Karachi.
All charts use Plotly for a consistent dark glassmorphic aesthetic.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme constants ───────────────────────────────────────────────────
_TRANSPARENT = "rgba(0,0,0,0)"
_GRID_COLOR   = "rgba(255,255,255,0.06)"
_TEXT_COLOR   = "#94a3b8"
_FONT_FAMILY  = "JetBrains Mono, monospace"

_LAYOUT_BASE = dict(
    plot_bgcolor  = _TRANSPARENT,
    paper_bgcolor = _TRANSPARENT,
    font          = dict(color=_TEXT_COLOR, family=_FONT_FAMILY, size=11),
    margin        = dict(l=10, r=10, t=48, b=10),
    legend        = dict(
        bgcolor     = "rgba(255,255,255,0.03)",
        bordercolor = "rgba(255,255,255,0.08)",
        borderwidth = 1,
        font        = dict(color="#cbd5e1", size=10),
    ),
)

_MODEL_COLORS = {
    "Random Forest": "#3b82f6",
    "Xgboost":       "#10b981",
    "Ridge":         "#f59e0b",
}


def _apply_axis_style(fig: go.Figure, x_title: str = "", y_title: str = "") -> None:
    """Apply consistent dark axis styling in-place."""
    axis_style = dict(
        showgrid    = True,
        gridcolor   = _GRID_COLOR,
        zeroline    = False,
        linecolor   = "rgba(255,255,255,0.08)",
        tickfont    = dict(color=_TEXT_COLOR, size=10),
    )
    fig.update_xaxes(**axis_style, title=dict(text=x_title, font=dict(color=_TEXT_COLOR, size=11)))
    fig.update_yaxes(**axis_style, title=dict(text=y_title, font=dict(color=_TEXT_COLOR, size=11)))


# ── 1. RMSE Progression ──────────────────────────────────────────────────────
def plot_error_progression(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Multi-model RMSE line chart across forecast horizons.
    Expects columns: Horizon, RMSE, Model
    """
    fig = go.Figure()

    for model, color in _MODEL_COLORS.items():
        df_m = metrics_df[metrics_df["Model"] == model]
        if df_m.empty:
            continue
        fig.add_trace(go.Scatter(
            x          = df_m["Horizon"],
            y          = df_m["RMSE"],
            name       = model,
            mode       = "lines+markers",
            line       = dict(color=color, width=2.5),
            marker     = dict(size=7, color=color, line=dict(color="#0a0d14", width=1.5)),
            hovertemplate = "<b>%{x}</b><br>RMSE: %{y:.2f}<extra>" + model + "</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = dict(text="RMSE Progression", x=0.01,
                     font=dict(color="#f1f5f9", size=14, family=_FONT_FAMILY)),
        height = 380,
    )
    _apply_axis_style(fig, x_title="Forecast Horizon", y_title="RMSE")
    return fig


# ── 2. R² Variance Profile ───────────────────────────────────────────────────
def plot_variance_matrix(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing R² scores per model across horizons.
    Expects columns: Horizon, R² Score, Model
    """
    fig = go.Figure()

    for model, color in _MODEL_COLORS.items():
        df_m = metrics_df[metrics_df["Model"] == model]
        if df_m.empty:
            continue
        fig.add_trace(go.Bar(
            x             = df_m["Horizon"],
            y             = df_m["R² Score"],
            name          = model,
            marker_color  = color,
            opacity       = 0.85,
            text          = df_m["R² Score"].round(3),
            textposition  = "outside",
            textfont      = dict(color=color, size=9),
            hovertemplate = "<b>%{x}</b><br>R²: %{y:.3f}<extra>" + model + "</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode = "group",
        bargap  = 0.25,
        title   = dict(text="R² Variance Profile", x=0.01,
                       font=dict(color="#f1f5f9", size=14, family=_FONT_FAMILY)),
        height  = 380,
    )
    _apply_axis_style(fig, x_title="Forecast Horizon", y_title="R² Score")
    fig.update_yaxes(range=[0, 1.12])
    return fig


# ── 3. AQI Gauge ─────────────────────────────────────────────────────────────
def plot_aqi_gauge(aqi_value: float, tier: dict, horizon_label: str) -> go.Figure:
    """
    Single-value gauge for a forecast horizon AQI.
    tier dict must have 'color' and 'label' keys (from alerts.get_epa_tier_details).
    """
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = aqi_value,
        number = dict(font=dict(color=tier["color"], size=42, family=_FONT_FAMILY)),
        title  = dict(text=f"{horizon_label}<br><span style='font-size:12px;color:{tier['color']}'>{tier['label']}</span>",
                      font=dict(color="#f1f5f9", size=13, family=_FONT_FAMILY)),
        gauge  = dict(
            axis       = dict(range=[0, 500], tickcolor=_TEXT_COLOR,
                              tickfont=dict(color=_TEXT_COLOR, size=9)),
            bar        = dict(color=tier["color"], thickness=0.25),
            bgcolor    = "rgba(255,255,255,0.03)",
            bordercolor= "rgba(255,255,255,0.06)",
            steps      = [
                dict(range=[0,   50],  color="rgba(0,230,118,0.12)"),
                dict(range=[50,  100], color="rgba(255,202,40,0.12)"),
                dict(range=[100, 150], color="rgba(255,145,0,0.12)"),
                dict(range=[150, 200], color="rgba(255,23,68,0.12)"),
                dict(range=[200, 300], color="rgba(213,0,249,0.12)"),
                dict(range=[300, 500], color="rgba(183,28,28,0.12)"),
            ],
            threshold  = dict(line=dict(color=tier["color"], width=3), thickness=0.8, value=aqi_value),
        ),
    ))
    fig.update_layout(
        plot_bgcolor  = _TRANSPARENT,
        paper_bgcolor = _TRANSPARENT,
        font          = dict(color=_TEXT_COLOR, family=_FONT_FAMILY, size=11),
        height        = 260,
        margin        = dict(l=20, r=20, t=60, b=10),
    )
    return fig