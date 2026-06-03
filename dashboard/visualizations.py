"""
Dashboard visualization utilities.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_error_progression(metrics_df: pd.DataFrame):
    """
    RMSE vs MAE progression chart.
    Matplotlib is fine here.
    """

    fig, ax = plt.subplots(figsize=(7, 4.5))

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(
        metrics_df["Horizon"],
        metrics_df["CV Mean RMSE"],
        marker="o",
        color="#3b82f6",
        linewidth=2.5,
        label="RMSE"
    )

    ax.plot(
        metrics_df["Horizon"],
        metrics_df["Test MAE"],
        marker="s",
        color="#10b981",
        linewidth=2.5,
        label="MAE"
    )

    ax.set_title(
        "Error Progression Matrix",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

    ax.tick_params(
        colors="#b0b3b8",
        labelsize=9
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.05,
        color="white"
    )

    legend = ax.legend(
        facecolor="#1c1c24",
        edgecolor="none"
    )

    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("#2d2d38")

    plt.tight_layout()

    return fig


def plot_variance_matrix(metrics_df: pd.DataFrame):
    """
    Interactive Plotly R² profile chart.
    Replaces problematic Matplotlib implementation.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics_df["Horizon"],
            y=metrics_df["Test R² Score"],
            marker_color="#3b82f6",
            opacity=0.8,
            text=metrics_df["Test R² Score"],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "R² Score: %{y:.3f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title={
            "text": "R² Variance Profile",
            "x": 0.02,
            "font": {
                "size": 18,
                "color": "white"
            }
        },

        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),

        xaxis=dict(
            title="Forecast Horizon",
            showgrid=False,
            zeroline=False
        ),

        yaxis=dict(
            title="R² Score",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False
        ),

        height=420
    )

    return fig