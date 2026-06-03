from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for production runs
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ─── System Paths & Configuration ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "feature_pipeline" / "data" / "processed" / "featured_dataset.csv"

# Local fallback configuration block
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).resolve().parent / "featured_dataset.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"   LAUNCHING ENTERPRISE EDA ENGINE FOR KARACHI AQI ANALYSIS PROFILE")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing primary source dataset file at: {DATA_PATH}")

# ─── 1. Load Data Asset ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Data Asset ingestion success. Working dimensions: {df.shape[0]:,} rows | {df.shape[1]} features.")

# Find chronological index anchor column
datetime_col = None
for candidate in ["datetime", "timestamp", "date", "Date", "Datetime"]:
    if candidate in df.columns:
        datetime_col = candidate
        break

if datetime_col:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    date_start = df[datetime_col].min().strftime('%Y-%m-%d')
    date_end = df[datetime_col].max().strftime('%Y-%m-%d')
    print(f"Temporal index confirmed via [{datetime_col}]. Coverage range: {date_start} to {date_end}")
else:
    print("WARNING: No valid datetime index recognized. Time-series sorting bypassed.")

# Fallback column safety protection guard
if "aqi" not in df.columns and "pm25" in df.columns:
    df["aqi"] = df["pm25"]  # Use pm25 directly if raw metric is unnamed

# ─── 2. High-Level Metrics Summary ─────────────────────────────────────────
print("\n" + "─" * 50 + "\n DATASET PROFILE METRICS \n" + "─" * 50)

overview_metrics = {
    "Total Record Rows": len(df),
    "Total Input Features": len(df.columns),
    "Global NaN Cell Count": int(df.isna().sum().sum()),
    "Missing Cells Percent": float((df.isna().sum().sum() / df.size) * 100),
    "AQI Arithmetic Mean": float(df["aqi"].mean()),
    "AQI Midpoint Median": float(df["aqi"].median()),
    "AQI Standard Deviation": float(df["aqi"].std()),
    "AQI Absolute Minimum": float(df["aqi"].min()),
    "AQI Absolute Maximum": float(df["aqi"].max()),
}

for metric_name, value in overview_metrics.items():
    formatter = f"{value:,.2f}%" if "Percent" in metric_name else f"{value:,.2f}" if "AQI" in metric_name or "Mean" in metric_name else f"{int(value):,}"
    print(f"  ├─ {metric_name:<30}: {formatter}")

# Save structured file for summary tables
pd.DataFrame(list(overview_metrics.items()), columns=["Metric", "Value"]).to_csv(
    OUTPUT_DIR / "dataset_overview.csv", index=False
)

# ─── 3. Extreme Event Spike Calculations ────────────────────────────────────
print("\n" + "─" * 50 + "\n EPA HAZARD SPIKE ANALYSIS \n" + "─" * 50)
spikes_150 = int((df["aqi"] > 150).sum())
spikes_200 = int((df["aqi"] > 200).sum())
spikes_300 = int((df["aqi"] > 300).sum())

print(f"  ├─ AQI > 150 (Unhealthy Zone)      : {spikes_150:,} counts ({100*spikes_150/len(df):.3f}%)")
print(f"  ├─ AQI > 200 (Very Unhealthy Zone) : {spikes_200:,} counts ({100*spikes_200/len(df):.3f}%)")
print(f"  └─ AQI > 300 (Hazardous Spike)     : {spikes_300:,} counts ({100*spikes_300/len(df):.3f}%)")

# ─── 4. Visual Quality Maps & Data Profiling ────────────────────────────────
missing_series = df.isna().sum().sort_values(ascending=False)
missing_series = missing_series[missing_series > 0]

if len(missing_series) > 0:
    fig, ax = plt.subplots(figsize=(12, min(8, len(missing_series) * 0.4)))
    sns.barplot(x=missing_series.values, y=missing_series.index, ax=ax, palette="flare")
    ax.set_title("Missing Cell Deficit Profile Across Feature Space Columns", fontsize=12, fontweight="bold")
    ax.set_xlabel("Missing Sample Counts")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=300)
    plt.close()

# AQI Value Log Distribution Density Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["aqi"], bins=60, kde=True, color="midnightblue", ax=ax)
ax.axvline(df["aqi"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df['aqi'].mean():.1f}")
ax.axvline(df["aqi"].median(), color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {df['aqi'].median():.1f}")
ax.set_title("AQI Density Distribution Curve & Asymmetry Profile", fontsize=12, fontweight="bold")
ax.set_xlabel("Calculated Continuous AQI Value")
ax.set_ylabel("Density Distribution Count")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aqi_distribution.png", dpi=300)
plt.close()

# ─── 5. Seasonality & Trend Profiling ───────────────────────────────────────
if datetime_col:
    # Continuous Chronological Baseline Time-Series Chart
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df[datetime_col], df["aqi"], color="teal", linewidth=0.6, alpha=0.85, label="Hourly AQI")
    ax.axhline(150, color="darkorange", linestyle=":", alpha=0.7, label="EPA Unhealthy (150)")
    ax.axhline(200, color="crimson", linestyle=":", alpha=0.7, label="EPA Very Unhealthy (200)")
    ax.set_title("System-Wide AQI Variance Timeline & Spike Trajectories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timeline Sequence Axis")
    ax.set_ylabel("AQI Metric Scale")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_timeseries.png", dpi=300)
    plt.close()

    # Capture Month-of-Year Cyclical Groups
    if "month" not in df.columns:
        df["month"] = df[datetime_col].dt.month

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="month", y="aqi", data=df, ax=ax, palette="viridis")
    ax.set_title("Monthly Seasonality Variations (AQI Aggregations)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calendar Month Index")
    ax.set_ylabel("AQI Metric Spread")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_monthly_boxplot.png", dpi=300)
    plt.close()

    # Capture Time-of-Day Diurnal Profiles
    if "hour" not in df.columns:
        df["hour"] = df[datetime_col].dt.hour

    hourly_mean_df = df.groupby("hour")["aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=hourly_mean_df, x="hour", y="aqi", marker="o", color="darkviolet", ax=ax, linewidth=2)
    ax.set_title("Diurnal Variations Profile (24-Hour Mean AQI Cycle)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day (Local Standard Time)")
    ax.set_ylabel("Mean Aggregated AQI")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_hourly_average.png", dpi=300)
    plt.close()

# ─── 6. Clean Target Features for Correlation Analysis ──────────────────────
# Excludes target variables and labels to prevent artificial correlation signals
all_possible_targets = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h"
]
clean_numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[t for t in all_possible_targets if t in df.columns], errors="ignore")

# Correlation Heatmap for Core Environmental Metrics
core_environmental_metrics = ["aqi", "pm25", "pm10", "humidity", "temperature", "wind_speed"]
corr_subset_cols = [col for col in core_environmental_metrics if col in clean_numeric_df.columns]

if len(corr_subset_cols) >= 2:
    fig, ax = plt.subplots(figsize=(8, 7))
    correlation_matrix = clean_numeric_df[corr_subset_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Core Meteorological & Ambient Pollution Matrix Correlations", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

# ─── 7. Isolate Top 25 Highest-Impact Correlated Features ─────────────────────
if "aqi" in clean_numeric_df.columns:
    target_correlations = clean_numeric_df.corr()["aqi"].drop(index="aqi", errors="ignore")
    sorted_absolute_correlations = target_correlations.abs().sort_values(ascending=False)
    
    # Extract top 25 features to map regional data dynamics
    top_25_features = sorted_absolute_correlations.head(25)
    
    fig, ax = plt.subplots(figsize=(11, 8))
    # Map signed correlation coefficients to show inverse/positive trends
    signed_correlation_values = target_correlations.loc[top_25_features.index]
    
    sns.barplot(x=signed_correlation_values.values, y=signed_correlation_values.index, ax=ax, palette="coolwarm")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Top 25 Engineered Predictive Features Correlated with AQI", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient ($R$)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlated_features.png", dpi=300)
    plt.close()
    
    # Save files to disk for review
    signed_correlation_values.to_csv(OUTPUT_DIR / "top_correlated_features.csv")

# ─── 8. Save Descriptive Summary Files ─────────────────────────────────────
df.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

print("\n" + "=" * 80)
print(f" EDA LOG PIPELINE COMPLETE. EVALUATION GRAPHICS EXPORTED TO:\n {OUTPUT_DIR.resolve()}")
print("=" * 80)