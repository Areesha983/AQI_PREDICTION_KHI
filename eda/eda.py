"""
Karachi AQI — Exploratory Data Analysis Pipeline
─────────────────────────────────────────────────
Data source : MongoDB Atlas  →  karachi_aqi.processed_features
Local cache : eda/data/featured_dataset.csv   (re-used on subsequent runs)
Outputs     : eda/outputs/  (PNG charts + CSV summaries)

Run once to pull + cache, then re-runs are instant (local CSV used).
The CSV is .gitignored — it never gets pushed to GitHub.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sns.set_style("whitegrid")

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
EDA_DIR    = Path(__file__).resolve().parent
DATA_DIR   = EDA_DIR / "data"
OUTPUT_DIR = EDA_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CSV = DATA_DIR / "featured_dataset.csv"

# Load .env from project root
load_dotenv(BASE_DIR / ".env")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME   = "karachi_aqi"
COL_NAME  = "processed_features"

print("=" * 80)
print("   KARACHI AQI — EDA ENGINE")
print("=" * 80)


# ─── 1. Fetch or Load Data ──────────────────────────────────────────────────
def fetch_from_mongo() -> pd.DataFrame:
    """Pull all documents from processed_features and return as DataFrame."""
    from pymongo import MongoClient
    if not MONGO_URI:
        raise ValueError(
            "MONGODB_URI not found in environment.\n"
            "Add it to your .env file: MONGODB_URI=\"mongodb+srv://...\""
        )
    print("Connecting to MongoDB Atlas …")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    db     = client[DB_NAME]
    col    = db[COL_NAME]

    total = col.count_documents({})
    print(f"  Found {total:,} documents in {DB_NAME}.{COL_NAME}")
    if total == 0:
        raise ValueError("Collection is empty — run the feature pipeline first.")

    docs = list(col.find({}, {"_id": 0}))
    client.close()
    print(f"  Fetched {len(docs):,} documents from MongoDB ✓")
    return pd.DataFrame(docs)


if LOCAL_CSV.exists():
    print(f"Local cache found → loading from:\n  {LOCAL_CSV}")
    df = pd.read_csv(LOCAL_CSV)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols from cache.")
else:
    print("No local cache — fetching from MongoDB …")
    df = fetch_from_mongo()
    df.to_csv(LOCAL_CSV, index=False)
    print(f"  Saved local cache → {LOCAL_CSV}  ({LOCAL_CSV.stat().st_size / 1e6:.1f} MB)")

print(
    "\n[TIP] Delete  eda/data/featured_dataset.csv  to force a fresh MongoDB pull.\n"
    "      The file is .gitignored — it will never be pushed to GitHub.\n"
)


# ─── 2. Datetime Index ──────────────────────────────────────────────────────
datetime_col = None
for candidate in ["datetime", "timestamp", "date", "Date", "Datetime"]:
    if candidate in df.columns:
        datetime_col = candidate
        break

if datetime_col:
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.sort_values(datetime_col).reset_index(drop=True)
    date_start = df[datetime_col].min().strftime("%Y-%m-%d")
    date_end   = df[datetime_col].max().strftime("%Y-%m-%d")
    print(f"Temporal index: [{datetime_col}]  {date_start} → {date_end}")
else:
    print("WARNING: No datetime column found — time-series charts skipped.")

# AQI column safety fallback
if "aqi" not in df.columns and "pm25" in df.columns:
    df["aqi"] = df["pm25"]


# ─── 3. High-Level Metrics Summary ──────────────────────────────────────────
print("\n" + "─" * 50)
print(" DATASET PROFILE METRICS")
print("─" * 50)

overview_metrics = {
    "Total Record Rows":        len(df),
    "Total Input Features":     len(df.columns),
    "Global NaN Cell Count":    int(df.isna().sum().sum()),
    "Missing Cells Percent":    float(df.isna().sum().sum() / df.size * 100),
    "AQI Arithmetic Mean":      float(df["aqi"].mean()),
    "AQI Midpoint Median":      float(df["aqi"].median()),
    "AQI Standard Deviation":   float(df["aqi"].std()),
    "AQI Absolute Minimum":     float(df["aqi"].min()),
    "AQI Absolute Maximum":     float(df["aqi"].max()),
}

for name, val in overview_metrics.items():
    if "Percent" in name:
        fmt = f"{val:,.2f}%"
    elif isinstance(val, float):
        fmt = f"{val:,.2f}"
    else:
        fmt = f"{int(val):,}"
    print(f"  ├─ {name:<30}: {fmt}")

pd.DataFrame(list(overview_metrics.items()), columns=["Metric", "Value"]).to_csv(
    OUTPUT_DIR / "dataset_overview.csv", index=False
)


# ─── 4. EPA Spike Analysis ──────────────────────────────────────────────────
print("\n" + "─" * 50)
print(" EPA HAZARD SPIKE ANALYSIS")
print("─" * 50)

spikes_150 = int((df["aqi"] > 150).sum())
spikes_200 = int((df["aqi"] > 200).sum())
spikes_300 = int((df["aqi"] > 300).sum())

print(f"  ├─ AQI > 150 (Unhealthy)       : {spikes_150:,}  ({100*spikes_150/len(df):.3f}%)")
print(f"  ├─ AQI > 200 (Very Unhealthy)  : {spikes_200:,}  ({100*spikes_200/len(df):.3f}%)")
print(f"  └─ AQI > 300 (Hazardous)       : {spikes_300:,}  ({100*spikes_300/len(df):.3f}%)")


# ─── 5. Missing Values Chart ────────────────────────────────────────────────
missing_series = df.isna().sum().sort_values(ascending=False)
missing_series = missing_series[missing_series > 0]

if len(missing_series) > 0:
    fig, ax = plt.subplots(figsize=(12, min(8, len(missing_series) * 0.4 + 2)))
    sns.barplot(x=missing_series.values, y=missing_series.index, ax=ax, palette="flare")
    ax.set_title("Missing Values by Feature", fontsize=12, fontweight="bold")
    ax.set_xlabel("Missing Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=300)
    plt.close()
    print(f"\n  Saved: missing_values.png  ({len(missing_series)} features with nulls)")
else:
    print("\n  No missing values detected — missing_values.png skipped.")


# ─── 6. AQI Distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["aqi"], bins=60, kde=True, color="midnightblue", ax=ax)
ax.axvline(df["aqi"].mean(),   color="red",    linestyle="--", linewidth=1.5, label=f"Mean: {df['aqi'].mean():.1f}")
ax.axvline(df["aqi"].median(), color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {df['aqi'].median():.1f}")
ax.set_title("AQI Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("AQI Value")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aqi_distribution.png", dpi=300)
plt.close()


# ─── 7. Time-Series Charts ──────────────────────────────────────────────────
if datetime_col:
    # Full timeline
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df[datetime_col], df["aqi"], color="teal", linewidth=0.6, alpha=0.85, label="Hourly AQI")
    ax.axhline(150, color="darkorange", linestyle=":", alpha=0.7, label="EPA Unhealthy (150)")
    ax.axhline(200, color="crimson",    linestyle=":", alpha=0.7, label="EPA Very Unhealthy (200)")
    ax.set_title("AQI Timeline", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_timeseries.png", dpi=300)
    plt.close()

    # Monthly boxplot
    if "month" not in df.columns:
        df["month"] = df[datetime_col].dt.month
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="month", y="aqi", data=df, ax=ax, palette="viridis")
    ax.set_title("Monthly AQI Seasonality", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_monthly_boxplot.png", dpi=300)
    plt.close()

    # Diurnal profile
    if "hour" not in df.columns:
        df["hour"] = df[datetime_col].dt.hour
    hourly_mean = df.groupby("hour")["aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=hourly_mean, x="hour", y="aqi", marker="o", color="darkviolet", ax=ax, linewidth=2)
    ax.set_title("Diurnal AQI Profile (24-Hour Mean)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean AQI")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aqi_hourly_average.png", dpi=300)
    plt.close()


# ─── 8. Correlation Analysis ────────────────────────────────────────────────
all_possible_targets = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
]
clean_df = df.select_dtypes(include=[np.number]).drop(
    columns=[t for t in all_possible_targets if t in df.columns], errors="ignore"
)

# Core environmental heatmap
core_cols = [c for c in ["aqi", "pm25", "pm10", "humidity", "temperature", "wind_speed"] if c in clean_df.columns]
if len(core_cols) >= 2:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(clean_df[core_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f",
                square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Core Feature Correlation Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

# Top-25 features correlated with AQI
if "aqi" in clean_df.columns:
    target_corr = clean_df.corr()["aqi"].drop(index="aqi", errors="ignore")
    top25 = target_corr.abs().sort_values(ascending=False).head(25)
    signed = target_corr.loc[top25.index]

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(x=signed.values, y=signed.index, ax=ax, palette="coolwarm")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 25 Features Correlated with AQI", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (R)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlated_features.png", dpi=300)
    plt.close()

    signed.to_csv(OUTPUT_DIR / "top_correlated_features.csv")


# ─── 9. Descriptive Statistics ──────────────────────────────────────────────
df.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

print("\n" + "=" * 80)
print(f" EDA COMPLETE. Outputs saved to:\n  {OUTPUT_DIR.resolve()}")
print("=" * 80)