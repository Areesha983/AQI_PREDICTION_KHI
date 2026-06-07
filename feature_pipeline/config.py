from datetime import datetime, timedelta

LATITUDE  = 24.8607
LONGITUDE = 67.0011

# Historical baseline anchor (used for the very first initialization)
HISTORICAL_START_DATE = "2022-08-04"

def get_end_date() -> str:
    """Returns yesterday's date to avoid incomplete trailing hours from API.
    Used by the main historical/training feature pipeline (fetch_air_quality.py,
    fetch_weather.py) — do NOT change this for the realtime path."""
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

def get_today_date() -> str:
    """Returns today's date (no lag).
    Used exclusively by update_realtime.py for the live dashboard path."""
    return datetime.today().strftime("%Y-%m-%d")