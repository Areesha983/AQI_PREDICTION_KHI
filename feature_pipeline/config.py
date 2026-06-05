from datetime import datetime, timedelta

LATITUDE  = 24.8607
LONGITUDE = 67.0011

# Historical baseline anchor (used for the very first initialization)
HISTORICAL_START_DATE = "2022-08-04"

def get_end_date() -> str:
    """Returns yesterday's date to avoid incomplete trailing hours from API."""
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")