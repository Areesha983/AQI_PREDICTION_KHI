from datetime import datetime, timedelta

LATITUDE  = 24.8607
LONGITUDE = 67.0011

# Use yesterday as the END to avoid incomplete trailing hours
END_DATE   = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
# Go back to Aug 2022 for a ~3-year training window
START_DATE = "2022-08-04"