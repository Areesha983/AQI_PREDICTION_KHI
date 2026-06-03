"""
Threshold rule-engine for Karachi AQI.
Translates numeric AQI predictions into EPA health tiers with full UI metadata.
"""


def get_epa_tier_details(aqi_value: float) -> dict:
    """
    Returns label, accent color, card background, and health advisory
    for any AQI value. Covers all six EPA breakpoints including Very Unhealthy.
    """
    if aqi_value <= 50:
        return {
            "label": "Good",
            "color": "#00e676",
            "bg": "rgba(0, 230, 118, 0.07)",
            "advice": "Air quality is satisfactory. Enjoy outdoor activities freely."
        }
    elif aqi_value <= 100:
        return {
            "label": "Moderate",
            "color": "#ffca28",
            "bg": "rgba(255, 202, 40, 0.07)",
            "advice": "Sensitive individuals should limit prolonged outdoor exertion."
        }
    elif aqi_value <= 150:
        return {
            "label": "Unhealthy for Sensitive Groups",
            "color": "#ff9100",
            "bg": "rgba(255, 145, 0, 0.07)",
            "advice": "N95 masks recommended for sensitive groups. Limit outdoor exposure."
        }
    elif aqi_value <= 200:
        return {
            "label": "Unhealthy",
            "color": "#ff1744",
            "bg": "rgba(255, 23, 68, 0.07)",
            "advice": "Avoid strenuous outdoor activities. Everyone should wear a mask."
        }
    elif aqi_value <= 300:
        return {
            "label": "Very Unhealthy",
            "color": "#d500f9",
            "bg": "rgba(213, 0, 249, 0.07)",
            "advice": "Health alert: significant risk for the general public. Stay indoors."
        }
    else:
        return {
            "label": "Hazardous",
            "color": "#b71c1c",
            "bg": "rgba(183, 28, 28, 0.07)",
            "advice": "HEALTH WARNING: Emergency conditions. Everyone must stay indoors."
        }