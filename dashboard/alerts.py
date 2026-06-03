"""
Threshold rule-engine analyzer for Karachi AQI.
Translates numeric predictions into health advisories.
"""


def get_epa_tier_details(aqi_value: float) -> dict:

    if aqi_value <= 50:
        return {
            "label": "Good",
            "color": "#00e400",
            "bg_gradient": "rgba(0,228,0,0.1)",
            "advice": "Air quality is satisfactory. Enjoy outdoor activities."
        }

    elif aqi_value <= 100:
        return {
            "label": "Moderate",
            "color": "#ffff00",
            "bg_gradient": "rgba(255,255,0,0.1)",
            "advice": "Sensitive individuals should limit prolonged outdoor exertion."
        }

    elif aqi_value <= 150:
        return {
            "label": "Unhealthy for Sensitive Groups",
            "color": "#ff7e00",
            "bg_gradient": "rgba(255,126,0,0.1)",
            "advice": "N95 masks recommended for sensitive groups. Limit outdoor exposure."
        }

    elif aqi_value <= 200:
        return {
            "label": "Unhealthy",
            "color": "#ff0000",
            "bg_gradient": "rgba(255,0,0,0.1)",
            "advice": "Avoid strenuous outdoor activities. Everyone should wear a mask."
        }

    else:
        return {
            "label": "Hazardous",
            "color": "#7e0023",
            "bg_gradient": "rgba(126,0,35,0.1)",
            "advice": "HEALTH WARNING: Emergency conditions. Everyone should stay indoors."
        }