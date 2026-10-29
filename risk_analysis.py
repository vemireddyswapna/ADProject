def get_risk_level(prediction):
    risk_levels = {
        "Dyskeratotic": "HIGH",
        "Koilocytotic": "MEDIUM",
        "Metaplastic": "LOW",
        "Parabasal": "LOW",
        "Superficial-Intermediate": "LOW"
    }

    return risk_levels.get(prediction, "UNKNOWN")