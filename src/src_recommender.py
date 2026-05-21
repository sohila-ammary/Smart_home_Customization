from .logger import get_logger

logger = get_logger("recommender")

def recommend_scenarios(predicted_activity, hour, temp_mean=None):
    logger.info(
        f"Generating recommendations for activity={predicted_activity}, hour={hour}, temp_mean={temp_mean}"
    )

    recs = []
    activity = str(predicted_activity).lower() if predicted_activity is not None else ""

    if "sleep" in activity:
        recs.extend([
            "Activate sleep scene: turn off unused lights",
            "Set thermostat to night comfort mode",
            "Silence non-essential notifications/devices"
        ])

    if "cook" in activity or "breakfast" in activity or "meal" in activity:
        recs.extend([
            "Activate kitchen scene: brighter lights",
            "Turn on kitchen ventilation if available",
            "Set comfortable cooking temperature"
        ])

    if "leave" in activity or "away" in activity:
        recs.extend([
            "Activate away mode",
            "Turn off unused devices",
            "Lower HVAC usage for energy saving"
        ])

    if 6 <= hour <= 9:
        recs.append("Suggest morning scene: lights up, comfortable temperature")

    if 21 <= hour <= 23:
        recs.append("Suggest bedtime scene: dim lights, reduce noise")

    if temp_mean is not None:
        try:
            if temp_mean >= 25:
                recs.append("Temperature is high: suggest cooling/AC scenario")
            elif temp_mean <= 19:
                recs.append("Temperature is low: suggest heating scenario")
        except Exception:
            logger.warning("Could not evaluate temperature-based recommendation")

    seen = set()
    result = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            result.append(r)

    logger.info(f"Generated {len(result[:5])} recommendations")
    return result[:5]
