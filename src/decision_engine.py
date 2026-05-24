from .logger import get_logger

logger = get_logger("decision_engine")

HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.50

AUTO_SCENARIOS = {
    "sleep_scene",
    "away_scene",
    "heating_adjustment",
    "cooling_adjustment",
}

def classify_confidence(top_predictions):
    if not top_predictions:
        return "low"

    top_conf = float(top_predictions[0]["confidence"])
    if top_conf >= HIGH_CONFIDENCE:
        return "high"
    elif top_conf >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"

def get_habit_fallback(dataset_name, hour, habit_profiles):
    profile = habit_profiles.get(dataset_name, {})
    dominant = profile.get("dominant_activity_by_hour", {}).get(str(hour))
    if dominant is None:
        dominant = profile.get("dominant_activity_by_hour", {}).get(hour)
    return dominant

def decide_trigger_mode(top_predictions, recommendations):
    if not top_predictions or not recommendations:
        return "none"

    conf_level = classify_confidence(top_predictions)
    top_conf = float(top_predictions[0]["confidence"])
    top_scenario = recommendations[0]["scenario"]
    top_score = float(recommendations[0]["score"])

    if conf_level == "high" and top_scenario in AUTO_SCENARIOS and top_score >= 0.80:
        return "auto"
    elif conf_level in {"high", "medium"} and top_score >= 0.15:
        return "suggest"
    return "none"

def build_decision_payload(dataset_name, top_predictions, recommendations, hour, habit_profiles):
    conf_level = classify_confidence(top_predictions)
    fallback = get_habit_fallback(dataset_name, hour, habit_profiles)
    trigger_mode = decide_trigger_mode(top_predictions, recommendations)

    payload = {
        "confidence_level": conf_level,
        "trigger_mode": trigger_mode,
        "habit_fallback": fallback,
        "top_prediction": top_predictions[0] if top_predictions else None,
        "top_predictions": top_predictions,
        "recommendations": recommendations,
    }

    logger.info(
        f"Decision payload built for dataset={dataset_name} with confidence={conf_level}, trigger_mode={trigger_mode}"
    )
    return payload
