from .logger import get_logger

logger = get_logger("advanced_recommender")

SCENARIO_RULES = [
    {
        "name": "sleep_scene",
        "keywords": ["sleep", "sleeping"],
        "actions": [
            "Turn off unused lights",
            "Set thermostat to night eco/comfort mode",
            "Mute non-essential notifications",
            "Lock doors if supported"
        ]
    },
    {
        "name": "cooking_scene",
        "keywords": ["cook", "breakfast", "lunch", "dinner", "meal_preparation", "eating"],
        "actions": [
            "Increase kitchen lighting",
            "Enable kitchen ventilation",
            "Set comfortable kitchen temperature",
            "Prepare appliance shortcuts"
        ]
    },
    {
        "name": "away_scene",
        "keywords": ["leave_home", "enter_home"],
        "actions": [
            "Enable away/return automation",
            "Turn off unnecessary devices",
            "Adjust HVAC for energy savings",
            "Arm security mode if available"
        ]
    },
    {
        "name": "work_scene",
        "keywords": ["work", "desk_activity", "read"],
        "actions": [
            "Set focused lighting",
            "Reduce distractions",
            "Optimize desk-area temperature",
            "Prepare work device profile"
        ]
    },
]

def score_scenarios(predicted_activity, hour, temp_mean=None, top_k=5):
    activity = str(predicted_activity).lower()
    logger.info(
        f"Scoring scenarios for activity={predicted_activity}, hour={hour}, temp_mean={temp_mean}"
    )

    scored = []

    for rule in SCENARIO_RULES:
        score = 0
        for kw in rule["keywords"]:
            if kw in activity:
                score += 5

        if rule["name"] == "sleep_scene" and 21 <= hour <= 23:
            score += 2
        if rule["name"] == "cooking_scene" and 6 <= hour <= 9:
            score += 1
        if rule["name"] == "work_scene" and 8 <= hour <= 18:
            score += 2

        if temp_mean is not None:
            if temp_mean > 25:
                score += 1
            elif temp_mean < 19:
                score += 1

        if score > 0:
            scored.append({
                "scenario": rule["name"],
                "score": score,
                "actions": rule["actions"]
            })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Generated {len(scored)} ranked scenarios")
    return scored
