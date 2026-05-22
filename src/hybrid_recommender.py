from .logger import get_logger

logger = get_logger("hybrid_recommender")

def generate_hybrid_recommendations(
    dataset_name,
    predicted_activities,
    confidence_scores,
    hour,
    temp_mean,
    habit_profiles,
    top_k=5
):
    logger.info(
        f"Generating hybrid recommendations for dataset={dataset_name}, hour={hour}, temp_mean={temp_mean}"
    )

    profile = habit_profiles.get(dataset_name, {})
    recs = []

    for activity, conf in zip(predicted_activities, confidence_scores):
        act = str(activity)

        if act == "Sleep":
            recs.append({
                "scenario": "sleep_scene",
                "score": float(conf) + (0.15 if hour in profile.get("sleep_hours", []) else 0.0),
                "reason": f"Predicted Sleep with confidence {conf:.2f}",
                "actions": [
                    "Turn off unused lights",
                    "Set thermostat to night mode",
                    "Reduce noise / mute non-essential alerts"
                ]
            })

        elif act == "Meal":
            recs.append({
                "scenario": "meal_scene",
                "score": float(conf) + (0.15 if hour in profile.get("meal_hours", []) else 0.0),
                "reason": f"Predicted Meal-related activity with confidence {conf:.2f}",
                "actions": [
                    "Increase kitchen/dining lighting",
                    "Start ventilation if available",
                    "Prepare common appliance shortcuts"
                ]
            })

        elif act == "Away":
            recs.append({
                "scenario": "away_scene",
                "score": float(conf) + (0.10 if hour in profile.get("away_hours", []) else 0.0),
                "reason": f"Predicted Away/transition activity with confidence {conf:.2f}",
                "actions": [
                    "Turn off unused devices",
                    "Adjust HVAC for efficiency",
                    "Enable away/home automation"
                ]
            })

        elif act == "Work":
            recs.append({
                "scenario": "focus_scene",
                "score": float(conf) + (0.10 if hour in profile.get("work_hours", []) else 0.0),
                "reason": f"Predicted Work/focus activity with confidence {conf:.2f}",
                "actions": [
                    "Set focused lighting",
                    "Reduce distractions",
                    "Optimize workspace comfort"
                ]
            })

        elif act == "Relax":
            recs.append({
                "scenario": "relax_scene",
                "score": float(conf) + (0.10 if hour in profile.get("relax_hours", []) else 0.0),
                "reason": f"Predicted Relax activity with confidence {conf:.2f}",
                "actions": [
                    "Dim lights",
                    "Enable evening comfort mode",
                    "Prepare entertainment environment"
                ]
            })

        elif act == "Bathroom":
            recs.append({
                "scenario": "bathroom_comfort",
                "score": float(conf),
                "reason": f"Predicted Bathroom-related activity with confidence {conf:.2f}",
                "actions": [
                    "Enable bathroom ventilation",
                    "Adjust bathroom light brightness",
                    "Maintain comfort temperature"
                ]
            })

    if temp_mean is not None:
        try:
            if temp_mean >= 25:
                recs.append({
                    "scenario": "cooling_adjustment",
                    "score": 0.35,
                    "reason": f"Indoor temperature is high ({temp_mean:.1f}°C)",
                    "actions": [
                        "Lower cooling setpoint",
                        "Start fan or ventilation"
                    ]
                })
            elif temp_mean <= 19:
                recs.append({
                    "scenario": "heating_adjustment",
                    "score": 0.35,
                    "reason": f"Indoor temperature is low ({temp_mean:.1f}°C)",
                    "actions": [
                        "Increase heating setpoint",
                        "Activate comfort heating mode"
                    ]
                })
        except Exception:
            logger.warning("Temperature handling failed in recommender")

    best = {}
    for r in recs:
        name = r["scenario"]
        if name not in best or r["score"] > best[name]["score"]:
            best[name] = r

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Generated {len(ranked)} hybrid recommendations")
    return ranked
