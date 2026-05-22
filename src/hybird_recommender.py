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
        act = str(activity).lower()

        if "sleep" in act:
            recs.append({
                "scenario": "sleep_scene",
                "score": float(conf) + (0.15 if hour in profile.get("sleep_hours", []) else 0.0),
                "reason": f"Predicted sleep-related activity '{activity}' with confidence {conf:.2f}",
                "actions": [
                    "Turn off unused lights",
                    "Set thermostat to night comfort mode",
                    "Reduce noise / mute non-essential devices"
                ]
            })

        if any(k in act for k in ["breakfast", "lunch", "dinner", "eat", "meal", "cook"]):
            recs.append({
                "scenario": "meal_scene",
                "score": float(conf) + (0.15 if hour in profile.get("meal_hours", []) else 0.0),
                "reason": f"Predicted meal-related activity '{activity}' with confidence {conf:.2f}",
                "actions": [
                    "Increase kitchen/dining lighting",
                    "Turn on ventilation if available",
                    "Prepare preferred appliance shortcuts"
                ]
            })

        if any(k in act for k in ["leave_home", "enter_home"]):
            recs.append({
                "scenario": "away_scene",
                "score": float(conf) + (0.10 if hour in profile.get("away_hours", []) else 0.0),
                "reason": f"Predicted transition activity '{activity}' with confidence {conf:.2f}",
                "actions": [
                    "Enable away/home mode",
                    "Switch off unused devices",
                    "Adjust HVAC for efficiency"
                ]
            })

        if any(k in act for k in ["work", "desk", "read"]):
            recs.append({
                "scenario": "focus_scene",
                "score": float(conf) + (0.10 if hour in profile.get("work_hours", []) else 0.0),
                "reason": f"Predicted focus-related activity '{activity}' with confidence {conf:.2f}",
                "actions": [
                    "Set focused lighting",
                    "Reduce distractions",
                    "Optimize workspace temperature"
                ]
            })

        if any(k in act for k in ["relax", "watch_tv"]):
            recs.append({
                "scenario": "relax_scene",
                "score": float(conf) + (0.10 if hour in profile.get("relax_hours", []) else 0.0),
                "reason": f"Predicted relax-related activity '{activity}' with confidence {conf:.2f}",
                "actions": [
                    "Dim lights",
                    "Enable entertainment comfort mode",
                    "Set comfortable evening temperature"
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

    # Deduplicate by scenario, keep highest score
    best = {}
    for r in recs:
        name = r["scenario"]
        if name not in best or r["score"] > best[name]["score"]:
            best[name] = r

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Generated {len(ranked)} hybrid recommendations")
    return ranked
