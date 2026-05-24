from .logger import get_logger

logger = get_logger("recommendation_fusion")

MIN_RECOMMENDATION_SCORE = 0.08

def fuse_topk_predictions_with_habits(
    dataset_name,
    top_predictions,
    hour,
    temp_mean,
    habit_profiles
):
    logger.info(f"Fusing top-k predictions with habit profile for dataset={dataset_name}")

    profile = habit_profiles.get(dataset_name, {})
    recs = []

    for item in top_predictions:
        activity = item["activity"]
        conf = float(item["confidence"])

        if conf < 0.02:
            continue

        if activity == "Sleep":
            score = conf + (0.10 if hour in profile.get("sleep_hours", []) else 0.0)
            recs.append({
                "scenario": "sleep_scene",
                "score": score,
                "reason": f"Sleep predicted with confidence {conf:.2f}" + (
                    ", and this hour matches usual sleep habit" if hour in profile.get("sleep_hours", []) else ""
                ),
                "actions": [
                    "Turn off unused lights",
                    "Set thermostat to night mode",
                    "Reduce noise / mute non-essential alerts"
                ]
            })

        elif activity == "Meal":
            score = conf + (0.10 if hour in profile.get("meal_hours", []) else 0.0)
            recs.append({
                "scenario": "meal_scene",
                "score": score,
                "reason": f"Meal predicted with confidence {conf:.2f}" + (
                    ", and this hour matches usual meal habit" if hour in profile.get("meal_hours", []) else ""
                ),
                "actions": [
                    "Increase kitchen/dining lighting",
                    "Start ventilation if available",
                    "Prepare appliance shortcuts"
                ]
            })

        elif activity == "Work":
            score = conf + (0.10 if hour in profile.get("work_hours", []) else 0.0)
            recs.append({
                "scenario": "focus_scene",
                "score": score,
                "reason": f"Work predicted with confidence {conf:.2f}",
                "actions": [
                    "Set focused lighting",
                    "Reduce distractions",
                    "Optimize workspace comfort"
                ]
            })

        elif activity == "Relax":
            score = conf + (0.10 if hour in profile.get("relax_hours", []) else 0.0)
            recs.append({
                "scenario": "relax_scene",
                "score": score,
                "reason": f"Relax predicted with confidence {conf:.2f}",
                "actions": [
                    "Dim lights",
                    "Enable comfort entertainment mode",
                    "Adjust evening temperature"
                ]
            })

        elif activity == "Away":
            score = conf + (0.10 if hour in profile.get("away_hours", []) else 0.0)
            recs.append({
                "scenario": "away_scene",
                "score": score,
                "reason": f"Away-related activity predicted with confidence {conf:.2f}",
                "actions": [
                    "Turn off unused devices",
                    "Adjust HVAC for efficiency",
                    "Enable away automation"
                ]
            })

        elif activity == "Bathroom":
            score = conf
            recs.append({
                "scenario": "bathroom_scene",
                "score": score,
                "reason": f"Bathroom-related activity predicted with confidence {conf:.2f}",
                "actions": [
                    "Enable bathroom ventilation",
                    "Adjust lighting brightness",
                    "Maintain comfort temperature"
                ]
            })

        elif activity == "NightBehavior":
            score = conf
            recs.append({
                "scenario": "night_safety_scene",
                "score": score,
                "reason": f"Night behavior predicted with confidence {conf:.2f}",
                "actions": [
                    "Enable low-level path lighting",
                    "Keep temperature comfortable",
                    "Avoid loud notifications"
                ]
            })

        elif activity == "Wake":
            score = conf
            recs.append({
                "scenario": "wake_scene",
                "score": score,
                "reason": f"Wake activity predicted with confidence {conf:.2f}",
                "actions": [
                    "Increase light gradually",
                    "Adjust morning temperature",
                    "Prepare morning device profile"
                ]
            })

    if temp_mean is not None:
        if temp_mean >= 25:
            recs.append({
                "scenario": "cooling_adjustment",
                "score": 0.40,
                "reason": f"Indoor temperature is high ({temp_mean:.1f}°C)",
                "actions": [
                    "Lower cooling setpoint",
                    "Start fan or ventilation"
                ]
            })
        elif temp_mean <= 19:
            recs.append({
                "scenario": "heating_adjustment",
                "score": 0.40,
                "reason": f"Indoor temperature is low ({temp_mean:.1f}°C)",
                "actions": [
                    "Increase heating setpoint",
                    "Activate comfort heating mode"
                ]
            })

    best = {}
    for r in recs:
        if r["score"] < MIN_RECOMMENDATION_SCORE:
            continue
        name = r["scenario"]
        if name not in best or r["score"] > best[name]["score"]:
            best[name] = r

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    logger.info(f"Generated {len(ranked)} fused recommendations")
    return ranked[:5]
