import json
from collections import defaultdict
from .logger import get_logger

logger = get_logger("habit_profile")

def build_habit_profiles(activity_hour_profile_df):
    logger.info("Building habit profiles from activity hour profile dataframe")

    profiles = {}

    if activity_hour_profile_df.empty:
        logger.warning("Empty activity_hour_profile_df received")
        return profiles

    for dataset_name, ds in activity_hour_profile_df.groupby("dataset"):
        profile = {
            "dataset": dataset_name,
            "top_hours_by_activity": {},
            "dominant_activity_by_hour": {},
            "sleep_hours": [],
            "meal_hours": [],
            "away_hours": [],
            "work_hours": [],
            "relax_hours": [],
        }

        for activity, act_df in ds.groupby("label"):
            top_hours = (
                act_df.sort_values("count", ascending=False)["hour"]
                .astype(int)
                .tolist()[:5]
            )
            profile["top_hours_by_activity"][activity] = top_hours

        for hour, hour_df in ds.groupby("hour"):
            top_row = hour_df.sort_values("count", ascending=False).iloc[0]
            profile["dominant_activity_by_hour"][int(hour)] = {
                "activity": top_row["label"],
                "count": int(top_row["count"])
            }

        def collect_hours(keyword_list):
            hours = []
            for activity, top_hours in profile["top_hours_by_activity"].items():
                act_l = activity.lower()
                if any(k in act_l for k in keyword_list):
                    hours.extend(top_hours)
            return sorted(list(set(hours)))

        profile["sleep_hours"] = collect_hours(["sleep"])
        profile["meal_hours"] = collect_hours(["breakfast", "lunch", "dinner", "eat", "meal", "cook"])
        profile["away_hours"] = collect_hours(["leave_home", "enter_home"])
        profile["work_hours"] = collect_hours(["work", "desk", "read"])
        profile["relax_hours"] = collect_hours(["relax", "watch_tv"])

        profiles[dataset_name] = profile
        logger.info(f"Habit profile built for dataset={dataset_name}")

    return profiles

def save_habit_profiles(profiles, output_dir):
    logger.info("Saving habit profiles to JSON")
    for dataset_name, profile in profiles.items():
        path = output_dir / f"habit_profile_{dataset_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Saved habit profile: {path}")
