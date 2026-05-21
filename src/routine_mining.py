import pandas as pd
from .logger import get_logger

logger = get_logger("routines")

def mine_basic_routines(events_df: pd.DataFrame):
    logger.info("Mining basic routines from event timestamps")

    if events_df.empty:
        logger.warning("Empty events dataframe passed to routine mining")
        return pd.DataFrame()

    df = events_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    routines = (
        df.groupby(["dataset", "sensor_id", "hour"])
          .size()
          .reset_index(name="count")
          .sort_values(["dataset", "sensor_id", "count"], ascending=[True, True, False])
    )

    top_routines = (
        routines.groupby(["dataset", "sensor_id"])
        .head(3)
        .reset_index(drop=True)
    )

    logger.info(
        f"Routine mining complete: total_patterns={len(routines)}, top_patterns={len(top_routines)}"
    )
    return top_routines

def activity_hour_profile(features_df: pd.DataFrame):
    logger.info("Building activity hour profile")

    if features_df.empty or "label" not in features_df.columns:
        logger.warning("Feature dataframe empty or missing 'label'")
        return pd.DataFrame()

    df = features_df.dropna(subset=["label"]).copy()
    profile = (
        df.groupby(["dataset", "label", "hour"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "label", "count"], ascending=[True, True, False])
    )

    logger.info(f"Activity hour profile complete: rows={len(profile)}")
    return profile
