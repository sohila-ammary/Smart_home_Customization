import pandas as pd

def mine_basic_routines(events_df: pd.DataFrame):
    if events_df.empty:
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
    return top_routines

def activity_hour_profile(features_df: pd.DataFrame):
    if features_df.empty or "label" not in features_df.columns:
        return pd.DataFrame()

    df = features_df.dropna(subset=["label"]).copy()
    profile = (
        df.groupby(["dataset", "label", "hour"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "label", "count"], ascending=[True, True, False])
    )
    return profile