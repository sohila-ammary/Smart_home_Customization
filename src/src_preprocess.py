import pandas as pd
import numpy as np

def build_activity_intervals(df: pd.DataFrame):
    """
    Build labeled intervals from rows like:
    timestamp sensor value activity phase
    ... Sleeping begin
    ... Sleeping end
    """
    if df.empty or "activity" not in df.columns:
        return pd.DataFrame(columns=["start", "end", "activity", "dataset"])

    intervals = []
    active_map = {}

    labeled = df.dropna(subset=["activity"]).copy()
    labeled["phase"] = labeled["phase"].fillna("point")

    for _, row in labeled.iterrows():
        key = (row["dataset"], row["activity"])
        phase = row["phase"]

        if phase == "begin":
            active_map[key] = row["timestamp"]
        elif phase == "end":
            if key in active_map:
                intervals.append({
                    "dataset": row["dataset"],
                    "activity": row["activity"],
                    "start": active_map[key],
                    "end": row["timestamp"]
                })
                del active_map[key]
        else:
            # point label
            intervals.append({
                "dataset": row["dataset"],
                "activity": row["activity"],
                "start": row["timestamp"],
                "end": row["timestamp"]
            })

    return pd.DataFrame(intervals)

def assign_window_label(window_start, window_end, intervals_df, dataset_name):
    if intervals_df.empty:
        return None
    sub = intervals_df[intervals_df["dataset"] == dataset_name]
    overlaps = sub[(sub["start"] <= window_end) & (sub["end"] >= window_start)]
    if overlaps.empty:
        return None

    overlaps = overlaps.copy()
    overlaps["overlap_seconds"] = overlaps.apply(
        lambda r: (min(window_end, r["end"]) - max(window_start, r["start"])).total_seconds(),
        axis=1
    )
    overlaps = overlaps.sort_values("overlap_seconds", ascending=False)
    return overlaps.iloc[0]["activity"]

def time_based_split(df, test_ratio=0.15, val_ratio=0.15):
    n = len(df)
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test