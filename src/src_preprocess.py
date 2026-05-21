import pandas as pd
from .logger import get_logger

logger = get_logger("preprocess")

def build_activity_intervals(df: pd.DataFrame):
    logger.info("Building activity intervals from begin/end annotations")

    if df.empty or "activity" not in df.columns:
        logger.warning("Input dataframe empty or missing 'activity' column")
        return pd.DataFrame(columns=["start", "end", "activity", "dataset"])

    intervals = []
    active_map = {}

    labeled = df.dropna(subset=["activity"]).copy()
    labeled["phase"] = labeled["phase"].fillna("point")

    logger.info(f"Labeled rows found: {len(labeled)}")

    begin_count = 0
    end_count = 0
    point_count = 0

    for _, row in labeled.iterrows():
        key = (row["dataset"], row["activity"])
        phase = row["phase"]

        if phase == "begin":
            active_map[key] = row["timestamp"]
            begin_count += 1
        elif phase == "end":
            end_count += 1
            if key in active_map:
                intervals.append({
                    "dataset": row["dataset"],
                    "activity": row["activity"],
                    "start": active_map[key],
                    "end": row["timestamp"]
                })
                del active_map[key]
        else:
            point_count += 1
            intervals.append({
                "dataset": row["dataset"],
                "activity": row["activity"],
                "start": row["timestamp"],
                "end": row["timestamp"]
            })

    intervals_df = pd.DataFrame(intervals)

    logger.info(
        f"Activity interval build complete: begins={begin_count}, ends={end_count}, "
        f"points={point_count}, intervals_created={len(intervals_df)}, open_intervals_left={len(active_map)}"
    )

    if not intervals_df.empty:
        logger.info(
            f"Unique activities discovered: {sorted(intervals_df['activity'].dropna().unique().tolist())[:20]}"
        )

    return intervals_df

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
    logger.info(
        f"Creating time-based split with val_ratio={val_ratio}, test_ratio={test_ratio}"
    )

    n = len(df)
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        f"Split sizes: total={n}, train={len(train)}, val={len(val)}, test={len(test)}"
    )

    return train, val, test
