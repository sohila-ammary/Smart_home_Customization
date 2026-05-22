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
                start_ts = active_map[key]
                end_ts = row["timestamp"]
                if end_ts >= start_ts:
                    intervals.append({
                        "dataset": row["dataset"],
                        "activity": row["activity"],
                        "start": start_ts,
                        "end": end_ts
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

def get_window_label_info(window_start, window_end, intervals_df, dataset_name):
    if intervals_df.empty:
        return None, 0.0, {}

    sub = intervals_df[intervals_df["dataset"] == dataset_name]
    overlaps = sub[(sub["start"] <= window_end) & (sub["end"] >= window_start)]

    if overlaps.empty:
        return None, 0.0, {}

    overlap_map = {}
    for _, r in overlaps.iterrows():
        overlap_seconds = (
            min(window_end, r["end"]) - max(window_start, r["start"])
        ).total_seconds()
        if overlap_seconds > 0:
            overlap_map[r["activity"]] = overlap_map.get(r["activity"], 0.0) + overlap_seconds

    if not overlap_map:
        return None, 0.0, {}

    best_label = max(overlap_map, key=overlap_map.get)
    best_overlap = overlap_map[best_label]
    window_seconds = max((window_end - window_start).total_seconds(), 1.0)
    best_fraction = best_overlap / window_seconds

    return best_label, best_fraction, overlap_map

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
