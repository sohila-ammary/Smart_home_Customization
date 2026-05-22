import pandas as pd
import numpy as np
from tqdm import tqdm
from .preprocess import build_activity_intervals, get_window_label_info
from .logger import get_logger

logger = get_logger("features")

def _encode_last_sensor(ds):
    sensors = sorted(ds["sensor_id"].astype(str).unique().tolist())
    sensor_to_idx = {s: i + 1 for i, s in enumerate(sensors)}
    return sensor_to_idx

def _build_dataset_windows_fast(
    ds: pd.DataFrame,
    dataset_name: str,
    intervals_df: pd.DataFrame,
    window_size="5min",
    step_size="1min",
    min_label_fraction=0.6
):
    ds = ds.sort_values("timestamp").reset_index(drop=True).copy()
    ds["minute_bin"] = ds["timestamp"].dt.floor("min")

    start_time = ds["minute_bin"].min()
    end_time = ds["minute_bin"].max().ceil("min")
    window_starts = pd.date_range(start=start_time, end=end_time, freq=step_size)

    logger.info(
        f"[{dataset_name}] V3 window generation: events={len(ds)}, sensors={ds['sensor_id'].nunique()}, "
        f"candidate_windows={len(window_starts)}, min_label_fraction={min_label_fraction}"
    )

    sensor_to_idx = _encode_last_sensor(ds)

    minute_sensor_counts = (
        ds.groupby(["minute_bin", "sensor_id"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    minute_on = (
        ds.assign(is_on=(ds["value"].astype(str).str.upper() == "ON").astype(int))
        .groupby("minute_bin")["is_on"]
        .sum()
    )

    minute_off = (
        ds.assign(is_off=(ds["value"].astype(str).str.upper() == "OFF").astype(int))
        .groupby("minute_bin")["is_off"]
        .sum()
    )

    minute_motion = (
        ds.assign(is_motion=(ds["sensor_type"] == "motion").astype(int))
        .groupby("minute_bin")["is_motion"]
        .sum()
    )

    minute_temp_events = (
        ds.assign(is_temp=(ds["sensor_type"] == "temperature").astype(int))
        .groupby("minute_bin")["is_temp"]
        .sum()
    )

    minute_door = (
        ds.assign(is_door=(ds["sensor_type"] == "door").astype(int))
        .groupby("minute_bin")["is_door"]
        .sum()
    )

    numeric_df = ds[pd.to_numeric(ds["value"], errors="coerce").notna()].copy()
    numeric_df["value_num"] = pd.to_numeric(numeric_df["value"], errors="coerce")
    temp_stats = numeric_df.groupby("minute_bin")["value_num"].agg(["mean", "std", "min", "max"])
    temp_stats.columns = ["temp_mean", "temp_std", "temp_min", "temp_max"]

    minute_index = pd.date_range(start=start_time, end=end_time, freq="1min")
    minute_sensor_counts = minute_sensor_counts.reindex(minute_index, fill_value=0)
    minute_on = minute_on.reindex(minute_index, fill_value=0)
    minute_off = minute_off.reindex(minute_index, fill_value=0)
    minute_motion = minute_motion.reindex(minute_index, fill_value=0)
    minute_temp_events = minute_temp_events.reindex(minute_index, fill_value=0)
    minute_door = minute_door.reindex(minute_index, fill_value=0)
    temp_stats = temp_stats.reindex(minute_index)

    window_minutes = int(pd.Timedelta(window_size).total_seconds() // 60)

    sensor_roll = minute_sensor_counts.rolling(window=window_minutes, min_periods=1).sum()
    on_roll = minute_on.rolling(window=window_minutes, min_periods=1).sum()
    off_roll = minute_off.rolling(window=window_minutes, min_periods=1).sum()
    motion_roll = minute_motion.rolling(window=window_minutes, min_periods=1).sum()
    temp_event_roll = minute_temp_events.rolling(window=window_minutes, min_periods=1).sum()
    door_roll = minute_door.rolling(window=window_minutes, min_periods=1).sum()

    temp_mean_roll = temp_stats["temp_mean"].rolling(window=window_minutes, min_periods=1).mean()
    temp_std_roll = temp_stats["temp_mean"].rolling(window=window_minutes, min_periods=1).std()
    temp_min_roll = temp_stats["temp_min"].rolling(window=window_minutes, min_periods=1).min()
    temp_max_roll = temp_stats["temp_max"].rolling(window=window_minutes, min_periods=1).max()

    rows = []

    for ws in tqdm(window_starts, desc=f"windows-{dataset_name}"):
        we = ws + pd.Timedelta(window_size)
        idx = ws.floor("min")

        if idx not in sensor_roll.index:
            continue

        sensor_counts = sensor_roll.loc[idx]
        total_events = float(sensor_counts.sum())
        if total_events <= 0:
            continue

        chunk = ds[(ds["timestamp"] >= ws) & (ds["timestamp"] < we)]
        if chunk.empty:
            continue

        label, label_fraction, overlap_map = get_window_label_info(ws, we, intervals_df, dataset_name)

        # Only keep clean enough labels
        if label is not None and label_fraction < min_label_fraction:
            label = None

        active_sensor_count = int((sensor_counts > 0).sum())
        last_sensor = str(chunk.iloc[-1]["sensor_id"])
        prev_sensor = str(chunk.iloc[-2]["sensor_id"]) if len(chunk) > 1 else last_sensor

        last_event_ts = chunk.iloc[-1]["timestamp"]
        first_event_ts = chunk.iloc[0]["timestamp"]
        window_span = max((last_event_ts - first_event_ts).total_seconds(), 1.0)

        event_rate = total_events / max((we - ws).total_seconds(), 1.0)
        burst_rate = total_events / window_span

        hour_float = ws.hour + ws.minute / 60.0
        hour_angle = 2 * np.pi * hour_float / 24.0

        feat = {
            "dataset": dataset_name,
            "window_start": ws,
            "window_end": we,
            "hour": ws.hour,
            "minute": ws.minute,
            "dayofweek": ws.dayofweek,
            "is_weekend": int(ws.dayofweek >= 5),
            "hour_sin": float(np.sin(hour_angle)),
            "hour_cos": float(np.cos(hour_angle)),
            "num_events": total_events,
            "unique_sensors": active_sensor_count,
            "motion_events": float(motion_roll.loc[idx]),
            "temp_events": float(temp_event_roll.loc[idx]),
            "door_events": float(door_roll.loc[idx]),
            "on_count": float(on_roll.loc[idx]),
            "off_count": float(off_roll.loc[idx]),
            "motion_ratio": float(motion_roll.loc[idx] / total_events),
            "door_ratio": float(door_roll.loc[idx] / total_events),
            "on_off_ratio": float(on_roll.loc[idx] / max(off_roll.loc[idx], 1.0)),
            "temp_mean": float(temp_mean_roll.loc[idx]) if pd.notna(temp_mean_roll.loc[idx]) else 0.0,
            "temp_std": float(temp_std_roll.loc[idx]) if pd.notna(temp_std_roll.loc[idx]) else 0.0,
            "temp_min": float(temp_min_roll.loc[idx]) if pd.notna(temp_min_roll.loc[idx]) else 0.0,
            "temp_max": float(temp_max_roll.loc[idx]) if pd.notna(temp_max_roll.loc[idx]) else 0.0,
            "event_rate_per_sec": float(event_rate),
            "burst_rate": float(burst_rate),
            "label_fraction": float(label_fraction),
            "last_sensor_idx": float(sensor_to_idx.get(last_sensor, 0)),
            "prev_sensor_idx": float(sensor_to_idx.get(prev_sensor, 0)),
            "last_equals_prev": float(last_sensor == prev_sensor),
        }

        # sensor count features
        for sid, val in sensor_counts.items():
            if val > 0:
                feat[f"count_{sid}"] = float(val)

        # top sensor share
        top_sensor_count = float(sensor_counts.max()) if len(sensor_counts) else 0.0
        feat["top_sensor_share"] = top_sensor_count / max(total_events, 1.0)

        feat["label"] = label
        rows.append(feat)

    out = pd.DataFrame(rows)
    logger.info(
        f"[{dataset_name}] V3 feature generation complete: windows={len(out)}, labeled={out['label'].notna().sum() if not out.empty else 0}"
    )
    return out

def build_window_features(df: pd.DataFrame, window_size="5min", step_size="1min", min_label_fraction=0.6):
    logger.info(
        f"Starting V3 window feature generation with window_size={window_size}, "
        f"step_size={step_size}, min_label_fraction={min_label_fraction}"
    )

    if df.empty:
        logger.warning("Empty dataframe received for feature generation")
        return pd.DataFrame()

    intervals_df = build_activity_intervals(df)

    outputs = []
    for dataset_name, ds in df.groupby("dataset"):
        out = _build_dataset_windows_fast(
            ds=ds,
            dataset_name=dataset_name,
            intervals_df=intervals_df,
            window_size=window_size,
            step_size=step_size,
            min_label_fraction=min_label_fraction
        )
        if not out.empty:
            outputs.append(out)

    if not outputs:
        logger.warning("No features generated")
        return pd.DataFrame()

    feat_df = pd.concat(outputs, ignore_index=True).sort_values(
        ["dataset", "window_start"]
    ).reset_index(drop=True)

    logger.info(
        f"All feature generation complete: total_windows={len(feat_df)}, "
        f"labeled_windows={feat_df['label'].notna().sum()}, columns={len(feat_df.columns)}"
    )
    return feat_df
