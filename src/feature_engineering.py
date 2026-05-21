import pandas as pd
import numpy as np
from tqdm import tqdm
from .preprocess import build_activity_intervals, assign_window_label
from .logger import get_logger

logger = get_logger("features")

def _build_dataset_windows_fast(ds: pd.DataFrame, dataset_name: str, intervals_df: pd.DataFrame,
                                window_size="5min", step_size="1min"):
    ds = ds.sort_values("timestamp").reset_index(drop=True).copy()
    ds["minute_bin"] = ds["timestamp"].dt.floor("min")

    start_time = ds["minute_bin"].min()
    end_time = ds["minute_bin"].max().ceil("min")
    window_starts = pd.date_range(start=start_time, end=end_time, freq=step_size)

    logger.info(
        f"[{dataset_name}] fast window generation: events={len(ds)}, sensors={ds['sensor_id'].nunique()}, "
        f"candidate_windows={len(window_starts)}"
    )

    # Aggregate per minute per sensor
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

    numeric_df = ds[pd.to_numeric(ds["value"], errors="coerce").notna()].copy()
    numeric_df["value_num"] = pd.to_numeric(numeric_df["value"], errors="coerce")
    temp_stats = numeric_df.groupby("minute_bin")["value_num"].agg(["mean", "std", "min", "max"])
    temp_stats.columns = ["temp_mean", "temp_std", "temp_min", "temp_max"]

    # Reindex all per-minute tables
    minute_index = pd.date_range(start=start_time, end=end_time, freq="1min")
    minute_sensor_counts = minute_sensor_counts.reindex(minute_index, fill_value=0)
    minute_on = minute_on.reindex(minute_index, fill_value=0)
    minute_off = minute_off.reindex(minute_index, fill_value=0)
    minute_motion = minute_motion.reindex(minute_index, fill_value=0)
    minute_temp_events = minute_temp_events.reindex(minute_index, fill_value=0)
    temp_stats = temp_stats.reindex(minute_index)

    # Rolling over minute-level tables
    window_minutes = int(pd.Timedelta(window_size).total_seconds() // 60)

    sensor_roll = minute_sensor_counts.rolling(window=window_minutes, min_periods=1).sum()
    on_roll = minute_on.rolling(window=window_minutes, min_periods=1).sum()
    off_roll = minute_off.rolling(window=window_minutes, min_periods=1).sum()
    motion_roll = minute_motion.rolling(window=window_minutes, min_periods=1).sum()
    temp_event_roll = minute_temp_events.rolling(window=window_minutes, min_periods=1).sum()

    temp_mean_roll = temp_stats["temp_mean"].rolling(window=window_minutes, min_periods=1).mean()
    temp_std_roll = temp_stats["temp_mean"].rolling(window=window_minutes, min_periods=1).std()
    temp_min_roll = temp_stats["temp_min"].rolling(window=window_minutes, min_periods=1).min()
    temp_max_roll = temp_stats["temp_max"].rolling(window=window_minutes, min_periods=1).max()

    rows = []
    step_minutes = int(pd.Timedelta(step_size).total_seconds() // 60)

    for ws in tqdm(window_starts, desc=f"windows-{dataset_name}"):
        idx = ws.floor("min")
        if idx not in sensor_roll.index:
            continue

        sensor_counts = sensor_roll.loc[idx]
        total_events = float(sensor_counts.sum())

        if total_events <= 0:
            continue

        feat = {
            "dataset": dataset_name,
            "window_start": ws,
            "window_end": ws + pd.Timedelta(window_size),
            "hour": ws.hour,
            "minute": ws.minute,
            "dayofweek": ws.dayofweek,
            "is_weekend": int(ws.dayofweek >= 5),
            "num_events": total_events,
            "unique_sensors": int((sensor_counts > 0).sum()),
            "motion_events": float(motion_roll.loc[idx]),
            "temp_events": float(temp_event_roll.loc[idx]),
            "on_count": float(on_roll.loc[idx]),
            "off_count": float(off_roll.loc[idx]),
            "temp_mean": float(temp_mean_roll.loc[idx]) if pd.notna(temp_mean_roll.loc[idx]) else 0.0,
            "temp_std": float(temp_std_roll.loc[idx]) if pd.notna(temp_std_roll.loc[idx]) else 0.0,
            "temp_min": float(temp_min_roll.loc[idx]) if pd.notna(temp_min_roll.loc[idx]) else 0.0,
            "temp_max": float(temp_max_roll.loc[idx]) if pd.notna(temp_max_roll.loc[idx]) else 0.0,
        }

        for sid, val in sensor_counts.items():
            if val > 0:
                feat[f"count_{sid}"] = float(val)

        feat["label"] = assign_window_label(
            feat["window_start"], feat["window_end"], intervals_df, dataset_name
        )
        rows.append(feat)

    out = pd.DataFrame(rows)
    logger.info(
        f"[{dataset_name}] fast feature generation complete: windows={len(out)}, labeled={out['label'].notna().sum() if not out.empty else 0}"
    )
    return out

def build_window_features(df: pd.DataFrame, window_size="5min", step_size="1min"):
    logger.info(
        f"Starting FAST window feature generation with window_size={window_size}, step_size={step_size}"
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
            step_size=step_size
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

def prepare_ml_table(features_df: pd.DataFrame):
    logger.info("Preparing ML table from feature dataframe")

    df = features_df.copy()
    total_windows = len(df)
    df = df.dropna(subset=["label"]).copy()
    labeled_windows = len(df)

    logger.info(
        f"Label filtering complete: total_windows={total_windows}, labeled_windows={labeled_windows}, "
        f"dropped_unlabeled={total_windows - labeled_windows}"
    )

    if df.empty:
        logger.warning("No labeled windows available after filtering")
        return pd.DataFrame(), pd.Series(dtype=object)

    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["label"].copy()

    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    logger.info(
        f"ML table ready: X_shape={X.shape}, y_shape={y.shape}, num_classes={y.nunique()}"
    )
    logger.info(f"Class distribution:\n{y.value_counts().to_string()}")

    return X, y
