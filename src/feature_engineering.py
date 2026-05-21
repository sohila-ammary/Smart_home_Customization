import pandas as pd
import numpy as np
from .preprocess import build_activity_intervals, assign_window_label
from .logger import get_logger

logger = get_logger("features")

def build_window_features(df: pd.DataFrame, window_size="5min", step_size="1min"):
    logger.info(
        f"Starting window feature generation with window_size={window_size}, step_size={step_size}"
    )

    all_windows = []
    if df.empty:
        logger.warning("Empty dataframe received for feature generation")
        return pd.DataFrame()

    intervals_df = build_activity_intervals(df)

    for dataset_name, ds in df.groupby("dataset"):
        ds = ds.sort_values("timestamp").reset_index(drop=True)

        start_time = ds["timestamp"].min().floor("min")
        end_time = ds["timestamp"].max().ceil("min")
        window_starts = pd.date_range(start=start_time, end=end_time, freq=step_size)

        logger.info(
            f"[{dataset_name}] generating windows: events={len(ds)}, sensors={ds['sensor_id'].nunique()}, "
            f"time_range=({start_time} -> {end_time}), candidate_windows={len(window_starts)}"
        )

        generated_count = 0
        labeled_count = 0

        for i, ws in enumerate(window_starts, start=1):
            we = ws + pd.Timedelta(window_size)
            chunk = ds[(ds["timestamp"] >= ws) & (ds["timestamp"] < we)]

            if chunk.empty:
                continue

            feat = {
                "dataset": dataset_name,
                "window_start": ws,
                "window_end": we,
                "hour": ws.hour,
                "minute": ws.minute,
                "dayofweek": ws.dayofweek,
                "is_weekend": int(ws.dayofweek >= 5),
                "num_events": len(chunk),
                "unique_sensors": chunk["sensor_id"].nunique(),
                "motion_events": int((chunk["sensor_type"] == "motion").sum()),
                "temp_events": int((chunk["sensor_type"] == "temperature").sum()),
            }

            on_count = 0
            off_count = 0
            temp_vals = []

            for _, row in chunk.iterrows():
                sid = str(row["sensor_id"])
                val = row["value"]

                feat[f"count_{sid}"] = feat.get(f"count_{sid}", 0) + 1

                if isinstance(val, str) and val.upper() == "ON":
                    on_count += 1
                    feat[f"{sid}_on"] = feat.get(f"{sid}_on", 0) + 1
                elif isinstance(val, str) and val.upper() == "OFF":
                    off_count += 1
                    feat[f"{sid}_off"] = feat.get(f"{sid}_off", 0) + 1
                elif isinstance(val, (int, float)):
                    temp_vals.append(float(val))
                    feat[f"{sid}_last_numeric"] = float(val)

            feat["on_count"] = on_count
            feat["off_count"] = off_count

            if temp_vals:
                feat["temp_mean"] = float(np.mean(temp_vals))
                feat["temp_std"] = float(np.std(temp_vals))
                feat["temp_min"] = float(np.min(temp_vals))
                feat["temp_max"] = float(np.max(temp_vals))
            else:
                feat["temp_mean"] = np.nan
                feat["temp_std"] = np.nan
                feat["temp_min"] = np.nan
                feat["temp_max"] = np.nan

            feat["label"] = assign_window_label(ws, we, intervals_df, dataset_name)

            if feat["label"] is not None:
                labeled_count += 1

            all_windows.append(feat)
            generated_count += 1

            if i % 10000 == 0:
                logger.info(
                    f"[{dataset_name}] progress: scanned_windows={i}, generated_nonempty={generated_count}, labeled={labeled_count}"
                )

        logger.info(
            f"[{dataset_name}] feature generation complete: nonempty_windows={generated_count}, labeled_windows={labeled_count}"
        )

    feat_df = pd.DataFrame(all_windows)
    if feat_df.empty:
        logger.warning("No features generated")
        return feat_df

    feat_df = feat_df.sort_values(["dataset", "window_start"]).reset_index(drop=True)

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
