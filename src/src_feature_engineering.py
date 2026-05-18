import pandas as pd
import numpy as np
from .preprocess import build_activity_intervals, assign_window_label
from .utils import is_on_off

def build_window_features(df: pd.DataFrame, window_size="5min", step_size="1min"):
    all_windows = []
    if df.empty:
        return pd.DataFrame()

    intervals_df = build_activity_intervals(df)

    for dataset_name, ds in df.groupby("dataset"):
        ds = ds.sort_values("timestamp").reset_index(drop=True)
        start_time = ds["timestamp"].min().floor("min")
        end_time = ds["timestamp"].max().ceil("min")

        window_starts = pd.date_range(start=start_time, end=end_time, freq=step_size)

        for ws in window_starts:
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
            all_windows.append(feat)

    feat_df = pd.DataFrame(all_windows)
    if feat_df.empty:
        return feat_df

    feat_df = feat_df.sort_values(["dataset", "window_start"]).reset_index(drop=True)
    return feat_df

def prepare_ml_table(features_df: pd.DataFrame):
    df = features_df.copy()
    df = df.dropna(subset=["label"]).copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=object)

    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["label"].copy()

    X = X.fillna(0)

    # ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y