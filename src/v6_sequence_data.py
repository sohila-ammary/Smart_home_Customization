import pandas as pd
import numpy as np
from .logger import get_logger
from .label_mapping import apply_coarse_mapping

logger = get_logger("v6_sequence_data")

def prepare_event_level_labels(events_df, features_df, dataset_name):
    logger.info(f"[{dataset_name}] preparing event-level labels from window labels")

    feat = features_df[
        (features_df["dataset"] == dataset_name) & (features_df["label"].notna())
    ].copy()

    if feat.empty:
        return pd.DataFrame()

    feat["window_start"] = pd.to_datetime(feat["window_start"])
    feat["window_end"] = pd.to_datetime(feat["window_end"])

    events = events_df[events_df["dataset"] == dataset_name].copy()
    events = events.sort_values("timestamp").reset_index(drop=True)

    labels = []
    feat_idx = 0
    feat = feat.sort_values("window_start").reset_index(drop=True)

    for _, row in events.iterrows():
        ts = row["timestamp"]

        assigned = None
        while feat_idx < len(feat):
            ws = feat.iloc[feat_idx]["window_start"]
            we = feat.iloc[feat_idx]["window_end"]
            if ws <= ts < we:
                assigned = feat.iloc[feat_idx]["label"]
                break
            elif ts >= we:
                feat_idx += 1
            else:
                break
        labels.append(assigned)

    events["label"] = labels
    events = events.dropna(subset=["label"]).copy()
    logger.info(f"[{dataset_name}] event-level labeled rows: {len(events)}")
    return events

def build_vocab(events_df):
    sensors = sorted(events_df["sensor_id"].astype(str).unique().tolist())
    sensor_types = sorted(events_df["sensor_type"].astype(str).unique().tolist())

    sensor_vocab = {s: i + 1 for i, s in enumerate(sensors)}
    type_vocab = {t: i + 1 for i, t in enumerate(sensor_types)}

    return sensor_vocab, type_vocab

def encode_value(val):
    s = str(val).upper()
    if s == "ON":
        return 1
    if s == "OFF":
        return 2
    try:
        return 3  # numeric bucket placeholder
    except Exception:
        return 0

def build_sequences(events_df, seq_len=50):
    logger.info(f"Building event sequences with seq_len={seq_len}")

    events_df = events_df.sort_values("timestamp").reset_index(drop=True).copy()
    sensor_vocab, type_vocab = build_vocab(events_df)

    X_sensor = []
    X_type = []
    X_value = []
    X_delta = []
    X_hour = []
    y = []

    timestamps = pd.to_datetime(events_df["timestamp"]).reset_index(drop=True)

    for i in range(seq_len, len(events_df)):
        chunk = events_df.iloc[i-seq_len:i]
        target = events_df.iloc[i]["label"]

        sensor_seq = [sensor_vocab.get(str(x), 0) for x in chunk["sensor_id"]]
        type_seq = [type_vocab.get(str(x), 0) for x in chunk["sensor_type"]]
        value_seq = [encode_value(x) for x in chunk["value"]]

        delta_seq = []
        chunk_ts = pd.to_datetime(chunk["timestamp"]).tolist()
        for j in range(len(chunk_ts)):
            if j == 0:
                delta_seq.append(0.0)
            else:
                delta = (chunk_ts[j] - chunk_ts[j-1]).total_seconds()
                delta_seq.append(min(delta, 3600.0))

        hour_seq = [pd.to_datetime(t).hour for t in chunk["timestamp"]]

        X_sensor.append(sensor_seq)
        X_type.append(type_seq)
        X_value.append(value_seq)
        X_delta.append(delta_seq)
        X_hour.append(hour_seq)
        y.append(target)

    return {
        "X_sensor": np.array(X_sensor, dtype=np.int64),
        "X_type": np.array(X_type, dtype=np.int64),
        "X_value": np.array(X_value, dtype=np.int64),
        "X_delta": np.array(X_delta, dtype=np.float32),
        "X_hour": np.array(X_hour, dtype=np.int64),
        "y": np.array(y, dtype=object),
        "sensor_vocab": sensor_vocab,
        "type_vocab": type_vocab,
    }
