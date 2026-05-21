import re
import pandas as pd
from pathlib import Path
from .utils import sensor_prefix, normalize_value
from .logger import get_logger

logger = get_logger("parser")

LABEL_PHASES = {"begin", "end"}

def _parse_line(line: str):
    line = line.strip()
    if not line:
        return None

    parts = re.split(r"\s+|\t+", line)
    parts = [p for p in parts if p != ""]

    if len(parts) >= 4 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
        if re.match(r"\d{2}:\d{2}:\d{2}", parts[1]):
            timestamp = f"{parts[0]} {parts[1]}"
            sensor = parts[2]
            value = parts[3]
            rest = parts[4:]
            activity = None
            phase = None

            if len(rest) >= 2 and rest[-1].lower() in LABEL_PHASES:
                phase = rest[-1].lower()
                activity = " ".join(rest[:-1]).strip() or None
            elif len(rest) >= 1:
                activity = " ".join(rest).strip() or None

            return {
                "timestamp": timestamp,
                "sensor_id": sensor,
                "value": normalize_value(value),
                "activity": activity,
                "phase": phase,
            }

    return None

def parse_casas_file(file_path):
    file_path = Path(file_path)
    logger.info(f"Parsing file: {file_path}")

    rows = []
    total_lines = 0
    parsed_lines = 0
    bad_lines = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)
                parsed_lines += 1
            else:
                bad_lines += 1

            if total_lines % 50000 == 0:
                logger.info(
                    f"[{file_path.name}] progress: total_lines={total_lines}, parsed={parsed_lines}, skipped={bad_lines}"
                )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning(f"No valid rows parsed from file: {file_path}")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    after_drop = len(df)

    df["sensor_type"] = df["sensor_id"].astype(str).apply(sensor_prefix)
    df["dataset"] = file_path.stem.lower()

    logger.info(
        f"Finished parsing {file_path.name}: total_lines={total_lines}, parsed={parsed_lines}, "
        f"invalid_timestamp_dropped={before_drop - after_drop}, final_rows={after_drop}"
    )

    logger.info(
        f"{file_path.name} summary: sensors={df['sensor_id'].nunique()}, "
        f"activities={df['activity'].dropna().nunique()}, "
        f"time_range=({df['timestamp'].min()} -> {df['timestamp'].max()})"
    )

    return df

def parse_multiple(files_dict):
    logger.info("Starting multi-file parse")
    dfs = []

    for name, path in files_dict.items():
        path = Path(path)
        if path.exists():
            logger.info(f"Found dataset '{name}' at {path}")
            df = parse_casas_file(path)
            if not df.empty:
                df["dataset"] = name
                dfs.append(df)
                logger.info(f"Dataset '{name}' added with {len(df)} rows")
            else:
                logger.warning(f"Dataset '{name}' produced empty dataframe")
        else:
            logger.warning(f"Dataset '{name}' not found at {path}")

    if not dfs:
        logger.error("No datasets could be loaded")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Finished multi-file parse: datasets_loaded={len(dfs)}, total_rows={len(merged)}, "
        f"unique_sensors={merged['sensor_id'].nunique()}"
    )
    return merged
