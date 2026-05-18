import re
import pandas as pd
from pathlib import Path
from .utils import sensor_prefix, normalize_value

LABEL_PHASES = {"begin", "end"}

def _parse_line(line: str):
    line = line.strip()
    if not line:
        return None

    parts = re.split(r"\s+|\t+", line)
    parts = [p for p in parts if p != ""]

    # Case A: combined date+time in first two columns
    # 2009-06-10 00:00:00.024668 T003 19
    # 2010-11-04 00:03:50.209589 M003 ON Sleeping begin
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

    # Case B: date and time are separated by tabs but still split fine
    # 2009-09-25 12:49:51.047872 T001 22
    if len(parts) >= 4 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
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
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["sensor_type"] = df["sensor_id"].astype(str).apply(sensor_prefix)
    df["dataset"] = file_path.stem.lower()
    return df

def parse_multiple(files_dict):
    dfs = []
    for name, path in files_dict.items():
        if Path(path).exists():
            df = parse_casas_file(path)
            if not df.empty:
                df["dataset"] = name
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)