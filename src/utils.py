import os
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def sensor_prefix(sensor_id: str) -> str:
    if not sensor_id:
        return "unknown"
    if sensor_id.startswith("M"):
        return "motion"
    if sensor_id.startswith("D"):
        return "door"
    if sensor_id.startswith("T"):
        return "temperature"
    return "other"

def is_on_off(value: str) -> bool:
    return str(value).upper() in {"ON", "OFF"}

def normalize_value(value):
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() in {"ON", "OFF"}:
        return s.upper()
    try:
        return float(s)
    except Exception:
        return s
