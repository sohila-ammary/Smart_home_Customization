from .logger import get_logger

logger = get_logger("ablation")

TIME_FEATURES = {
    "hour", "minute", "dayofweek", "is_weekend", "hour_sin", "hour_cos"
}

META_EXCLUDE = {"dataset", "window_start", "window_end", "label"}

def get_feature_subsets(columns):
    cols = [c for c in columns if c not in META_EXCLUDE]

    time_only = [c for c in cols if c in TIME_FEATURES]
    sensor_only = [c for c in cols if c not in TIME_FEATURES and c not in {"label_fraction"}]
    full = cols

    subsets = {
        "time_only": time_only,
        "sensor_only": sensor_only,
        "full": full,
    }

    logger.info(
        f"Ablation subsets built: time_only={len(time_only)}, sensor_only={len(sensor_only)}, full={len(full)}"
    )
    return subsets
