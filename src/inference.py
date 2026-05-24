import joblib
import pandas as pd
import json
from pathlib import Path
from .config import OUTPUT_DIR
from .logger import get_logger
from .recommendation_fusion import fuse_topk_predictions_with_habits
from .decision_engine import build_decision_payload

logger = get_logger("inference")

def load_habit_profile(dataset_name):
    path = OUTPUT_DIR / f"habit_profile_{dataset_name}.json"
    if not path.exists():
        logger.warning(f"Habit profile not found for dataset={dataset_name}: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_top_k(model, X_row, label_encoder=None, top_k=3):
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X_row)[0]
        return [{"activity": pred, "confidence": 1.0}]

    probs = model.predict_proba(X_row)[0]
    top_idx = probs.argsort()[::-1][:top_k]
    top_probs = probs[top_idx]

    if label_encoder is not None:
        labels = label_encoder.inverse_transform(top_idx)
    else:
        labels = model.classes_[top_idx]

    return [
        {"activity": str(a), "confidence": float(p)}
        for a, p in zip(labels, top_probs)
    ]

def _safe_load_model(dataset_name, model_name="best"):
    if model_name == "best":
        model_path = OUTPUT_DIR / f"best_model_{dataset_name}.pkl"
        encoder_path = OUTPUT_DIR / f"best_label_encoder_{dataset_name}.pkl"

        if model_path.exists():
            logger.info(f"Loading best model from {model_path}")
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None
            return model, label_encoder

        logger.warning(f"Best model not found for dataset={dataset_name}: {model_path}")

        # fallback order
        fallback_paths = [
            (OUTPUT_DIR / f"boost_{dataset_name}.pkl", OUTPUT_DIR / f"label_encoder_{dataset_name}.pkl"),
            (OUTPUT_DIR / f"rf_{dataset_name}.pkl", None),
        ]

        for mpath, epath in fallback_paths:
            if mpath.exists():
                logger.warning(f"Falling back to model: {mpath}")
                model = joblib.load(mpath)
                label_encoder = joblib.load(epath) if epath is not None and Path(epath).exists() else None
                return model, label_encoder

        raise FileNotFoundError(
            f"No usable model found for dataset={dataset_name} in {OUTPUT_DIR}"
        )

    elif model_name == "boost":
        model_path = OUTPUT_DIR / f"boost_{dataset_name}.pkl"
        encoder_path = OUTPUT_DIR / f"label_encoder_{dataset_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Boost model not found: {model_path}")
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None
        return model, label_encoder

    else:
        model_path = OUTPUT_DIR / f"rf_{dataset_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"RF model not found: {model_path}")
        model = joblib.load(model_path)
        return model, None

def _extract_temp_mean(sample_features: pd.DataFrame):
    if "temp_mean" not in sample_features.columns:
        return None

    val = pd.to_numeric(sample_features.iloc[0]["temp_mean"], errors="coerce")
    if pd.isna(val):
        return None

    val = float(val)

    # Treat invalid zero / extreme values as missing for recommendation purposes
    if val <= 0 or val < 5 or val > 45:
        return None

    return val

def _extract_hour(sample_features: pd.DataFrame):
    if "hour" not in sample_features.columns:
        return 12
    val = pd.to_numeric(sample_features.iloc[0]["hour"], errors="coerce")
    if pd.isna(val):
        return 12
    return int(val)

def run_inference(sample_features: pd.DataFrame, dataset_name: str, model_name="best"):
    logger.info(f"Running V4 inference for dataset={dataset_name} using model={model_name}")

    model, label_encoder = _safe_load_model(dataset_name, model_name=model_name)

    habit_profile = load_habit_profile(dataset_name)
    habit_profiles = {dataset_name: habit_profile}

    X = sample_features.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    top_predictions = predict_top_k(model, X, label_encoder=label_encoder, top_k=3)

    temp_mean = _extract_temp_mean(sample_features)
    hour = _extract_hour(sample_features)

    recommendations = fuse_topk_predictions_with_habits(
        dataset_name=dataset_name,
        top_predictions=top_predictions,
        hour=hour,
        temp_mean=temp_mean,
        habit_profiles=habit_profiles
    )

    decision = build_decision_payload(
        dataset_name=dataset_name,
        top_predictions=top_predictions,
        recommendations=recommendations,
        hour=hour,
        habit_profiles=habit_profiles
    )

    result = {
        "dataset": dataset_name,
        "hour": hour,
        "temp_mean": temp_mean,
        **decision
    }

    logger.info(f"V4 inference complete for dataset={dataset_name}")
    return result
