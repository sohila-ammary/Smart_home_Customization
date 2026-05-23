import joblib
import pandas as pd
import json
from .config import OUTPUT_DIR
from .logger import get_logger
from .recommendation_fusion import fuse_topk_predictions_with_habits
from .decision_engine import build_decision_payload

logger = get_logger("inference")

def load_habit_profile(dataset_name):
    path = OUTPUT_DIR / f"habit_profile_{dataset_name}.json"
    if not path.exists():
        logger.warning(f"Habit profile not found for dataset={dataset_name}")
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

def run_inference(sample_features: pd.DataFrame, dataset_name: str, model_name="best"):
    logger.info(f"Running V4 inference for dataset={dataset_name} using model={model_name}")

    if model_name == "best":
        model = joblib.load(OUTPUT_DIR / f"best_model_{dataset_name}.pkl")
        label_encoder_path = OUTPUT_DIR / f"best_label_encoder_{dataset_name}.pkl"
        label_encoder = joblib.load(label_encoder_path) if label_encoder_path.exists() else None
    elif model_name == "boost":
        model = joblib.load(OUTPUT_DIR / f"boost_{dataset_name}.pkl")
        label_encoder = joblib.load(OUTPUT_DIR / f"label_encoder_{dataset_name}.pkl")
    else:
        model = joblib.load(OUTPUT_DIR / f"rf_{dataset_name}.pkl")
        label_encoder = None

    habit_profile = load_habit_profile(dataset_name)
    habit_profiles = {dataset_name: habit_profile}

    X = sample_features.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    top_predictions = predict_top_k(model, X, label_encoder=label_encoder, top_k=3)

    temp_mean = float(sample_features.iloc[0]["temp_mean"]) if "temp_mean" in sample_features.columns else None
    hour = int(sample_features.iloc[0]["hour"]) if "hour" in sample_features.columns else 12

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
