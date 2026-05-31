import joblib
import pandas as pd
import json
from .config import OUTPUT_DIR
from .logger import get_logger
from .recommendation_fusion import fuse_topk_predictions_with_habits
from .decision_engine import build_decision_payload

logger = get_logger("inference_v51")

def load_habit_profile(dataset_name):
    path = OUTPUT_DIR / f"habit_profile_{dataset_name}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_inference_v51(sample_features: pd.DataFrame, dataset_name: str):
    logger.info(f"Running V5.1 inference for dataset={dataset_name}")

    model = joblib.load(OUTPUT_DIR / f"v51_best_model_{dataset_name}.pkl")
    encoder = joblib.load(OUTPUT_DIR / f"v51_label_encoder_{dataset_name}.pkl")
    kept_features = joblib.load(OUTPUT_DIR / f"v51_kept_features_{dataset_name}.pkl")

    habit_profiles = {dataset_name: load_habit_profile(dataset_name)}

    X = sample_features.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    X = X[[c for c in kept_features if c in X.columns]].copy()

    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[::-1][:3]
    top_predictions = [
        {"activity": str(encoder.inverse_transform([i])[0]), "confidence": float(probs[i])}
        for i in top_idx
    ]

    hour = int(sample_features.iloc[0]["hour"]) if "hour" in sample_features.columns else 12
    temp_mean = None
    if "temp_mean" in sample_features.columns:
        v = pd.to_numeric(sample_features.iloc[0]["temp_mean"], errors="coerce")
        if pd.notna(v):
            v = float(v)
            if 5 <= v <= 45:
                temp_mean = v

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

    logger.info(f"V5.1 inference complete for dataset={dataset_name}")
    return result
