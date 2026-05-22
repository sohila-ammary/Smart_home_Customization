import joblib
import pandas as pd
import json
from .config import OUTPUT_DIR
from .logger import get_logger
from .hybrid_recommender import generate_hybrid_recommendations

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
        return [pred], [1.0]

    probs = model.predict_proba(X_row)[0]
    top_idx = probs.argsort()[::-1][:top_k]
    top_probs = probs[top_idx]

    if label_encoder is not None:
        labels = label_encoder.inverse_transform(top_idx)
    else:
        labels = model.classes_[top_idx]

    return labels.tolist(), top_probs.tolist()

def run_inference(sample_features: pd.DataFrame, dataset_name: str, model_name="boosting"):
    logger.info(f"Running inference for dataset={dataset_name} using model={model_name}")

    if model_name == "boosting":
        model = joblib.load(OUTPUT_DIR / "boosting.pkl")
        label_encoder = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    else:
        model = joblib.load(OUTPUT_DIR / "random_forest.pkl")
        label_encoder = None

    habit_profile = load_habit_profile(dataset_name)
    habit_profiles = {dataset_name: habit_profile}

    X = sample_features.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    labels, probs = predict_top_k(model, X, label_encoder=label_encoder, top_k=3)

    temp_mean = None
    hour = None
    if "temp_mean" in sample_features.columns:
        temp_mean = float(sample_features.iloc[0]["temp_mean"])
    if "hour" in sample_features.columns:
        hour = int(sample_features.iloc[0]["hour"])

    recs = generate_hybrid_recommendations(
        dataset_name=dataset_name,
        predicted_activities=labels,
        confidence_scores=probs,
        hour=hour if hour is not None else 12,
        temp_mean=temp_mean,
        habit_profiles=habit_profiles,
        top_k=5
    )

    result = {
        "dataset": dataset_name,
        "top_predictions": [
            {"activity": a, "confidence": float(p)}
            for a, p in zip(labels, probs)
        ],
        "recommendations": recs
    }

    logger.info(f"Inference complete for dataset={dataset_name}")
    return result
