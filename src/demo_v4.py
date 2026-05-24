import pandas as pd
from .config import OUTPUT_DIR
from .inference import run_inference
from .logger import get_logger

logger = get_logger("demo_v4")

def main():
    logger.info("Loading V4 feature file")
    df = pd.read_csv(OUTPUT_DIR / "window_features_v31_coarse.csv")

    for dataset_name in ["aruba", "cairo", "milan", "tulum1"]:
        sample = df[(df["dataset"] == dataset_name) & (df["label"].notna())].head(1).copy()
        if sample.empty:
            continue

        X = sample.drop(columns=["dataset", "window_start", "window_end", "label"], errors="ignore")
        result = run_inference(X, dataset_name=dataset_name, model_name="best")

        print(f"\n===== DATASET: {dataset_name.upper()} =====")
        print("Confidence level:", result["confidence_level"])
        print("Trigger mode:", result["trigger_mode"])
        print("Top predictions:")
        for p in result["top_predictions"]:
            print(f"  - {p['activity']}: {p['confidence']:.4f}")

        print("Recommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec['scenario']} | score={rec['score']:.3f}")
            print(f"    reason: {rec['reason']}")
            for action in rec["actions"]:
                print(f"      * {action}")

if __name__ == "__main__":
    main()

