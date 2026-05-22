import pandas as pd
from .config import OUTPUT_DIR
from .inference import run_inference
from .logger import get_logger

logger = get_logger("demo_v3")

def main():
    logger.info("Loading V3 coarse feature file")
    df = pd.read_csv(OUTPUT_DIR / "window_features_v3_coarse.csv")

    dataset_name = "aruba"
    sample = df[(df["dataset"] == dataset_name) & (df["label"].notna())].head(1).copy()
    if sample.empty:
        raise ValueError(f"No labeled sample available for dataset={dataset_name}")

    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = sample.drop(columns=drop_cols, errors="ignore")

    result = run_inference(X, dataset_name=dataset_name, model_name="boost")

    print("\n=== TOP PREDICTIONS ===")
    for item in result["top_predictions"]:
        print(f"- {item['activity']}: {item['confidence']:.4f}")

    print("\n=== RECOMMENDATIONS ===")
    for rec in result["recommendations"]:
        print(f"- Scenario: {rec['scenario']} | score={rec['score']:.3f}")
        print(f"  Reason: {rec['reason']}")
        for action in rec["actions"]:
            print(f"    * {action}")

if __name__ == "__main__":
    main()
