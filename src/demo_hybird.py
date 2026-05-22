import pandas as pd
from .config import OUTPUT_DIR
from .inference import run_inference
from .logger import get_logger

logger = get_logger("demo_hybrid")

def main():
    logger.info("Loading a sample row from window_features.csv for hybrid inference demo")
    df = pd.read_csv(OUTPUT_DIR / "window_features.csv")

    # choose one dataset sample
    ds = "aruba"
    sample = df[df["dataset"] == ds].dropna(subset=["label"]).head(1).copy()

    if sample.empty:
        raise ValueError(f"No sample found for dataset={ds}")

    # keep only model feature columns
    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = sample.drop(columns=drop_cols, errors="ignore")

    result = run_inference(X, dataset_name=ds, model_name="boosting")

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
