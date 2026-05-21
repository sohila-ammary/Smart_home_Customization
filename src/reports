import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .logger import get_logger

logger = get_logger("reports")

def build_per_dataset_report(test_df, y_true, y_pred, out_path):
    logger.info("Building per-dataset report")

    report_rows = []
    tmp = test_df.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred

    for dataset_name, ds in tmp.groupby("dataset"):
        row = {
            "dataset": dataset_name,
            "samples": len(ds),
            "accuracy": accuracy_score(ds["y_true"], ds["y_pred"]),
            "macro_f1": f1_score(ds["y_true"], ds["y_pred"], average="macro"),
            "num_classes_true": ds["y_true"].nunique(),
            "num_classes_pred": ds["y_pred"].nunique(),
        }
        report_rows.append(row)

    out = pd.DataFrame(report_rows).sort_values("dataset").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Per-dataset report saved to {out_path}")
    return out
