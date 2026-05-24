import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .logger import get_logger

logger = get_logger("baselines")

def majority_class_baseline(y_train, y_test):
    majority = y_train.value_counts().idxmax()
    preds = [majority] * len(y_test)

    return {
        "baseline": "majority_class",
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "report": classification_report(y_test, preds, zero_division=0)
    }

def hour_of_day_baseline(train_df, test_df, label_col="label"):
    logger.info("Running hour-of-day baseline")

    hour_to_majority = (
        train_df.groupby("hour")[label_col]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    global_majority = train_df[label_col].value_counts().idxmax()

    preds = []
    for _, row in test_df.iterrows():
        preds.append(hour_to_majority.get(row["hour"], global_majority))

    y_test = test_df[label_col]

    return {
        "baseline": "hour_of_day",
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "report": classification_report(y_test, preds, zero_division=0)
    }
