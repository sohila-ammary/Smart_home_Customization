import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from .config import DATASETS, OUTPUT_DIR, RANDOM_STATE, TEST_RATIO, VAL_RATIO
from .utils import ensure_dir
from .parser import parse_multiple
from .feature_engineering import build_window_features, prepare_ml_table
from .preprocess import time_based_split

def get_boosting_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
    except Exception:
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

def train_pipeline():
    ensure_dir(OUTPUT_DIR)

    events = parse_multiple(DATASETS)
    if events.empty:
        raise ValueError("No data found. Put dataset files in data/raw/.")

    features = build_window_features(events)
    X, y = prepare_ml_table(features)

    if len(X) == 0:
        raise ValueError("No labeled windows found. Some datasets may not contain enough begin/end activity annotations.")

    full_df = X.copy()
    full_df["label"] = y.values

    train_df, val_df, test_df = time_based_split(full_df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    boost = get_boosting_model()
    try:
        boost.fit(X_train, y_train_enc)
        boost_pred = le.inverse_transform(boost.predict(X_test))
    except Exception:
        boost.fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

    rf_pred = rf.predict(X_test)

    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "macro_f1": f1_score(y_test, rf_pred, average="macro"),
        "report": classification_report(y_test, rf_pred)
    }

    boost_metrics = {
        "accuracy": accuracy_score(y_test, boost_pred),
        "macro_f1": f1_score(y_test, boost_pred, average="macro"),
        "report": classification_report(y_test, boost_pred)
    }

    joblib.dump(rf, OUTPUT_DIR / "random_forest.pkl")
    joblib.dump(boost, OUTPUT_DIR / "boosting.pkl")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
    features.to_csv(OUTPUT_DIR / "window_features.csv", index=False)

    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("=== RANDOM FOREST ===\n")
        f.write(str(rf_metrics["accuracy"]) + "\n")
        f.write(str(rf_metrics["macro_f1"]) + "\n")
        f.write(rf_metrics["report"] + "\n\n")
        f.write("=== BOOSTING ===\n")
        f.write(str(boost_metrics["accuracy"]) + "\n")
        f.write(str(boost_metrics["macro_f1"]) + "\n")
        f.write(boost_metrics["report"] + "\n")

    print("Training finished.")
    print("\n=== RANDOM FOREST ===")
    print("Accuracy:", rf_metrics["accuracy"])
    print("Macro F1:", rf_metrics["macro_f1"])
    print(rf_metrics["report"])

    print("\n=== BOOSTING ===")
    print("Accuracy:", boost_metrics["accuracy"])
    print("Macro F1:", boost_metrics["macro_f1"])
    print(boost_metrics["report"])

if __name__ == "__main__":
    train_pipeline()