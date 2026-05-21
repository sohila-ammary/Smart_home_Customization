import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import DATASETS, OUTPUT_DIR, RANDOM_STATE, TEST_RATIO, VAL_RATIO, WINDOW_SIZE, STEP_SIZE
from .utils import ensure_dir
from .parser import parse_multiple
from .feature_engineering import build_window_features, prepare_ml_table
from .preprocess import time_based_split
from .routine_mining import mine_basic_routines, activity_hour_profile
from .logger import get_logger

logger = get_logger("train")

def get_boosting_model():
    logger.info("Trying to initialize XGBoost model")
    try:
        from xgboost import XGBClassifier
        logger.info("XGBoost is available, using XGBClassifier")
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
    except Exception as e:
        logger.warning(f"XGBoost unavailable, falling back to GradientBoostingClassifier. Reason: {e}")
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

def train_pipeline():
    logger.info("========== CASAS SMART HOME MVP TRAINING STARTED ==========")
    ensure_dir(OUTPUT_DIR)
    logger.info(f"Output directory ensured at: {OUTPUT_DIR}")

    logger.info("Step 1/7: Parsing datasets")
    events = parse_multiple(DATASETS)
    if events.empty:
        logger.error("No data found. Put dataset files in data/raw/")
        raise ValueError("No data found. Put dataset files in data/raw/.")

    logger.info(
        f"Parsed events summary: rows={len(events)}, datasets={events['dataset'].nunique()}, "
        f"sensors={events['sensor_id'].nunique()}"
    )

    logger.info("Step 2/7: Mining routines from raw events")
    routines = mine_basic_routines(events)
    if not routines.empty:
        routines.to_csv(OUTPUT_DIR / "top_routines.csv", index=False)
        logger.info(f"Saved top routines to: {OUTPUT_DIR / 'top_routines.csv'}")
    else:
        logger.warning("No routines were mined")

    logger.info("Step 3/7: Building sliding-window features")
    features = build_window_features(events, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    if features.empty:
        logger.error("Feature generation returned an empty dataframe")
        raise ValueError("Feature generation returned an empty dataframe")

    logger.info(f"Feature dataframe shape: {features.shape}")
    features.to_csv(OUTPUT_DIR / "window_features.csv", index=False)
    logger.info(f"Saved window features to: {OUTPUT_DIR / 'window_features.csv'}")

    logger.info("Step 4/7: Building activity hour profiles")
    profile = activity_hour_profile(features)
    if not profile.empty:
        profile.to_csv(OUTPUT_DIR / "activity_hour_profile.csv", index=False)
        logger.info(f"Saved activity hour profile to: {OUTPUT_DIR / 'activity_hour_profile.csv'}")
    else:
        logger.warning("No activity hour profile generated")

    logger.info("Step 5/7: Preparing ML table")
    X, y = prepare_ml_table(features)
    if len(X) == 0:
        logger.error("No labeled windows found. Not enough begin/end activity annotations.")
        raise ValueError("No labeled windows found. Some datasets may not contain enough begin/end activity annotations.")

    full_df = X.copy()
    full_df["label"] = y.values

    logger.info("Step 6/7: Time-based split")
    train_df, val_df, test_df = time_based_split(full_df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    logger.info(
        f"Train/Val/Test shapes => X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    logger.info("Step 7/7: Training models")
    logger.info("Training RandomForestClassifier")
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
    logger.info("Random forest training completed")

    logger.info("Generating Random Forest predictions on test set")
    rf_pred = rf.predict(X_test)
    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "macro_f1": f1_score(y_test, rf_pred, average="macro"),
        "report": classification_report(y_test, rf_pred)
    }

    logger.info("Training boosting model")
    boost = get_boosting_model()
    boosting_uses_encoded_labels = False

    try:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        boost.fit(X_train, y_train_enc)
        boost_pred_enc = boost.predict(X_test)
        boost_pred = le.inverse_transform(boost_pred_enc)
        boosting_uses_encoded_labels = True

        joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
        logger.info(f"Saved label encoder to: {OUTPUT_DIR / 'label_encoder.pkl'}")

    except Exception as e:
        logger.warning(f"Encoded-label boosting path failed, retrying with raw labels. Reason: {e}")
        boost.fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

    logger.info("Boosting model training completed")
    boost_metrics = {
        "accuracy": accuracy_score(y_test, boost_pred),
        "macro_f1": f1_score(y_test, boost_pred, average="macro"),
        "report": classification_report(y_test, boost_pred)
    }

    logger.info("Saving trained artifacts")
    joblib.dump(rf, OUTPUT_DIR / "random_forest.pkl")
    joblib.dump(boost, OUTPUT_DIR / "boosting.pkl")
    logger.info(f"Saved random forest model to: {OUTPUT_DIR / 'random_forest.pkl'}")
    logger.info(f"Saved boosting model to: {OUTPUT_DIR / 'boosting.pkl'}")

    metrics_path = OUTPUT_DIR / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== RANDOM FOREST ===\n")
        f.write(str(rf_metrics["accuracy"]) + "\n")
        f.write(str(rf_metrics["macro_f1"]) + "\n")
        f.write(rf_metrics["report"] + "\n\n")
        f.write("=== BOOSTING ===\n")
        f.write(str(boost_metrics["accuracy"]) + "\n")
        f.write(str(boost_metrics["macro_f1"]) + "\n")
        f.write(boost_metrics["report"] + "\n")

    logger.info(f"Saved metrics to: {metrics_path}")

    logger.info("========== TRAINING SUMMARY ==========")
    logger.info(f"Random Forest Accuracy: {rf_metrics['accuracy']:.4f}")
    logger.info(f"Random Forest Macro F1: {rf_metrics['macro_f1']:.4f}")
    logger.info(f"Boosting Accuracy: {boost_metrics['accuracy']:.4f}")
    logger.info(f"Boosting Macro F1: {boost_metrics['macro_f1']:.4f}")
    logger.info(f"Boosting used encoded labels: {boosting_uses_encoded_labels}")

    print("\n=== RANDOM FOREST ===")
    print("Accuracy:", rf_metrics["accuracy"])
    print("Macro F1:", rf_metrics["macro_f1"])
    print(rf_metrics["report"])

    print("\n=== BOOSTING ===")
    print("Accuracy:", boost_metrics["accuracy"])
    print("Macro F1:", boost_metrics["macro_f1"])
    print(boost_metrics["report"])

    logger.info("========== CASAS SMART HOME MVP TRAINING FINISHED ==========")

if __name__ == "__main__":
    train_pipeline()
