import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from .config import (
    DATASETS, OUTPUT_DIR, RANDOM_STATE, WINDOW_SIZE, STEP_SIZE,
    VAL_RATIO, TEST_RATIO, MIN_CLASS_SAMPLES,
    SEQUENCE_LENGTH, LSTM_BATCH_SIZE, LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_LEARNING_RATE
)

from .utils import ensure_dir
from .parser import parse_multiple
from .feature_engineering import build_window_features
from .habit_profile import build_habit_profiles, save_habit_profiles
from .routine_mining import mine_basic_routines, activity_hour_profile
from .logger import get_logger
from .plots import save_confusion_matrix, save_top_class_barplot
from .reports import build_per_dataset_report
from .lstm_model import train_lstm

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
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
    except Exception as e:
        logger.warning(f"XGBoost unavailable, falling back to GradientBoostingClassifier. Reason: {e}")
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

def clean_labels(df: pd.DataFrame):
    logger.info("Cleaning labels")
    out = df.copy()
    out["label"] = out["label"].astype(str).str.strip()
    out["label"] = out["label"].str.replace(r"\s+", " ", regex=True)
    out["label"] = out["label"].replace({
        "Work_Bedroom_2 begub": "Work_Bedroom_2",
        "Bed_Toilet_Transition": "Bed_to_Toilet",
    })
    logger.info(f"Label cleaning complete. Unique labels now: {out['label'].nunique()}")
    return out

def filter_rare_classes(df: pd.DataFrame, min_samples=20):
    logger.info(f"Filtering rare classes with min_samples={min_samples}")
    counts = df["label"].value_counts()
    keep = counts[counts >= min_samples].index
    filtered = df[df["label"].isin(keep)].copy()
    logger.info(
        f"Rare class filtering complete: kept_classes={len(keep)}, remaining_rows={len(filtered)}"
    )
    return filtered

def prepare_labeled_table(features: pd.DataFrame):
    logger.info("Preparing labeled table with dataset retained")
    df = features.copy()
    df = df.dropna(subset=["label"]).copy()
    if df.empty:
        raise ValueError("No labeled rows after filtering null labels")

    df = clean_labels(df)
    df = filter_rare_classes(df, min_samples=MIN_CLASS_SAMPLES)

    plot_path = OUTPUT_DIR / "class_distribution.png"
    save_top_class_barplot(df["label"].value_counts(), plot_path, "Top Class Distribution", top_n=25)

    logger.info(
        f"Labeled table ready: rows={len(df)}, classes={df['label'].nunique()}, datasets={df['dataset'].nunique()}"
    )
    return df

def split_per_dataset(df: pd.DataFrame, val_ratio=0.15, test_ratio=0.15):
    logger.info(
        f"Performing per-dataset time split with val_ratio={val_ratio}, test_ratio={test_ratio}"
    )

    train_parts, val_parts, test_parts = [], [], []

    for dataset_name, ds in df.groupby("dataset"):
        ds = ds.sort_values("window_start").reset_index(drop=True)

        n = len(ds)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        train_ds = ds.iloc[:train_end].copy()
        val_ds = ds.iloc[train_end:val_end].copy()
        test_ds = ds.iloc[val_end:].copy()

        logger.info(
            f"[{dataset_name}] split sizes: total={n}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        train_parts.append(train_ds)
        val_parts.append(val_ds)
        test_parts.append(test_ds)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    logger.info(
        f"Combined split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df

def keep_known_classes(train_df, eval_df, split_name):
    known = set(train_df["label"].unique())
    before = len(eval_df)
    eval_df = eval_df[eval_df["label"].isin(known)].copy()
    after = len(eval_df)
    logger.info(
        f"{split_name}: filtered unseen classes against train labels -> before={before}, after={after}, removed={before-after}"
    )
    return eval_df

def split_xy(df: pd.DataFrame):
    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df["label"].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

def train_pipeline():
    logger.info("========== CASAS SMART HOME V2 TRAINING STARTED ==========")
    ensure_dir(OUTPUT_DIR)

    logger.info("Step 1/10: Parsing datasets")
    events = parse_multiple(DATASETS)
    if events.empty:
        raise ValueError("No data found. Put dataset files in data/raw/.")

    logger.info("Step 2/10: Mining routines from raw events")
    routines = mine_basic_routines(events)
    if not routines.empty:
        routines.to_csv(OUTPUT_DIR / "top_routines.csv", index=False)

    logger.info("Step 3/10: Building fast sliding-window features")
    features = build_window_features(events, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    if features.empty:
        raise ValueError("Feature generation returned empty dataframe")

    features.to_csv(OUTPUT_DIR / "window_features.csv", index=False)

    logger.info("Step 4/10: Building activity hour profiles")
    profile = activity_hour_profile(features)
    if not profile.empty:
        profile.to_csv(OUTPUT_DIR / "activity_hour_profile.csv", index=False)
        habit_profiles = build_habit_profiles(profile)
        save_habit_profiles(habit_profiles, OUTPUT_DIR)

    logger.info("Step 5/10: Preparing labeled dataset")
    labeled_df = prepare_labeled_table(features)

    logger.info("Step 6/10: Per-dataset split")
    train_df, val_df, test_df = split_per_dataset(labeled_df, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    val_df = keep_known_classes(train_df, val_df, "Validation")
    test_df = keep_known_classes(train_df, test_df, "Test")

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    logger.info(
        f"Final ML shapes => X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    logger.info("Step 7/10: Train Random Forest")
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
    rf_pred = rf.predict(X_test)

    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "macro_f1": f1_score(y_test, rf_pred, average="macro"),
        "report": classification_report(y_test, rf_pred, zero_division=0)
    }

    logger.info("Step 8/10: Train Boosting")
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    boost = get_boosting_model()
    boost.fit(X_train, y_train_enc)
    boost_pred_enc = boost.predict(X_test)
    boost_pred = le.inverse_transform(boost_pred_enc.astype(int))

    boost_metrics = {
        "accuracy": accuracy_score(y_test, boost_pred),
        "macro_f1": f1_score(y_test, boost_pred, average="macro"),
        "report": classification_report(y_test, boost_pred, zero_division=0)
    }

    logger.info("Step 9/10: Save reports and plots")
    save_confusion_matrix(
        y_test, rf_pred,
        labels=sorted(y_test.unique().tolist()),
        out_path=OUTPUT_DIR / "rf_confusion_matrix.png",
        title="Random Forest Confusion Matrix"
    )
    save_confusion_matrix(
        y_test, boost_pred,
        labels=sorted(y_test.unique().tolist()),
        out_path=OUTPUT_DIR / "boost_confusion_matrix.png",
        title="Boosting Confusion Matrix"
    )

    build_per_dataset_report(
        test_df=test_df,
        y_true=y_test.values,
        y_pred=rf_pred,
        out_path=OUTPUT_DIR / "rf_per_dataset_report.csv"
    )
    build_per_dataset_report(
        test_df=test_df,
        y_true=y_test.values,
        y_pred=boost_pred,
        out_path=OUTPUT_DIR / "boost_per_dataset_report.csv"
    )

    logger.info("Step 10/10: Optional LSTM training")
    lstm_metrics = None
    try:
        lstm_model, lstm_metrics, y_true_lstm, y_pred_lstm = train_lstm(
            X_train=X_train.values,
            y_train=y_train_enc,
            X_test=X_test.values,
            y_test=y_test_enc,
            seq_len=SEQUENCE_LENGTH,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_size=LSTM_BATCH_SIZE,
            epochs=LSTM_EPOCHS,
            lr=LSTM_LEARNING_RATE
        )

        torch_path = OUTPUT_DIR / "lstm_model.pt"
        import torch
        torch.save(lstm_model.state_dict(), torch_path)
        logger.info(f"Saved LSTM model to {torch_path}")

    except Exception as e:
        logger.warning(f"LSTM training skipped/failed: {e}")

    joblib.dump(rf, OUTPUT_DIR / "random_forest.pkl")
    joblib.dump(boost, OUTPUT_DIR / "boosting.pkl")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")

    metrics_path = OUTPUT_DIR / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== RANDOM FOREST ===\n")
        f.write(f"Accuracy: {rf_metrics['accuracy']}\n")
        f.write(f"Macro F1: {rf_metrics['macro_f1']}\n")
        f.write(rf_metrics["report"] + "\n\n")

        f.write("=== BOOSTING ===\n")
        f.write(f"Accuracy: {boost_metrics['accuracy']}\n")
        f.write(f"Macro F1: {boost_metrics['macro_f1']}\n")
        f.write(boost_metrics["report"] + "\n\n")

        if lstm_metrics is not None:
            f.write("=== LSTM ===\n")
            f.write(f"Accuracy: {lstm_metrics['accuracy']}\n")
            f.write(f"Macro F1: {lstm_metrics['macro_f1']}\n")
            f.write(lstm_metrics["report"] + "\n")

    logger.info("========== TRAINING SUMMARY ==========")
    logger.info(f"RF Accuracy={rf_metrics['accuracy']:.4f}, MacroF1={rf_metrics['macro_f1']:.4f}")
    logger.info(f"Boost Accuracy={boost_metrics['accuracy']:.4f}, MacroF1={boost_metrics['macro_f1']:.4f}")
    if lstm_metrics is not None:
        logger.info(f"LSTM Accuracy={lstm_metrics['accuracy']:.4f}, MacroF1={lstm_metrics['macro_f1']:.4f}")

    print("\n=== RANDOM FOREST ===")
    print("Accuracy:", rf_metrics["accuracy"])
    print("Macro F1:", rf_metrics["macro_f1"])
    print(rf_metrics["report"])

    print("\n=== BOOSTING ===")
    print("Accuracy:", boost_metrics["accuracy"])
    print("Macro F1:", boost_metrics["macro_f1"])
    print(boost_metrics["report"])

    if lstm_metrics is not None:
        print("\n=== LSTM ===")
        print("Accuracy:", lstm_metrics["accuracy"])
        print("Macro F1:", lstm_metrics["macro_f1"])
        print(lstm_metrics["report"])

    logger.info("========== CASAS SMART HOME V2 TRAINING FINISHED ==========")

if __name__ == "__main__":
    train_pipeline()
