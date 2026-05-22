import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from .logger import get_logger

logger = get_logger("per_home_models")

def get_boosting_model(random_state=42):
    try:
        from xgboost import XGBClassifier
        logger.info("Using XGBoost for per-home model")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state
        )
    except Exception as e:
        logger.warning(f"XGBoost unavailable, using GradientBoostingClassifier. Reason: {e}")
        return GradientBoostingClassifier(random_state=random_state)

def split_per_dataset_single(df, val_ratio=0.15, test_ratio=0.15):
    df = df.sort_values("window_start").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def build_xy(df):
    drop_cols = ["dataset", "window_start", "window_end", "label"]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df["label"].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

def train_per_home_models(features_df, output_dir, random_state=42, min_class_samples=30):
    logger.info("Training per-home models")

    all_rows = []

    for dataset_name, ds in features_df.groupby("dataset"):
        logger.info(f"[{dataset_name}] preparing per-home training")

        ds = ds.dropna(subset=["label"]).copy()
        counts = ds["label"].value_counts()
        keep_labels = counts[counts >= min_class_samples].index
        ds = ds[ds["label"].isin(keep_labels)].copy()

        logger.info(
            f"[{dataset_name}] rows={len(ds)}, classes={ds['label'].nunique()}, "
            f"kept_labels={sorted(ds['label'].unique().tolist())}"
        )

        if len(ds) < 500 or ds["label"].nunique() < 2:
            logger.warning(f"[{dataset_name}] skipped due to insufficient data")
            continue

        train_df, val_df, test_df = split_per_dataset_single(ds)
        X_train, y_train = build_xy(train_df)
        X_val, y_val = build_xy(val_df)
        X_test, y_test = build_xy(test_df)

        rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average="macro")

        le = LabelEncoder()
        le.fit(y_train)

        # keep only known classes
        known = set(y_train.unique())
        mask_test = y_test.isin(known)
        X_test_boost = X_test[mask_test].copy()
        y_test_boost = y_test[mask_test].copy()

        X_train_boost = X_train.copy()
        y_train_boost = y_train.copy()

        y_train_enc = le.transform(y_train_boost)
        y_test_enc = le.transform(y_test_boost)

        boost = get_boosting_model(random_state=random_state)
        boost.fit(X_train_boost, y_train_enc)
        boost_pred_enc = boost.predict(X_test_boost)
        boost_pred = le.inverse_transform(boost_pred_enc.astype(int))

        boost_acc = accuracy_score(y_test_boost, boost_pred)
        boost_f1 = f1_score(y_test_boost, boost_pred, average="macro")

        joblib.dump(rf, output_dir / f"rf_{dataset_name}.pkl")
        joblib.dump(boost, output_dir / f"boost_{dataset_name}.pkl")
        joblib.dump(le, output_dir / f"label_encoder_{dataset_name}.pkl")

        with open(output_dir / f"report_{dataset_name}.txt", "w", encoding="utf-8") as f:
            f.write("=== RANDOM FOREST ===\n")
            f.write(f"Accuracy: {rf_acc}\n")
            f.write(f"Macro F1: {rf_f1}\n")
            f.write(classification_report(y_test, rf_pred, zero_division=0))
            f.write("\n\n=== BOOSTING ===\n")
            f.write(f"Accuracy: {boost_acc}\n")
            f.write(f"Macro F1: {boost_f1}\n")
            f.write(classification_report(y_test_boost, boost_pred, zero_division=0))

        all_rows.append({
            "dataset": dataset_name,
            "rf_accuracy": rf_acc,
            "rf_macro_f1": rf_f1,
            "boost_accuracy": boost_acc,
            "boost_macro_f1": boost_f1,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "num_classes": ds["label"].nunique(),
        })

        logger.info(
            f"[{dataset_name}] RF acc={rf_acc:.4f}, RF macro_f1={rf_f1:.4f}, "
            f"Boost acc={boost_acc:.4f}, Boost macro_f1={boost_f1:.4f}"
        )

    report_df = pd.DataFrame(all_rows)
    if not report_df.empty:
        report_df.to_csv(output_dir / "per_home_summary.csv", index=False)
        logger.info(f"Saved per-home summary to {output_dir / 'per_home_summary.csv'}")

    return report_df
