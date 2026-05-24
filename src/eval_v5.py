import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .config import OUTPUT_DIR, RANDOM_STATE
from .logger import get_logger
from .baselines import majority_class_baseline, hour_of_day_baseline
from .ablation import get_feature_subsets
from .sanity_checks import shuffled_label_test
from .rolling_cv import evaluate_rolling_folds

logger = get_logger("eval_v5")

def load_v31_features():
    path = OUTPUT_DIR / "window_features_v31_coarse.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)

def _numeric_X(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

def train_test_split_single(df):
    df = df.sort_values("window_start").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def evaluate_dataset(dataset_name, ds_df):
    logger.info(f"[{dataset_name}] starting V5 evaluation")

    ds_df = ds_df.dropna(subset=["label"]).copy()
    counts = ds_df["label"].value_counts()
    ds_df = ds_df[ds_df["label"].isin(counts[counts >= 30].index)].copy()

    if len(ds_df) < 1000 or ds_df["label"].nunique() < 2:
        logger.warning(f"[{dataset_name}] skipped due to insufficient data")
        return None

    train_df, val_df, test_df = train_test_split_single(ds_df)

    known = set(train_df["label"].unique())
    val_df = val_df[val_df["label"].isin(known)].copy()
    test_df = test_df[test_df["label"].isin(known)].copy()

    subsets = get_feature_subsets(ds_df.columns.tolist())

    baseline_majority = majority_class_baseline(train_df["label"], test_df["label"])
    baseline_hour = hour_of_day_baseline(train_df, test_df, label_col="label")

    rows = []

    for subset_name, feature_cols in subsets.items():
        if len(feature_cols) == 0:
            continue

        X_train = _numeric_X(train_df, feature_cols)
        y_train = train_df["label"].copy()
        X_test = _numeric_X(test_df, feature_cols)
        y_test = test_df["label"].copy()

        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rows.append({
            "dataset": dataset_name,
            "evaluation": subset_name,
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
        })

    # rolling CV with ExtraTrees on full features
    full_features = subsets["full"]
    rolling_df = evaluate_rolling_folds(
        ds_df,
        model_builder=lambda: ExtraTreesClassifier(
            n_estimators=200,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        feature_cols=full_features,
        label_col="label",
        n_folds=3
    )
    if not rolling_df.empty:
        rolling_df.to_csv(OUTPUT_DIR / f"rolling_cv_{dataset_name}.csv", index=False)

    # sanity test
    X_train_full = _numeric_X(train_df, full_features)
    y_train_full = train_df["label"].copy()
    X_test_full = _numeric_X(test_df, full_features)
    y_test_full = test_df["label"].copy()

    sanity = shuffled_label_test(
        model_builder=lambda: RandomForestClassifier(
            n_estimators=200,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        X_train=X_train_full,
        y_train=y_train_full,
        X_test=X_test_full,
        y_test=y_test_full,
        random_state=RANDOM_STATE
    )

    baseline_path = OUTPUT_DIR / f"baselines_{dataset_name}.txt"
    with open(baseline_path, "w", encoding="utf-8") as f:
        f.write("=== MAJORITY CLASS ===\n")
        f.write(str(baseline_majority) + "\n\n")
        f.write("=== HOUR OF DAY ===\n")
        f.write(str(baseline_hour) + "\n\n")
        f.write("=== SHUFFLED LABEL SANITY ===\n")
        f.write(str(sanity) + "\n")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / f"ablation_{dataset_name}.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "majority_accuracy": baseline_majority["accuracy"],
        "majority_macro_f1": baseline_majority["macro_f1"],
        "hour_accuracy": baseline_hour["accuracy"],
        "hour_macro_f1": baseline_hour["macro_f1"],
        "shuffled_accuracy": sanity["accuracy"],
        "shuffled_macro_f1": sanity["macro_f1"],
        "rolling_macro_f1_mean": rolling_df["macro_f1"].mean() if not rolling_df.empty else None,
        "rolling_macro_f1_std": rolling_df["macro_f1"].std() if not rolling_df.empty else None,
    }

    logger.info(f"[{dataset_name}] V5 evaluation complete")
    return summary

def main():
    logger.info("========== V5 EVALUATION STARTED ==========")
    df = load_v31_features()

    summaries = []
    for dataset_name, ds_df in df.groupby("dataset"):
        summary = evaluate_dataset(dataset_name, ds_df)
        if summary is not None:
            summaries.append(summary)

    if summaries:
        out = pd.DataFrame(summaries)
        out.to_csv(OUTPUT_DIR / "v5_evaluation_summary.csv", index=False)
        print("\n=== V5 EVALUATION SUMMARY ===")
        print(out.to_string(index=False))

    logger.info("========== V5 EVALUATION FINISHED ==========")

if __name__ == "__main__":
    main()
