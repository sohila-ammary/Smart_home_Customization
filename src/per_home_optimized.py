import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .logger import get_logger
from .model_zoo import get_model_candidates
from .metrics_ext import classification_metrics, top_k_accuracy_from_proba
from .sample_weights import inverse_frequency_sample_weights
from .feature_importance import save_feature_importance

logger = get_logger("per_home_optimized")

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

def fit_model(model_name, model, X_train, y_train, sample_weight=None):
    if model_name in {"xgboost", "lightgbm", "catboost", "hist_gb"} and sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return model
        except TypeError:
            pass
        except Exception:
            pass

    try:
        model.fit(X_train, y_train)
    except TypeError:
        model.fit(X_train, y_train)
    return model

def train_per_home_optimized(features_df, output_dir, random_state=42, min_class_samples=30):
    logger.info("Training optimized per-home models")

    final_rows = []

    for dataset_name, ds in features_df.groupby("dataset"):
        logger.info(f"[{dataset_name}] preparing optimized search")

        ds = ds.dropna(subset=["label"]).copy()
        counts = ds["label"].value_counts()
        keep_labels = counts[counts >= min_class_samples].index
        ds = ds[ds["label"].isin(keep_labels)].copy()

        if len(ds) < 500 or ds["label"].nunique() < 2:
            logger.warning(f"[{dataset_name}] skipped due to insufficient data")
            continue

        train_df, val_df, test_df = split_per_dataset_single(ds)
        X_train, y_train = build_xy(train_df)
        X_val, y_val = build_xy(val_df)
        X_test, y_test = build_xy(test_df)

        le = LabelEncoder()
        le.fit(y_train)

        known = set(y_train.unique())
        val_mask = y_val.isin(known)
        test_mask = y_test.isin(known)

        X_val = X_val[val_mask].copy()
        y_val = y_val[val_mask].copy()
        X_test = X_test[test_mask].copy()
        y_test = y_test[test_mask].copy()

        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        weights = inverse_frequency_sample_weights(y_train_enc)

        candidates = get_model_candidates(random_state=random_state)

        best_model_name = None
        best_model = None
        best_val_macro = -1
        best_val_metrics = None

        for model_name, model in candidates.items():
            logger.info(f"[{dataset_name}] training candidate={model_name}")

            # use encoded labels for boosting-like models
            if model_name in {"xgboost", "lightgbm", "catboost"}:
                fit_model(model_name, model, X_train, y_train_enc, sample_weight=weights)
                val_pred_enc = model.predict(X_val)
                val_pred = le.inverse_transform(val_pred_enc.astype(int))

                val_metrics = classification_metrics(y_val, val_pred)
            else:
                fit_model(model_name, model, X_train, y_train)
                val_pred = model.predict(X_val)
                val_metrics = classification_metrics(y_val, val_pred)

            logger.info(
                f"[{dataset_name}] {model_name}: val_acc={val_metrics['accuracy']:.4f}, "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_macro:
                best_val_macro = val_metrics["macro_f1"]
                best_model_name = model_name
                best_model = model
                best_val_metrics = val_metrics

        logger.info(f"[{dataset_name}] best model={best_model_name} with val_macro_f1={best_val_macro:.4f}")

        # test evaluation
        if best_model_name in {"xgboost", "lightgbm", "catboost"}:
            test_pred_enc = best_model.predict(X_test)
            test_pred = le.inverse_transform(test_pred_enc.astype(int))
            proba = best_model.predict_proba(X_test)
            top3 = top_k_accuracy_from_proba(y_test_enc, proba, k=3)
        else:
            test_pred = best_model.predict(X_test)
            top3 = None
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(X_test)
                class_to_idx = {c: i for i, c in enumerate(best_model.classes_)}
                y_test_idx = [class_to_idx[v] for v in y_test]
                top3 = top_k_accuracy_from_proba(y_test_idx, proba, k=3)

        test_metrics = classification_metrics(y_test, test_pred)

        joblib.dump(best_model, output_dir / f"best_model_{dataset_name}.pkl")
        joblib.dump(le, output_dir / f"best_label_encoder_{dataset_name}.pkl")

        importance_path = output_dir / f"feature_importance_{dataset_name}.csv"
        save_feature_importance(best_model, X_train.columns.tolist(), importance_path, top_n=60)

        report_path = output_dir / f"optimized_report_{dataset_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Best model: {best_model_name}\n")
            f.write(f"Validation Accuracy: {best_val_metrics['accuracy']}\n")
            f.write(f"Validation Macro F1: {best_val_metrics['macro_f1']}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']}\n")
            f.write(f"Test Macro F1: {test_metrics['macro_f1']}\n")
            f.write(f"Test Weighted F1: {test_metrics['weighted_f1']}\n")
            f.write(f"Top-3 Accuracy: {top3}\n\n")
            f.write(test_metrics["report"])

        final_rows.append({
            "dataset": dataset_name,
            "best_model": best_model_name,
            "val_accuracy": best_val_metrics["accuracy"],
            "val_macro_f1": best_val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "top3_accuracy": top3,
            "train_rows": len(train_df),
            "test_rows": len(X_test),
            "num_classes": ds["label"].nunique(),
        })

        logger.info(
            f"[{dataset_name}] final best={best_model_name}, test_acc={test_metrics['accuracy']:.4f}, "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}, top3={top3}"
        )

    summary = pd.DataFrame(final_rows)
    if not summary.empty:
        summary.to_csv(output_dir / "per_home_optimized_summary.csv", index=False)
        logger.info(f"Saved optimized summary to {output_dir / 'per_home_optimized_summary.csv'}")
    return summary
