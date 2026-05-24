import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .logger import get_logger

logger = get_logger("rolling_cv")

def build_rolling_folds(df, n_folds=3):
    df = df.sort_values("window_start").reset_index(drop=True)
    n = len(df)

    folds = []
    for i in range(n_folds):
        train_end = int(n * (0.5 + i * 0.1))
        val_end = int(n * (0.7 + i * 0.05))
        test_end = int(n * (0.9 + i * 0.03))

        train_end = min(train_end, n)
        val_end = min(val_end, n)
        test_end = min(test_end, n)

        if not (train_end < val_end < test_end):
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:test_end].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        folds.append((train_df, val_df, test_df))

    logger.info(f"Built {len(folds)} rolling folds")
    return folds

def evaluate_rolling_folds(df, model_builder, feature_cols, label_col="label", n_folds=3):
    logger.info("Evaluating rolling folds")
    folds = build_rolling_folds(df, n_folds=n_folds)

    rows = []

    for i, (train_df, val_df, test_df) in enumerate(folds, start=1):
        # keep only shared classes
        known = set(train_df[label_col].unique())
        test_df = test_df[test_df[label_col].isin(known)].copy()
        if test_df.empty:
            continue

        X_train = train_df[feature_cols].copy()
        y_train = train_df[label_col].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[label_col].copy()

        model = model_builder()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rows.append({
            "fold": i,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
        })

    return pd.DataFrame(rows)
