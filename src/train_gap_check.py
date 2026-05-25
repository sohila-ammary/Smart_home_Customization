import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from .config import OUTPUT_DIR, RANDOM_STATE
from .logger import get_logger

logger = get_logger("train_gap_check")

def _numeric_X(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

def main():
    path = OUTPUT_DIR / "window_features_v31_coarse.csv"
    df = pd.read_csv(path)

    rows = []

    for dataset_name, ds in df.groupby("dataset"):
        ds = ds.dropna(subset=["label"]).copy()
        counts = ds["label"].value_counts()
        ds = ds[ds["label"].isin(counts[counts >= 30].index)].copy()
        ds = ds.sort_values("window_start").reset_index(drop=True)

        n = len(ds)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = ds.iloc[:train_end].copy()
        val_df = ds.iloc[train_end:val_end].copy()
        test_df = ds.iloc[val_end:].copy()

        features = [c for c in ds.columns if c not in {"dataset", "window_start", "window_end", "label"}]

        known = set(train_df["label"].unique())
        val_df = val_df[val_df["label"].isin(known)].copy()
        test_df = test_df[test_df["label"].isin(known)].copy()

        X_train = _numeric_X(train_df, features)
        y_train = train_df["label"]
        X_val = _numeric_X(val_df, features)
        y_val = val_df["label"]
        X_test = _numeric_X(test_df, features)
        y_test = test_df["label"]

        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            preds = model.predict(X_split)
            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "accuracy": accuracy_score(y_split, preds),
                "macro_f1": f1_score(y_split, preds, average="macro"),
                "weighted_f1": f1_score(y_split, preds, average="weighted"),
                "rows": len(y_split),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "train_val_test_gap.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
