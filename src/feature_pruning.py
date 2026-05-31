import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from .logger import get_logger

logger = get_logger("feature_pruning")

META_COLS = {"dataset", "window_start", "window_end", "label"}

def prune_constant_features(df: pd.DataFrame):
    logger.info("Pruning constant / near-constant features")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)

    kept_cols = X.columns[selector.get_support()].tolist()
    pruned_cols = [c for c in feature_cols if c not in kept_cols]

    logger.info(f"Feature pruning complete: kept={len(kept_cols)}, removed={len(pruned_cols)}")

    out = df[[c for c in df.columns if c in META_COLS or c in kept_cols]].copy()
    return out, kept_cols, pruned_cols

def prune_low_importance_features(df: pd.DataFrame, importance_csv_path, threshold=0.001):
    logger.info(f"Pruning low-importance features using {importance_csv_path}")

    imp_df = pd.read_csv(importance_csv_path)
    keep = imp_df[imp_df["importance"] >= threshold]["feature"].tolist()

    cols = [c for c in df.columns if c in META_COLS or c in keep]
    out = df[cols].copy()

    logger.info(f"Low-importance pruning complete: kept_features={len(keep)}")
    return out, keep
