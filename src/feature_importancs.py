import pandas as pd
from .logger import get_logger

logger = get_logger("feature_importance")

def save_feature_importance(model, feature_names, out_path, top_n=50):
    logger.info(f"Saving feature importance to {out_path}")

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            importances = abs(coef).mean(axis=0)
        else:
            importances = abs(coef)

    if importances is None:
        logger.warning("Model does not expose feature importances")
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    df.to_csv(out_path, index=False)
    logger.info(f"Saved feature importance csv: {out_path}")
    return df
