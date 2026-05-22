from .logger import get_logger

logger = get_logger("label_mapping_v31")

def dataset_specific_post_mapping(df):
    logger.info("Applying dataset-specific post-mapping")
    out = df.copy()

    # Milan is a bit harder and may benefit from slight consolidation
    milan_mask = out["dataset"] == "milan"
    out.loc[milan_mask & (out["label"] == "Dining"), "label"] = "Meal"
    out.loc[milan_mask & (out["label"] == "Kitchen"), "label"] = "Meal"

    return out
