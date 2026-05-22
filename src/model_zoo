from .logger import get_logger

logger = get_logger("model_zoo")

def get_model_candidates(random_state=42):
    models = {}

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["hist_gb"] = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=random_state
    )

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state
        )
        logger.info("XGBoost available")
    except Exception as e:
        logger.warning(f"XGBoost unavailable: {e}")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state
        )
        logger.info("LightGBM available")
    except Exception as e:
        logger.warning(f"LightGBM unavailable: {e}")

    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False
        )
        logger.info("CatBoost available")
    except Exception as e:
        logger.warning(f"CatBoost unavailable: {e}")

    return models
