from .logger import get_logger

logger = get_logger("model_zoo_v51")

def get_regularized_model_candidates(random_state=42):
    models = {}

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

    models["random_forest_reg"] = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["extra_trees_reg"] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["hist_gb_reg"] = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.04,
        max_iter=250,
        random_state=random_state
    )

    try:
        from xgboost import XGBClassifier
        models["xgboost_reg"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state
        )
        logger.info("Regularized XGBoost available")
    except Exception as e:
        logger.warning(f"Regularized XGBoost unavailable: {e}")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm_reg"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state
        )
        logger.info("Regularized LightGBM available")
    except Exception as e:
        logger.warning(f"Regularized LightGBM unavailable: {e}")

    try:
        from catboost import CatBoostClassifier
        models["catboost_reg"] = CatBoostClassifier(
            iterations=250,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False
        )
        logger.info("Regularized CatBoost available")
    except Exception as e:
        logger.warning(f"Regularized CatBoost unavailable: {e}")

    return models
