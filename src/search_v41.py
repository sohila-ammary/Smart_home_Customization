import pandas as pd
from .config import DATASETS, OUTPUT_DIR, RANDOM_STATE
from .parser import parse_multiple
from .feature_engineering import build_window_features
from .label_mapping import apply_coarse_mapping
from .label_mapping_v31 import dataset_specific_post_mapping
from .per_home_optimized import train_per_home_optimized
from .search_config import WINDOW_SIZE_GRID, MIN_LABEL_FRACTION_GRID, MIN_CLASS_SAMPLES_GRID, SEARCH_DATASETS
from .logger import get_logger

logger = get_logger("search_v41")

def run_search():
    logger.info("========== CASAS SMART HOME V4.1 SEARCH STARTED ==========")

    events = parse_multiple(DATASETS)
    if events.empty:
        raise ValueError("No data found")

    events = events[events["dataset"].isin(SEARCH_DATASETS)].copy()
    logger.info(f"Running search only on datasets={SEARCH_DATASETS}")

    results = []

    trial_id = 0
    for window_size in WINDOW_SIZE_GRID:
        for min_label_fraction in MIN_LABEL_FRACTION_GRID:
            trial_id += 1
            logger.info(
                f"Trial {trial_id}: window_size={window_size}, min_label_fraction={min_label_fraction}"
            )

            features = build_window_features(
                events,
                window_size=window_size,
                step_size="1min",
                min_label_fraction=min_label_fraction
            )

            if features.empty:
                logger.warning("Features empty, skipping trial")
                continue

            features = apply_coarse_mapping(features, label_col="label")
            features = dataset_specific_post_mapping(features)

            for min_class_samples in MIN_CLASS_SAMPLES_GRID:
                logger.info(
                    f"Trial {trial_id}: evaluating min_class_samples={min_class_samples}"
                )

                trial_out_dir = OUTPUT_DIR / f"search_trial_ws_{window_size}_lf_{str(min_label_fraction).replace('.', '_')}_mcs_{min_class_samples}"
                trial_out_dir.mkdir(parents=True, exist_ok=True)

                summary = train_per_home_optimized(
                    features_df=features,
                    output_dir=trial_out_dir,
                    random_state=RANDOM_STATE,
                    min_class_samples=min_class_samples
                )

                if summary.empty:
                    continue

                summary["window_size"] = window_size
                summary["min_label_fraction"] = min_label_fraction
                summary["min_class_samples"] = min_class_samples
                results.append(summary)

    if not results:
        logger.warning("No search results generated")
        return pd.DataFrame()

    final = pd.concat(results, ignore_index=True)
    final.to_csv(OUTPUT_DIR / "v41_search_results.csv", index=False)

    best = (
        final.sort_values(["dataset", "test_macro_f1"], ascending=[True, False])
        .groupby("dataset")
        .head(5)
        .reset_index(drop=True)
    )
    best.to_csv(OUTPUT_DIR / "v41_best_configs.csv", index=False)

    logger.info(f"Saved all search results to {OUTPUT_DIR / 'v41_search_results.csv'}")
    logger.info(f"Saved best configs to {OUTPUT_DIR / 'v41_best_configs.csv'}")
    logger.info("========== CASAS SMART HOME V4.1 SEARCH FINISHED ==========")

    return final
