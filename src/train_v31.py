from .config import DATASETS, OUTPUT_DIR, RANDOM_STATE
from .utils import ensure_dir
from .parser import parse_multiple
from .feature_engineering import build_window_features
from .routine_mining import mine_basic_routines, activity_hour_profile
from .habit_profile import build_habit_profiles, save_habit_profiles
from .label_mapping import apply_coarse_mapping
from .label_mapping_v31 import dataset_specific_post_mapping
from .per_home_optimized import train_per_home_optimized
from .logger import get_logger

logger = get_logger("train_v31")

def train_pipeline_v31():
    logger.info("========== CASAS SMART HOME V3.1 TRAINING STARTED ==========")
    ensure_dir(OUTPUT_DIR)

    logger.info("Step 1/6: Parsing datasets")
    events = parse_multiple(DATASETS)
    if events.empty:
        raise ValueError("No data found. Put dataset files in data/raw/.")

    logger.info("Step 2/6: Mining routines")
    routines = mine_basic_routines(events)
    if not routines.empty:
        routines.to_csv(OUTPUT_DIR / "top_routines_v31.csv", index=False)

    logger.info("Step 3/6: Building features")
    features = build_window_features(
        events,
        window_size="5min",
        step_size="1min",
        min_label_fraction=0.6
    )
    if features.empty:
        raise ValueError("Feature generation returned empty dataframe")

    logger.info("Step 4/6: Habit profiles")
    profile = activity_hour_profile(features)
    if not profile.empty:
        profile.to_csv(OUTPUT_DIR / "activity_hour_profile_v31.csv", index=False)
        habit_profiles = build_habit_profiles(profile)
        save_habit_profiles(habit_profiles, OUTPUT_DIR)

    logger.info("Step 5/6: Coarse mapping")
    features = apply_coarse_mapping(features, label_col="label")
    features = dataset_specific_post_mapping(features)
    features.to_csv(OUTPUT_DIR / "window_features_v31_coarse.csv", index=False)

    logger.info("Step 6/6: Optimized per-home training")
    summary = train_per_home_optimized(
        features_df=features,
        output_dir=OUTPUT_DIR,
        random_state=RANDOM_STATE,
        min_class_samples=30
    )

    if not summary.empty:
        print("\n=== OPTIMIZED PER-HOME SUMMARY ===")
        print(summary.to_string(index=False))

    logger.info("========== CASAS SMART HOME V3.1 TRAINING FINISHED ==========")

if __name__ == "__main__":
    train_pipeline_v31()
