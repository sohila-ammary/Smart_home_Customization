from .config import DATASETS, OUTPUT_DIR, RANDOM_STATE
from .utils import ensure_dir
from .parser import parse_multiple
from .feature_engineering import build_window_features
from .routine_mining import activity_hour_profile
from .habit_profile import build_habit_profiles, save_habit_profiles
from .label_mapping import apply_coarse_mapping
from .label_mapping_v31 import dataset_specific_post_mapping
from .per_home_v51 import train_per_home_v51
from .logger import get_logger

logger = get_logger("train_v51")

def main():
    logger.info("========== CASAS SMART HOME V5.1 TRAINING STARTED ==========")
    ensure_dir(OUTPUT_DIR)

    logger.info("Step 1/5: Parsing datasets")
    events = parse_multiple(DATASETS)

    logger.info("Step 2/5: Building features")
    features = build_window_features(
        events,
        window_size="5min",
        step_size="1min",
        min_label_fraction=0.6
    )
    features.to_csv(OUTPUT_DIR / "window_features_v51.csv", index=False)

    logger.info("Step 3/5: Habit profiles")
    profile = activity_hour_profile(features)
    if not profile.empty:
        profile.to_csv(OUTPUT_DIR / "activity_hour_profile_v51.csv", index=False)
        habit_profiles = build_habit_profiles(profile)
        save_habit_profiles(habit_profiles, OUTPUT_DIR)

    logger.info("Step 4/5: Coarse mapping")
    features = apply_coarse_mapping(features, label_col="label")
    features = dataset_specific_post_mapping(features)
    features.to_csv(OUTPUT_DIR / "window_features_v51_coarse.csv", index=False)

    logger.info("Step 5/5: Train V5.1 models")
    summary = train_per_home_v51(
        features_df=features,
        output_dir=OUTPUT_DIR,
        random_state=RANDOM_STATE,
        min_class_samples=30
    )

    if not summary.empty:
        print("\n=== V5.1 SUMMARY ===")
        print(summary.to_string(index=False))

    logger.info("========== CASAS SMART HOME V5.1 TRAINING FINISHED ==========")

if __name__ == "__main__":
    main()
