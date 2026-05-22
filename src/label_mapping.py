from .logger import get_logger

logger = get_logger("label_mapping")

COARSE_LABEL_MAP = {
    "Sleeping": "Sleep",
    "Sleep": "Sleep",
    "R1_Sleep": "Sleep",
    "R2_Sleep": "Sleep",
    "R1_Sleeping_in_Bed": "Sleep",
    "R2_Sleeping_in_Bed": "Sleep",

    "Breakfast": "Meal",
    "Lunch": "Meal",
    "Dinner": "Meal",
    "Eating": "Meal",
    "Cook_Breakfast": "Meal",
    "Cook_Lunch": "Meal",
    "Meal_Preparation": "Meal",
    "Wash_Dishes": "Meal",
    "R1_Eat_Breakfast": "Meal",
    "R2_Eat_Breakfast": "Meal",
    "R1_Snack": "Meal",

    "Bathing": "Bathroom",
    "Personal_Hygiene": "Bathroom",
    "Bed_to_Toilet": "Bathroom",
    "Bed_Toilet_Transition": "Bathroom",
    "Master_Bathroom": "Bathroom",
    "Guest_Bathroom": "Bathroom",

    "Work": "Work",
    "Read": "Work",
    "Desk_Activity": "Work",
    "R1_Work": "Work",
    "Work_Bedroom_1": "Work",
    "Work_Bedroom_2": "Work",
    "Work_LivingRm": "Work",
    "Work_Table": "Work",

    "Relax": "Relax",
    "Watch_TV": "Relax",

    "Leave_Home": "Away",
    "Enter_Home": "Away",

    "Kitchen_Activity": "Kitchen",
    "Housekeeping": "Chores",
    "Chores": "Chores",
    "Laundry": "Chores",

    "R1_Wake": "Wake",
    "R2_Wake": "Wake",

    "Morning_Meds": "Medication",
    "Eve_Meds": "Medication",
    "R2_Take_Medicine": "Medication",

    "Night_Wandering": "NightBehavior",
    "Group_Meeting": "Social",
    "Dining_Rm_Activity": "Dining",
    "Meditate": "Wellness",
    "Respirate": "Wellness",
    "Yoga": "Wellness",
}

def map_to_coarse_label(label: str):
    if label is None:
        return None
    return COARSE_LABEL_MAP.get(label, label)

def apply_coarse_mapping(df, label_col="label"):
    logger.info("Applying coarse label mapping")
    out = df.copy()
    out[label_col] = out[label_col].apply(map_to_coarse_label)
    logger.info(f"Coarse mapping complete. Unique labels: {out[label_col].nunique()}")
    return out
