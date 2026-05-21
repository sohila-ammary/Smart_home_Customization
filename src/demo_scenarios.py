from .advanced_recommender import score_scenarios
from .logger import get_logger

logger = get_logger("demo_scenarios")

def main():
    logger.info("Running smart-home scenario demo")

    examples = [
        ("Sleeping", 22, 18.5),
        ("Cook_Breakfast", 7, 22.0),
        ("Work", 11, 24.0),
        ("Leave_Home", 9, 23.0),
    ]

    for activity, hour, temp in examples:
        print(f"\nActivity={activity}, hour={hour}, temp={temp}")
        ranked = score_scenarios(activity, hour, temp_mean=temp, top_k=3)
        for i, item in enumerate(ranked, start=1):
            print(f"{i}. {item['scenario']} (score={item['score']})")
            for action in item["actions"]:
                print(f"   - {action}")

if __name__ == "__main__":
    main()
