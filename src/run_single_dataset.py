import argparse
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .utils import ensure_dir
from .config import OUTPUT_DIR
from .dataset_selector import select_datasets
from .parser import parse_multiple
from .feature_engineering import build_window_features
from .label_mapping import apply_coarse_mapping
from .label_mapping_v31 import dataset_specific_post_mapping
from .routine_mining import activity_hour_profile
from .habit_profile import build_habit_profiles, save_habit_profiles
from .v6_sequence_data import prepare_event_level_labels, build_sequences
from .v6_gru_model import EventGRUClassifier
from .v6_dataset import EventSequenceDataset
from .logger import get_logger

logger = get_logger("run_single_dataset")

def time_split_sequences(data_dict, y, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = {k: v[:train_end] if k.startswith("X_") else v for k, v in data_dict.items()}
    val_data = {k: v[train_end:val_end] if k.startswith("X_") else v for k, v in data_dict.items()}
    test_data = {k: v[val_end:] if k.startswith("X_") else v for k, v in data_dict.items()}

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    return train_data, val_data, test_data, y_train, y_val, y_test

def build_features_for_dataset(dataset_name, window_size="5min", step_size="1min", min_label_fraction=0.6):
    logger.info(f"========== BUILD FEATURES FOR {dataset_name.upper()} STARTED ==========")
    ensure_dir(OUTPUT_DIR)

    datasets = select_datasets([dataset_name])

    logger.info(f"Parsing dataset={dataset_name}")
    events = parse_multiple(datasets)
    if events.empty:
        raise ValueError(f"No data found for dataset={dataset_name}")

    logger.info(f"Building features for dataset={dataset_name}")
    features = build_window_features(
        events,
        window_size=window_size,
        step_size=step_size,
        min_label_fraction=min_label_fraction
    )
    if features.empty:
        raise ValueError(f"No features generated for dataset={dataset_name}")

    raw_path = OUTPUT_DIR / f"{dataset_name}_window_features.csv"
    features.to_csv(raw_path, index=False)
    logger.info(f"Saved raw features to {raw_path}")

    logger.info(f"Building profiles for dataset={dataset_name}")
    profile = activity_hour_profile(features)
    if not profile.empty:
        profile_path = OUTPUT_DIR / f"{dataset_name}_activity_hour_profile.csv"
        profile.to_csv(profile_path, index=False)
        logger.info(f"Saved activity hour profile to {profile_path}")

        habit_profiles = build_habit_profiles(profile)
        save_habit_profiles(habit_profiles, OUTPUT_DIR)

    logger.info(f"Applying coarse label mapping for dataset={dataset_name}")
    features = apply_coarse_mapping(features, label_col="label")
    features = dataset_specific_post_mapping(features)

    coarse_path = OUTPUT_DIR / f"{dataset_name}_window_features_coarse.csv"
    features.to_csv(coarse_path, index=False)
    logger.info(f"Saved coarse features to {coarse_path}")

    logger.info(f"========== BUILD FEATURES FOR {dataset_name.upper()} FINISHED ==========")
    return events, features

def train_v6_for_dataset(
    dataset_name,
    seq_len=50,
    batch_size=256,
    epochs=8,
    lr=1e-3,
    window_size="5min",
    step_size="1min",
    min_label_fraction=0.6,
    min_label_count=50
):
    logger.info(f"========== V6 TRAINING FOR {dataset_name.upper()} STARTED ==========")
    ensure_dir(OUTPUT_DIR)

    datasets = select_datasets([dataset_name])

    logger.info(f"Parsing dataset={dataset_name}")
    events = parse_multiple(datasets)
    if events.empty:
        raise ValueError(f"No data found for dataset={dataset_name}")

    logger.info(f"Building features for dataset={dataset_name}")
    features = build_window_features(
        events,
        window_size=window_size,
        step_size=step_size,
        min_label_fraction=min_label_fraction
    )
    features = apply_coarse_mapping(features, label_col="label")
    features = dataset_specific_post_mapping(features)

    logger.info(f"Preparing event-level labels for dataset={dataset_name}")
    labeled_events = prepare_event_level_labels(events, features, dataset_name=dataset_name)
    if labeled_events.empty:
        raise ValueError(f"No event-level labels found for dataset={dataset_name}")

    counts = labeled_events["label"].value_counts()
    labeled_events = labeled_events[labeled_events["label"].isin(counts[counts >= min_label_count].index)].copy()

    logger.info(f"Building event sequences for dataset={dataset_name}")
    seq_data = build_sequences(labeled_events, seq_len=seq_len)

    le = LabelEncoder()
    y_enc = le.fit_transform(seq_data["y"])

    train_data, val_data, test_data, y_train, y_val, y_test = time_split_sequences(seq_data, y_enc)

    train_ds = EventSequenceDataset(train_data, y_train)
    val_ds = EventSequenceDataset(val_data, y_val)
    test_ds = EventSequenceDataset(test_data, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device={device}")

    model = EventGRUClassifier(
        num_sensors=len(seq_data["sensor_vocab"]),
        num_types=len(seq_data["type_vocab"]),
        num_values=3,
        num_hours=24,
        num_classes=len(le.classes_),
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"Training V6 GRU for dataset={dataset_name}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb_sensor, xb_type, xb_value, xb_delta, xb_hour, yb in train_loader:
            xb_sensor = xb_sensor.to(device)
            xb_type = xb_type.to(device)
            xb_value = xb_value.to(device)
            xb_delta = xb_delta.to(device)
            xb_hour = xb_hour.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb_sensor, xb_type, xb_value, xb_delta, xb_hour)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch}/{epochs} loss={total_loss / len(train_loader):.4f}")

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb_sensor, xb_type, xb_value, xb_delta, xb_hour, yb in test_loader:
            xb_sensor = xb_sensor.to(device)
            xb_type = xb_type.to(device)
            xb_value = xb_value.to(device)
            xb_delta = xb_delta.to(device)
            xb_hour = xb_hour.to(device)

            logits = model(xb_sensor, xb_type, xb_value, xb_delta, xb_hour)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(pred.tolist())
            targets.extend(yb.numpy().tolist())

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    report = classification_report(targets, preds, zero_division=0)

    model_path = OUTPUT_DIR / f"v6_gru_{dataset_name}.pt"
    encoder_path = OUTPUT_DIR / f"v6_gru_label_encoder_{dataset_name}.pkl"
    report_path = OUTPUT_DIR / f"v6_gru_report_{dataset_name}.txt"

    torch.save(model.state_dict(), model_path)
    joblib.dump(le, encoder_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Macro F1: {macro_f1}\n")
        f.write(report)

    logger.info(f"Saved V6 model to {model_path}")
    logger.info(f"Saved V6 label encoder to {encoder_path}")
    logger.info(f"Saved V6 report to {report_path}")
    logger.info(f"V6 {dataset_name} Accuracy={acc:.4f}, MacroF1={macro_f1:.4f}")
    logger.info(f"========== V6 TRAINING FOR {dataset_name.upper()} FINISHED ==========")

def main():
    parser = argparse.ArgumentParser(description="Run single-dataset smart home pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. tulum2, aruba, milan")
    parser.add_argument("--mode", required=True, choices=["features", "train_v6"], help="What to run")
    parser.add_argument("--window_size", default="5min", help="Window size for feature generation")
    parser.add_argument("--step_size", default="1min", help="Step size for feature generation")
    parser.add_argument("--min_label_fraction", type=float, default=0.6, help="Minimum dominant label fraction")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for V6")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for V6")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs for V6")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for V6")
    parser.add_argument("--min_label_count", type=int, default=50, help="Minimum label support for V6")

    args = parser.parse_args()

    if args.mode == "features":
        build_features_for_dataset(
            dataset_name=args.dataset,
            window_size=args.window_size,
            step_size=args.step_size,
            min_label_fraction=args.min_label_fraction
        )
    elif args.mode == "train_v6":
        train_v6_for_dataset(
            dataset_name=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            window_size=args.window_size,
            step_size=args.step_size,
            min_label_fraction=args.min_label_fraction,
            min_label_count=args.min_label_count
        )

if __name__ == "__main__":
    main()
