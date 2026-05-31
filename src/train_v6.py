import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import OUTPUT_DIR
from .parser import parse_multiple
from .config import DATASETS
from .feature_engineering import build_window_features
from .label_mapping import apply_coarse_mapping
from .label_mapping_v31 import dataset_specific_post_mapping
from .v6_sequence_data import prepare_event_level_labels, build_sequences
from .v6_gru_model import EventGRUClassifier
from .v6_dataset import EventSequenceDataset
from .logger import get_logger

logger = get_logger("train_v6")

def time_split_sequences(data_dict, y, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    def slice_dict(d, a, b):
        return {k: v[a:b] if hasattr(v, "__len__") and k.startswith("X_") else v for k, v in d.items()}

    train_data = {k: v[:train_end] if k.startswith("X_") else v for k, v in data_dict.items()}
    val_data = {k: v[train_end:val_end] if k.startswith("X_") else v for k, v in data_dict.items()}
    test_data = {k: v[val_end:] if k.startswith("X_") else v for k, v in data_dict.items()}

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    return train_data, val_data, test_data, y_train, y_val, y_test

def main():
    logger.info("========== V6 EVENT-LEVEL GRU TRAINING STARTED ==========")

    dataset_name = "tulum2"  # start with strongest candidate
    seq_len = 50
    batch_size = 256
    epochs = 8
    lr = 1e-3

    events = parse_multiple(DATASETS)
    features = build_window_features(events, window_size="5min", step_size="1min", min_label_fraction=0.6)
    features = apply_coarse_mapping(features, label_col="label")
    features = dataset_specific_post_mapping(features)

    labeled_events = prepare_event_level_labels(events, features, dataset_name=dataset_name)
    if labeled_events.empty:
        raise ValueError("No event-level labels found")

    counts = labeled_events["label"].value_counts()
    labeled_events = labeled_events[labeled_events["label"].isin(counts[counts >= 50].index)].copy()

    seq_data = build_sequences(labeled_events, seq_len=seq_len)

    le = LabelEncoder()
    y_enc = le.fit_transform(seq_data["y"])

    train_data, val_data, test_data, y_train, y_val, y_test = time_split_sequences(seq_data, y_enc)

    train_ds = EventSequenceDataset(train_data, y_train)
    val_ds = EventSequenceDataset(val_data, y_val)
    test_ds = EventSequenceDataset(test_data, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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

    torch.save(model.state_dict(), OUTPUT_DIR / f"v6_gru_{dataset_name}.pt")
    joblib.dump(le, OUTPUT_DIR / f"v6_gru_label_encoder_{dataset_name}.pkl")

    with open(OUTPUT_DIR / f"v6_gru_report_{dataset_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Macro F1: {macro_f1}\n")
        f.write(report)

    logger.info(f"V6 GRU Accuracy={acc:.4f}, MacroF1={macro_f1:.4f}")
    logger.info("========== V6 EVENT-LEVEL GRU TRAINING FINISHED ==========")

if __name__ == "__main__":
    main()
