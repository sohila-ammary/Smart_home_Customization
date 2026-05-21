import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .logger import get_logger

logger = get_logger("lstm")

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len-1], dtype=torch.long)
        )

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def train_lstm(
    X_train, y_train, X_test, y_test,
    seq_len=30,
    hidden_size=128,
    num_layers=2,
    batch_size=256,
    epochs=8,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training LSTM on device={device}")

    train_ds = SequenceDataset(X_train, y_train, seq_len)
    test_ds = SequenceDataset(X_test, y_test, seq_len)

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Sequence dataset is empty. Reduce sequence length or check input sizes.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(
        input_dim=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(np.unique(y_train))
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"LSTM epoch {epoch}/{epochs} - loss={total_loss / len(train_loader):.4f}")

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(yb.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "report": classification_report(targets, preds, zero_division=0)
    }

    logger.info(f"LSTM Accuracy={metrics['accuracy']:.4f}, MacroF1={metrics['macro_f1']:.4f}")
    return model, metrics, np.array(targets), np.array(preds)
