import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .logger import get_logger

logger = get_logger("metrics_ext")

def top_k_accuracy_from_proba(y_true_enc, proba, k=3):
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = [(yt in row) for yt, row in zip(y_true_enc, topk)]
    return float(np.mean(hits))

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, zero_division=0)
    }
