import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .logger import get_logger

logger = get_logger("plots")

def save_confusion_matrix(y_true, y_pred, labels, out_path, title="Confusion Matrix", max_labels=25):
    logger.info(f"Saving confusion matrix to {out_path}")

    # show only top labels if too many
    if len(labels) > max_labels:
        labels = labels[:max_labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_top_class_barplot(series, out_path, title, top_n=20):
    logger.info(f"Saving class distribution plot to {out_path}")
    s = series.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=s.values, y=s.index)
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
