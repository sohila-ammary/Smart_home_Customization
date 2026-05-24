import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .logger import get_logger

logger = get_logger("sanity_checks")

def shuffled_label_test(model_builder, X_train, y_train, X_test, y_test, random_state=42):
    logger.info("Running shuffled-label sanity test")

    rng = np.random.default_rng(random_state)
    y_shuffled = y_train.copy().to_numpy()
    rng.shuffle(y_shuffled)

    model = model_builder()
    model.fit(X_train, y_shuffled)
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    }
