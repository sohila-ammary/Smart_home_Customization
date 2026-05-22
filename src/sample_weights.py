import numpy as np
from .logger import get_logger

logger = get_logger("sample_weights")

def inverse_frequency_sample_weights(y):
    values, counts = np.unique(y, return_counts=True)
    freq = dict(zip(values, counts))
    weights = np.array([1.0 / freq[v] for v in y], dtype=float)
    weights = weights / weights.mean()
    logger.info("Generated inverse-frequency sample weights")
    return weights
