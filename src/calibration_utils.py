import joblib
from sklearn.calibration import CalibratedClassifierCV
from .logger import get_logger

logger = get_logger("calibration_utils")

def calibrate_model(model, X_val, y_val, method="sigmoid"):
    logger.info(f"Calibrating model with method={method}")
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated

def save_model(model, path):
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")
