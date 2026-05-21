from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

WINDOW_SIZE = "5min"
STEP_SIZE = "1min"

RANDOM_STATE = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_CLASS_SAMPLES = 20

SEQUENCE_LENGTH = 30
LSTM_BATCH_SIZE = 256
LSTM_EPOCHS = 8
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_LEARNING_RATE = 1e-3

DATASETS = {
    "aruba": DATA_DIR / "aruba.txt",
    "cairo": DATA_DIR / "cairo.txt",
    "milan": DATA_DIR / "milan.txt",
    "tulum1": DATA_DIR / "tulum1.txt",
    "tulum2": DATA_DIR / "tulum2.txt",
}
