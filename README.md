# CASAS Smart Home MVP

This project builds a minimum viable AI system for:

1. **Activity recognition** from CASAS smart-home sensor streams
2. **Routine mining**
3. **Personalized smart-home scenario recommendations**

## Supported datasets

- Aruba
- Cairo
- Milan
- Tulum1
- Tulum2

## Expected data format

Place the raw dataset files in:

```text
data/raw/
```

Expected file names:

```text
aruba.txt
cairo.txt
milan.txt
tulum1.txt
tulum2.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m src.train
```

## Outputs

Generated inside `outputs/`:

- `random_forest.pkl`
- `boosting.pkl`
- `label_encoder.pkl`
- `window_features.csv`
- `metrics.txt`

## How it works

### 1. Parsing
The parser reads CASAS event logs with variants like:

- timestamp + sensor + ON/OFF
- timestamp + sensor + numeric temperature
- optional labels like `Sleeping begin`

### 2. Weak label extraction
Activity labels are converted into time intervals using `begin` / `end`.

### 3. Sliding window features
The system builds 5-minute windows with 1-minute stride and extracts:

- number of events
- per-sensor counts
- motion counts
- temperature stats
- temporal features

### 4. Models
- Random Forest
- XGBoost if available, otherwise Gradient Boosting

### 5. Recommendations
The recommendation layer maps predicted activity + context to suggested smart-home scenarios.

## Notes

- Time-based split is used to reduce leakage.
- If accuracy is weak, next upgrades should be:
  1. better label alignment
  2. richer temporal features
  3. sequence models (LSTM/Transformer)
  4. per-home personalization