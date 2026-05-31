from .config import DATASETS

def select_datasets(dataset_names=None):
    if dataset_names is None:
        return DATASETS

    selected = {}
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        selected[name] = DATASETS[name]
    return selected
