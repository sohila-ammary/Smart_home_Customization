import torch
from torch.utils.data import Dataset

class EventSequenceDataset(Dataset):
    def __init__(self, data_dict, y_encoded):
        self.X_sensor = torch.tensor(data_dict["X_sensor"], dtype=torch.long)
        self.X_type = torch.tensor(data_dict["X_type"], dtype=torch.long)
        self.X_value = torch.tensor(data_dict["X_value"], dtype=torch.long)
        self.X_delta = torch.tensor(data_dict["X_delta"], dtype=torch.float32)
        self.X_hour = torch.tensor(data_dict["X_hour"], dtype=torch.long)
        self.y = torch.tensor(y_encoded, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_sensor[idx],
            self.X_type[idx],
            self.X_value[idx],
            self.X_delta[idx],
            self.X_hour[idx],
            self.y[idx]
        )
