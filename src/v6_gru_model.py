import torch
import torch.nn as nn

class EventGRUClassifier(nn.Module):
    def __init__(
        self,
        num_sensors,
        num_types,
        num_values,
        num_hours,
        num_classes,
        sensor_emb_dim=32,
        type_emb_dim=8,
        value_emb_dim=4,
        hour_emb_dim=8,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.sensor_emb = nn.Embedding(num_sensors + 1, sensor_emb_dim, padding_idx=0)
        self.type_emb = nn.Embedding(num_types + 1, type_emb_dim, padding_idx=0)
        self.value_emb = nn.Embedding(num_values + 1, value_emb_dim, padding_idx=0)
        self.hour_emb = nn.Embedding(num_hours + 1, hour_emb_dim, padding_idx=0)

        input_dim = sensor_emb_dim + type_emb_dim + value_emb_dim + hour_emb_dim + 1

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_sensor, x_type, x_value, x_delta, x_hour):
        s = self.sensor_emb(x_sensor)
        t = self.type_emb(x_type)
        v = self.value_emb(x_value)
        h = self.hour_emb(x_hour)

        x_delta = x_delta.unsqueeze(-1)
        x = torch.cat([s, t, v, h, x_delta], dim=-1)

        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits
