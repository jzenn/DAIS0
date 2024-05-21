import pandas as pd
import torch


class IonosphereDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)

        assert self.data.shape[1] == 35

        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(
            self.data.iloc[:, -1].map({"g": 1, "b": 0}).values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
