import numpy as np
import pandas as pd
import torch


class SonarDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)

        assert data.shape[1] == 61

        X = data.values[:, :-1].astype(np.float32)
        y = data.values[:, -1]
        y = np.where(y == "M", 0, 1)

        self.features = torch.from_numpy(X)
        self.labels = torch.from_numpy(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
