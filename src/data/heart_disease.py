import numpy as np
import pandas as pd
import torch


class HeartDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)

        data = data.dropna()

        columns_to_rescale = [
            "male",
            "age",
            "education",
            "currentSmoker",
            "cigsPerDay",
            "BPMeds",
            "prevalentStroke",
            "prevalentHyp",
            "diabetes",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose",
        ]
        data[columns_to_rescale] -= data[columns_to_rescale].min()
        data[columns_to_rescale] /= data[columns_to_rescale].max()

        X = data.values[:, :-1].astype(np.float32)
        y = data.values[:, -1]

        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), torch.tensor(self.labels[index])
