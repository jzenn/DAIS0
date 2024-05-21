import numpy as np
import pandas as pd
import torch


class LoanDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)

        data = data.dropna()
        data = data.drop(columns="Loan_ID")

        data["Married"] = data["Married"].apply(lambda x: int(x == "Yes"))
        data["Self_Employed"] = data["Self_Employed"].apply(lambda x: int(x == "Yes"))
        data["Education"] = data["Education"].apply(lambda x: int(x == "Graduate"))
        data["Gender"] = data["Gender"].apply(lambda x: int(x == "Female"))
        data["Dependents"] = data["Dependents"].apply(
            lambda x: x if len(x) == 1 else x[0]
        )
        data["Property_Area"] = data["Property_Area"].apply(
            lambda x: {"Rural": 1, "Semiurban": 0.5, "Urban": 0}[x]
        )

        data = data.astype({"Dependents": float})

        columns_to_rescale = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ]
        data[columns_to_rescale] -= data[columns_to_rescale].min()
        data[columns_to_rescale] /= data[columns_to_rescale].max()

        X = data.values[:, :-1].astype(np.float32)
        y = data.values[:, -1]
        y = np.where(y == "N", 0, 1)

        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), torch.tensor(self.labels[index])
