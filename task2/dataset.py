import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, feature_path, label_path, device="cpu"):
        self.device = device
        if (type(feature_path) == str):
            self.features = pd.read_csv(feature_path)
        else:
            self.features = feature_path
        
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        self.features = pd.DataFrame(imputer.fit_transform(self.features))

        if (type(label_path) == str):
            self.labels = pd.read_csv(label_path).iloc[:, [1]]
        else:
            self.labels = label_path

    def __len__(self):
        return len(self.labels)

    def get_num_of_features(self):
        return len(self.features.columns)

    def __getitem__(self, idx):
        data = torch.tensor(self.features.iloc[idx], dtype=torch.float).to(self.device)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float).to(self.device)
        return (data, label)
