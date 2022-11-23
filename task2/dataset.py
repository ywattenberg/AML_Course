import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.impute import SimpleImputer


class TrainDataset(Dataset):
    def __init__(self, feature_path, label_path, device="cpu"):
        self.device = device
        self.features = pd.read_csv(feature_path).iloc[:, 1:]
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        self.features = pd.DataFrame(imputer.fit_transform(self.features))

        self.labels = pd.read_csv(label_path).iloc[:, 1:]

    def __len__(self):
        return len(self.labels)

    def get_num_of_features(self):
        return len(self.features.columns)

    def __getitem__(self, idx):
        data = torch.tensor(self.features.iloc[idx], dtype=torch.float).to(self.device)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float).to(self.device)
        return (data, label)
