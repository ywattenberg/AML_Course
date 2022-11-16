import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch


class TrainDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = pd.read_csv(feature_path)
        self.labels = pd.read_csv(label_path)
        # print(self.labels)

    def __len__(self):
        return len(self.labels)

    def get_num_of_features(self):
        return len(self.features.columns)

    def __getitem__(self, idx):
        data = torch.tensor(self.features.iloc[idx], dtype=torch.float)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)
        return (data, label)
