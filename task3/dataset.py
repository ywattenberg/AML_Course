import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import numpy as np
from data_aug import augment_sample


class HeartDataset(Dataset):
    def __init__(self, data, path=None, transform=None, device=None, unpack_frames=False):

        if path is None:
            self.data = data
        else:
            self.data = np.load(path, allow_pickle=True)
            # self.data = utils.load_zipped_pickle(path)

        if transform is None:
            self.transform = utils.get_transforms()
        else:
            self.transform = transform

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if unpack_frames:
            self.data = self.unpack_frames()

    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        unpacked_data = []
        for entry in data["arr_0"]:
            for frame in entry["frames"]:
                unpacked_data.append(
                    {
                        "name": entry["name"],
                        "frame": entry["nmf"][frame, :, :].numpy().astype(np.float64),
                        "frames": [frame],
                        "box": entry["box"],
                        "label": entry["label"][:, :, frame],
                        "dataset": entry["dataset"],
                    }
                )
        return unpacked_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_full_data=False):
        if return_full_data:
            return self.data[idx]
        else:
            item = self.data[idx]
            item = augment_sample(item)
            return (item["frame"], item["label"])
