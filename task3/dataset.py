import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils


class HeartDataset(Dataset):
    def __init__(
        self, data, path=None, transform=None, device=None, unpack_frames=False
    ):

        if path is None:
            self.data = data
        else:
            self.data = utils.load_zipped_pickle(path)

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
        for entry in data:
            for frame in entry["frames"]:
                unpacked_data.append(
                    {
                        "name": entry["name"],
                        "video": entry["video"][:, :, frame],
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
            #print(type(item["video"]))
            #print(item["label"])
            pic = utils.transform_data(item["video"])
            lab = utils.transform_label(item["label"])
            return (pic, lab)
