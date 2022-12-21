import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import numpy as np
from data_aug import augment_sample, augment_transfrom

BOX_SHAPE = (256, 256)


class BoxDataset(Dataset):
    def __init__(
        self, path, n_batches=1, unpack_frames=False, device="cpu", test=False, stride=1
    ):
        # path without ending and without batch number
        # for file test_data_5_112_0.npz the path is test_data_5_112

        if n_batches > 1:
            for i in range(n_batches):
                if i == 0:
                    self.data = np.load(f"{path}_{i}.npz", allow_pickle=True)["arr_0"]
                else:
                    self.data = np.concatenate(
                        (
                            self.data,
                            np.load(f"{path}_{i}.npz", allow_pickle=True)["arr_0"],
                        )
                    )
        else:
            self.data = np.load(f"{path}_{0}.npz", allow_pickle=True)["arr_0"]
        if not test:
            self.data = self.data[6:]
        else:
            self.data = self.data[0:6]
        # self.data = utils.load_zipped_pickle(path)
        self.device = device

        if unpack_frames:
            self.data = self.unpack_frames(stride=stride)

        self.device = device

    def unpack_frames(self, data=None, stride=1):
        if data is None:
            data = self.data

        unpacked_data = []
        for entry in data:
            number_of_frames = entry["nmf"].shape[0]
            for i in range(0, number_of_frames, stride):
                box_transformed = self.transform_box(BOX_SHAPE, entry["box"])
                frame = entry["nmf"][i, :, :].numpy().astype(np.float64) * 255
                frame = torch.Tensor(
                    frame.astype(np.uint8),
                ).to(self.device)
                unpacked_data.append(
                    {
                        "name": entry["name"],
                        "frame": frame,
                        "frames": [i],
                        "box": torch.Tensor(box_transformed).to(self.device).int(),
                        "box_coordinates": self.get_coordinates_box(box_transformed),
                        "dataset": entry["dataset"],
                    }
                )
        return unpacked_data

    def transform_box(self, size, box):
        x, y = size
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((x, y)),
            ]
        )

        return transform(box)

    def get_coordinates_box(self, box):
        box = box.squeeze()
        rows, cols = np.where(box == 1)
        r_min, r_max, c_min, c_max = (
            np.min(rows),
            np.max(rows),
            np.min(cols),
            np.max(cols),
        )
        return [(r_min, c_min), (r_max, c_max)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_full_data=False):
        if return_full_data:
            return self.data[idx]
        else:
            tmp = augment_transfrom(self.data[idx], has_label=False)
            return tmp["frame"], tmp["box"]
