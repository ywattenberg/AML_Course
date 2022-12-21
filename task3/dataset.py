import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import numpy as np
from data_aug import augment_sample

BOX_SHAPE = (256, 256)


class HeartDataset(Dataset):
    def __init__(self, path, n_batches=1, unpack_frames=False, device="cpu"):
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
        # self.data = utils.load_zipped_pickle(path)

        if unpack_frames:
            self.data = self.unpack_frames()

        self.device = device

    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        unpacked_data = []
        for entry in data:
            for frame in entry["frames"]:
                unpacked_data.append(
                    {
                        "name": entry["name"],
                        "frame": entry["nmf"][frame, :, :].numpy().astype(np.float64),
                        "frames": [frame],
                        "box": self.transform_box(BOX_SHAPE, entry["box"]),
                        "label": entry["label"][frame, :, :],
                        "dataset": entry["dataset"],
                    }
                )
        return unpacked_data

    def __len__(self):
        return len(self.data)

    def transform_box(self, size, box):
        x, y = size
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((x, y)),
            ]
        )

        return transform(box)

    def __getitem__(self, idx, return_full_data=False):
        if return_full_data:
            return self.data[idx]
        else:
            item = self.data[idx]
            box = item["box"].to(self.device)
            item = augment_sample(item)
            item["frame"] = torch.Tensor(item["frame"]).to(self.device)
            item["label"] = torch.Tensor(item["label"]).to(self.device)
            return (item["frame"], item["label"], box)


class InterpolatedHeartDataset(HeartDataset):
    def __getitem__(self, idx):
        if idx == 0:
            items = [
                self.data[idx],
                self.data[idx],
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
            ]
        if idx == 1:
            items = [
                self.data[idx - 1],
                self.data[idx - 1],
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
            ]
        elif idx == len(self.data) - 1:
            items = [
                self.data[idx - 2],
                self.data[idx - 1],
                self.data[idx],
                self.data[idx],
                self.data[idx],
            ]
        elif idx == len(self.data) - 2:
            items = [
                self.data[idx - 2],
                self.data[idx - 1],
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 1],
            ]
        else:
            items = [
                self.data[idx - 2],
                self.data[idx - 1],
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
            ]

        frames = torch.Tensor(5, 256, 256)

        for i in range(len(items)):
            items[i] = augment_sample(items[i], augment_label=False)
            frames[i] = torch.Tensor(items[i]["frame"])

        label = torch.Tensor(self.data[idx]["label"])
        label = label.unsqueeze(1).permute(1, 0, 2)
        frames = frames.to(self.device)
        label = label.to(self.device)
        return (frames, label, torch.Tensor())


class HeartTestDataset(Dataset):
    def __init__(
        self, path, n_batches=1, unpack_frames=False, return_full_data=False, device="cpu"
    ):
        # path without ending and without batch number
        # for file test_data_5_112_0.npz the path is test_data_5_112
        self.return_full_data = return_full_data
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
        # self.data = utils.load_zipped_pickle(path)

        if unpack_frames:
            self.data = self.unpack_frames()

        self.device = device

    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        unpacked_data = []
        for entry in data:
            for frame in range(len(entry["nmf"][:, 0, 0])):
                unpacked_data.append(
                    {
                        "name": entry["name"],
                        "frame": entry["nmf"][frame, :, :].numpy().astype(np.float64),
                    }
                )
        return unpacked_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_full_data:
            return self.data[idx]
        else:
            item = self.data[idx]
            # item = augment_sample(item)
            item["frame"] = torch.Tensor(item["frame"]).to(self.device)
            return (item["frame"].unsqueeze(0), None)


class InterpolationSet(HeartDataset):
    def __init__(self, path, n_batches=1, unpack_frames=False, device="cpu", interpol_size=0, focus_on_middle_frame=1):
        # path without ending and without batch number
        # for file test_data_5_112_0.npz the path is test_data_5_112
        if interpol_size % 2 == 0 or type(interpol_size) != int:
            raise ValueError("interpol_size must be odd")

        super.__init__(path, n_batches, unpack_frames, device)
        self.interpol_size = interpol_size
        self.focus_on_middle_frame = focus_on_middle_frame


    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        half = self.interpol_size // 2

        unpacked_data = []
        for vid in data:
            for labeled_frame in vid["frames"]:
                stacked_frames = []
                for i in range(half):
                    stacked_frames.append(vid["video"][labeled_frame - half + i])
                    stacked_frames.append(vid["video"][labeled_frame + half - i])
                
                for i in range(self.focus_on_middle_frame):
                    stacked_frames.append(vid["video"][labeled_frame])

                print(stacked_frames.shape)
                unpacked_data.append(
                    {
                        "name": vid["name"],
                        "frame_sequence": stacked_frames,
                        "label": vid["labels"][labeled_frame],
                        "box": vid["boxes"][labeled_frame]
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
            print(item["frame_sequence"].shape)


            # item["frame"] = torch.Tensor(item["frame"]).to(self.device)
            # item["label"] = torch.Tensor(item["label"]).to(self.device)
            # return (item["frame"], item["label"], box)

    
    