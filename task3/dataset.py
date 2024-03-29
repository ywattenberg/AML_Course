import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import numpy as np
from data_aug import augment_sample
from data_aug import augment_transfrom

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
            a = torch.Tensor(item["frame"]).to(self.device)
            b = torch.Tensor(item["label"]).to(self.device)
            dic = {"frame": a.unsqueeze(0), "label": b.unsqueeze(0)}
            
            res = augment_transfrom([dic], has_box=False)[0]
            return (res["frame"], res["label"])



class HeartTestDataset(Dataset):
    def __init__(
        self, path, n_batches=1, unpack_frames=False, return_full_data=False, device="cpu", interpol_size=0, focus_on_middle_frame=1
    ):
        # path without ending and without batch number
        # for file test_data_5_112_0.npz the path is test_data_5_112
        if interpol_size % 2 == 0 or type(interpol_size) != int:
            raise ValueError("interpol_size must be odd")


        self.return_full_data = return_full_data
        self.interpol_size = interpol_size
        self.focus_on_middle_frame = focus_on_middle_frame
        self.device = device

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



    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        unpacked_data = []
        for entry in data:
            frame = entry["nmf"].numpy().astype(np.float64)*255
            frame = torch.Tensor(frame).to(self.device)
            length = len(frame)
            
            half = self.interpol_size // 2
            begin = frame[0].repeat(half, 1, 1)
            end = frame[-1].repeat(half, 1, 1)
            frame = torch.cat([begin, frame, end])

            unpacked_data.append(
                {
                    "name": entry["name"],
                    "frame": frame,
                }
            )

        return unpacked_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        half = self.interpol_size // 2

        res = []
        name = self.data[idx]["name"]
        frame = self.data[idx]["frame"]
        for i in range(half, len(frame)-half):
            # item = augment_sample(item)
            res.append(frame[i-half:i+half+1, :, :])
        return (res, name)


class InterpolationSet(Dataset):
    def __init__(self, path, n_batches=1, unpack_frames=False, device="cpu", interpol_size=0, focus_on_middle_frame=1):
        # path without ending and without batch number
        # for file test_data_5_112_0.npz the path is test_data_5_112
        if interpol_size % 2 == 0 or type(interpol_size) != int:
            raise ValueError("interpol_size must be odd")

        self.interpol_size = interpol_size
        self.focus_on_middle_frame = focus_on_middle_frame

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
        self.device = device

        if unpack_frames:
            self.data = self.unpack_frames()
        
        



    def unpack_frames(self, data=None):
        if data is None:
            data = self.data

        half = self.interpol_size // 2

        unpacked_data = []
        for vid in data:
            for labeled_frame in vid["frames"]:
                # print(labeled_frame)
                stacked_frames = []
                # print(vid["nmf"].shape)
                for i in range(half):
                    if labeled_frame - half + i < 0:
                        stacked_frames.append(vid["nmf"][0, :, :])
                    else:
                        stacked_frames.append(vid["nmf"][labeled_frame - half + i, :, :])
                    
                    # print(len(vid["nmf"][:, 0, 0]))
                    if labeled_frame + half - i >= len(vid["nmf"][:, 0, 0]):
                        stacked_frames.append(vid["nmf"][-1, :, :])
                    else:
                        stacked_frames.append(vid["nmf"][labeled_frame + half - i, :, :])
                
                for i in range(self.focus_on_middle_frame):
                    stacked_frames.append(vid["nmf"][labeled_frame])

                stacked_frames = torch.stack(stacked_frames, dim=0) * 255
                stacked_frames = stacked_frames.type(torch.uint8)
                stacked_frames = stacked_frames.unsqueeze(1)
            
                
                unpacked_data.append(
                    {
                        "name": vid["name"],
                        "frame": stacked_frames.to(self.device),
                        "label": torch.tensor(vid["label"][labeled_frame]).unsqueeze(0).to(self.device),
                        "box": vid["box"]
                    }
                )
                    

        return unpacked_data
        


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx, return_full_data=False):
        if return_full_data:
            return self.data[idx]
        else:
            tmp = {
                "frame": self.data[idx]["frame"].clone(),
                "label": self.data[idx]["label"].clone(),
            }
            
            tmp = augment_transfrom([tmp], has_box=False, is_batched=True)[0]
            tmp["frame"] = tmp["frame"].squeeze(1)
            # print(tmp["frame"].shape)
            return (tmp["frame"].to(self.device), tmp["label"].to(self.device))


            # item["frame"] = torch.Tensor(item["frame"]).to(self.device)
            # item["label"] = torch.Tensor(item["label"]).to(self.device)
            # return (item["frame"], item["label"], box)

    
    