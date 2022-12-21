import os
import numpy as np
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.transforms as transforms

import utils

TEST = False
DEVICE = "cuda"
IMAGE_SIZE = 512
REG_VAL = 1
SAVING_BATCHES = 4

if DEVICE == "cuda":
    from torch_functions import robust_nmf

    device = torch.device("cuda:0")
elif DEVICE == "cpu":
    from numpy_functions import robust_nmf

    device = torch.device("cpu")
elif DEVICE == "mps":
    from numpy_functions import robust_nmf

    device = torch.device("mps")


def augment_data(train_data, test_data):
    return train_data, test_data


def transform_data(data, img_size=IMAGE_SIZE):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
        ]
    )
    return transform(data)


def transform_label(label, img_size=IMAGE_SIZE):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
            ),
        ]
    )
    return transform(label)


def main():
    train_file = "data/train.pkl"
    test_file = "data/test.pkl"
    train_data = utils.load_zipped_pickle(train_file)
    test_data = utils.load_zipped_pickle(test_file)

    train_data, test_data = augment_data(train_data, test_data)

    # get maximum frames over all videos
    num_frames = np.zeros(len(train_data) + len(test_data), dtype=np.int32)
    for i in range(len(train_data)):
        l = train_data[i]["video"].shape[-1]
        num_frames[i] = l
    for i in range(len(test_data)):
        l = test_data[i]["video"].shape[-1]
        num_frames[i + len(train_data)] = l
    max_length = max(num_frames)
    print(f"max_length: {max_length}")

    # rnmf and padding
    for i in tqdm(range(len(train_data))):
        video = train_data[i]["video"]
        # resize all frames to match IMAGE_SIZE
        video = transform_data(video, img_size=IMAGE_SIZE).numpy()
        # resize label to also match IMAGE_SIZE
        label = transform_label(train_data[i]["label"], img_size=IMAGE_SIZE).numpy()

        frame_shape = video.shape[1:]
        orig_len = video.shape[0]
        # print(f"orig_len: {orig_len}")

        # all frames are flattened
        video = video.reshape(video.shape[0], -1)

        basis, coeff, outlier, obj = robust_nmf(
            data=video,
            rank=2,
            init="NMF",
            reg_val=REG_VAL,
            beta=1,
            max_iter=1000,
            sum_to_one=False,
            tol=1e-7,
            print_every=100,
        )

        nmf = outlier.reshape((orig_len, frame_shape[0], frame_shape[1])).cpu()

        train_data[i]["nmf"] = nmf
        train_data[i]["label"] = label
        train_data[i].pop("video")

        if TEST:
            print(f"reg_val: {REG_VAL}")
            video = train_data[0][f"padded_nmf"]
            print(video.shape)
            video = np.moveaxis(video, 0, -1)

            utils.produce_gif(video, "tmp_train.gif")

    print(f"saving train_data in {SAVING_BATCHES} batches...")
    for i in tqdm(range(0, SAVING_BATCHES)):
        start_idx = i * int(len(train_data) / SAVING_BATCHES)
        end_idx = (i + 1) * int(len(train_data) / SAVING_BATCHES)
        if end_idx > len(train_data):
            end_idx = len(train_data)

        np.savez(
            f"data/train_data_{REG_VAL}_{IMAGE_SIZE}_{i}.npz",
            train_data[start_idx:end_idx],
        )
    print(f"finished saving train_data")

    for i in tqdm(range(len(test_data))):
        video = test_data[i]["video"]

        # resize all frames to match IMAGE_SIZE
        video = transform_data(video, img_size=IMAGE_SIZE).numpy()

        frame_shape = video.shape[1:]
        orig_len = video.shape[0]
        # all frames are flattened
        video = video.reshape(video.shape[0], -1)

        basis, coeff, outlier, obj = robust_nmf(
            data=video,
            rank=2,
            init="NMF",
            reg_val=REG_VAL,
            beta=1,
            max_iter=1000,
            sum_to_one=False,
            tol=1e-7,
            print_every=100,
        )

        nmf = outlier.reshape((orig_len, frame_shape[0], frame_shape[1])).cpu()

        test_data[i]["nmf"] = nmf
        test_data[i].pop("video")

        if TEST:
            print(f"reg_val: {REG_VAL}")
            video = test_data[0][f"padded_nmf"]
            print(video.shape)

            video = np.moveaxis(video, 0, -1)

            utils.produce_gif(video, "tmp_test.gif")

    print(f"saving test_data in {SAVING_BATCHES} batches...")
    for i in tqdm(range(0, SAVING_BATCHES)):
        start_idx = i * int(len(test_data) / SAVING_BATCHES)
        end_idx = (i + 1) * int(len(test_data) / SAVING_BATCHES)
        if end_idx > len(test_data):
            end_idx = len(test_data)

        np.savez(
            f"data/test_data_{REG_VAL}_{IMAGE_SIZE}_{i}.npz",
            test_data[start_idx:end_idx],
        )
    print(f"saved test_data finished")


if __name__ == "__main__":
    main()
