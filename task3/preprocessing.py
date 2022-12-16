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
IMAGE_SIZE = 256
REG_VAL = 15

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
            transforms.Resize((img_size, img_size)),
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

        nmf = outlier.reshape((orig_len, frame_shape[0], frame_shape[1]))

        padded_nmf = np.zeros((max_length, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

        padded_nmf[: nmf.shape[0]] = nmf
        train_data[i]["padded_nmf"] = padded_nmf
        train_data[i][f"nmf"] = nmf

        if TEST:
            print(f"reg_val: {REG_VAL}")
            video = train_data[0][f"padded_nmf"]
            print(video.shape)
            if DEVICE == "cuda":
                video = video.permute(1, 2, 0).cpu().numpy()
            elif DEVICE == "cpu":
                video = np.moveaxis(video, 0, -1)

            fig = plt.figure(frameon=False)
            # plt.margins(0,0)
            im = plt.imshow(video[:, :, 0], animated=True, cmap=plt.cm.bone)

            def fig_update(i):
                i = i % video.shape[0]
                im.set_array(video[:, :, i])
                return [im]

            anim = animation.FuncAnimation(
                fig,
                fig_update,
                frames=video.shape[-1],
            )
            anim.save("img/tmp.gif", fps=20)

        utils.save_zipped_pickle(train_data[:i], f"data/train_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl")

    utils.save_zipped_pickle(train_data, f"data/train_data_{REG_VAL}_{IMAGE_SIZE}.pkl")
    if os.path.exists(f"data/train_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl"):
        os.remove(f"data/train_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl")

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

        nmf = outlier.reshape((orig_len, frame_shape[0], frame_shape[1]))

        padded_nmf = np.zeros((max_length, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)

        padded_nmf[: nmf.shape[0]] = nmf
        test_data[i]["padded_nmf"] = padded_nmf
        test_data[i][f"nmf"] = nmf

        if TEST:
            print(f"reg_val: {REG_VAL}")
            video = test_data[0][f"padded_nmf"]
            print(video.shape)

            if DEVICE == "cuda":
                video = video.permute(1, 2, 0).cpu().numpy()
            elif DEVICE == "cpu":
                video = np.moveaxis(video, 0, -1)

            fig = plt.figure(frameon=False)
            # plt.margins(0,0)
            im = plt.imshow(video[:, :, 0], animated=True, cmap=plt.cm.bone)

            def fig_update(i):
                i = i % video.shape[0]
                im.set_array(video[:, :, i])
                return [im]

            anim = animation.FuncAnimation(
                fig,
                fig_update,
                frames=video.shape[-1],
            )
            anim.save("img/tmp.gif", fps=20)

        utils.save_zipped_pickle(test_data[:i], f"data/test_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl")

    utils.save_zipped_pickle(test_data, f"data/test_data_{REG_VAL}_{IMAGE_SIZE}.pkl")
    if os.path.exists(f"data/test_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl"):
        os.remove(f"data/test_data_{REG_VAL}_{IMAGE_SIZE}_tmp.pkl")


if __name__ == "__main__":
    main()
