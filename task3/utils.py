import pickle
import gzip
import numpy as np
import os
import torchvision.transforms as transforms
import torch
import PIL
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import convolve
import robust_nfm
import torch_functions
import cv2
from segmentation_mask_overlay import overlay_masks


warnings.filterwarnings("ignore")


def post_process_mask(mask, size, erode_it=1, dilate_it=1):
    T = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    tmp = T(mask).permute(1, 2, 0).cpu().detach().numpy()
    tmp = np.where(tmp > 0.6, 255, 0).astype(np.uint8)
    kernel = np.ones((6, 6), np.uint8)
    tmp = cv2.erode(tmp, kernel, iterations=erode_it)
    tmp = cv2.dilate(tmp, kernel, iterations=dilate_it)
    tmp = tmp > 200
    return tmp


def get_transforms():
    return transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8)),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def transform_data(data):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            # transforms.Normalize(
            #     mean=[0.44531356896770125],
            #     std=[0.2692461874154524],
            # ),
        ]
    )
    normalized_img = transform(data / 255)
    return normalized_img


def transform_label(label, width=256, height=256):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (height, width), interpolation=transforms.InterpolationMode.NEAREST
            ),
        ]
    )
    normalized_img = transform(label)
    return normalized_img


def load_zipped_pickle(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f, 2)


def test_pred():
    # load data
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    samples = load_zipped_pickle("sample.pkl")
    # make prediction for test
    predictions = []
    for d in test_data:
        prediction = np.array(np.zeros_like(d["video"]), dtype=np.bool)
        height = prediction.shape[0]
        width = prediction.shape[1]
        prediction[
            int(height / 2) - 50 : int(height / 2 + 50),
            int(width / 2) - 50 : int(width / 2 + 50),
        ] = True

        # DATA Strucure
        predictions.append({"name": d["name"], "prediction": prediction})
        # save in correct format

    save_zipped_pickle(predictions, "my_predictions.pkl")


# function to produce a gif from a numpy array
# inputs:
#   data: numpy array of shape (height, width, frames)
#   name: name of the gif
def produce_gif(data, name, is_int=False):
    if not is_int:
        data *= 255
    data = data.astype(np.uint8)
    with imageio.get_writer(name, mode="I") as writer:
        for i in range(data.shape[2]):
            image = data[:, :, i]
            writer.append_data(image)

def produce_gif_colour(data, name):
    with imageio.get_writer(name, mode="I") as writer:
        for i in range(data.shape[0]):
            image = data[i, :, :, :]
            writer.append_data(image)


# function to apply NMF to a video
# video: numpy array of shape (height, width, frames)
# components: number of components to use in NMF
# method: possible options ‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’, default: ‘nndsvd’
# returns: numpy array of shape (height, width, frames) with the NMF applied
def apply_NMF(video, components, method="nndsvd"):
    model = NMF(n_components=components, init=method, random_state=0)
    W = model.fit_transform(video.reshape(-1, video.shape[2]))
    H = model.components_

    # print(W.shape)
    # print(H.shape)
    # print(W)
    # print(H)
    # return W, H
    return (W @ H).reshape(video.shape)


def apply_PCA(video, components):
    scaler = StandardScaler()
    video = scaler.fit_transform(video.reshape(video.shape[2], -1))
    print(video.shape)
    pca = PCA(n_components=components)
    W = pca.fit_transform(video)
    W_inverted = pca.inverse_transform(W)

    return W_inverted


def gaussian_filter(shape, sigma):
    """
    Returns a 2D gaussian filter specified by its shape and standard deviation.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# function to apply a high pass filter to a video
# video: numpy array of shape (height, width, frames)
# shape: shape of the gaussian filter
# sigma: standard deviation of the gaussian filter
def high_pass_filter(video, dimensions, sigma):
    x, y = dimensions
    gauss_filter = gaussian_filter((x, y), sigma)
    unary_impulse = np.zeros((x, y))
    unary_impulse[int(x / 2), int(y / 2)] = 1
    high_pass_filter = unary_impulse - gauss_filter

    # apply filter to each frame
    for i in range(video.shape[2]):
        video[:, :, i] = convolve(video[:, :, i], high_pass_filter)

    return video


def modify_data_with_rnfm(filename, regularization, reg_parameter, new_filename):
    # load data
    train_data = load_zipped_pickle(f"data/{filename}.pkl")

    # apply NMF to train data
    for i in range(len(train_data)):
        video_prep = transform_data(train_data[i]["video"]).permute(1, 2, 0).numpy()
        print(video_prep.shape)
        video_prep = video_prep.reshape(-1, video_prep.shape[2])
        print(video_prep.shape)
        # video_prep = torch.Tensor(video_prep)
        _, _, tmp, _ = robust_nfm.robust_nmf(
            data=video_prep,
            rank=2,
            max_iter=1000,
            tol=1e-7,
            beta=regularization,
            init="NMF",
            reg_val=reg_parameter,
            sum_to_one=False,
        )
        print(tmp)
        print(tmp.shape)
        print(video_prep.shape)
        train_data[i]["video"] = tmp.reshape(256, 256, video_prep.shape[1])
        produce_gif(train_data[i]["video"], "tmp.gif")

        labels = train_data[i]["label"]
        train_data[i]["label"] = transform_label(labels)

        save_zipped_pickle(train_data[:i], f"data/{new_filename}_tmp.pkl")

    save_zipped_pickle(train_data, f"data/{new_filename}.pkl")
    # apply NMF to test data


def get_windows(frame, window_size, stride):
    # data: numpy array of shape (height, width, frames)
    # window_size: size of the window
    # stride: stride of the window
    # returns: numpy array of shape (height, width, frames, windows)
    windows = []
    x,y = window_size
    stride_x, stride_y = stride

    for i in range(0, frame.shape[0] - x, stride_x):
        for j in range(0, frame.shape[1] - y, stride_y):
            windows.append(frame[i : i + window_size, j : j + window_size, :])
    
    return np.stack(windows, axis=2)




def find_roi(frame, window_size, stride):
    # windows = get_windows(frame, window_size, stride)
    x,y = window_size
    stride_x, stride_y = stride

    max_norm = 0
    max_window = None
    max_region = None
    for i in range(0, frame.shape[0] - x, stride_x):
        for j in range(0, frame.shape[1] - y, stride_y):
            tmp = frame[i : i + x, j : j + y]
            fb_norm = np.linalg.norm(tmp, ord='fro')

            if fb_norm > max_norm:
                max_norm = fb_norm
                max_window = (i, j, i + x, j + y)
                max_region = tmp
    
    # plt.imshow(max_region, cmap='gray')
    # plt.show()

    return max_window, max_region


def overlay_segmentation(frame, segmentation, filename, box=None, true_label=None):
    masks = []
    masks.append(segmentation)
    mask_labels = ["segmentation"]
    layers = 1
    if box is not None:
        layers += 1
        masks.append(box)
        mask_labels.append("box")
    
    if true_label is not None:
        layers += 1
        masks.append(true_label)
        mask_labels.append("true label")
    
    cmap = plt.cm.tab20(np.arange(layers))
    
    fig = overlay_masks(frame, masks, colors=cmap, mask_alpha=0.5)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

    return frame
