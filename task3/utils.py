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

warnings.filterwarnings("ignore")


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


def transform_label(label):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
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


def train_loop(model, train_loader, loss_fn, optimizer):

    for batch, (x, y) in enumerate(train_loader):
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 10) == 0:
            print(f"Batch: {batch}, Loss: {loss.item():.4f}")
            break


def test_loop(model, test_loader, loss_fn, epoch):
    size = len(test_loader.dataset)
    test_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += loss_fn(output, y).item()

    test_loss /= size
    print(f"Test Error: {test_loss:>8f} \n")

    print(y.shape)

    output = (output > 0.6).float()
    produce_gif(output[0].permute(1, 2, 0).detach().numpy(), f"img/output_{epoch}.gif")
    produce_gif(y[0].permute(1, 2, 0).int().detach().numpy(), f"img//label_{epoch}.gif")
    # print picture
    # r = np.random.randint(0, len(test_loader))
    # x, y = test_loader.dataset[r]

    # with torch.no_grad():
    #     output = model(x.unsqueeze(0))
    # plt.imshow(output.permute(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(y.permute(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    # plt.show()


# function to produce a gif from a numpy array
# inputs:
#   data: numpy array of shape (height, width, frames)
#   name: name of the gif
def produce_gif(data, name):
    with imageio.get_writer(name, mode="I") as writer:
        for i in range(data.shape[2]):
            image = data[:, :, i]
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

    save_zipped_pickle(train_data, f"data/{new_filename}.pkl")
    # apply NMF to test data
