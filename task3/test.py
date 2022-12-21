import utils
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from torch.nn import Conv2d
from PIL import Image, ImageFilter
from matplotlib import cm
from scipy.ndimage import convolve
import robust_nfm
import os
import cv2
import dataset

import warnings

warnings.filterwarnings("always")


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


# sub = utils.load_zipped_pickle("submission_256_1_200.pkl")
# print(sub)

# data = utils.load_zipped_pickle("data/train.pkl")

data_interpol = dataset.InterpolationSet(
    path=f"data/train_data_{1}_{256}",
    n_batches=2,
    unpack_frames=True,
    device="gpu",
    interpol_size=11,
    focus_on_middle_frame=1,
)
for i in range(10):
    tmp = (
        torch.concat(
            [data_interpol[i][0], data_interpol[i][1].repeat(11, 1, 1, 1)],
            dim=2,
        )
        .squeeze()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    utils.produce_gif(tmp, f"pred_img/pred_{i}.gif")


# folder = "results/0"

# images = []
# for filename in os.listdir(folder):
#     if not filename.startswith("."):
#         img = cv2.imread(os.path.join(folder,filename))
#         images.append(img)

# # print(np.array(images).shape)
# images = np.array(images)
# print(images.shape)

# utils.produce_gif_colour(images, "sub.gif")

quit()

# for i in range(1):
#     fig = plt.figure(figsize=(10, 10))
#     pic = data[i]['video'][:,:,0]/255
#     fig.add_subplot(2, 3, 1)
#     plt.imshow(pic, cmap='gray')


#     gauss = gaussian_filter((5,5),3)
#     unary_impulse = np.zeros((5,5))
#     unary_impulse[2,2] = 1
#     high_pass = unary_impulse - gauss
#     sobel_vert = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#     sobel_horz = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

#     pic_filtered = convolve(pic, gauss)
#     pic_sobel_vert = convolve(pic, sobel_vert)
#     pic_sobel_horz = convolve(pic, sobel_horz)
#     pic_sobel = np.sqrt(np.square(pic_sobel_vert) + np.square(pic_sobel_horz**2))
#     pic_sobel *= 255.0 / pic_sobel.max()

#     fig.add_subplot(2, 3, 2)
#     plt.imshow(pic_filtered, cmap='gray')


#     pic_filtered = convolve(pic, high_pass)
#     fig.add_subplot(2, 3, 3)
#     plt.imshow(pic_filtered, cmap='gray')


#     pic_sobel_vert = convolve(pic, sobel_vert)
#     fig.add_subplot(2, 3, 4)
#     plt.imshow(pic_sobel_vert, cmap='gray')

#     pic_sobel_horz = convolve(pic, sobel_horz)
#     fig.add_subplot(2, 3, 5)
#     plt.imshow(pic_sobel_horz, cmap='gray')

#     pic_sobel = np.sqrt(np.square(pic_sobel_vert) + np.square(pic_sobel_horz**2))
#     pic_sobel *= 255.0 / pic_sobel.max()

#     fig.add_subplot(2, 3, 6)
#     plt.imshow(pic_sobel, cmap='gray')


#     plt.show()


# filtered_video = utils.high_pass_filter(data[0]["video"], (15,15), 3)
# utils.produce_gif(filtered_video, "high_pass_filter.gif")

# W = utils.apply_NMF(filtered_video/255, 2)
# utils.produce_gif(filtered_video/255 - W, "NMF_high_pass.gif")


utils.modify_data_with_rnfm("train", 1, 50, "rnfm_50_train")
# video = data[0]["video"]
# print(video.shape)
# video_prep = video.reshape(112*112, 334)
# print(video.shape)
# W, H, S, obj = robust_nfm.robust_nmf(data=video_prep, rank=2, max_iter=1000, tol=1e-7, beta=1, init='NMF', reg_val=20, sum_to_one=False)
# print(W.shape)
# print(H.shape)
# print(S.shape)
# print(obj)

# print(W)
# print(H)
# print(S)

# S = S.reshape(112, 112, 334)
# utils.produce_gif(S, "RNFM_20.gif")

quit()
pic = Image.fromarray(np.uint8(pic))
pic_filtered = pic.filter(ImageFilter.GaussianBlur)

pic.show()

quit()

# for i in data[0]["frames"]:
#     print(i)
#     print(data[0]["video"][:, :, i]/255)
#     print(np.mean(((data[0]["video"][:, :, i]/255).flatten())))
#     print(data[0]["label"][:, :, i])

# W = utils.apply_NMF(data[0]["video"]/255, 2)

# print(W.shape)
# print(H.shape)

# print(W)
# print(H)

# frame = data[0]['frames'][0]
# print(type(frame))
# plt.imshow(data[0]['video'][:, :, frame], cmap='gray')
# plt.show()
# plt.imshow(data[0]['label'][:, :, frame], cmap='gray')
# plt.show()

W = utils.apply_PCA(data[0]["video"], 2)
print(W.shape)
W = W.reshape(112, 112, 334)
plt.imshow(W[:, :, [4]], cmap="gray", vmin=0, vmax=1)
plt.show()


# print(tmp.shape)
# print(tmp)

with imageio.get_writer("PCA_2.gif", mode="I") as writer:
    print(data[0]["video"].shape[2])
    for i in range(data[0]["video"].shape[2]):
        image = W[:, :, i]
        writer.append_data(image)

W = data[0]["video"] / 255 - W
with imageio.get_writer("PCA_2_neg.gif", mode="I") as writer:
    print(data[0]["video"].shape[2])
    for i in range(data[0]["video"].shape[2]):
        image = W[:, :, i]
        writer.append_data(image)
