import utils
import numpy as np
import matplotlib.pyplot as plt
import imageio

import warnings

warnings.filterwarnings("always")


data = utils.load_zipped_pickle("data/train.pkl")

# for i in data[0]["frames"]:
#     print(i)
#     print(data[0]["video"][:, :, i]/255)
#     print(np.mean(((data[0]["video"][:, :, i]/255).flatten())))
#     print(data[0]["label"][:, :, i])

W = utils.apply_NMF(data[0]["video"]/255, 2)

print(W.shape)
# print(H.shape)

print(W)
# print(H)


# with imageio.get_writer('NFM_90.gif', mode='I') as writer:
#         print(data[0]['video'].shape[2])
#         for i in range (data[0]['video'].shape[2]):
#             image = W[:, :, i]
#             writer.append_data(image)

# W = data[0]["video"]/255 - W
# with imageio.get_writer('NFM_90_neg.gif', mode='I') as writer:
#         print(data[0]['video'].shape[2])
#         for i in range (data[0]['video'].shape[2]):
            # image = W[:, :, i]
            # writer.append_data(image)



    