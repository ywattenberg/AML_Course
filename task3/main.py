import torch
import dataset
import loss
import numpy as np
import utils
import matplotlib.pyplot as plt
from unet import UNet
import torchmetrics
from torch.autograd import Variable


def main():
    # model = torch.hub.load(
    #     "mateuszbuda/brain-segmentation-pytorch",
    #     "unet",
    #     in_channels=1,
    #     out_channels=1,
    #     init_features=32,
    #     pretrained=False,
    # )
    model = UNet(in_channels=1, out_channels=1, init_features=32)
    model.double()

    data_train = dataset.HeartDataset(data=None, path="data/train.pkl", unpack_frames=True)

    print(len(data_train))
    # for i in range(len(data_train)):
    #     print("video: ", data_train[i][0].shape)
    #     print("label: ", data_train[i][1].shape)

    #     print(data_train[i][0])

    #     plt.imshow(data_train[i][0].permute(1, 2, 0), cmap='gray', vmin=0, vmax= 1)
    #     plt.show()

    # quit()

    pretrain_length = int(len(data_train) * 0.8)
    val_length = len(data_train) - pretrain_length
    data_pretrain, data_val = torch.utils.data.random_split(
        data_train, [pretrain_length, val_length]
    )

    train_loader = torch.utils.data.DataLoader(data_pretrain, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=8, shuffle=True)
    torch.set_grad_enabled(True)

    # loss_fn = torchmetrics.JaccardIndex(num_classes=2)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        utils.train_loop(model, train_loader, loss_fn, optimizer)
        utils.test_loop(model, val_loader, loss_fn, epoch)


if __name__ == "__main__":
    main()
