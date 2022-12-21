import numpy as np
import torch
import tqdm
import dataset_roi
import loss
import utils
import rcnn
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE


REG_VAL = 1
IMAGE_SIZE = 256
EPOCHS = 10


def train_loop(model, train_loader, loss_fn, optimizer):

    for batch, (x, y) in enumerate(train_loader):
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 50) == 0:
            print(f"Batch: {batch}, Loss: {loss.item():.8f}")
            # break


def test_loop(model, test_loader, loss_fn, epoch):
    test_loss = 0
    size = 0

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += loss_fn(output, y).item()
            size += 1

    test_loss /= size
    print(f"Test Error:     {test_loss:.8f}")

    # output = (output > 0.6).float()

    utils.produce_gif(x[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/input.gif")
    utils.produce_gif(
        output[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/output.gif"
    )
    utils.produce_gif(y[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/label.gif")


def main():
    train_data = dataset_roi.BoxDataset(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=2,
        unpack_frames=True,
        device=DEVICE,
    )

    val_data = dataset_roi.BoxDataset(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=1,
        unpack_frames=True,
        device=DEVICE,
        test=True,
    )
    # pretrain_length = int(len(train_data) * 0.8)
    # val_length = len(train_data) - pretrain_length
    # train_data, val_data = torch.utils.data.random_split(
    #     train_data, [pretrain_length, val_length]
    # )
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True)

    model = rcnn.RCnn(in_channels=1, out_channels=1, init_features=32)
    model.to(DEVICE)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        print(f"--------------------------")
        print("Epoch: {}".format(epoch))
        train_loop(model, train_loader, loss_fn, optimizer)
        test_loop(model, val_loader, loss_fn, epoch)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"model_box_all_data_{epoch}.pth")

    torch.save(model.state_dict(), "model_box_all_data_final.pth")


if __name__ == "__main__":
    main()
