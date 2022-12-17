import torch
import dataset
import loss
import utils
from unet import UNet

DEVICE = "cuda"


def train_loop(model, train_loader, loss_fn, optimizer):

    for batch, (x, y) in enumerate(train_loader):
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 10) == 0:
            print(f"Batch: {batch}, Loss: {loss.item():.4f}")
            # break


def test_loop(model, test_loader, loss_fn, epoch):
    size = len(test_loader.dataset)
    test_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += loss_fn(output, y).item()

    test_loss /= size
    print(f"Test Error: {test_loss:>8f} \n")

    # output = (output > 0.6).float()

    x = x.cpu()
    y = y.cpu()
    output = output.cpu()

    utils.produce_gif(x[0].permute(1, 2, 0).detach().numpy(), f"img/input.gif")
    utils.produce_gif(output[0].permute(1, 2, 0).detach().numpy(), f"img/output.gif")
    utils.produce_gif(y[0].permute(1, 2, 0).int().detach().numpy(), f"img/label.gif")


def main():
    model = UNet(in_channels=1, out_channels=1, init_features=32)

    if DEVICE == "mps":
        model.float()
    else:
        model.double()

    model.to(DEVICE)

    data_train = dataset.HeartDataset(
        path="data/train_data_1_256", n_batches=3, unpack_frames=True, device=DEVICE
    )

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
        train_loop(model, train_loader, loss_fn, optimizer)
        test_loop(model, val_loader, loss_fn, epoch)


if __name__ == "__main__":
    main()
