import torch
import unet
import dataset
import utils
import loss
import interpol_net


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE

REG_VAL = 1
IMAGE_SIZE = 256
EPOCHS_PRETRAIN = 0
EPOCHS_INTERPOL = 100
INTERPOL_SIZE = 5  # range of the interpolation --> has to be odd
FOCUS_ON_MIDDLE_FRAME = 1  # how often we take the middle frame in the interpolated dataset


def train_loop(unet_pretrain_model, train_loader, loss_fn, optimizer):

    for batch, (x, y) in enumerate(train_loader):
        output = unet_pretrain_model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 10) == 0:
            print(f"Batch: {batch}, Loss: {loss.item():.8f}")
            # break


def test_loop(unet_pretrain_model, test_loader, loss_fn):
    test_loss = 0
    size = 0

    with torch.no_grad():
        for x, y in test_loader:
            output = unet_pretrain_model(x)
            test_loss += loss_fn(output, y).item()
            size += 1

    test_loss /= size
    print(f"Test Error:     {test_loss:.8f}")

    # output = (output > 0.6).float()

    utils.produce_gif(x[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/input.gif")
    utils.produce_gif(output[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/output.gif")
    utils.produce_gif(y[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/label.gif")

    return test_loss


def main(train_full=False):
    unet_pretrain_model = unet.UNet(in_channels=5, out_channels=1, init_features=32)
    # unet_pretrain_model = Generic_UNetPlusPlus(1, base_num_features=32, num_classes=1)
    unet_pretrain_model.to(DEVICE)

    data_train = dataset.HeartDataset(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=4,
        unpack_frames=True,
        device=DEVICE,
    )

    # data_test = dataset.HeartTestDataset(
    #     path=f"data/test_data_{REG_VAL}_{IMAGE_SIZE}",
    #     n_batches=4,
    #     unpack_frames=False,
    #     return_full_data=True,
    #     device=DEVICE,
    # )

    pretrain_length = int(len(data_train) * 0.8)
    val_length = len(data_train) - pretrain_length
    pretrain, validation = torch.utils.data.random_split(data_train, [pretrain_length, val_length])
    train_loader = torch.utils.data.DataLoader(pretrain, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=16, shuffle=True)

    unet_pretrain_model = unet.UNet(in_channels=1, out_channels=1, init_features=32)
    unet_pretrain_model.to(DEVICE)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(unet_pretrain_model.parameters(), lr=1e-4)

    epochs = EPOCHS_PRETRAIN
    best_loss_pretrain = 1
    best_epoch_pretrain = 0

    for i in range(epochs):
        print(f"Pretrain Epoch: {i}")
        train_loop(unet_pretrain_model, train_loader, loss_fn, optimizer)
        test_loss = test_loop(unet_pretrain_model, val_loader, loss_fn)

        if test_loss < best_loss_pretrain:
            best_loss_pretrain = test_loss
            best_epoch_pretrain = i + 1
            torch.save(
                unet_pretrain_model.state_dict(),
                f"unet_pretrain_model_{REG_VAL}_{IMAGE_SIZE}_best.pth",
            )

    print(f"Best epoch: {best_epoch_pretrain}, Best loss: {best_loss_pretrain:.8f}")

    unet_pretrain_model.load_state_dict(
        torch.load(f"unet_pretrain_model_{REG_VAL}_{IMAGE_SIZE}_best.pth")
    )
    # freeze the unet_pretrain_model
    for param in unet_pretrain_model.parameters():
        param.requires_grad = False

    # create the interpolated dataset
    data_interpol = dataset.InterpolationSet(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=4,
        unpack_frames=True,
        device=DEVICE,
        interpol_size=INTERPOL_SIZE,
        focus_on_middle_frame=FOCUS_ON_MIDDLE_FRAME,
    )

    number_of_frames = INTERPOL_SIZE + FOCUS_ON_MIDDLE_FRAME - 1
    print(number_of_frames)
    train_interpol_length = int(len(data_interpol) * 0.8)
    val_interpol_length = len(data_interpol) - train_interpol_length
    train_interpol, val_interpol = torch.utils.data.random_split(
        data_interpol, [train_interpol_length, val_interpol_length]
    )
    train_interpol_loader = torch.utils.data.DataLoader(train_interpol, batch_size=16, shuffle=True)
    val_interpol_loader = torch.utils.data.DataLoader(val_interpol, batch_size=16, shuffle=True)

    interpol_model = interpol_net.InterpolNet(
        unet_model=unet_pretrain_model,
        in_channels=number_of_frames,
        out_channels=1,
        init_features=32,
    )
    interpol_model.to(DEVICE)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(interpol_model.parameters(), lr=1e-4)

    epochs = EPOCHS_INTERPOL
    best_loss_interpol = 1
    best_epoch_interpol = 0

    for i in range(epochs):
        print(f"Epoch: {i}")
        train_loop(interpol_model, train_interpol_loader, loss_fn, optimizer)
        test_loss = test_loop(interpol_model, val_interpol_loader, loss_fn)

        if test_loss < best_loss_interpol:
            best_loss_interpol = test_loss
            best_epoch_interpol = i + 1
            torch.save(
                interpol_model.state_dict(),
                f"unet_2_interpol_model_{REG_VAL}_{IMAGE_SIZE}_{INTERPOL_SIZE}_{FOCUS_ON_MIDDLE_FRAME}_best.pth",
            )

    print(f"Best epoch: {best_epoch_pretrain}, Best loss: {best_loss_pretrain:.8f}")
    print(f"Best epoch: {best_epoch_interpol}, Best loss: {best_loss_interpol:.8f}")


if __name__ == "__main__":
    main()
