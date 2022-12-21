import torch
import dataset
import loss
import utils
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE

REG_VAL = 1
IMAGE_SIZE = 256
EPOCHS = 400
IMAGE_SIZE = 256
EPOCHS = 400


def train_loop(model, train_loader, loss_fn, optimizer):

    for batch, (x, y, _) in enumerate(train_loader):
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 10) == 0:
            print(f"Batch: {batch}, Loss: {loss.item():.8f}")
            # break


def test_loop(model, test_loader, loss_fn, epoch):
    test_loss = 0
    size = 0

    with torch.no_grad():
        for x, y, box in test_loader:
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


def main(train=True, do_evaluation=False, create_submission=False):

    model = UNet(in_channels=5, out_channels=1, init_features=32)
    # model = Generic_UNetPlusPlus(1, base_num_features=32, num_classes=1)
    model.to(DEVICE)

    data_train = dataset.InterpolatedHeartDataset(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=4,
        unpack_frames=True,
        device=DEVICE,
    )
    data_test = dataset.HeartTestDataset(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=2,
        path=f"data/test_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=4,
        unpack_frames=False,
        return_full_data=True,
        device=DEVICE,
    )

    pretrain_length = int(len(data_train) * 0.8)
    val_length = len(data_train) - pretrain_length
    data_pretrain, data_val = torch.utils.data.random_split(
        data_train, [pretrain_length, val_length]
    )

    train_loader = torch.utils.data.DataLoader(
        data_pretrain, batch_size=8, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=8, shuffle=True)
    torch.set_grad_enabled(True)

    # loss_fn = torchmetrics.JaccardIndex(num_classes=2)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if train:
        print(EPOCHS)
        for epoch in range(EPOCHS):
            print(f"--------------------------")
            print("Epoch: {}".format(epoch))
            train_loop(model, train_loader, loss_fn, optimizer)
            test_loop(model, val_loader, loss_fn, epoch)
            if epoch % 100 == 0 and epoch != 0:
                torch.save(
                    model.state_dict(), f"model_{IMAGE_SIZE}_{REG_VAL}_{epoch}.pth"
                )
        torch.save(model.state_dict(), f"model_{IMAGE_SIZE}_{REG_VAL}_{EPOCHS}.pth")
    else:
        model.load_state_dict(
            torch.load(
                f"model_{IMAGE_SIZE}_{REG_VAL}_{EPOCHS}.pth", map_location=DEVICE
            )
        )

    if do_evaluation:
        model.eval()
        print("Evaluating model")
        test_loop(model, val_loader, loss_fn, 0)
    if create_submission:
        submit(model, data_test, create_gif=True)


def submit(
    model,
    test,
    batch_size=8,
    create_gif=False,
):
    model.eval()

    torch.set_grad_enabled(False)
    submission = []
    for data in test:
        name = data["name"]
        output = []
        for frame in range(0, data["nmf"].shape[0], batch_size):
            lim = min(frame + batch_size, data["nmf"].shape[0])
            output.append(model(data["nmf"][frame:lim].unsqueeze(1).to(DEVICE)))

        output = torch.cat(output, dim=0)
        output = output.squeeze()
        submission.append({"name": name, "prediction": output})

    original = utils.load_zipped_pickle("data/test.pkl")
    for original_data in original:
        for submission_data in submission:
            if original_data["name"] == submission_data["name"]:
                # submission_data["mask"] = original_data["mask"]
                size = original_data["video"].shape[0:2]
                name = submission_data["name"]
                submission_data["prediction"] = utils.post_process_mask(
                    submission_data["prediction"], size, erode_it=20, dilate_it=5
                )
                break
    if create_gif:
        for i in range(len(submission)):
            name = submission[i]["name"]
            assert submission[i]["name"] == original[i]["name"]
            mask = submission[i]["prediction"] * 1.0
            video = original[i]["video"]
            stacked = np.concatenate([video, mask], axis=1)
            utils.produce_gif(stacked, f"pred_img/{name}.gif")

    utils.save_zipped_pickle(
        submission, f"submission_{IMAGE_SIZE}_{REG_VAL}_{EPOCHS}.pkl"
    )


if __name__ == "__main__":
    main(train=True, do_evaluation=False, create_submission=False)
    # evaluate()
    # model = UNet(in_channels=1, out_channels=1, init_features=32)
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()
    # model.to(DEVICE)
    # test = dataset.HeartTestDataset(
    #     path="data/test_data_1_256", unpack_frames=True, device=DEVICE
    # )
    # predict_and_save(model, test)
    # create_submission(True)
