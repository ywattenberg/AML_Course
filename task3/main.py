import torch
import dataset
import loss
import utils
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from unet import UNet

DEVICE = "cpu"


def train_loop(model, train_loader, loss_fn, optimizer):

    for batch, (x, y) in enumerate(train_loader):
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
    model = UNet(in_channels=1, out_channels=1, init_features=32)
    model.to(DEVICE)

    data_train = dataset.HeartDataset(
        path="data/train_data_1_256", n_batches=2, unpack_frames=True, device=DEVICE
    )

    pretrain_length = int(len(data_train) * 0.8)
    val_length = len(data_train) - pretrain_length
    data_pretrain, data_val = torch.utils.data.random_split(
        data_train, [pretrain_length, val_length]
    )

    train_loader = torch.utils.data.DataLoader(
        data_pretrain, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=32, shuffle=True)
    torch.set_grad_enabled(True)

    # loss_fn = torchmetrics.JaccardIndex(num_classes=2)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100

    for epoch in range(epochs):
        print(f"--------------------------")
        print("Epoch: {}".format(epoch))
        train_loop(model, train_loader, loss_fn, optimizer)
        test_loop(model, val_loader, loss_fn, epoch)

    torch.save(model.state_dict(), "model.pth")


def evaluate():
    model = UNet(in_channels=1, out_channels=1, init_features=32)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(DEVICE)

    data_train = dataset.HeartDataset(
        path="data/train_data_1_256", n_batches=2, unpack_frames=True, device=DEVICE
    )

    test_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True)
    torch.set_grad_enabled(False)

    loss_fn = loss.JaccardLoss()
    print("Evaluating model")
    test_loop(model, test_loader, loss_fn, 0)


def predict_and_save(model, test_loader):
    model.eval()
    with torch.no_grad():
        for i in range(0, 501, 100):
            output = model(test_loader[i][0].unsqueeze(0).to(DEVICE))
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(test_loader[i][0].squeeze().cpu().detach().numpy())
            axarr[1].imshow(output.squeeze().cpu().detach().numpy())
            plt.savefig(f"pred_img/pred_{i}.png")


def create_submission(create_gif=False):
    model = UNet(in_channels=1, out_channels=1, init_features=32)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(DEVICE)
    test = dataset.HeartTestDataset(
        path="data/test_data_1_256",
        n_batches=2,
        unpack_frames=False,
        return_full_data=True,
        device=DEVICE,
    )
    torch.set_grad_enabled(False)
    submission = []
    for data in test:
        name = data["name"]
        output = model(data["nmf"].unsqueeze(1).to(DEVICE))
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
                    submission_data["prediction"], size
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

    utils.save_zipped_pickle(submission, "submission.pkl")


if __name__ == "__main__":
    main()
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
