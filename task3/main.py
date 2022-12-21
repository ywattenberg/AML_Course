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
EPOCHS = 200


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
    utils.produce_gif(
        y[0].type(torch.float).permute(1, 2, 0).cpu().detach().numpy(), f"img/label.gif"
    )
    return test_loss


def main(train=True, do_evaluation=False, create_submission=False):

    model = UNet(in_channels=3, out_channels=1, init_features=32)
    # model = torch.hub.load(
    #     "mateuszbuda/brain-segmentation-pytorch",
    #     "unet",
    #     in_channels=3,
    #     out_channels=1,
    #     init_features=32,
    #     pretrained=True,
    # )
    # model = Generic_UNetPlusPlus(1, base_num_features=32, num_classes=1)
    model.to(DEVICE)

    data_train = dataset.InterpolationSet(
        path=f"data/train_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=2,
        unpack_frames=True,
        device=DEVICE,
        interpol_size=3,
    )
    data_test = dataset.HeartTestDataset(
        path=f"data/test_data_{REG_VAL}_{IMAGE_SIZE}",
        n_batches=2,
        unpack_frames=True,
        device=DEVICE,
        interpol_size=3,
    )

    pretrain_length = int(len(data_train) * 0.8)
    val_length = len(data_train) - pretrain_length
    data_pretrain, data_val = torch.utils.data.random_split(
        data_train, [pretrain_length, val_length]
    )

    train_loader = torch.utils.data.DataLoader(
        data_pretrain, batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=16, shuffle=True)
    torch.set_grad_enabled(True)

    # loss_fn = torchmetrics.JaccardIndex(num_classes=2)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_loss = None
    if train:
        print(EPOCHS)
        for epoch in range(EPOCHS):
            print(f"--------------------------")
            print("Epoch: {}".format(epoch))
            train_loop(model, train_loader, loss_fn, optimizer)
            test_loss = test_loop(model, val_loader, loss_fn, epoch)

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), f"best_model_{IMAGE_SIZE}_{REG_VAL}.pth")

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
    create_gif=False,
):
    model.eval()

    torch.set_grad_enabled(False)
    submission = []
    for data, name in test:
        output = []
        print(name)
        for x in data:
            x = x.unsqueeze(0)
            output.append(model(x))

        output = torch.cat(output, dim=0)

        output = output.squeeze()
        submission.append({"name": name, "prediction": output})

    original = utils.load_zipped_pickle("data/test.pkl")
    found = 0
    for original_data in original:
        for submission_data in submission:
            if original_data["name"] == submission_data["name"]:
                # submission_data["mask"] = original_data["mask"]
                found += 1
                size = original_data["video"].shape[0:2]
                name = submission_data["name"]
                submission_data["prediction"] = utils.post_process_mask(
                    submission_data["prediction"], size, erode_it=0, dilate_it=0
                )
                break
    print(f"Found {found} out of {len(original)}")
    if create_gif:
        for original_data in original:
            for submission_data in submission:
                if original_data["name"] == submission_data["name"]:
                    name = submission_data["name"]
                    mask = submission_data["prediction"] * 1.0
                    video = original_data["video"]
                    stacked = np.concatenate([video, mask], axis=1)
                    utils.produce_gif(stacked, f"pred_img/{name}.gif")

    utils.save_zipped_pickle(
        submission, f"submission_{IMAGE_SIZE}_{REG_VAL}_{EPOCHS}.pkl"
    )


if __name__ == "__main__":
    main(train=False, do_evaluation=False, create_submission=True)
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
