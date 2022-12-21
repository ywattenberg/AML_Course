import torch
import unet
import dataset
import utils
import loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE

REG_VAL = 1
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


def test_loop(model, test_loader, loss_fn):
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

    # utils.produce_gif(x[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/input.gif")
    # utils.produce_gif(output[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/output.gif")
    # utils.produce_gif(y[0].permute(1, 2, 0).cpu().detach().numpy(), f"img/label.gif")

    return test_loss



def main(train_full=False):
    model = unet.UNet(in_channels=5, out_channels=1, init_features=32)
    # model = Generic_UNetPlusPlus(1, base_num_features=32, num_classes=1)
    model.to(DEVICE)

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

    model = unet.UNet(in_channels=1, out_channels=1, init_features=32)
    model.to(DEVICE)
    loss_fn = loss.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = EPOCHS
    best_loss = 1
    best_epoch = 0

    for i in range(epochs):
        print(f"Epoch: {i}")
        train_loop(model, train_loader, loss_fn, optimizer)
        test_loss = test_loop(model, val_loader, loss_fn)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = i+1
            torch.save(model.state_dict(), f"model_{REG_VAL}_{IMAGE_SIZE}_best.pth")
        
    print(f"Best epoch: {best_epoch}, Best loss: {best_loss:.8f}")


    model.load_state_dict(torch.load(f"model_{REG_VAL}_{IMAGE_SIZE}_best.pth"))
    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    
    

    
    


    






if __name__ == '__main__':
    main()
