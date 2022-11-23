import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax

import utils
import dataset
import model


def train_loop(dataloader, model, loss_fn, optimizer):
    sm = Softmax(dim=1)
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x).type(torch.float)
        # print("pred: ", pred)
        pred_sm = sm(pred.squeeze(1))
        y = y.long()
        # print(pred)
        # print(y.squeeze(1))
        # print(type(pred_sm))
        # print(type(y.squeeze(1)))
        # print(pred_sm.shape)
        # print(pred_sm)
        # print(pred_sm.shape)
        # pred_sm = pred_sm.squeeze(1)
        # print(pred_sm.argmax(1))
        loss = loss_fn(pred_sm, y.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x) * 3
            print(f"loss: {loss:>7f}  [{current:>5d} /{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    predicition = [0, 0, 0, 0]
    sm = Softmax(dim=1)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (x, y) in dataloader:
            pred = model(x).type(torch.float)
            pred_sm = sm(pred.squeeze(1))
            loss = loss_fn(pred_sm, y.squeeze(1).long())
            test_loss += loss
            correct += (pred_sm.argmax(1) == y).type(torch.float).sum().item()

            preds = pred_sm.argmax(1)

            for pred in preds.int():
                print(pred)
                print(type(pred))
                predicition[int(pred)] += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}")
    print(predicition)


if __name__ == "__main__":
    train_data = dataset.TrainDataset(
        feature_path="data/X_train.csv", label_path="data/y_train.csv"
    )

    num_of_features = train_data.get_num_of_features()
    print(len(train_data))

    length_pretain_data = int(len(train_data) * 0.8)
    length_val_data = len(train_data) - length_pretain_data
    train_data, val_data = torch.utils.data.random_split(
        train_data, [length_pretain_data, length_val_data]
    )

    dataloader_pretrain = DataLoader(train_data, batch_size=64, shuffle=True)
    dataloader_val = DataLoader(val_data, batch_size=64, shuffle=True)

    nn_model = model.Model(num_of_features=num_of_features)
    loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 100

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(dataloader_pretrain, nn_model, loss, optimizer)
        test_loop(dataloader_val, nn_model, loss)
        print()

    print("Done!")
