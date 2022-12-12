import pickle
import gzip
import numpy as np
import os
import torchvision.transforms as transforms
import torch
import PIL
import matplotlib.pyplot as plt


def get_transforms():
    return transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8)),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

def transform_data(data):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Normalize(
        mean=[0.44531356896770125],
        std=[0.2692461874154524],
    ),
    ])
    normalized_img = transform(data/255)
    return normalized_img

def transform_label(label):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    ])
    normalized_img = transform(label)
    return normalized_img

def load_zipped_pickle(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f, 2)


def test_pred():
    # load data
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    samples = load_zipped_pickle("sample.pkl")
    # make prediction for test
    predictions = []
    for d in test_data:
        prediction = np.array(np.zeros_like(d["video"]), dtype=np.bool)
        height = prediction.shape[0]
        width = prediction.shape[1]
        prediction[
            int(height / 2) - 50 : int(height / 2 + 50),
            int(width / 2) - 50 : int(width / 2 + 50),
        ] = True

        # DATA Strucure
        predictions.append({"name": d["name"], "prediction": prediction})
        # save in correct format

    save_zipped_pickle(predictions, "my_predictions.pkl")


def train_loop(model, train_loader, loss_fn, optimizer):
    
    for batch, (x,y) in enumerate(train_loader):
        print(x.DoubleTensor())
        output = model(x)
        loss = loss_fn(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 10) == 0:
            print("Batch: {}, Loss: {}".format(batch, loss.item().round(8)))


def test_loop(model, test_loader, loss_fn):
    size = len(test_loader.dataset)
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += loss_fn(output, y).item()
    
    test_loss /= size
    print(f"Test Error: {test_loss:>8f} \n")

    # print picture
    r = np.random.randint(0, len(test_loader))
    x, y = test_loader.dataset[r]
    output = model(x)
    plt.imshow(output.permute(1, 2, 0), cmap='gray', vmin=0, vmax= 1)
    plt.show()
    plt.imshow(y.permute(1, 2, 0), cmap='gray', vmin=0, vmax= 1)
    plt.show()
    

