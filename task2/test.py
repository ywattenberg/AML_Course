import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch.nn import (AvgPool1d, AvgPool2d, Conv1d, Dropout, Linear, MaxPool2d,
                      Module, ReLU, Sequential, Sigmoid, Softmax)


class Model(Module):
    def __init__(self, num_of_features) -> None:
        super().__init__()

        self.num_of_features = num_of_features

        self.conv = Sequential(
            Conv1d(in_channels=1,out_channels=2,kernel_size=2), #out_dim = (num_of_features-2+1) x 2
            AvgPool1d(kernel_size=2, stride=1), #out_dim = ((num_of_features-2+1)-2+1) x 2 = (num_of_features-2) x 2
            Conv1d(in_channels=2, out_channels=5,kernel_size=2), #out_dim = (num_of_features-2-2+1) x 5 = (num_of_features-3) x 5
            MaxPool2d(kernel_size=2), #out_dim = (num_of_features-3)
        )

        self.linear = Sequential(
            Linear(num_of_features - 3 - ((num_of_features+1)%2), 1),
        )

    def forward(self,x):
        x_tmp = self.conv(x)
        print("x_tmp: ", x_tmp)
        x_tmp = x_tmp.flatten()
        print(x_tmp)
        return self.linear(x_tmp)


if __name__ == "__main__":

    # torch.manual_seed(0)

    # model = Model(num_of_features=6)

    # x = [1,5,4,3,2,2]


    # x_new = model(tensor(x).type(torch.float).reshape(1,1,-1))
    
    # print(x_new)

    y_train = pd.read_csv("data/y_train.csv")
    y_train = y_train["y"].to_numpy().flatten()

    print("0: ", np.count_nonzero(y_train == 0), "percentage of 0: ", 100 * np.count_nonzero(y_train == 0)/len(y_train))
    print("1: ", np.count_nonzero(y_train == 1), "percentage of 1: ", 100 * np.count_nonzero(y_train == 1)/len(y_train))
    print("2: ", np.count_nonzero(y_train == 2), "percentage of 2: ", 100 * np.count_nonzero(y_train == 2)/len(y_train))
    print("3: ", np.count_nonzero(y_train == 3), "percentage of 3: ", 100 * np.count_nonzero(y_train == 3)/len(y_train))

    y_pred = pd.read_csv("predictions.csv")
    y_pred = y_pred["y"].to_numpy().flatten()

    print("neural network")

    print("0: ", np.count_nonzero(y_pred == 0), "percentage of 0: ", 100 * np.count_nonzero(y_pred == 0)/len(y_pred))
    print("1: ", np.count_nonzero(y_pred == 1), "percentage of 1: ", 100 * np.count_nonzero(y_pred == 1)/len(y_pred))
    print("2: ", np.count_nonzero(y_pred == 2), "percentage of 2: ", 100 * np.count_nonzero(y_pred == 2)/len(y_pred))
    print("3: ", np.count_nonzero(y_pred == 3), "percentage of 3: ", 100 * np.count_nonzero(y_pred == 3)/len(y_pred))

    print("best version")
    y_pred = pd.read_csv("out.csv")

    print("0: ", np.count_nonzero(y_pred == 0), "percentage of 0: ", 100 * np.count_nonzero(y_pred == 0)/len(y_pred))
    print("1: ", np.count_nonzero(y_pred == 1), "percentage of 1: ", 100 * np.count_nonzero(y_pred == 1)/len(y_pred))
    print("2: ", np.count_nonzero(y_pred == 2), "percentage of 2: ", 100 * np.count_nonzero(y_pred == 2)/len(y_pred))
    print("3: ", np.count_nonzero(y_pred == 3), "percentage of 3: ", 100 * np.count_nonzero(y_pred == 3)/len(y_pred))

    print(y_train)

  


