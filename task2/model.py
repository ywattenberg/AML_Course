from torch.nn import (AvgPool1d, Conv1d, Dropout, Flatten, Linear, MaxPool1d,
                      Module, ReLU, Sequential, Sigmoid, Softmax)


class Model(Module):
    def __init__(self, num_of_features):
        super().__init__()

        self.num_of_features = num_of_features

        self.conv = Sequential(
            Conv1d(
                in_channels=1, out_channels=5, kernel_size=16
            ),  # out_dim = (num_of_features-5+1) x 32 = num_of_features-4 x 32
            Conv1d(in_channels=5, out_channels=25, kernel_size=16),
            MaxPool1d(
                kernel_size=5, stride=1
            ),  # out_dim = num_of_features-4-5 + 1 x 32 = (num_of_features-9)/2 + 1 x 32
            Conv1d(
                in_channels=25, out_channels=25, kernel_size=16
            ),  # out_dim = num_of_features -8 - 4 x 32
            Conv1d(in_channels=25, out_channels=5, kernel_size=16),
            MaxPool1d(
                kernel_size=16, stride=1
            ),  # out_dim = num_of_features-12 -4 x 32 = num_of_features-16 x 32
        )

        self.linear = Sequential(
            Flatten(),
            Linear(88640, 5000),
            ReLU(),
            Dropout(0.2),
            Linear(5000, 5000),
            ReLU(),
            Dropout(0.2),
            Linear(5000, 1024),
            ReLU(),
            Dropout(0.2),
            Linear(1024, 4),
            Sigmoid(),
        )
        # self.nn = Sequential(
        #     #1-d signal --> in_channel=1, cannot be changed in the first conv
        #     Conv1d(in_channels=1, out_channels=32, kernel_size=5), #32 filters --> output dim: (num_of_features-kernel-size + 1) x 32
        #     AvgPool1d(kernel_size=5),
        #     Linear(num_of_features, num_of_features),
        #     ReLU(),
        #     Dropout(0.5),
        #     Linear(num_of_features, 6000),
        #     ReLU(),
        #     Dropout(0.5),
        #     Linear(6000, 1024),
        #     ReLU(),
        #     Dropout(0.5),
        #     Linear(1024, 256),
        #     ReLU(),
        #     Dropout(0.5),
        #     Linear(256, 64),
        #     ReLU(),
        #     Dropout(0.5),
        #     Linear(64, 4),
        #     Sigmoid(),
        # )

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 1, self.num_of_features)
        # print(x.shape)
        x_tmp = self.conv(x)
        # .reshape(
        #     64, self.num_of_features - 16 - ((self.num_of_features + 1) % 2)
        # )
        return self.linear(x_tmp)
