from torch.nn import (
    Module,
    Linear,
    ReLU,
    Sequential,
    Dropout,
    Sigmoid,
    Softmax,
    Conv1d,
    AvgPool1d,
    MaxPool1d
)


class Model(Module):
    def __init__(self, num_of_features):
        super().__init__()

        self.num_of_features = num_of_features

        self.conv = Sequential(
            Conv1d(in_channels=1, out_channels=32, kernel_size=5),  # out_dim = (num_of_features-5+1) x 32 = num_of_features-4 x 32
            ReLU(),
            MaxPool1d(kernel_size=5, stride=1),  # out_dim = num_of_features-4-5 + 1 x 32 = (num_of_features-9)/2 + 1 x 32
            Dropout(0.2),
            Conv1d(in_channels=32, out_channels=32, kernel_size=5), # out_dim = num_of_features -8 - 4 x 32
            ReLU(),
            MaxPool1d(kernel_size=5, stride=1), # out_dim = num_of_features-12 -4 x 32 = num_of_features-16 x 32
            Dropout(0.2),
        )

        self.linear = Sequential(
            Linear(num_of_features - 16 - ((num_of_features+1)%2), 5000),
            ReLU(),
            Dropout(0.2),
            Linear(5000, 5000),
            ReLU(),
            Dropout(0.2),
            Linear(5000, 1024),
            ReLU(),
            Dropout(0.2),
            Linear(1024, 4),
            Sigmoid()
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
        x_tmp = self.conv().flatten()
        return self.linear(x_tmp)
  
