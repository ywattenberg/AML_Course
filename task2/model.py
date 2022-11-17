from torch.nn import (
    Module,
    Linear,
    ReLU,
    Sequential,
    Dropout,
    Sigmoid,
    Softmax,
)


class Model(Module):
    def __init__(self, num_of_features):
        super().__init__()

        self.nn = Sequential(
            Linear(num_of_features, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, 256),
            ReLU(),
            Dropout(0.5),
            Linear(256, 4),
            Sigmoid(),
        )

    def forward(self, x):
        return self.nn(x)
