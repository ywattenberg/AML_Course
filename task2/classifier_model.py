import torch


class ClassifierModel(torch.nn.Module):
    def __init__(self, num_features):
        super(ClassifierModel, self).__init__()
        
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(num_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 4),
            torch.nn.Sigmoid(),
        )

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.nn(x)
        x = self.softmax(x)
        return x