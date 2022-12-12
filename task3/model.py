from torch.nn import Module
import torch.nn as nn


class Model(Module):
    def __init__(self, channels, classes, bilinear=True):
        super().__init__()

        self.num_of_features = num_of_features

        # Double conv params
        in_channels_dconv = channels
        mid_channels_dconv = 64
        out_channels_dconv = 64
        nn.Sequential(
            # Double conv
            nn.Conv2d(in_channels_dconv, mid_channels_dconv, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels_dconv),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels_dconv, mid_channels_dconv, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels_dconv),
            nn.ReLU(inplace=True),
        )
