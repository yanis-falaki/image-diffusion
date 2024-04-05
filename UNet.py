import torch.nn.functional as F
from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

class DownsampleBlock(nn.Module):
    def __init__(self, input_channels):
        super(DownsampleBlock).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self, input_channels):
        super(UpsampleBlock).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels * 2, input_channels , 3, 1, 1) # UNet paper uses a 2x2 convolution, but it dosent work with my dimensions as it wont maintain feature map size
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(input_channels * 2, input_channels * 2, 3, 1, 1)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))

        # concat equivalent feature map from downsample and x
        x = torch.concat((x, skip_connection), dim=1)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class UNet(nn.Module):
    def __init__(self, ) -> None:
        super(UNet).__init__()
        # Convolutional output dimensions formula (in each depth slice): W_new = (W-F + 2P)/S + 1 where W=input_shape, F=kernel_shape, P=padding_amount, S=stride_amount
        self.downsample1 = DownsampleBlock(3) # ends up with 6 channels, 32 x 32
        self.downsample2 = DownsampleBlock(6) # ends up with 12, 16 x 16
        self.downsample3 = DownsampleBlock(12) # ends up with 24, 8 x 8

        self.upsample1 = UpsampleBlock(24) # ends up with 12, 16 x 16
        self.upsample2 = UpsampleBlock(12) # ends up with 6, 32 x 32
        self.upsample3 = UpsampleBlock(6) # ends up with 3, 64 x 64

        self.fc1 = nn.Linear(64 * 64 * 3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, positional_encoding):
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        x = self.downsample3(x2)

        x = self.upsample1(x, x)
        x = self.upsample2(x, x2)
        x = self.upsample3(x, x1)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
