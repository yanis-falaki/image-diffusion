import torch.nn.functional as F
from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, time_emb_dim):
        super(DownsampleBlock, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, input_channels * 2)

        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, t_emb):
        t_emb = F.relu(self.time_mlp(t_emb))
        # Extending last 2 dimensions so that it can take shape N C 1 1, then can be added to x
        t_emb = t_emb[(..., ) + (None, ) * 2]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x + t_emb # Time embedding has a value for each channel, every pixel in the channel is summed with the corresponding embedding-channel value
        return x
    
class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, time_emb_dim):
        super(UpsampleBlock, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, input_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2 , 3, 1, 1) # UNet paper uses a 2x2 convolution, but it dosent work with my dimensions as it wont maintain feature map size
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, 3, 1, 1)

    def forward(self, x, skip_connection, t_emb):
        # Getting time embedding
        t_emb = F.relu(self.time_mlp(t_emb))
        # Extending last 2 dimensions so that it can take shape N C 1 1, then can be added to x
        t_emb = t_emb[(..., ) + (None, ) * 2]

        x = self.upsample(x)

        x = F.relu(self.conv1(x))

        # concat equivalent feature map from downsample and x
        x = torch.concat((x, skip_connection), dim=1)

        # Adding time embedding
        x = x + t_emb

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class UNetModel(nn.Module):
    def __init__(self, time_emb_dim) -> None:
        super(UNetModel, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Convolutional output dimensions formula (in each depth slice): W_new = (W-F + 2P)/S + 1 where W=input_shape, F=kernel_shape, P=padding_amount, S=stride_amount
        self.downsample1 = DownsampleBlock(3, time_emb_dim) # ends up with 6 channels, 32 x 32
        self.downsample2 = DownsampleBlock(6, time_emb_dim) # ends up with 12, 16 x 16
        self.downsample3 = DownsampleBlock(12, time_emb_dim) # ends up with 24, 8 x 8

        self.upsample1 = UpsampleBlock(24, time_emb_dim) # ends up with 12, 16 x 16
        self.upsample2 = UpsampleBlock(12, time_emb_dim) # ends up with 6, 32 x 32
        self.upsample3 = UpsampleBlock(6, time_emb_dim) # ends up with 3, 64 x 64

        self.fc1 = nn.Linear(64 * 64 * 3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, t):
        # Get time embedding
        time_emb = self.time_mlp(t)

        x1 = self.downsample1(x, time_emb)
        x2 = self.downsample2(x1, time_emb)
        x3 = self.downsample3(x2, time_emb)

        x3 = self.upsample1(x3, x2, time_emb)
        x3 = self.upsample2(x3, x1, time_emb)
        x3 = self.upsample3(x3, x, time_emb)

        x3 = torch.flatten(x3, start_dim=1)

        x3 = F.relu(self.fc1(x3))
        x3 = self.fc2(x3)

        return x3
        