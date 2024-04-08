import torch.nn.functional as F
from torch import nn
import torch
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

class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        groups = channels // 4
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(Downsample, self).__init__()
        groups = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, 2, 2, 0)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU()
        )


    def forward(self, x, t):
        x = F.relu(self.gn1(self.conv1(x)))
        t = self.time_mlp(t)
        t = t[(..., ) + (None, ) * 2] # Extending last 2 dimensions so that it can take shape N C 1 1, then can be added to x
        x = x + t
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(Upsample, self).__init__()
        groups = out_channels // 4
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.downconv = nn.ConvTranspose2d(out_channels * 2, out_channels, 3, 1, 1) # Multiplying out channels by 2 as this takes the concatenated tensor as input
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip_connection, t):
        x = F.relu(self.gn1(self.upconv(x)))
        x = torch.cat((x, skip_connection), dim=1) # concatenate along channel dimension. This doubles channels, next layer reduces it back to output size
        x = F.relu(self.gn2(self.downconv(x)))
        t = self.time_mlp(t)
        t = t[(..., ) + (None, ) * 2] # Extending last 2 dimensions so that it can take shape N C 1 1, then can be added to x
        x = x + t
        return x

class GoodUNet(nn.Module):
    def __init__(self, n=1, input_channels=3, time_emb_dim=32, channel_sequence=[32, 64, 128, 256]):
        super(GoodUNet, self).__init__()
        self.channel_sequence = channel_sequence
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(input_channels, channel_sequence[0], 3, 1, 1)
        self.gn1 = nn.GroupNorm(channel_sequence[0] // 4, channel_sequence[0])

        self.downblocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channel_sequence) - 1): # We dont downsample on the bottleneck, thus the length minus 1
            self.downblocks.append(nn.Sequential(*[Block(channel_sequence[i]) for block in range(n)]))
            self.downsamples.append(Downsample(channel_sequence[i], channel_sequence[i+1], time_emb_dim))

        self.upblocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # Reverse the channel sequence for upsampling
        reversed_channels = list(reversed(channel_sequence))
        for i in range(len(reversed_channels)):
            self.upblocks.append(nn.Sequential(*[Block(reversed_channels[i]) for block in range(n)]))
            if i != len(reversed_channels)-1:
                self.upsamples.append(Upsample(reversed_channels[i], reversed_channels[i+1], time_emb_dim))

        self.outputconv = nn.Conv2d(channel_sequence[0], input_channels, 1, 1, 0)
    
    def forward(self, x, t):
        skip_connections = []
        t = self.time_mlp(t) # Time embedding

        x = F.relu(self.gn1(self.conv1(x)))
        skip_connections.append(x)

        for i in range(len(self.channel_sequence) - 1):
            x = self.downblocks[i](x)
            x = self.downsamples[i](x, t)
            if i != len(self.channel_sequence) - 2: # We dont have a residual connection for the last block as its going into the bottleneck
                skip_connections.append(x)

        for i in range(len(self.channel_sequence)):
            x = self.upblocks[i](x) # When i is zero, this is the bottleneck
            if i != len(self.channel_sequence) - 1: # We dont upsample on the last block
                x = self.upsamples[i](x, skip_connections.pop(), t)

        x = self.outputconv(x)
        return x
