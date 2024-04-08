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
    def __init__(self, n=1, input_channels=3, initial_channels=32, time_emb_dim=32):
        super(GoodUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(input_channels, initial_channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(initial_channels // 4, initial_channels)

        self.blocks1 = nn.Sequential(*[Block(initial_channels) for block in range(n)])
        self.downsample1 = Downsample(initial_channels, initial_channels * 2, time_emb_dim)
        self.blocks2 = nn.Sequential(*[Block(initial_channels * 2) for block in range(n)])
        self.downsample2 = Downsample(initial_channels * 2, initial_channels * 4, time_emb_dim)
        self.blocks3 = nn.Sequential(*[Block(initial_channels * 4) for block in range(n)])
        self.downsample3 = Downsample(initial_channels * 4, initial_channels * 8, time_emb_dim)
        self.blocks4 = nn.Sequential(*[Block(initial_channels * 8) for block in range(n)]) # Bottleneck
        self.upsample1 = Upsample(initial_channels * 8, initial_channels * 4, time_emb_dim)
        self.blocks5 = nn.Sequential(*[Block(initial_channels * 4) for block in range(n)])
        self.upsample2 = Upsample(initial_channels * 4, initial_channels * 2, time_emb_dim)
        self.blocks6 = nn.Sequential(*[Block(initial_channels * 2) for block in range(n)])
        self.upsample3 = Upsample(initial_channels * 2, initial_channels, time_emb_dim)
        self.blocks7 = nn.Sequential(*[Block(initial_channels) for block in range(n)])

        self.outputconv = nn.Conv2d(initial_channels, input_channels, 1, 1, 0)
    
    def forward(self, x, t):
        skip_connections = []
        t = self.time_mlp(t) # Time embedding

        x = F.relu(self.gn1(self.conv1(x)))
        skip_connections.append(x)
        x = self.blocks1(x)
        x = self.downsample1(x, t)
        skip_connections.append(x)
        x = self.blocks2(x)
        x = self.downsample2(x, t)
        skip_connections.append(x)
        x = self.blocks3(x)
        x = self.downsample3(x, t)
        x = self.blocks4(x)
        x = self.upsample1(x, skip_connections.pop(), t)
        x = self.blocks5(x)
        x = self.upsample2(x, skip_connections.pop(), t)
        x = self.blocks6(x)
        x = self.upsample3(x, skip_connections.pop(), t)
        x = self.blocks7(x)
        x = self.outputconv(x)
        return x
