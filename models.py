import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x)

class ForceFieldPredictor(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, n_blocks=4):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(3, embed_dim), nn.SiLU())
        self.backbone = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 3)
    def forward(self, x):
        x = self.input_layer(x)
        return self.head(self.backbone(x))

class ErrorPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
    def forward(self, x):
        return self.stack(x) + 1e-4