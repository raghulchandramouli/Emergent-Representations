import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """
    A tiny Lightweight CNN backbone.
    Input: (B, 3, H, W)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), # 16 -> 8
            nn.Conv2d(128, 256, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # -> (B, 256, 1, 1)
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        feat = self.encoder(x).flatten(1) # (B, 256)
        return self.fc(feat) # (B, embed_dim)


