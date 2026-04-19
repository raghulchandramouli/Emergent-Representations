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

# --- v2/v2 ViT-Tiny Backbone ---

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.
    """

    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, 
                              embed_dim,
                              kernel_size = patch_size,
                              stride      = patch_size
        )

    def forward(self, x):
        # X: (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1,2)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim,
                                           num_heads,
                                           dropout     = dropout,
                                           batch_first = True ) 
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim,
                     mlp_dim),
                     nn.GELU(),
            
            nn.Linear(mlp_dim, 
                     embed_dim)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )

        x = x + self.drop(attn_out)

        # MLP with residual
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, attn_weights

        