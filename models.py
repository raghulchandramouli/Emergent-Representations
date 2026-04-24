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

class ViTTiny(nn.Module):
    """
    ViT-Tiny: 5.7M parameters, works on 4GB VRAM.
    embed_dim = 192, depth = 12, heads = 3
    """

    def __init__(self, img_size  = 64, 
                patch_size       = 8, 
                embed_dim        = 192, 
                depth            = 6, 
                num_heads        = 3):

        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
          TransformerBlock(embed_dim, num_heads) for _ in range(depth)  
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.__init__weights()

    def _init__weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn=False):
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        
        all_attn = []
        for block in self.blocks:
            x, attn = block(x)
            all_attn.append(attn)
            
        x = self.norm(x)
        cls_out = x[:, 0] # CLS token embedding

        if return_attn:
            return cls_out, all_attn # for visualization
        return cls_out

# Projection Head
class ProjectionHead(nn.Module):
    """
    v1  : linear probing
    v2+ : 3-layer with bottleneck (like original DINO paper) 
    """
        
    def __init__(
        self, 
        in_dim,
        hidden_dim = 512,
        out_dim    = 128,
        use_mlp    = False
    ):
        super().__init__()
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)

        # L2 normalize output (critical for DINO)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.proj(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# Full DINO network + Backbone + Head
class DINONetwork(nn.Module):
    def __init__(self, backbone, proj_head):
        super().__init__()
        self.backbone = backbone
        self.head     = proj_head

    def forward(self, x, return_attn=False):
        if return_attn and hasattr(self.backbone, 'forward'):
            try:
                feat, attn = self.backbone(x, return_attn = True)
                return self.head(feat), attn
            except TypeError:
                pass
        feat = self.backbone(x)
        return self.head(feat)
