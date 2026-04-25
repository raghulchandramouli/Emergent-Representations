"""
This file contains the code to visualize the embeddings obtained from the trained model

This Includes:
1. PCA Plot - 2D Projection of Embeddings
2. TSNE Plot - 2D Projection of Embeddings
3. Att_maps  - Extract from ViTs. only for DINOv3 
"""

from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F

def plot_pca(feats, labels, class_name=None, title="PCA Embeddings"):
    pca  = PCA(n_components=2)
    proj = pca.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8,6))
    cmap = cm.get_cmap("tab10", len(np.unique(labels)))
    
    for cls in np.unique(labels):
        mask = labels == cls
        name = class_name[cls] if class_name else str(cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], s = 20, alpha=0.7,
                   color = cmap(cls), label = name)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig("pca_embedding.png", dpi=150) 
    plt.show()
    print('saved: pca_embedding.png')

def plot_tsne(feats, labels, class_name=None, title="TSNE Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    proj = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(9,7))
    cmap = cm.get_cmap("tab10", len(np.unique(labels)))
    
    for cls in np.unique(labels):
        mask = labels == cls
        name = class_name[cls] if class_name else str(cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], s = 20, alpha=0.7,
                   color = cmap(cls), label = name)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig("tsne_embedding.png", dpi=150) 
    plt.show()
    print('saved: tsne_embedding.png')

# Attention Map visualization for  v3 ViT
@torch.no_grad()
def get_attention_maps(vit_model, img_tensor, head_idx=0, layer_idx=-1):
    """
    Extract attention maps from ViT

    img_tensor : (1, 3, H, w)
    Returns    : attention maps (num_patches_h, num_patches_w)
    """ 

    vit_model.eval()
    _, all_attn = vit_model.get_all_attns(img_tensor, return_attn = True)

    # Pick last transformer layer attention
    attn = all_attn[layer_idx]
    attn = attn[0]

    # CLS Token
    cls_attn = attn[head_idx, 0, 1:]

    num_patches = cls_attn.shape[0]
    grid_size   = int(num_patches ** 0.5)
    attn_map    = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    return attn_map

def visualize_attention(model, img_tensor, img_size=64, patch_suze=8):
    """
    side-by-side: original image | attention map per head
    """

    n_heads = model.backbone.blocks[-1].attn.num_heads
    fig, axes = plt.subplots(1, n_heads + 1, figsize=(3 * (n_heads + 1), 3))

    # Original image (denormalized)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_show = (img_tensor[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)

    axes[0].imshow(img_show)
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis('off')

    for h in range(n_heads):
        attn = get_attention_maps(model, img_tensor, head_idx=h)
        attn_up = F.interpolate(
            torch.tensor(attn).unsqueeze(0).unsqueeze(0).float(),
            size = (img_size, img_size), mode='bilinear', align_corners=False
        )[0, 0].numpy()

        axes[h + 1].imshow(img_show) 
        im = axes[h + 1].imshow(attn_up, alpha = 0.6, cmap = 'hot')
        axes[h + 1].set_title(f"Head {h}", fontsize = 9)
        axes[h + 1].axis('off')

    plt.suptitle("DINO self attn maps", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("attn_map.png", dpi=150, bbox_inches='tight')
    plt.show()
    print('saved: attn_map.png')

def plot_loss(histories: dict):
    """histories = {"v1": [...], "v2": [...], "v3": [...]}"""
    plt.figure(figsize=(8, 4))
    for name, h in histories.items():
        plt.plot(h, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("DINO Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()