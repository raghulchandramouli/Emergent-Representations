"""
interpret.py - DINO Interpretabilty 

Usage:
    1. LRP - Layer-wise Relavance Propagation - (CNN & ViT)
    2. Attention Score - raw, rollout, head variance
    3. Attention Tracking - per-layer evolution across depth
    4. GradCam (bonus, works on CNN backbone)
    5. summary dashboard - single call to plot it all
"""

from IPython.core.pylabtools import figsize
import torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

# Helpers
def denormalize(img_tensor):
    """(1, 3, H, W) normalized tensor -> (H, W, 3) numpy for display"""

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img  = (img_tensor[0].cpu() * std + mean).clamp(0, 1)
    return img.permute(1,2,0).numpy()

def overlay_heatmap(ax, img_np, heatmap, title="", alpha=0.55, cmap="jet"):
    "Overlay a heatmap on the original image."

    ax.imshow(img_np)
    ax.imshow(heatmap, alpha=alpha, cmap=cmap,
             vmin=heatmap.min(), vmax=heatmap.max())
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def to_spatial(flat_map, img_size, patch_size):
    """Upsample flat patch relevance to image resolution."""
    grid = int(flat_map.shape[-1] ** 0.5)
    spatial = flat_map.reshape(1, 1, grid, grid).float()
    return F.interpolate(spatial, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)[0, 0].numpy()

# Custom colormap: blue (negative) → white → red (positive)
_BWR = LinearSegmentedColormap.from_list(
    "bwr_custom", ["#2166ac", "#f7f7f7", "#d6604d"], N=256
)

# LRP - Layer-wise Relavance Propagation

class LRPExplainer:
    """
    LRP for CNN using the epsilon-rule.

    Theory:
        Relevance flows backward through layers.
        Each neuron receives 'relevance' from the layer above and its prop to its activation.
    
    Supports:
        Linear, Conv2D, BatchNorm2D, ReLU
    """

    def __init__(self, model, device='cpu', eps=1e-6):
        self.model  = model.eval()
        self.device = device

    def explain(self, img_tensor):
        """
        Returns (H, W) relevance map for ViT
        """

        img = img_tensor.to(self.device)
        
        # Forward with gradient tracking
        img.requires_grad_(True)
        cls_feat, all_attn = self.model.backbone(img, return_attn = True)

        # Use gradients of sum(CLS Embeddings) wrt input
        loss = cls_feat.sum()
        loss.backward()

        grad = img.grad[0].abs().mean(0).cpu()

        # Also compute attention rollout
        rollout     = self._attention_rollout(all_attn)         #(num_patches,)
        img_size    = img_tensor.shape[-1] 
        patch_size  = img_size // int(rollout.shape[0] ** 0.5)
        rollout_map = to_spatial(rollout, img_size, patch_size) #(H, W)

        # Combine: element-wise product of rollout and gradient map
        combined = rollout_map * grad.numpy()
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        return combined

    @staticmethod
    def _attention_rollout(all_attn):
        """
        Attention Rollout 
        Propagates attention through all layers to get.
        total information flow from CLS token to each patch.

        all_attn: list of attention tensors (num_layers, B, n_heads, n_tokens, n_tokens)

        Returns: (num_patches,) relavance
        """

        # each attn: (1, num_heads, N+1, N+1)
        rollout = torch.eye(all_attn[0].shape[-1])

        for attn in all_attn:
            # Average over heads
            a = attn[0].mean(0) # (N+1, N+1)
            # Add identity (residual connection)
            a = a + torch.eye(a.shape[0], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True) # row-norm
            rollout = a @ rollout

        # CLS token row: how much attention flows from CLS to each patch
        cls_rollout = rollout[0, 1:] # exclude CLS token itself
        return cls_rollout.detach().cpu()

# Attnetion Score analysis
class AttentionAnalyzer:
    """
    Full attention score analysis for ViT-Tiny (v3).

    Methods:
        raw_attention()     -> per-head attention maps.
        head_variance()     -> which heads are specialized
        attention_entropy() -> how focused is each head?
        rollout()           -> attention rollout for each head
    """
    def __init__(self, model, device = 'cpu'):
        self.model  = model.eval()
        self.device = device

    @torch.no_grad()
    def _get_all_attn(self, img_tensor):
        """
        Forward a batch of images and return all attention maps.

        input: tensor of shape (B, 3, H, W)
        output: all_attn (num_layers, B, num_heads, N+1, N+1)
        """

        img_tensor = img_tensor.to(self.device)
        _, all_attn = self.model.backbone(img_tensor, return_attn = True)
        return all_attn # list of (1, num_heads, N+1, N+1)

    def raw_attention(self, img_tensor, layer_idx=-1):
        """
        Raw CLS -> patch attention for each head at a given layer.
        Returns: (num_heads, num_patches) numpy array.
        """

        all_attn = self._get_all_attn(img_tensor)
        attn     = all_attn[layer_idx][0]     # (H, N + 1, N + 1)
        cls_attn = attn[0, 1:].cpu().numpy()  # (N+1,) -> (N,)
        return cls_attn

    def head_variance(self, img_tensor):
        """
        Variance of CLS -> patch attention across patches per head per layer
        High Variance = head is focused / specilized
        Low  Variance = head is diffuse / attends uniformely.

        Returns: (n_layer, n_heads) numpy array
        """

        all_attn = self._get_all_attn(img_tensor)
        variance = []

        for attn in all_attn:
            a = attn[0: :, 0, 1:].cpu().numpy() # (n_heads, N)
            variance.append(a.var(axis=-1))     # (n_heads,)
        return np.stack(variance)               # (n_layers, n_heads)

    def attention_entropy(self, img_tensor):
        """
        Shannon entropy of CLS -> patch attention per head per layer

        Low entropy  = focused attention (interpretable)
        High entropy = diffuse attention (uniform)

        Returns: (n_layer, n_heads)
        """

        all_attn = self._get_all_attn(img_tensor)
        entropies = []

        for attn in all_attn:
            a   = attn[0, :, 0, 1:]                     # (n_heads, N)
            a   = a / (a.sum(-1, keepdim=True) + 1e-8)  # normalize
            ent = -(a * (a + 1e-8).log()).sum(-1)       # entropy for each head
            entropies.append(ent.cpu().numpy())
        return np.stack(entropies) 

    def rollout(self, img_tensor):
        """
        Attention rollout across all layer -> (num_patches,).
        """

        all_attn = self._get_all_attn(img_tensor)
        return LRPExplainer.rollout(all_attn).numpy()

# Attention Tracking - Per-Layer Evolution

class AttentionTracker:
    """
    Track how attn patterns evolve across transformer depth.

    Useful for understanding:
        - Which layer attend locally vs globally
        - At what depth semantic features emerge
        - Head specialization patters
    """
    
    def __init__(self, model, device='cpu'):
        self.analyzer = AttentionAnalyzer(model, device)
        self.model    = model
        self.device   = device
        self.img_size = None

    @torch.no_grad()
    def track(self, img_tensor, img_size=64, patch_size=8):
        """
        Compute per-layer attention maps for all heads

        Returns: dict with keys:
            `per_layer_maps` : list of (n_heads, H, W) spatial maps
            `entropy`        : (n_layers, n_heads)
            `variance`       : (n_layers, n_heads)
            `rollout`        : (num_patches,)
        """  

        self.img_size   = img_size
        self.patch_size = patch_size

        all_attn = self.analyzer._get_all_attn(img_tensor)
        n_layers = len(all_attn) 
        n_heads  = all_attn[0].shape[1] # 6 

        per_layer_maps = []
        for layer_idx, attn in enumerate(all_attn):

            head_maps = []
            for h in range(n_heads):
                cls_attn = attn[0, h, 0, 1:].cpu() # (N,)
                spatial  = to_spatial(cls_attn, img_size, patch_size)
                spatial  = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-6)
                head_maps.append(spatial)

            per_layer_maps.append(head_maps) # [n_layer][n_heads] -> (H, W)

            rollout_flat = LRPExplainer._attention_rollout(all_attn).numpy()
            rollout_map  = to_spatial(torch.tensor(rollout_flat), img_size, patch_size)

            return {
                "per_layer_maps" : per_layer_maps,
                "entropy"        : self.analyzer.attention_entropy(img_tensor),
                "variance"       : self.analyzer.head_variance(img_tensor),
                "rollout"        : rollout_map
            }

    def plot_layer_evolution(self,
                            img_tensor,
                            img_size=64,
                            patch_size=8,
                            heads_to_show=(0,1,2),
                            save="attn_tracking.png"):

        """

        Grid plot: selected heads, cols = transformer layers.
        shows how each head's attn changes with depth.
        """

        data      = self.track(img_tensor, img_size, patch_size)
        per_layer = data['per_layer_maps']
        n_layer   = len(per_layer)
        img_np    = denormalize(img_tensor)
        n_heads_show = len(heads_to_show)

        fig, axes = plt.subplots(
            n_heads_show, n_layer + 1,
            figsize=(2.5 * (n_layer + 1), 2.5 * n_heads_show)
        ) 

        if n_heads_show == 1:
            axes = axes[np.newaxis, :] 
        
        for row, h in enumerate(heads_to_show):
            # Show input image
            axes[row, 0].imshow(img_np) 
            axes[row, 0].set_title(f"Head {h}\n(input)", fontsize=7)
            axes[row, 0].axis('off')

            for col, layer_idx in enumerate(range(n_layer)):
                # Show attention heatmap for this head at this layer
                attn_map = per_layer[layer_idx][h]
                overlay_heatmap(axes[row, col + 1], img_np, attn_map,
                                title=f"L{layer_idx}", alpha=0.6, cmap='inferno')

        plt.suptitle("Attention Evolution Across Transformer Depth", fontsize=11, y = 1.01)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"saved {save}")
        return data
    
    def plot_entropy_heatmap(self,
                            img_tensor,
                            save = "entropy_heatmap.png"):
        """
        Heatmap of attention entropy across layers for each head.
        Low entropy = focused; High = diffuse.
        """

        entropy = self.analyzer.attention_entropy(img_tensor)

        fig, ax = plt.subplots(figsize=(max(6, entropy.shape[0]), 3))
        im      = ax.imshow(entropy.T, aspect='auto', cmap = "YlOrRd")
        ax.set_xlabel("Transfomer Layer")
        ax.set_ylabel("Attention Head")
        ax.set_xticks(range(entropy.shape[0]))
        ax.set_yticks(range(entropy.shape[1]))
        plt.colorbar(im, ax=ax, label="Entropy (nats)")
        plt.tight_layout()
        plt.savefig(save, dpi=150)
        plt.show()
        print(f"saved {save}")
