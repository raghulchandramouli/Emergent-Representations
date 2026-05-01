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

class LRPExplainer:
    """
    LRP for CNN backbones (v1/v2) using the epsilon-rule.
    Theory:
        Relevance flows backwards through layers.
        Each neuron receives relevance proportional to its activation.
        R_j = Σ_k [ (z_jk / Σ_j z_jk + ε) * R_k ]
    Supports: Linear, Conv2d, BatchNorm2d (merged), ReLU (pass-through).
    """
    def __init__(self, model, device="cpu", eps=1e-6):
        self.model  = model.eval()
        self.device = device
        self.eps    = eps
        self._hooks = []
        self._acts  = {}   # layer_name → activation
        self._grads = {}   # layer_name → gradient

    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        self._acts.clear()

        def make_hook(name):
            def hook(module, inp, out):
                self._acts[name] = out.detach()
            return hook

        for name, module in self.model.backbone.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        
    @staticmethod
    def _lrp_linear(layer, activation, relevance, eps=1e-6):
        """
        Epsilon-LRP for a Linear layer.
        z = W·a   (pre-activation)
        R_j = a_j * Σ_k [ W_jk * R_k / (z_k + ε·sign(z_k)) ]
        """
        W = layer.weight.data                                   # (out, in)
        b = layer.bias.data if layer.bias is not None else 0
        z  = activation @ W.T + b                               # (B, out)
        s  = relevance / (z + eps * z.sign().clamp(min=1e-3))   # (B, out)
        c  = s @ W                                              # (B, in)
        return activation * c                                   # (B, in)

    @staticmethod
    def _lrp_conv(layer, inp, relevance, eps=1e-6):
        """
        Epsilon-LRP for a Conv2d layer (via gradient trick).
        Gradient trick: R_in = inp * ∂(z·s)/∂inp
        where s = R / (z + ε)
        """
        inp = inp.detach().requires_grad_(True)
        z   = layer(inp)
        s   = (relevance / (z + eps)).detach()
        (z * s).sum().backward()
        return (inp * inp.grad).detach()

    def _lrp_pass(self, inp, relevance):
        """Pass-through rule for activations / pooling."""
        return relevance  # same shape assumed
    
    @torch.no_grad()
    def explain(self, img_tensor):
        """
        Compute pixel-level relevance map.
        img_tensor: (1, 3, H, W)
        Returns:    (H, W) numpy array — relevance per pixel
        """
        img = img_tensor.to(self.device)
        # Forward pass to get all activations
        self._register_hooks()
        with torch.enable_grad():
            img.requires_grad_(True)
            out = self.model.backbone(img)           # (1, embed_dim)
        self._remove_hooks()

        # Start relevance from output (all equal)
        R = out.detach()                             # (1, embed_dim)
        # Walk through backbone layers in reverse

        layers = [(n, m) for n, m in self.model.backbone.named_modules()
                  if isinstance(m, (nn.Linear, nn.Conv2d))]
        # Propagate through linear layers (fc at end)

        for name, layer in reversed(layers):
            act = self._acts.get(name)
            if act is None:
                continue
            if isinstance(layer, nn.Linear):
                R = self._lrp_linear(layer, act, R, self.eps)
            elif isinstance(layer, nn.Conv2d):
                # Need inp for conv — approximate with activation of prev layer
                # (For a cleaner impl, track inputs too)
                inp_approx = act  # approximation
                R_spatial   = R.view(act.shape) if R.numel() == act.numel() else R

                try:
                    R = self._lrp_conv(layer, act, R_spatial, self.eps)
                except Exception:
                    pass  # shape mismatch — skip

        # R now has shape close to input; aggregate channels
        with torch.enable_grad():
            img2 = img_tensor.to(self.device).requires_grad_(True)
            loss = self.model.backbone(img2).sum()
            loss.backward()
            saliency = img2.grad[0].abs().mean(0).cpu().numpy()  # (H, W)
        return saliency  # fallback to gradient-based if LRP shape mismatch

# LRP - Layer-wise Relavance Propagation
class LRP_ViT:
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
        return LRP_ViT.rollout(all_attn).numpy()

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

            rollout_flat = LRP_ViT._attention_rollout(all_attn).numpy()
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

# GradCAM (CNN backbone - v1/v2)

class GradCAM:
    """
    Gradient-weighted Class Activation Map for CNN.
    Works on any Conv2D layer. 

    Viz: which spatial region most activates the repr?
    """

    def __init__(self, model, target_layer_name = None, device = 'cpu'):
        self.model  = model.eval()
        self.device = device 
        self._feat  = None
        self._grad  = None 

        # Auto-select last conv layer
        self.target_layer = self._find_layer(target_layer_name)
        self._register()

    def _find_layer(self, name = None):
        last_conv = None
        for n, m in self.model.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _register(self):
        def fwd(module, inp, out):
            self._feat = out

        def bwd(module, grad_in, grad_out):
            self._grad = grad_out[0]

        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def explain(self, img_tensor):
        """
        Returns (H, W) GradCAM heatmap.
        """

        img = img_tensor.to(self.device).requires_grad_(True)
        out = self.model.backbone(img) # (1, embed_dim)

        self.model.zero_grad()
        out.sum().backward()

        # Weight feature maps by global-averaged gradient
        weights = self._grad.mean(dim=[2, 3], keepdim=True)        # (1, C, 1, 1)
        cam     = (weights * self._feat).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)
        
        # Upsample to input size
        H = img_tensor.shape[-1]
        cam = F.interpolate(cam, size=(H, H),
                            mode="bilinear", align_corners=False) 
        
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam      

# Master Interpreter - SEP

class DINOInterpreter:

    """
    One-stop interpretabilty for any DINO Version

        interp = DINOInterpreter(student_model, "dino_v1.pt", device='cpu')
        interp.explain(img_tensor, label='dog')
        interp.track_attention(img_tensor)
        interp.plot_attn_evolution(img_tensor)
    """

    def __init__(self, model, device='cpu', version='v3', img_size=64, patch_size=8):

        self.model     = model.eval().to(device)
        self.device    = device
        self.version   = version
        self.img_size  = img_size
        self.patch_size = patch_size

        # init LRP
        if version == 'v3':
            self.lrp = LRP_ViT(model, device)
        else:
            self.lrp = LRPExplainer(model, device)

        # init GradCAM
        self.gradcam      = GradCAM(model, device) if version != 'v3' else None

        # Attn init
        self.attn_tracker = AttentionTracker(model, device) if version == 'v3' else None

    # Main dashboard
    def explain(self, img_tensor, label="", save='explain_dashboard.png'):
        """
        Full dashboard showing multiple explanations for a single image
            - Original
            - LRP relevance map
            - GradCAM (CNN) or Attention Rollout (ViT)
            - Combined overlay
        """ 

        img_np = denormalize(img_tensor)

        # Compute maps
        lrp_map = self.lrp.explain(img_tensor.to(self.device))

        if self.gradcam:
            cam_map   = self.gradcam.explain(img_tensor)
            cam_title = "GradCAM"
        elif self.attn_tracker:
            cam_map = AttentionAnalyzer(
                self.model, self.device
            ).rollout(img_tensor.to(self.device))

            cam_map = to_spatial(torch.tensor(cam_map),
                                 self.img_size,
                                 self.patch_size) 

            cam_title = "Attention Rollout"

        else:
            cam_map   = lrp_map
            cam_title = "LRP"

        # Norm maps
        def norm(m):
            return (m-m.min()) / (m.max() - m.min() + 1e-8) 
        
        lrp_map = norm(lrp_map)
        cam_map = norm(cam_map)

        combined = norm(0.5 * lrp_map + 0.5 * cam_map)

        # plot 
        fig, axes = plt.subplots(1, 4, figsize = (14, 3.5))
        fig.suptitle(f"DINO {self.version} Interpretability - {label}",
                     fontsize = 12, fontweight = "bold")

        axes[0].imshow(img_np)
        axes[0].set_title("Original", fontsize = 10)
        axes[0].axis('off')

        overlay_heatmap(axes[1], img_np, lrp_map, title="LRP Relevance", cmap=_BWR)
        overlay_heatmap(axes[2], img_np, cam_map, title=cam_title, cmap="jet")
        overlay_heatmap(axes[3], img_np, combined, title="LRP + Rollout (combined)", cmap="hot")

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches = 'tight')
        plt.show()
        print(f"saved {save}")

        return {"LRP" : lrp_map, 
                "CAM" : cam_map,
                "combined" : combined,
        }

        # Attention Tracking

        def tracking_attention(
            self, img_tensor, heads_to_show=(0, 1, 2)
        ):

            """
            Plot attention evolution for specific heads
            """
            if self.attn_tracker is None:
                return 

            self.attn_tracker.plot_layer_evolution(
                img_tensor.to(self.device),
                img_size = self.img_size,
                patch_size = self.patch_size,
                heads_to_show = heads_to_show,
            )

        def entropy_analysis(self, img_tensor):
            """
            Entropy heatmap + across layer x heads
            """

            if self.attn_tracker is None:
                return
            
            self.attn_tracker.plot_entropy_heatmap(
                img
            )