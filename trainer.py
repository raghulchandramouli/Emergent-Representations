"""
This file contains the implementation of the DINO trainer class.

It contains:
- build_dino
- update_teacher
- train_one_epoch
- train_dino
"""

from copy import copy
import torch, copy
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from models import TinyCNN, ViTTiny, ProjectionHead, DINONetwork
from loss import DINOLoss


def build_dino(version   = 'v1',
              embed_dim  = 256,
              proj_dim   = 128,
              image_size = 64):

    """
    Factory: build student + teacher for v1/v2/v3

    v1 : TinyCNN + linear head, no head,
    v2 : TinyCNN + MLP head, centering ON
    v3 : ViT-Tiny + MLP head, centering ON
    """

    if version in ('v1', 'v2'):
        backbone = TinyCNN(embed_dim=embed_dim)
    
    else: # v3
        backbone = ViTTiny(img_size   = image_size,
                           patch_size = 8,
                           embed_dim  = 192,
                           depth      = 6,
                           num_heads  = 3)
        
        embed_dim = 192 # ViT output dim

    use_mlp = (version != 'v1')
    head    = ProjectionHead(
        embed_dim,
        hidden_dim = 512,
        out_dim    = proj_dim,
        use_mlp    = use_mlp
    )

    student = DINONetwork(backbone, head)
    teacher = copy.deepcopy(student)

    for p in teacher.parameters():
        p.requires_grad_(False) # teacher::no grad!
    
    use_centering = (version != 'v1')
    criterion = DINOLoss(
        out_dim       = proj_dim,
        use_centering = use_centering
    )

    return student, teacher, criterion

