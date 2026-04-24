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

@torch.no_grad()
def update_teacher(student, teacher, momentum = 0.996):
    """EMA update: θ_t ← m·θ_t + (1-m)·θ_s"""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)

def train_one_epoch(
    student,
    teacher,
    criterion,
    loader,
    optimizer, 
    scaler,
    device,
    accum_steps      = 4, # gradients accumlation::simulate batch~64
    teacher_momentum = 0.996,
    n_global_views   = 2 # first 2 views are global 
):

    student.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (views, _) in enumerate(loader):

        # views::list of (B, C, H, W) tensor
        views = [v.to(device) for v in views]
        global_views = views[:n_global_views]

        with autocast(): # fp16 - cuts VRAM ~50%

            # Student: all views
            student_out = [student(v) for v in views]

            # Teacher: global views only
            with torch.no_grad():
                teacher_out = [teacher(v) for v in global_views]

            loss = criterion(student_out, teacher_out)
            loss = loss / accum_steps # normalize for accumalation
        
        scaler.scale(loss).backward()

        # Accumulate gradients for `accum_steps`:
        if (step + 1) % accum_steps == 0:

            # Gradients clip (prevents explosions)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_morm=3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update teacher
            update_teacher(
                student,
                teacher,
                teacher_momentum
            )

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)

def train_dino(
    version = 'v1',
    dataloader = None,
    epochs = 100,
    lr = 1e-3,
    device = 'cuda',
    teacher_momentum = 0.996,
    save_path = 'dino_weights.pt',
):

    student, teacher, criterion = build_dino(version = version)
    student   = student.to(device)
    teacher   = teacher.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(student.parameters(),
                                 lr = lr,
                                 weight_decay = 0.04)
    scheluler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max = epochs)
    scaler    = GradScaler()

    history   = []

    for epoch in range(epochs):

        loss = train_one_epoch(
            student,
            teacher,
            criterion,
            dataloader,
            optimizer,
            scaler,
            device,
            teacher_momentum = teacher_momentum
        )

        scheluler.step()
        history.append(loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1:3d}/{epochs}] Loss: {loss:.4f}")

    torch.save({
        "student" : student.state_dict(),
        "teacher" : teacher.state_dict(),
        "version" : version
    }, save_path)

    print(f'saved to {save_path}')
    return student, teacher, history