"""
This file contains the code to start the training of `run.py`

Usage:
 python run.py 
"""

from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms as T

from augmentations import DINOAugmentations
from trainer import train_dino
from evaluate import run_eval
from visualize import plot_pca, plot_tsne, plot_loss

# Global configs:

VERSION  = 'v1'
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
BATCH    = 16   # effective = BATCH * accum_steps (4) = 64
EPOCHS   = 50
LR       = 1e-3
N_SUBSET = 2000 # how many Cifar images to use 

print(f"Training DINO {VERSION} | Device : {DEVICE} | Subset : {N_SUBSET}")

# Dataset
aug = DINOAugmentations(
    image_size = IMG_SIZE,
    n_local    = 1
)

class MultiViewCIFAR(torch.utils.data.Dataset):
    """
    Wraps a CIFAR subset so __getitem__ returns multi-view augmented tensors
    """

    def __init__(self, base_dataset, multi_view_transform):
        self.ds = base_dataset
        self.mv = multi_view_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        views      = self.mv(img)
        return views, label

def collate_views(batch):
    """
    Stack each view postition across the batch.

    input: list of (views, label) pairs
    output: tensor batch ready for DINO loss
    """ 

    views_list, labels = zip(*batch)

    # views_list is a list of lists/tensors: [[global_view1,...],[global_view2,...]]

    n_views = len(views_list[0]) # number of augmented views per sample
    stacked = [torch.stack([v[i] for v in views_list]) for i in range(n_views)]
    return stacked, torch.tensor(labels)

# Download CIFAR-10 (auto-download) Mode
raw_train = CIFAR10(root     = "./cifar10_data",
                    train    = True,
                    download = True)

raw_test  = CIFAR10(root    = "./cifar10_data",
                    train   = False,
                    download = True)

CLASS_NAME = raw_train.classes

# Random subset 
torch.manual_seed(42)
train_indices = torch.randperm(len(raw_train))[:N_SUBSET].tolist()
train_subset  = Subset(raw_train, train_indices)

# Training Loader (multi-view augmentation)
train_ds = MultiViewCIFAR(train_subset, aug)
train_loader = DataLoader(train_ds,
                         batch_size   = BATCH,
                         shuffle      = True,
                         collate_fn   = collate_views,
                         num_workers  = 0,
                         pin_memory   = True,
                         drop_last    = True)

# Evals
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_train_ds = CIFAR10(root      = './cifar10_data', 
                        train     = True, 
                        download  = True, 
                        transform = eval_transform)

eval_test_ds  = CIFAR10(root      = './cifar10_data',
                        train     = False,
                        download  = False,
                        transform = eval_transform)

eval_train_loader = DataLoader(
    Subset(eval_train_ds, train_indices),
    batch_size=64, num_workers=2,
)
eval_test_loader = DataLoader(
    eval_test_ds,
    batch_size=64, num_workers=2,
)

# Train DINO

student, teacher, history = train_dino(
    version    = VERSION,
    dataloader = train_loader,
    epochs     = EPOCHS,
    student_lr = LR,
    device     = DEVICE,
    save_path  = f"dino_{VERSION}.pt"
)

# Evaluate
train_feats, test_feats, train_labels, test_labels = run_eval(
    student, eval_train_loader, eval_test_loader, device=DEVICE
)

# Visualize
plot_pca(
    test_feats, test_labels,
    CLASS_NAME, title=f"PCA {VERSION} ({N_SUBSET} imgs)"
)

plot_tsne(test_feats, test_labels, CLASS_NAME,
          title=f"t-SNE — DINO {VERSION} ({N_SUBSET} imgs)")

plot_loss({VERSION: history})

# Interpretabilty
