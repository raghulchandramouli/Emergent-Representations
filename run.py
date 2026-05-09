"""
This file contains the code to start the training of `run.py`

Usage:
 python run.py 
"""

from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
# pyrefly: ignore [missing-import]
from datasets import load_dataset

from augmentations import DINOAugmentations
from interpret import DINOInterpreter
from trainer import train_dino
from evaluate import run_eval
from visualize import plot_pca, plot_tsne, plot_loss

# Global configs:
VERSION  = 'v3'
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
BATCH    = 16   # effective = BATCH * accum_steps (4) = 64
EPOCHS   = 100
LR       = 1e-4
N_SUBSET = 10000 # how many Cifar images to use 

print(f"Training DINO {VERSION} | Device : {DEVICE} | Subset : {N_SUBSET}")


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

class HF_CIFAR10(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace CIFAR-10 split into a PyTorch Dataset.
    Returns (PIL image, int label) — compatible with torchvision transforms.
    """
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform
        self.classes = CIFAR_CLASSES
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item  = self.ds[idx]
        img   = item["img"]       # PIL Image (HF provides this directly)
        label = item["label"]     # int 0–9
        if self.transform:
            img = self.transform(img)
        return img, label


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


# Download via HFHub
hf_train = load_dataset("cifar10", split="train")
hf_test  = load_dataset("cifar10", split="test")

# Download CIFAR-10 (auto-download) Mode
raw_train = HF_CIFAR10(hf_train)

raw_test  = HF_CIFAR10(hf_test)
CLASS_NAME = raw_train.classes

# Random subset 
torch.manual_seed(42)
train_indices = torch.randperm(len(raw_train))[:N_SUBSET].tolist()
train_subset  = Subset(raw_train, train_indices)

# Training Loader (multi-view augmentation)
aug     = DINOAugmentations(image_size=IMG_SIZE, n_local=1)
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

eval_train_ds = HF_CIFAR10(hf_train, transform = eval_transform)
eval_test_ds  = HF_CIFAR10(hf_test, transform = eval_transform)

eval_train_loader = DataLoader(
    Subset(eval_train_ds, train_indices),
    batch_size=64, num_workers=0,
)

eval_test_loader = DataLoader(
    eval_test_ds,
    batch_size=64, num_workers=0,
)

# Train DINO
student, teacher, history = train_dino(
    version    = VERSION,
    dataloader = train_loader,
    epochs     = EPOCHS,
    lr         = LR,
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
    CLASS_NAME,
    title=f"PCA {VERSION} ({N_SUBSET} imgs)",
    save=f"pca_embeddings_{VERSION}.png",
)

plot_tsne(test_feats, test_labels, CLASS_NAME,
          title=f"t-SNE - DINO {VERSION} ({N_SUBSET} imgs)",
          save=f"tsne_embeddings_{VERSION}.png")

plot_loss({VERSION: history})

# Interpretabilty
interp = DINOInterpreter(
    student, device = DEVICE,
    version = VERSION, img_size=IMG_SIZE, patch_size=8,
)

# Grab sample test images
sample_imgs   = [eval_test_ds[i][0].unsqueeze(0) for i in range(6)]
sample_labels = [CIFAR_CLASSES[eval_test_ds[i][1]] for i in range(6)]

# single-image dashboard
interp.explain(
    sample_imgs[0],
    label=sample_labels[0],
    save=f"explain_dashboard_{VERSION}.png",
)

# Batch explaination grid
interp.explain_batch(
    sample_imgs,
    sample_labels,
    save=f"batch_explain_{VERSION}.png",
)

if VERSION == "v3":
    interp.tracking_attention(
        sample_imgs[0],
        heads_to_show=(0, 1, 2),
        save=f"attn_tracking_{VERSION}.png",
    )
    interp.entropy_analysis(sample_imgs[0], save=f"entropy_heatmap_{VERSION}.png")

print("PNGs saved in directory")
