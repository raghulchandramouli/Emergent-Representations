"""
This file contains the evaluation metrics for the emergent representations.

This Includes:
1. extract-embeddings
2. knn-evals
3. linear-probing
"""
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Extract (embedding, label) pairs for a model over a loader

    Args:
        model: The model to extract embeddings from
        loader: DataLoader for the dataset
        device: Device to run the model on

    Returns:
        embeddings: numpy array of shape (N, dim)
        labels: numpy array of shape (N,)
    """

    model.eval()
    all_feats, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)

        # Use backbone only (not projection head)
        feats = model.backbone(imgs)
        feats = F.normalize(feats, dim=-1) # unit norm

        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    return (np.concatenate(all_feats, axis=0),
            np.concatenate(all_labels, axis=0))

