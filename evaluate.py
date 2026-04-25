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

def knn_eval(train_feats, train_labels, test_feats, test_labels, k = 5):
    """
    knn evaluation

    Args:
        train_feats: Training embeddings
        train_labels: Training labels
        test_feats: Test embeddings
        test_labels: Test labels
        k: Number of neighbors

    Returns:
        Accuracy score
    """
    # Normalize features
    knn = KNeighborsClassifier(n_neighbors = k, metric='cosine')
    knn.fit(train_feats, train_labels)
    preds = knn.predict(test_feats)
    acc = accuracy_score(test_labels, preds)
    print(f" KNN (k={k}) Accuracy: {acc*100:.2f}%")
    return acc

def linear_probe(train_feats, train_labels, test_feats, test_labels):
    """
    linear probing
    
    Args:
        train_feats: Training embeddings
        train_labels: Training labels
        test_feats: Test embeddings
        test_labels: Test labels
    
    Returns:
        Accuracy score
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats)
    X_test = scaler.transform(test_feats)
    
    clf = LogisticRegression(max_iter = 1000, c = 1.0)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    print(f" Linear Probing Accuracy: {acc*100:.2f}%")
    return acc