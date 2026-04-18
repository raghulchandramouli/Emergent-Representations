"""
This file contains the augmentations used for DINO. (my version)
Not so much for a faithful implementation of the paper.
"""

import torchvision.transforms as T
import torch

class DINOAugmentations:
    """
    Multi-view augmentations for DINO.
    Each Image -> [global_1, global_2, global_3] crops.
    """

    def __init__(self, image_size=64, n_local=1):
        self.n_local = n_local

        # Global Crops: large random crops (> 50% of crops)
        self.global_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=image_size // 10 * 2 + 1, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

        # Local Crops: Small random crops (< 50% of images) ~ Optional with respect to the paper
        self.local_transform = T.Compose([
            T.RandomResizedCrop(image_size // 2, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        views = [
            self.global_transform(image),
            self.global_transform(image)
        ]

        for _ in range(self.n_local):
            views.append(self.local_transform(image))
        return views # list of tensor
        