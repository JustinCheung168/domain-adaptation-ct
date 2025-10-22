#!/usr/bin/env python3
import os
import sys

# Add root of repo to Python path.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Enable logging.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF


# Transform to resize and convert to 3 channels
def get_transform(rotation=0):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.rotate(img, rotation)),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Source: standard MNIST
source_dataset = datasets.MNIST(
    root="/data/mnist_data", train=True, download=True, transform=get_transform(rotation=0)
)

# Target: rotated MNIST
target_dataset = datasets.MNIST(
    root="/data/mnist_data", train=True, download=True, transform=get_transform(rotation=45)
)

# Dataloaders
batch_size = 32
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
