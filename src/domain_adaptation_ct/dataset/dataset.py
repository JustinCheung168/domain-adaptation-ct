from typing import Dict

import numpy as np
import torch
from torchvision import transforms

class OneLabelDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels1):
        self.images = images
        self.labels1 = labels1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img = self.images[idx]
        label1 = int(self.labels1[idx])

        if self.transform:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "labels1": label1,
        }

class TwoLabelDataset(OneLabelDataset):
    def __init__(self, images, labels1, labels2):
        super().__init__(images, labels1)
        self.labels2 = labels2

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        label2 = int(self.labels2[idx])
        sample["labels2"] = label2
        return sample

def one_label_dataset_load(file_path):
    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels1 = data['labels1']

    return OneLabelDataset(images, labels1)

def two_label_dataset_load(file_path):
    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels1 = data['labels1']
    labels2 = data['labels2']

    return TwoLabelDataset(images, labels1, labels2)
