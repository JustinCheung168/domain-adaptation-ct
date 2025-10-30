import numpy as np
import torch
from torchvision import transforms
from typing import Optional

from domain_adaptation_ct.logging.log_mixin import LogMixin

class BaseImageDataset(torch.utils.data.Dataset, LogMixin):
    """Base dataset class handling image loading and transforms."""
    
    def __init__(
        self, 
        images: np.ndarray,
        convert_grayscale_to_rgb: bool,
        additional_transforms: Optional[transforms.Compose] = None
    ):
        """
        Args:
            images: Input image array
            convert_grayscale_to_rgb: Whether to convert 1-channel grayscale images to 3-channel (ResNet input format)
            additional_transforms: Optional transforms to apply after base transforms
        """
        self.images = images
        
        # Build transform pipeline
        transform_list = [transforms.ToTensor()]
        
        if convert_grayscale_to_rgb:
            transform_list.append(
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            )
            
        if additional_transforms:
            transform_list.append(additional_transforms)
            
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.images)

class OneLabelDataset(BaseImageDataset):
    def __init__(
        self, 
        images: np.ndarray,
        labels1: np.ndarray,
        convert_grayscale_to_rgb: bool,
        additional_transforms: Optional[transforms.Compose] = None
    ):
        super().__init__(images, convert_grayscale_to_rgb, additional_transforms)
        self.labels1 = labels1

    def __getitem__(self, idx: int) -> dict[str, int]:
        img = self.images[idx]
        img = self.transform(img)
        
        return {
            "pixel_values": img,
            "labels1": int(self.labels1[idx]),
        }

    @classmethod
    def load(cls, file_path: str, convert_grayscale_to_rgb: bool) -> 'OneLabelDataset':
        """Load from an npz file"""
        data = np.load(file_path, allow_pickle=True)
        return cls(
            data['images'], 
            data['labels1'],
            convert_grayscale_to_rgb
        )

    def save(self, file_path: str) -> None:
        """Save to an npz file"""
        np.savez_compressed(
            file_path,
            images=self.images,
            labels1=self.labels1
        )

class TwoLabelDataset(BaseImageDataset):
    def __init__(
        self,
        images: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        convert_grayscale_to_rgb: bool,
        additional_transforms: Optional[transforms.Compose] = None
    ):
        super().__init__(images, convert_grayscale_to_rgb, additional_transforms)
        self.labels1 = labels1
        self.labels2 = labels2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = self.images[idx]
        img = self.transform(img)
        
        return {
            "pixel_values": img,
            "labels1": int(self.labels1[idx]),
            "labels2": int(self.labels2[idx]),
        }

    @classmethod
    def load(cls, file_path: str, convert_grayscale_to_rgb: bool) -> 'TwoLabelDataset':
        """Load from an npz file"""
        data = np.load(file_path, allow_pickle=True)
        return cls(
            data['images'],
            data['labels1'],
            data['labels2'],
            convert_grayscale_to_rgb
        )
    
    def save(self, file_path: str) -> None:
        """Save to an npz file"""
        np.savez_compressed(
            file_path,
            images=self.images,
            labels1=self.labels1,
            labels2=self.labels2
        )

class MultifoldDataset(torch.utils.data.Dataset, LogMixin):
    """
    A wrapper that combines multiple BaseImageDataset-like objects and
    provides a unified interface for __getitem__ and __len__.

    Behaves as if you concatenate the input datasets, without needing copies.
    """
    def __init__(self, datasets: list[BaseImageDataset]):
        if not datasets:
            raise ValueError("No datasets provided.")
        self.datasets = datasets

        # Precompute cumulative lengths for indexing
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)

    def __len__(self) -> int:
        """Combined length of all datasets included in this multifold dataset."""
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve item idx from the appropriate dataset."""
        ds_idx, local_idx = self._locate_dataset(idx)
        return self.datasets[ds_idx][local_idx]

    def _locate_dataset(self, idx: int):
        for dataset_num, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if dataset_num == 0:
                    prev_cum = 0
                else:
                    prev_cum = self.cumulative_lengths[dataset_num - 1]
                return dataset_num, idx - prev_cum
        raise IndexError("Index out of range")

DATASET_REGISTRY: dict[str, type[BaseImageDataset]] = {
    "OneLabelDataset": OneLabelDataset,
    "TwoLabelDataset": TwoLabelDataset,
}
