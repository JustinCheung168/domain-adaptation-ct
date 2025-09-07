import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, images, labels1, labels2, transform=None):
        # Need to add shuffling
        self.images = images  # Should be torch.Tensor of shape [N, 3, 224, 224]
        self.labels1 = labels1
        self.labels2 = labels2

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label1 = int(self.labels1[idx])
        label2 = int(self.labels2[idx])

        if self.transform:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "labels1": int(label1) if torch.is_tensor(label1) else label1,
            "labels2": int(label2) if torch.is_tensor(label2) else label2,
        }

