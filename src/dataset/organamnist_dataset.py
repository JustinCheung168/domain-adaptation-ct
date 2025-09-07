from torch.utils.data import Dataset
from torchvision import transforms
from medmnist import OrganAMNIST

# Prepare OrganAMNIST Dataset with dummy secondary labels for now
class OrganAMNISTDataset(Dataset):
    def __init__(self, split="train"):
        dataset = OrganAMNIST(split=split, size=224, download=True)
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label1 = self.data[idx]  # Unpack the tuple
        image = self.transform(image)
        label1 = int(label1)
        label2 = label1 % 5  # Dummy secondary label
        return {"pixel_values": image, "labels1": label1, "labels2": label2}

# Prepare OrganAMNIST Dataset with dummy secondary labels for now
class OrganAMNISTDataset_2(Dataset):
    def __init__(self, split="train"):
        dataset = OrganAMNIST(split=split, size=224, download=True)
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {
            "pixel_values": self.transform(image),
            "labels": int(label)
        }
