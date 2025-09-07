import numpy as np

from src.dataset.custom_image_dataset import CustomImageDataset

def dataset_load(file_path):

    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels1 = data['labels1']
    labels2 = data['labels2']

    return CustomImageDataset(images, labels1, labels2)

