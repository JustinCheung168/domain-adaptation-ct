import os

import numpy as np


def import_data(directory, save_path=None, save=False):

    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            loaded_data = np.load(file_path)
            data.append(loaded_data)

    # concatenate the data from all files
    all_data = {}
    for key in data[0].keys():
        all_data[key] = np.concatenate([d[key] for d in data], axis=0)

    # check the shape of the concatenated data
    for key, value in all_data.items():
        print(f"{key}: {value.shape}")  

    if save:
        if save_path is None:
            save_path = f'datasets/{directory}_concatenated_data.npz'
        np.savez(save_path, **all_data)
        print(f"Data saved to {save_path}")

    return all_data