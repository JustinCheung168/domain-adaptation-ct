import os

import numpy as np

def combine_data(orig_data, new_data):
    combined_data = dict(orig_data)

    combined_data['Ring_Artifact_v1'] = new_data['Ring_Artifact_v1']
    combined_data['ring_labels'] = new_data['label']

    return combined_data

def combine_npzs(data_dir):
    combined_data = {}
    order = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'last']
    # Process files in the order specified by the 'order' list
    for pos in order:
        for filename in os.listdir(data_dir):
            file_split = filename.split('_')
            if pos in file_split and filename.endswith('.npz'):
                print(f"Processing file: {filename}")
                file_path = os.path.join(data_dir, filename)
                data = np.load(file_path)
                for key in data.files:
                    if key in combined_data:
                        combined_data[key] = np.concatenate((combined_data[key], data[key]), axis=0)
                    else:
                        combined_data[key] = data[key]
    return combined_data