def preprocess_data(data, distortions, include_original=True, save_data = False, save_path=None):

    keys = list(data.keys())

    if 'Ring_Artifact_v1' in distortions:
        ring_flag = True
        distortions.remove('Ring_Artifact_v1')
    else:
        ring_flag = False

    if include_original:
        images = [data[keys[0]]]
    else:
        images = []

    for distortion in distortions:
        images.append(data[distortion])

    labels = data[keys[1]]
    ring_labels = data[keys[-1]]

    normalized_images = []
    for image in images:
        normalized_images.append(normalize_images(image))

    zero_labels = np.zeros_like(labels)
    one_labels = np.ones_like(labels)

    if include_original:
        domain_label_list = [zero_labels]
        expanded_label_list = [labels]
    else:
        domain_label_list = []
        expanded_label_list = []

    for _ in distortions:
        domain_label_list.append(one_labels)
        expanded_label_list.append(labels)

    if domain_label_list != []:
        domain_labels = np.concatenate(domain_label_list, axis=0)
        
    if expanded_label_list != []:
        expanded_labels = np.concatenate(expanded_label_list, axis=0)

    concatenated_images = np.concatenate(normalized_images, axis=0)

    if ring_flag:
        ring_images = data['Ring_Artifact_v1']
        ring_labels = data['ring_labels']
        ring_images = normalize_images(ring_images)
        concatenated_images = np.concatenate((concatenated_images, ring_images), axis=0)
        domain_labels = np.concatenate((domain_labels, one_labels), axis=0)
        expanded_labels = np.concatenate((expanded_labels, ring_labels), axis=0)


    print(len(concatenated_images), len(expanded_labels), len(domain_labels))
    assert len(concatenated_images) == len(expanded_labels) == len(domain_labels), "Dataset length mismatch!"

    # Shuffle the concatenated images and labels
    seed = 42
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(concatenated_images))
    concatenated_images = concatenated_images[shuffled_indices]
    expanded_labels = expanded_labels[shuffled_indices]
    domain_labels = domain_labels[shuffled_indices]

    if save_data:
        if save_path is None:
            raise ValueError("save_path must be specified if save_data is True")
        np.savez_compressed(save_path, images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

    dataset = CustomImageDataset(images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

    return dataset