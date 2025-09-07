

def normalize_image(image, mean=0.5, std=0.5):
    """
    Normalize an image tensor to have a mean and standard deviation.
    """
    return (image - mean) / std

def normalize_images(images, mean=0.5, std=0.5):
    """
    Normalize a list of images.
    """
    return [normalize_image(image, mean, std) for image in images]
