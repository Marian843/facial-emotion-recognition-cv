from torchvision import transforms


def get_train_transforms():
    """
    Transforms for training images.

    Simple augmentation:
    - ensure grayscale with 1 channel
    - small random horizontal flip
    - convert to tensor
    - normalize
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_eval_transforms():
    """
    Transforms for validation and test images.

    No random augmentation here.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])