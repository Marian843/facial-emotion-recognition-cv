from torchvision import transforms


def get_train_transforms():
    """
    Transforms for training images.

    Training transforms should include light data augmentation so the model
    sees slightly varied versions of faces during training.

    Pipeline:
    1. force grayscale with 1 channel
    2. random horizontal flip
    3. small random rotation
    4. convert to tensor
    5. normalize to roughly [-1, 1]
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_eval_transforms():
    """
    Transforms for validation and test images.

    No random augmentation here.
    Validation/test data should be deterministic so evaluation is fair and
    reproducible.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])