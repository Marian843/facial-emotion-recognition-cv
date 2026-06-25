from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from src.data.transforms import get_train_transforms, get_eval_transforms


def create_datasets(data_dir, val_split=0.1, seed=42):
    """
    Create train, validation, and test datasets from folder-based FER2013 data.

    Strategy:
    - load full training folder
    - split it into train + validation
    - load test folder separately
    """

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Full training dataset (with training transforms for now)
    full_train_dataset = ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )

    # Validation dataset needs deterministic transforms,
    # so we create a second copy of the same folder with eval transforms.
    full_train_dataset_eval = ImageFolder(
        root=train_dir,
        transform=get_eval_transforms()
    )

    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * val_split)
    train_size = total_train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    # Split indices using the same random seed so both datasets align
    train_subset, val_subset_train_view = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(
        full_train_dataset_eval,
        [train_size, val_size],
        generator=generator
    )

    test_dataset = ImageFolder(
        root=test_dir,
        transform=get_eval_transforms()
    )

    return train_subset, val_subset, test_dataset


def create_dataloaders(
    data_dir,
    batch_size=64,
    val_split=0.1,
    num_workers=0,
    seed=42
):
    """
    Create train, validation, and test DataLoaders.
    """
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader