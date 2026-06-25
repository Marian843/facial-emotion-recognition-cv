from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.baseline_cnn import BaselineCNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Returns:
        avg_loss: average training loss over the epoch
        accuracy: training accuracy over the epoch
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # 1) zero previous gradients
        optimizer.zero_grad()

        # 2) forward pass
        outputs = model(images)

        # 3) compute loss
        loss = criterion(outputs, labels)

        # 4) backpropagation
        loss.backward()

        # 5) update parameters
        optimizer.step()

        # track metrics
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test data.

    Returns:
        avg_loss: average loss
        accuracy: classification accuracy
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def compute_class_weights(dataset, num_classes):
    """
    Compute class weights from a training dataset.

    We want rarer classes to receive larger weights so that mistakes
    on underrepresented classes are penalized more during training.

    Expected dataset behavior:
    - dataset[i] returns (image, label)
    - label is an integer class index in [0, num_classes-1]

    Returns:
        class_weights: torch.FloatTensor of shape [num_classes]
    """
    label_counts = Counter()

    for i in range(len(dataset)):
        _, label = dataset[i]
        label_counts[label] += 1

    print("\nTraining class counts:")
    for class_idx in range(num_classes):
        print(f"Class {class_idx}: {label_counts[class_idx]}")

    total_samples = sum(label_counts.values())

    class_weights = []
    for class_idx in range(num_classes):
        count_i = label_counts[class_idx]

        if count_i == 0:
            weight_i = 0.0
        else:
            weight_i = total_samples / (num_classes * count_i)

        class_weights.append(weight_i)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("\nComputed class weights:")
    for class_idx, weight in enumerate(class_weights):
        print(f"Class {class_idx}: {weight.item():.4f}")

    return class_weights


def main():
    # -----------------------------
    # Config
    # -----------------------------
    data_dir = "data/raw"
    batch_size = 64
    val_split = 0.1
    num_workers = 0   # good default for Mac while starting
    num_epochs = 10
    learning_rate = 1e-3
    seed = 42
    num_classes = 7

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "baseline_cnn_weighted_best.pth"

    # -----------------------------
    # Device
    # -----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # Data
    # -----------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = BaselineCNN(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(
        train_loader.dataset,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # -----------------------------
    # Load best checkpoint before final test evaluation
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()