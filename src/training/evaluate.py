from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from src.data.dataset import create_dataloaders
from src.models.improved_cnn import ImprovedCNN


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """
    Run the model on a dataloader and collect:
    - predicted labels
    - true labels
    """
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return all_preds, all_labels


@torch.no_grad()
def collect_sample_predictions(model, dataloader, device, max_images=8):
    """
    Collect a small batch of images + predictions for visualization.
    """
    model.eval()

    images_out = []
    preds_out = []
    labels_out = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for i in range(images.size(0)):
            images_out.append(images[i].cpu())
            preds_out.append(preds[i].cpu().item())
            labels_out.append(labels[i].item())

            if len(images_out) >= max_images:
                return images_out, preds_out, labels_out

    return images_out, preds_out, labels_out


def compute_per_class_accuracy(cm):
    """
    Compute per-class accuracy (recall) from confusion matrix.

    cm shape: [num_classes, num_classes]
    rows = true labels
    cols = predicted labels
    """
    per_class_acc = {}

    for i, class_name in enumerate(CLASS_NAMES):
        total_true = cm[i].sum()
        correct = cm[i, i]

        if total_true == 0:
            acc = 0.0
        else:
            acc = correct / total_true

        per_class_acc[class_name] = acc

    return per_class_acc


def plot_sample_predictions(images, preds, labels):
    """
    Plot sample predictions from the test set.
    """
    num_images = len(images)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].squeeze(0).numpy()
        img = (img * 0.5) + 0.5  # undo normalization

        pred_name = CLASS_NAMES[preds[i]]
        true_name = CLASS_NAMES[labels[i]]

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {pred_name}\nTrue: {true_name}")
        axes[i].axis("off")

    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # Config
    # -----------------------------
    data_dir = "data/raw"
    batch_size = 64
    val_split = 0.1
    num_workers = 0
    seed = 42
    num_classes = 7

    checkpoint_path = Path("checkpoints/improved_cnn_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # -----------------------------
    # Device
    # -----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # Data
    # -----------------------------
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = ImprovedCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # -----------------------------
    # Collect predictions
    # -----------------------------
    preds, labels = collect_predictions(model, test_loader, device)

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Purples")
    plt.title("ImprovedCNN Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Per-class accuracy
    # -----------------------------
    per_class_acc = compute_per_class_accuracy(cm)

    print("\nPer-class accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"{class_name:>10}: {acc:.4f}")

    # -----------------------------
    # Classification report
    # -----------------------------
    print("\nClassification Report:")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    print(report)

    # -----------------------------
    # Sample predictions
    # -----------------------------
    sample_images, sample_preds, sample_labels = collect_sample_predictions(
        model, test_loader, device, max_images=8
    )

    plot_sample_predictions(sample_images, sample_preds, sample_labels)


if __name__ == "__main__":
    main()