from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.models.improved_cnn import ImprovedCNN
from src.data.transforms import get_eval_transforms


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_model(checkpoint_path, device, num_classes=7):
    """
    Load the trained ImprovedCNN model from a checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ImprovedCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    return model


def load_and_preprocess_image(image_path):
    """
    Load an image from disk and preprocess it for the model.

    Returns:
        original_image_pil: original PIL image for display
        image_tensor: tensor of shape [1, 1, 48, 48]
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_image = Image.open(image_path).convert("RGB")
    resized_image = original_image.resize((48, 48))

    transform = get_eval_transforms()
    image_tensor = transform(resized_image)
    image_tensor = image_tensor.unsqueeze(0)

    return original_image, image_tensor


@torch.no_grad()
def predict_emotion(model, image_tensor, device):
    """
    Predict class probabilities for a single image.

    Returns:
        pred_idx: predicted class index
        pred_label: predicted class name
        confidence: probability of predicted class
        probabilities: tensor of shape [7]
    """
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)

    pred_idx = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, pred_idx].item()
    pred_label = CLASS_NAMES[pred_idx]

    return pred_idx, pred_label, confidence, probabilities[0].cpu()


def get_top_k_predictions(probabilities, k=3):
    """
    Return the top-k predicted classes and probabilities.

    Args:
        probabilities: tensor of shape [num_classes]
        k: number of top predictions to return

    Returns:
        top_results: list of tuples (class_name, probability)
    """
    top_probs, top_indices = torch.topk(probabilities, k=k)

    top_results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        top_results.append((class_name, prob.item()))

    return top_results


def show_prediction(original_image, pred_label, confidence):
    """
    Display the image with predicted emotion.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title(f"Prediction: {pred_label} ({confidence:.2%})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict facial emotion from a single image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input face image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/improved_cnn_best.pth",
        help="Path to trained model checkpoint"
    )
    args = parser.parse_args()

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
    # Load model
    # -----------------------------
    model = load_model(args.checkpoint, device)

    # -----------------------------
    # Load and preprocess image
    # -----------------------------
    original_image, image_tensor = load_and_preprocess_image(args.image)
    print(f"Input tensor shape: {image_tensor.shape}")

    # -----------------------------
    # Predict
    # -----------------------------
    pred_idx, pred_label, confidence, probabilities = predict_emotion(
        model, image_tensor, device
    )

    print("\nPrediction result")
    print("-----------------")
    print(f"Predicted emotion: {pred_label}")
    print(f"Confidence: {confidence:.4f}")

    print("\nTop-3 predictions:")
    top3 = get_top_k_predictions(probabilities, k=3)
    for rank, (class_name, prob) in enumerate(top3, start=1):
        print(f"{rank}. {class_name:<10} - {prob:.4f}")

    print("\nAll class probabilities:")
    for class_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"{class_name:>10}: {prob.item():.4f}")

    # -----------------------------
    # Show image
    # -----------------------------
    show_prediction(original_image, pred_label, confidence)


if __name__ == "__main__":
    main()