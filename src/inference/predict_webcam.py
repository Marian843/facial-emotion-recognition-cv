from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.models.improved_cnn import ImprovedCNN


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def get_device():
    """
    Select the best available device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path, device):
    """
    Load the trained FER model checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ImprovedCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    return model


def get_inference_transform():
    """
    Transform a face crop into the same format used during training/evaluation.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


@torch.no_grad()
def predict_emotion(model, face_bgr, device, transform):
    """
    Predict emotion for a cropped face image.

    Input:
        face_bgr: face crop from OpenCV in BGR format

    Returns:
        pred_idx: predicted class index
        confidence: highest softmax probability
        probs: full probability vector
    """
    # Convert OpenCV BGR image -> RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL so torchvision transforms work nicely
    face_pil = Image.fromarray(face_rgb)

    # Transform to tensor and add batch dimension
    x = transform(face_pil).unsqueeze(0).to(device)  # [1, 1, 48, 48]

    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return pred_idx, confidence, probs.cpu().numpy()


def main():
    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    checkpoint_path = "checkpoints/improved_cnn_best.pth"
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    transform = get_inference_transform()

    # Haar cascade face detector bundled with OpenCV
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_detector.empty():
        raise RuntimeError("Failed to load Haar cascade face detector.")

    # --------------------------------------------------
    # Open webcam
    # --------------------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Make a copy for display
        display_frame = frame.copy()

        # Convert full frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]

            # Skip invalid crops
            if face_crop.size == 0:
                continue

            pred_idx, confidence, probs = predict_emotion(
                model=model,
                face_bgr=face_crop,
                device=device,
                transform=transform
            )

            emotion = CLASS_NAMES[pred_idx]
            label = f"{emotion} ({confidence:.2f})"

            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw label above the face
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(
                display_frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Facial Emotion Recognition - Webcam", display_frame)

        # Press q to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()