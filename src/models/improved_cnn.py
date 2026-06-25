import torch
import torch.nn as nn


class ImprovedCNN(nn.Module):
    """
    Improved CNN for facial emotion recognition.

    Upgrades over BaselineCNN:
    - BatchNorm after each convolution
    - one additional convolution block
    - slightly stronger feature extractor

    Input:
        [batch_size, 1, 48, 48]

    Output:
        [batch_size, num_classes]
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: [B, 1, 48, 48] -> [B, 32, 24, 24]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: [B, 32, 24, 24] -> [B, 64, 12, 12]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: [B, 64, 12, 12] -> [B, 128, 6, 6]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: [B, 128, 6, 6] -> [B, 256, 6, 6]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x