from src.models.improved_cnn import ImprovedCNN
import torch

model = ImprovedCNN(num_classes=7)
x = torch.randn(1, 1, 48, 48)
y = model(x)

print(model)
print("Input shape:", x.shape)
print("Output shape:", y.shape)