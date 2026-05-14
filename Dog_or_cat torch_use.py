""" Applies trained model to make predictions on new images."""

# --- Importing Libraries ---
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


""" --- Defining functions ---"""
class SmallCNN(nn.Module):
    """Creates a CNN model for image classification."""
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)






# Defining Params
MODEL_PATH = "best_model.pt"
IMG_PATH = "test_image.jpg"   # put test images here, filenames must contain labels or be in subfolders
IMG_SIZE = (224, 224)           # set to model input



""" --- Loading Model and Making Predictions --- """
# Example of loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = load_checkpoint(MODEL_PATH, device)

classes = checkpoint["classes"]
IMG_SIZE = (checkpoint["img_height"], checkpoint["img_width"])

loaded_model = SmallCNN(num_classes=len(classes))
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),    # IMPORTANT: float32, normalized, shaped (C, H, W)
])

img = Image.open(IMG_PATH).convert("RGB")
arr = transform(img)
arr = arr.unsqueeze(0).to(device)  # (1, C, H, W)

# Making predictions
with torch.no_grad():
    pred = loaded_model(arr)  # shape (1, num_classes), raw logits
    pred = torch.softmax(pred, dim=1)

print(pred.cpu().numpy())
pred = pred[0]
pred_class_index = int(torch.argmax(pred).item())
pred_confidence = float(pred[pred_class_index].item())

print(f"Predicted: {classes[pred_class_index]} ({pred_confidence:.3f})")
