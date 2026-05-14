""" Code to CREATE AND TRAIN an ML model to predict whether a photo is that of a cat or a dog
    using data from kaggle or google?"""

# PyTorch with CUDA support is recommended for GPU training on Windows

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import random
import sys
import signal




""" --- Defining functions ---"""
class SmallCNN(nn.Module):
    """Creates a CNN model for image classification."""
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Convolutional layer creates 32 3x3 filters, keeping image height/width the same because padding=1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Pooling 2D layer with pool size 2x2, reducing the image size from 224x224 to 112x112
            nn.MaxPool2d(2),

            # Convolutional layer creates 64 3x3 filters, keeping image height/width the same because padding=1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Pooling 2D layer with pool size 2x2, reducing the image size from 112x112 to 56x56
            nn.MaxPool2d(2),

            # Convolutional layer creates 128 3x3 filters, keeping image height/width the same because padding=1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Pooling 2D layer with pool size 2x2, reducing the image size from 56x56 to 28x28
            nn.MaxPool2d(2),

            # Convolutional layer creates 256 3x3 filters, keeping image height/width the same because padding=1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Pooling 2D layer with pool size 2x2, reducing the image size from 28x28 to 14x14
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            # Global pooling layer to reduce each feature map to one value, avoiding a huge Dense layer
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Dense layer with num_classes outputs. CrossEntropyLoss expects raw logits, so no softmax here.
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model_ox_cat_dog(input_shape, num_classes):
    """Creates a CNN model for image classification."""
    model = SmallCNN(num_classes)
    return model



""" Unused """
def convert_to_16bit(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB") # Ensure it's in RGB format
    img_16bit = np.array(img, dtype=np.uint16) * 257  # Simple upsampling
    return img_16bit

def normalize_image(image_path):
    img = Image.open(image_path)
    image = np.array(img)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported bit depth")

""" Unused Ends"""



# --- Data Augmentation Functions ---

def rotate_image(image, angle_in):
    """Rotates the image by a given angle."""
    return image.rotate(angle_in)

def augment_image(image):
    """Augments the image by random reflections, rotations, and colour changes."""

    # Reflections
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # X-axis reflection

    # Rotations
    angle = random.uniform(-10, 10)  # Random angle between -10 and 10 degrees
    image = rotate_image(image, angle)

    # ColorJitter
    if random.random() < 0.9:
        image = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        )(image)

    return image


# --- Getting image paths and labels from filenames ---
def get_image_paths_and_labels(data_dir):
    """
    Gets image paths and labels from filenames in a directory.

    Args:
        data_dir: Path to the directory containing the images.

    Returns:
        A tuple containing two lists:
        - image_paths: A list of full paths to the image files.
        - labels: A list of corresponding labels.
        Returns empty lists if no suitable files are found.
    """

    image_paths = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for common image extensions
            match = re.match(r"([a-zA-Z]+(?:_[a-zA-Z]+)*)_(\d+)\.(?:png|jpg|jpeg)", filename, re.IGNORECASE) #Use a regex to extract the label
            if match:
                label = match.group(1)

                try:
                    label = label.encode('latin-1').decode('utf-8') #Try decoding with utf-8, if it fails then try latin-1
                except UnicodeDecodeError:
                    label = label.encode('utf-8').decode('latin-1')

                image_path = os.path.join(data_dir, filename)
                image_paths.append(image_path)
                labels.append(label)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)

    if not combined:
        return [], []

    image_paths, labels = zip(*combined)

    return list(image_paths), list(labels)


# --- Loading and preprocessing images ---
class PetImageDataset(Dataset):
    """Creates a PyTorch Dataset from image paths and numeric labels."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


def load_image(image_path, label, transform):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    label = torch.tensor(label, dtype=torch.long)
    return image, label


# --- Creating torch Dataset and DataLoader ---
def create_dataset(image_paths, labels, img_height, img_width, batch_size):
    """Creates PyTorch DataLoaders from image paths and labels."""


    unique_labels = np.array(list(dict.fromkeys(labels)))
    num_classes = len(unique_labels)

    print("Unique labels:", unique_labels)

    label_to_index = {label: index for index, label in enumerate(unique_labels)} # dictionary of labels to index
    numeric_labels = [label_to_index[label] for label in labels] # list of indices for each label

    # CrossEntropyLoss uses integer class labels, so no one-hot encoding is needed
    train_size = int(0.8 * len(image_paths))
    if len(image_paths) > 1:
        train_size = max(1, min(train_size, len(image_paths) - 1))
    val_size = len(image_paths) - train_size

    train_image_paths = image_paths[:train_size]
    train_labels = numeric_labels[:train_size]
    test_image_paths = image_paths[train_size:]
    test_labels = numeric_labels[train_size:]

    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Lambda(augment_image),
        transforms.ToTensor(),  # Converts to float32 tensor shaped (C, H, W), normalized to [0, 1]
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),  # Converts to float32 tensor shaped (C, H, W), normalized to [0, 1]
    ])

    train_dataset = PetImageDataset(train_image_paths, train_labels, transform=train_transform)
    test_dataset = PetImageDataset(test_image_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(test_loader)}")

    return train_loader, test_loader, unique_labels, num_classes



# --- Checking for corrupted images ---
def check_image_old(filename):
  try:
    with Image.open(filename) as img:
      img.verify()  # Raises an exception if corrupt
      return True, None
  except (OSError, IOError, Exception) as e:  # Catch various exceptions
    return False, str(e)


def check_image(filename):
    try:
        with Image.open(filename) as img:
            img.load()
            if img.size == (0, 0):
                return False, "Image has zero dimensions"
            if img.format not in ("JPEG",):
                return False, f"Unexpected image format: {img.format}"
            #Check for other unusual properties
            if os.path.getsize(filename) < 100:
                return False, f"File is too small: {os.path.getsize(filename)} bytes"
            return True, None
    except (OSError, SyntaxError, UnidentifiedImageError) as e:
        return False, str(e)
    if os.path.getsize(filename) < 100: #Check for small files
        return False, f"File is too small: {os.path.getsize(filename)} bytes"

    return True, None

def find_corrupted_images(directory):
  """Finds corrupted JPEG images in a directory."""
  corrupted_files = []
  for filename in os.listdir(directory):
    if filename.lower().endswith(('.jpg', '.jpeg')):
      filepath = os.path.join(directory, filename)
      is_valid, error_message = check_image(filepath)
      if not is_valid:
        corrupted_files.append((filepath, error_message))
        print(f"File {filepath} is corrupted: {error_message}")
  return corrupted_files

def remove_corrupted(corrupted):
  if corrupted:
    print(f"Found {len(corrupted)} corrupted images:")
    for file, err in corrupted:
      print(f"- {file}: {err}")
  else:
    print("No corrupted images found in the specified directory.")

  # Optionally remove the corrupted files with enhanced error handling
  if corrupted:
    remove_corrupted = input("Do you want to remove these files? (y/n): ")
    if remove_corrupted.lower() == 'y':
      for file, _ in corrupted:
        try:
          os.remove(file)
          print(f"Removed: {file}")
        except (OSError, PermissionError) as e:
          print(f"Error removing {file}: {e}")
    else:
      print("Corrupted files were not removed.")


def calculate_metrics(predictions, targets, num_classes):
    predictions = torch.cat(predictions).cpu()
    targets = torch.cat(targets).cpu()

    accuracy = (predictions == targets).float().mean().item()

    precision_values = []
    recall_values = []
    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (targets == class_index)).sum().item()
        false_positive = ((predictions == class_index) & (targets != class_index)).sum().item()
        false_negative = ((predictions != class_index) & (targets == class_index)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        precision_values.append(precision)
        recall_values.append(recall)

    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))

    return accuracy, macro_precision, macro_recall


def save_checkpoint(epoch, checkpoint_path=None, best_val_loss=None):
  if checkpoint_path is None:
      checkpoint_path = model_path

  checkpoint = {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "classes": unique_labels.tolist(),
      "img_height": img_height,
      "img_width": img_width,
      "epoch": epoch,
      "history": history,
  }

  if "scheduler" in globals():
      checkpoint["scheduler_state_dict"] = scheduler.state_dict()

  if best_val_loss is not None:
      checkpoint["best_val_loss"] = best_val_loss

  torch.save(checkpoint, checkpoint_path)


def signal_handler(sig, frame):
  print('You pressed Ctrl+C!')
  # Your code to be run on interrupt
  try:
      print("Training interrupted by user. Saving model...")
      save_checkpoint(current_epoch)  # Save the model
      print("Model saved.")
  except Exception as e: # added exception handling
      print(f"Error saving model: {e}")
  finally:
      sys.exit(0)  # Exit cleanly after saving





""" ---
---
---
---
---
SCRIPT STARTS HERE
---
---
---
---
---
---
 ---"""


""" --- Defining Paths --- """
current_dir = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(current_dir, 'Training_data', 'images')  # Replace with the actual path
model_path = os.path.join(current_dir, "autosave_model.pt")
best_model_path = os.path.join(current_dir, "best_model.pt")

# Defining device and checkpoint frequency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_save_freq = 100  # Save every 100 batches
print(f"Using device: {device}")


""" --- Load data and create dataset ---"""
# Find corrupted images - remove them
corrupted = find_corrupted_images(data_directory)
remove_corrupted(corrupted)

# After removing corrupted images, then identify paths/labels
image_paths, labels = get_image_paths_and_labels(data_directory)

# Creating dataset
if image_paths: #check if any images were found
    print(f"Found {len(image_paths)} images.")
    img_height = 224
    img_width = 224
    batch_size = 32
    train_loader, test_loader, unique_labels, num_classes = create_dataset(image_paths, labels, img_height, img_width, batch_size)
    print(unique_labels)

    # Print shapes of a single batch shape = (batch_size, rgb_depth, x, y)
    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)
        print(labels.dtype)
        break
else:
    print("No images found in the directory.")
    sys.exit(0)






""" --- Creating the model  ---"""
# Defining training and evaluation data
train_ds = train_loader
test_ds = test_loader

# Define input shape
input_shape = (3, img_height, img_width) # Define the input shape
print('Num classes:', num_classes)

# Create model
model = create_model_ox_cat_dog(input_shape, num_classes)
model = model.to(device)

# Compile the model - loss/optimiser define how the model is trained,
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) # optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)
print(model) # Print model summary
total_params = sum(parameter.numel() for parameter in model.parameters())
trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

history = {
    "loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
}
current_epoch = 0
best_val_loss = float("inf")
signal.signal(signal.SIGINT, signal_handler)


""" --- Training Model --- """
# Train the model
epochs = 30  # Number of training epochs

for epoch in range(epochs):
    current_epoch = epoch + 1

    model.train()
    train_loss_total = 0.0
    train_count = 0
    train_predictions = []
    train_targets = []

    for batch_index, (images, labels) in enumerate(train_ds, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)
        train_count += images.size(0)

        predicted_classes = torch.argmax(outputs, dim=1)
        train_predictions.append(predicted_classes.detach().cpu())
        train_targets.append(labels.detach().cpu())

        if batch_index % checkpoint_save_freq == 0:
            save_checkpoint(current_epoch)
            print(f"Saved checkpoint to: {model_path}")

    train_loss = train_loss_total / train_count
    train_accuracy, train_precision, train_recall = calculate_metrics(train_predictions, train_targets, num_classes)

    model.eval()
    val_loss_total = 0.0
    val_count = 0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for images, labels in test_ds:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)
            val_count += images.size(0)

            predicted_classes = torch.argmax(outputs, dim=1)
            val_predictions.append(predicted_classes.detach().cpu())
            val_targets.append(labels.detach().cpu())

    val_loss = val_loss_total / val_count
    val_accuracy, val_precision, val_recall = calculate_metrics(val_predictions, val_targets, num_classes)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_accuracy)
    history["precision"].append(train_precision)
    history["recall"].append(train_recall)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - precision: {train_precision:.4f} - recall: {train_recall:.4f} - "
        f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f} - val_precision: {val_precision:.4f} - val_recall: {val_recall:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(current_epoch, checkpoint_path=best_model_path, best_val_loss=best_val_loss)
        print(f"Saved best model to: {best_model_path}")

    scheduler.step(val_loss)


# Saving model if training successful
save_checkpoint(current_epoch)
print(f"Model saved to: {model_path}") #Print the full path



""" --- Plotting the training history --- """
metrics_to_plot = ['accuracy', 'precision', 'recall']  # Metrics to plot - loss left out
num_metrics = len(metrics_to_plot)

plt.figure(figsize=(15, 5 * num_metrics))  # Adjust figure size for subplots

for i, metric_name in enumerate(metrics_to_plot):
    plt.subplot(num_metrics, 1, i + 1)  # Create subplots in a vertical layout

    metric_values = history[metric_name]
    plt.plot(metric_values, label=f'Training {metric_name.capitalize()}')

    if f'val_{metric_name}' in history: #check if validation data exists
        val_metric_values = history[f'val_{metric_name}']
        plt.plot(val_metric_values, label=f'Validation {metric_name.capitalize()}')

    plt.title(f'Model {metric_name.capitalize()}')
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Save metric values as NumPy arrays
for metric_name in history:
    metric_values = np.array(history[metric_name])
    np.save(f"{metric_name}_dogcat.npy", metric_values)
    print(f"saved {metric_name}")
