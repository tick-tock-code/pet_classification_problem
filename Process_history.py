
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import matplotlib.pyplot as plt



""" --- Plotting the training history --- """
metrics_to_plot = ['accuracy', 'precision', 'recall']  # Metrics you want to plot 'loss'
num_metrics = len(metrics_to_plot)

plt.figure(figsize=(15, 5 * num_metrics))  # Adjust figure size for subplots

for i, metric_name in enumerate(metrics_to_plot):
    plt.subplot(num_metrics, 1, i + 1)  # Create subplots in a vertical layout

    metric_values = history.history[metric_name]
    plt.plot(metric_values, label=f'Training {metric_name.capitalize()}')

    if f'val_{metric_name}' in history.history: #check if validation data exists
        val_metric_values = history.history[f'val_{metric_name}']
        plt.plot(val_metric_values, label=f'Validation {metric_name.capitalize()}')

    plt.title(f'Model {metric_name.capitalize()}')
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Save metric values as NumPy arrays
for metric_name in history.history:
    metric_values = np.array(history.history[metric_name])
    np.save(f"{metric_name}_dogcat.npy", metric_values)
    print(f"saved {metric_name}")