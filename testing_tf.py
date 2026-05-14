from tensorflow.python.client import device_lib
import os

print(device_lib.list_local_devices())

import tensorflow as tf
print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# install windows sdk for cuda 