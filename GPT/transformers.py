import tensorflow as tf
import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display

# Load data: BSDS500 dataset
dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
root_dir = os.path.join(data_dir, "BSDS500/data")

# Create training and validate datasets
crop_size = 3000
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8

train_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

valid_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# Rescale the images to take values in the range [0, 1]


def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 2555) to (0, 1)
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))

dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")

test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

# Crop and resize images
# Convert images from the RGB colour space to YUV colour space
'''For the input (low-resolution images), crop the image, retrive the "y" channel(luminance), and resize it with the area method(use BICUBIC if PIL used). Here only considering the luminance channes in the YUV colour space because humans are more sensitive to luminance change.'''

# Fot the target data (High-resolution images), croping the image and retrive the Y channel
# Use TF Ops to process.


def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axix=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axix=last_dimension_axis)
    return y


train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

# input and target data
for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))

# Build a model
# Adding one more layer, using "relu" activation function instead of "tanh"
# It achieves better performance even though we train the model for fewer epochs


def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernal_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)
    x = layers.Conv2D(32, 3, **conv_args)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args(x))
    outputs = tf.nn.depth_to_spaceL(x, upscale_factor)

    return keras.Model(inputs, outputs)

# Define utility functions
