import tensorflow as tf
import os
import math
import numpy as np

from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from IPython.display import display
from utils import plot_results, get_lowers_image, upscale_image

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

# Get the current working directory (where your code is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the "images" folder
images_folder = os.path.join(current_dir, "images")

# Now, you can use "images_folder" as the "dataset" path
dataset = images_folder
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
'''For the input (low-resolution images), crop the image, retrieve the "y" channel(luminance), and resize it with the area method(use BICUBIC if PIL used). Here only considering the luminance channes in the YUV colour space because humans are more sensitive to luminance change.'''

# Fot the target data (High-resolution images), croping the image and retrive the Y channel
# Use TF Ops to process.


def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
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
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)

# Define callbacks to monitor training

# ESPCN (Enhanced Super-Resolution Convolutional Network) model


class ESPCNCallback(keras.callbacks.Callback):
    '''The ESPCNCallback object will compute and display the PSNR metric that will use to evaluate super-resolution performance'''

    def __init__(self):
        super().__init__()
        self.test_img = get_lowers_image(
            load_img(test_img_paths[0]), upscale_factor)

        # Store PSNR value in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


# Define ModelChechpoint and EarlyStopping callbacks
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="loss", patience=10)

checkpoint_filepath = "/tmp/checkpoint"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()

callbacks = [ESPCNCallback(), early_stopping_callback,
             model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)


# Train the model
epochs = 100

model.compile(
    optimizer=optimizer, loss=loss_fn
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)


# The model weights (that are considered the best) are loaded into the model
model.load_weights(checkpoint_filepath)

# Run model prodection and plot the results
total_bicubic_psnr = 0.0
total_test_psnr = 0.0

for index, test_img_path in enumerate(test_img_paths[50:60]):
    img = load_img(test_img_path)
    lowres_input = get_lowers_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(
        lowres_img_arr, highres_img_arr, max_value=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print("PSNR of low resolution image and high resolution image is %.4f" %
          bicubic_psnr)
    print("PSNR fo predict and high resolution image is %.4f" % test_psnr)
    plot_results(lowres_img, index, "lowres")
    plot_results(highres_img, index, "highres")
    plot_results(prediction, index, "prediction")

print("Avg. PSNR of lowres image is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstruction is %.4f" % (total_test_psnr / 10))
