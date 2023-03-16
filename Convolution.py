import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reshape the image to match the expected input shape of the convolutional layer
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)

# Create the convolutional layer
conv_layer = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(None, None, 1))

# Apply the convolution to the image
result = conv_layer(image)

# Print the shape of the output tensor
print(result.shape)
