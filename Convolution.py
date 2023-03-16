import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os

# Load the image
print(os.getcwd())
image = cv2.imread('image1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image.shape

# Reshape the image to match the expected input shape of the convolutional layer
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)

# Create the convolutional layer
tensor_image = tf.convert_to_tensor(image)
tensor_image = tf.cast(tensor_image, dtype = tf.float32)
conv_layer = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(None, None, 1))

# Apply the convolution to the image
result = conv_layer(tensor_image)

# Print the shape of the output tensor
print(result.shape)
