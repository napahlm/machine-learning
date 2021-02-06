import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.__version__)

# Built-in test data set for recognizing clothes
fash_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fash_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# The neural network model
model = keras.Sequential([
    # The first layer takes the image as input which is 28x28 pixels
    # The last layer categorizes in 10 different labels
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The optimizer
model.compile(optimizer = keras.optimizers.Adam(),
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

# Fitting the data to the model
model.fit(train_images, train_labels, epochs=5)

# Testing the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)