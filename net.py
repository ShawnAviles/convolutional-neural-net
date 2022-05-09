# Shawn Aviles
# I pledge my honor that I have abided by the Stevens Honor System. -Shawn Aviles

# Quantitative Biology Final Project: Building a Convolutional Neural Network to determine how far a tree is
# Notes:
# total of 807 photos (.jpg) in the prescribed dataset
# taken from 3m, 4m, 5m, 6m, 7m, 8m, 9m, and 10m away (8 digits)
# 1200 x 800 pixels 
# had to be scaled down to 300 x 200 pixels to be run by my laptop
# RGB Value Pixels

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 200 # 800/4 
img_width = 300 # 1200/4
batch_size = 25
input_shape = (img_height, img_width, 3)

model = keras.Sequential([
    # keras.Input(shape = (img_height, img_width, 3)),
    layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),   # 32-out-channels, kernel size, activation
    layers.MaxPooling2D(pool_size=(2, 2)),                      # 2x2 pooling is default so it is not needed
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax'),
])

# using dataset from directory
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels = 'inferred',
    label_mode = 'int',  # can use categorical
    class_names = ['3', '4', '5', '6', '7', '8', '9', '10'],
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (img_height, img_width),   # reshape if not in this size
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = 'training',
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels = 'inferred',
    label_mode = 'int',  # can use categorical
    class_names = ['3', '4', '5', '6', '7', '8', '9', '10'],
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (img_height, img_width),   # reshape if not in this size
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = 'validation',
)

# augment function to add random transformations to the data for addings additional inputs to the dataset
def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    # TODO add more random transformations i.e. flip horizontally
    return image, y

# apply augmentaion to the dataset
#ds_train = ds_train.map(augment)
    
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ["accuracy"],
)

print("Training:")
model.fit(ds_train, batch_size = 25, epochs = 4, verbose = 2)
print("Testing:")
model.evaluate(ds_validation, batch_size = 25, verbose = 2)
print(model.summary())



