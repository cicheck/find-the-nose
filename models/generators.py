import os
import sys
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
utility_path = os.path.abspath(os.path.join('..'))
if utility_path not in sys.path:
    sys.path.append(utility_path)
from utills.augmentation import shift_colors, mirror
from tensorflow.keras.applications.resnet import preprocess_input


def build_cnn_generators(X_train, X_val):
    """Initiate and fit generators for cascade network."""

    max_rgb = 255.0
    train_gen = ImageDataGenerator(brightness_range=[0.5,1.0], channel_shift_range=5.0, preprocessing_function=lambda x: x / max_rgb)
    val_gen = ImageDataGenerator(preprocessing_function=lambda x: x / max_rgb)
    train_gen.fit(X_train)
    val_gen.fit(X_val)
    return train_gen, val_gen


def build_resnet_generators(X_train, X_val):
    """Initiate and fit generators for network using resnet as backbone."""

    train_gen = ImageDataGenerator(brightness_range=[0.5,1.0], preprocessing_function=preprocess_input)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen.fit(X_train)
    val_gen.fit(X_val)
    return train_gen, val_gen


def custom_generator(X, Y, batch_size, aug_prob):
    """For each batch in epoch yield augmented data."""

    max_rgb = 255.0
    picture_size = X[0, :].shape[0]
    colors_prob = aug_prob[0]
    mirror_prob = aug_prob[1]
    # Iterate over batches
    b = 0
    while (b + 1) * batch_size < len(X):
        x_batch = X[b * batch_size:(b + 1) * batch_size, :].copy()
        y_batch = Y[b * batch_size:(b + 1) * batch_size, :].copy()
        for i in range(len(x_batch)):
            r = random.random()
            if r < colors_prob:
                x_batch[i,:] = shift_colors(x_batch[i, :])
            if r < mirror_prob:
                x_batch[i, :], y_batch[i, :] = mirror(x_batch[i, :], y_batch[i, :])
        yield x_batch / max_rgb, y_batch / picture_size


def train_with_custom_augmentation(model, X_train, Y_train, batch_size, aug_prob, epochs):
    """Train model using custom_generator for augmentation"""

    for e in range(epochs):
        print("Epoch: ", e + 1)
        for x_batch, y_batch in custom_generator(X_train, Y_train, batch_size, aug_prob):
            model.fit(x_batch, y_batch, verbose=0)

