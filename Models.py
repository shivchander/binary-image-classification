#!/usr/bin/env python

"""

"""

__author__ = "Shivchander Sudalairaj"
__email__ = "sudalasr@mail.uc.edu"

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Models:
    def __init__(self, data_augmentation):
        self.data_augmentation = data_augmentation

    def vgg3(self):
        """
        Custom Base Model inspired from VGG3

        source: https://arxiv.org/abs/1409.1556
        """
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(128, 128, 3)))

        # data augmentation
        if 'RandomFlip' in self.data_augmentation:
            model.add(layers.experimental.preprocessing.RandomFlip(self.data_augmentation['RandomFlip']))
        if 'RandomRotation' in self.data_augmentation:
            model.add(layers.experimental.preprocessing.RandomRotation(self.data_augmentation['RandomRotation']))
        if 'RandomZoom' in self.data_augmentation:
            model.add(layers.experimental.preprocessing.RandomZoom(self.data_augmentation['RandomZoom']))

        # vgg block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # vgg block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # vgg block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Flatten and final layer
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def vgg16(self):
        """
        Transfer Learning from VGG-16
        Freeze all the layers from feature extractor part
        Fine tune the classifier part of the models

        source: https://arxiv.org/abs/1409.1556
        """
        inputs = keras.Input(shape=(224, 224, 3))

        # data augmentation
        if 'RandomFlip' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomFlip(self.data_augmentation['RandomFlip'])(inputs)
        if 'RandomRotation' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomRotation(self.data_augmentation['RandomRotation'])(inputs)
        if 'RandomZoom' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomZoom(self.data_augmentation['RandomZoom'])(inputs)

        model = tensorflow.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), input_tensor=inputs)

        # freeze the feature extractor layers
        for curr_layer in model.layers:
            curr_layer.trainable = False

        # add new classifier layers
        flat1 = layers.Flatten()(model.layers[-1].output)
        class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = layers.Dense(1, activation='sigmoid')(class1)

        # define new models
        model = keras.Model(inputs=model.inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def xception(self):
        """
        Transfer Learning from Xception
        Freeze all the layers from the feature extractor part
        Fine tune the classifier part of the models

        source: https://arxiv.org/abs/1610.02357
        """

        inputs = keras.Input(shape=(299, 299, 3))

        # data augmentation
        if 'RandomFlip' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomFlip(self.data_augmentation['RandomFlip'])(inputs)
        if 'RandomRotation' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomRotation(self.data_augmentation['RandomRotation'])(inputs)
        if 'RandomZoom' in self.data_augmentation:
            inputs = layers.experimental.preprocessing.RandomZoom(self.data_augmentation['RandomZoom'])(inputs)

        model = tensorflow.keras.applications.Xception(include_top=False, input_tensor=inputs)

        # freeze the feature extractor layers
        for curr_layer in model.layers:
            curr_layer.trainable = False

        # add new classifier layers
        flat1 = layers.Flatten()(model.layers[-1].output)
        class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = layers.Dense(1, activation='sigmoid')(class1)

        # define new models
        model = keras.Model(inputs=model.inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
