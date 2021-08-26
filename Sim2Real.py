#!/usr/bin/env python

"""

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Shivchander Sudalairaj"
__email__ = "sudalasr@mail.uc.edu"

import cv2
import numpy as np
import glob
import tensorflow
from sklearn.metrics import accuracy_score
from Models import Models


def load_and_pad_imgs(img_paths, target_height, target_width):
    """
    loads the images from a list of image paths into a numpy array. Then checks if the image size matches
    the target dims, if not, pads the image array
    """
    padded_imgs = []

    for _, img_path in enumerate(img_paths):
        img_array = cv2.imread(img_path)
        if img_array.shape[:-1] == (target_height, target_width):
            padded_imgs.append(img_array)
        else:
            padded_img = tensorflow.image.resize_with_pad(img_array, target_height, target_width, antialias=True)
            padded_imgs.append(np.array(padded_img, np.uint8))

    return np.array(padded_imgs)


def load_data_varying_training_size(X, y):
    varying_data = []
    for percent in np.arange(0.1, 1.1, 0.1):
        upper_limit = int(percent * len(X))
        xi = X[:upper_limit, :, :, :].copy()
        yi = y[:upper_limit].copy()
        varying_data.append((xi, yi))
    return varying_data


class Sim2Real:

    def load_data_and_save_X_Y(self, class0_dir, class1_dir, img_size=(128, 128)):
        """
        Load the images from the building and car directories and create two numpy arrays X and Y,
        where X is the images (i.e., each element is a 128x128x3 image) and Y are the labels (i.e., building or car)
        for each image
        """
        bldg_files = sorted(glob.glob('{}/*'.format(class0_dir)))
        car_files = sorted(glob.glob('{}/*'.format(class1_dir)))

        bldg_array = load_and_pad_imgs(bldg_files, target_height=img_size[0], target_width=img_size[1]) / 255
        print(bldg_array.shape)
        car_array = load_and_pad_imgs(car_files, target_height=img_size[0], target_width=img_size[1]) / 255
        print(car_array.shape)

        X = np.vstack((bldg_array, car_array))
        y = np.concatenate((np.zeros(len(bldg_files)), np.ones(len(car_files))))

        return X, y

    def sim_train(self, X_train, y_train, X_test, y_test, model='baseline', data_augment=True, epochs=5, validation_split=0.2,
                  early_stopping=True, save=False, save_dir='models', filename='model'):
        """
        Trains the specified model using X_train and y_train data
        """
        if data_augment:
            data_augmentation = {'RandomFlip': 'horizontal_and_vertical',
                                 'RandomZoom': 0.2,
                                 'RandomRotation': 0.2}
        else:
            data_augmentation = {}

        m = Models(data_augmentation)

        if model == 'baseline':
            curr_model = m.vgg3()
        elif model == 'VGG16':
            curr_model = m.vgg16()
        elif model == 'Xception':
            curr_model = m.xception()
        else:
            raise ValueError('Unsupported Model')

        if early_stopping:
            callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=1)]
        else:
            callbacks = []

        history = curr_model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

        y_probs = curr_model.predict(X_test)
        y_preds = np.where(y_probs > 0.5, 1, 0)

        print('Accuracy Score: {}'.format(accuracy_score(y_test, y_preds)))

        if save:
            curr_model.save("{}/{}".format(save_dir, filename))

        return history

    def load_model_predict(self, X_test, y_test, model='models/baseline'):
        """
        load models from disk and predict on test data, returns accuracy and confidence score
        """
        try:
            reconstructed_model = tensorflow.keras.models.load_model(model)

            # check if inputs shape of the data and model matches
            if reconstructed_model.input.shape[1:] == X_test.shape[1:]:
                y_probs = reconstructed_model.predict(X_test)
                y_preds = np.where(y_probs > 0.5, 1, 0)
                y_confidence = np.where(y_probs > 0.5, y_probs, 1 - y_probs)
                pred_and_confidence = [(p, c) for p, c in zip(y_preds.flatten(), y_confidence.flatten())]
                return accuracy_score(y_test, y_preds), pred_and_confidence
            else:
                raise ValueError('Input Shape is not supported by the model')

        except FileNotFoundError:
            print('Model not Found')


    def load_maze_scan_predict(self):
        """
        load the maze (maze.png) and find the bounding box of the car and building
        """
        pass


# x = Sim2Real()
# X, y = x.load_data_and_save_X_Y(class0_dir='building', class1_dir='car')
