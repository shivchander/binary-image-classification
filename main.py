#!/usr/bin/env python

"""

"""

__author__ = "Shivchander Sudalairaj"
__email__ = "sudalasr@mail.uc.edu"

import json
from Sim2Real import *
from sklearn.model_selection import train_test_split
import glob

MODEL_IN_SIZES = {'baseline': (128, 128), 'VGG16': (224, 244), 'Xception': (299, 299)}

if __name__ == '__main__':
    try:
        with open('config.json', 'r') as config_file:
            config_data = json.load(config_file)

        s_instance = Sim2Real()
        print(config_data)

        # use pretrained model to make predictions on the test dir
        # assuming the files have 'building' or 'car' in its name
        if config_data['use_pretrained']:
            test_file_paths = sorted(glob.glob('{}/*'.format(config_data['test_dir'])))
            target_height, target_width = MODEL_IN_SIZES[config_data['model_name']]
            model_path = '{}/{}'.format(config_data['model_dir'], config_data['model_name'])
            X_test = load_and_pad_imgs(test_file_paths, target_height, target_width)
            y_test = np.array([0 if 'building' in file_path else 1 for file_path in test_file_paths])

            accuracy, pred_confidence = s_instance.load_model_predict(X_test, y_test, model_path)

            print('Accuracy Score: ', accuracy)
            for path, pc in zip(test_file_paths, pred_confidence):
                print('\t File: {}'.format(path))
                print('\t Prediction: {}'.format(pc[0]))
                print('\t Confidence: {}'.format(pc[1]))
                print()

        else:
            X, y = s_instance.load_data_and_save_X_Y(config_data['class0_dir'], config_data['class1_dir'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            history = s_instance.sim_train(X_train, y_train, X_test, y_test, model=config_data['model_name'],
                                           data_augment=config_data['augment_data'], epochs=config_data['num_epochs'],
                                           validation_split=0.2, early_stopping=True, save=True,
                                           save_dir=config_data['model_dir'], filename=config_data['model_name'])

    except FileNotFoundError:
        print('Missing Files ... Exiting')

    except KeyError:
        print('Incomplete Config')