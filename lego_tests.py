#!/usr/bin/env python

"""

"""

__author__ = "Shivchander Sudalairaj"
__email__ = "sudalasr@mail.uc.edu"


import unittest
from Sim2Real import *


class TestFunction(unittest.TestCase):

    def test_load_and_pad_data(self):
        testfiles = glob.glob('test/*')

        with self.subTest():
            # check if it can load images to 128x128x3 - same size
            expected_array_size = (6, 128, 128, 3)
            result_array_size = load_and_pad_imgs(img_paths=testfiles, target_height=128, target_width=128).shape
            self.assertEqual(expected_array_size, result_array_size)

        with self.subTest():
            # check if it can load images to 256x256x3 - padded resize
            expected_array_size = (6, 256, 256, 3)
            result_array_size = load_and_pad_imgs(img_paths=testfiles, target_height=256, target_width=256).shape
            self.assertEqual(expected_array_size, result_array_size)

        with self.subTest():
            # check if it can load images to 64x64x3 - resize shrink
            expected_array_size = (6, 64, 64, 3)
            result_array_size = load_and_pad_imgs(img_paths=testfiles, target_height=64, target_width=64).shape
            self.assertEqual(expected_array_size, result_array_size)

    def test_load_data_varying_training_size(self):
        img_array = np.zeros((1000, 128, 128, 3))
        label_array = np.zeros(1000)
        result_list_of_array = load_data_varying_training_size(img_array, label_array)

        # check if all the arrays are of expected shape - variation by 10%
        for i, ele in enumerate(result_list_of_array):
            xi, yi = ele
            with self.subTest():
                expected_array_size = ((i+1) * 100, 128, 128, 3)
                self.assertEqual(expected_array_size, xi.shape)


if __name__ == '__main__':
    unittest.main()