import gzip
import pickle

import numpy as np
from keras import utils as utils

def load_mnist():
    """
    Loads the MNIST handwritten digits dataset into three tuples training_data/

    :return: Three tuples containing training data, validation data and test data
    """
    f = gzip.open('./data/mnist.pkl.gz')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def make_mnist_subset(data, digits):
    """
    Takes a list of numbers and returns the part of the MNIST dataset corresponding to these numbers. In addition,
    the labels are converted to a 1 out of 10 representation.

    :rtype: array, array
    :param tuple data: e.g. train_data, validation_data or test_data
    :param list digits: list of digits to be included in subset (will be converted into list if int is given)
    :return: a Nx8=784 array of images and a Nx10 array of labels
    """
    if isinstance(digits, int):
        digits = [digits]

    idx = [np.squeeze(np.argwhere(data[1] == i)) for i in digits]
    idx = np.sort(np.concatenate(idx, axis=0))

    images = data[0][idx]
    labels = utils.to_categorical(data[1][idx],num_classes=10) # convert labels to categorical
    return images, labels


def make_mnist_subset_categorical_labels(data, digits):
    """
    Takes a list of numbers and returns the part of the MNIST dataset corresponding to these numbers. In addition,
    the labels are converted to a 1 out of 10 representation.

    :rtype: array, array
    :param tuple data: e.g. train_data, validation_data or test_data
    :param list digits: list of digits to be included in subset (will be converted into list if int is given)
    :return: a Nx8=784 array of images and a Nx10 array of labels
    """
    if isinstance(digits, int):
        digits = [digits]

    idx = [np.squeeze(np.argwhere(data[1] == i)) for i in digits]
    idx = np.sort(np.concatenate(idx, axis=0))

    images = data[0][idx]
    labels = utils.to_categorical(data[1][idx],num_classes=10) # convert labels to categorical
    labels = labels[:,digits]
    return images, labels

