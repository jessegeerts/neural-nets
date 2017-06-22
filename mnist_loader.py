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

    idx = [np.squeeze(np.where(data[1] == i)) for i in digits]
    idx = np.sort(np.concatenate(idx, axis=0))

    images = data[0][idx]
    labels = utils.to_categorical(data[1][idx], num_classes=10)  # convert labels to categorical
    return images, labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


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

    idx = [np.squeeze(np.where(data[1] == i)) for i in digits]
    idx = np.sort(np.concatenate(idx, axis=0))

    images = data[0][idx]
    labels = utils.to_categorical(data[1][idx], num_classes=10)  # convert labels to categorical
    labels = labels[:, digits]
    return images, labels


class DataSet(object):
    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.training_data, self.validation_data, self.test_data = load_mnist()
        self._num_examples = self.training_data[0].shape[0]
        self._images = None
        self._labels = None

    def next_batch(self, batch_size, shuffle=False):
        """
        Return the next `batch_size` examples from this data set.

        :param batch_size:
        :param shuffle:
        :return:
        """
        start = self._index_in_epoch
        self._images = self.images
        self._labels = self.labels
        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # go to next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            else:
                self._images = self.images
                self._labels = self.labels
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def take_subset(self, digits=list(range(10))):
        """
        Takes a subset of the dataset

        :param digits: A list of digits from 0 to 10 you want to keep in the dataset
        :return:
        """
        idx = [np.where(self.labels == i)[0].tolist() for i in digits]
        flatlist_idx = [item for sublist in idx for item in sublist]
        self.images = self.images[flatlist_idx]
        self.labels = self.labels[flatlist_idx]
        self._num_examples = len(flatlist_idx)

    def to_one_hot(self):
        self.labels = dense_to_one_hot(self.labels, num_classes=len(set(self.labels)))

    def restore(self):
        __init__(self)


class TrainingSet(DataSet):
    def __init__(self, one_hot=False):
        DataSet.__init__(self)
        self.images = self.training_data[0]
        self.labels = self.training_data[1]
        if one_hot is True:
            self.labels = dense_to_one_hot(self.labels, num_classes=len(set(self.labels)))


class ValidationSet(DataSet):
    def __init__(self, one_hot=False):
        DataSet.__init__(self)
        self.images = self.validation_data[0]
        self.labels = self.validation_data[1]
        if one_hot is True:
            self.labels = dense_to_one_hot(self.labels, num_classes=len(set(self.labels)))


class TestSet(DataSet):
    def __init__(self, one_hot=False):
        DataSet.__init__(self)
        self.images = self.test_data[0]
        self.labels = self.test_data[1]
        if one_hot is True:
            self.labels = dense_to_one_hot(self.labels, num_classes=len(set(self.labels)))
