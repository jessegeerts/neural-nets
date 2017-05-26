import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_loader import load_mnist, make_mnist_subset_categorical_labels
from multi_network import Network

from plotting import plot_average_split_mnist_scores

training_data, validation_data, test_data = load_mnist()

# Create data subsets
train_images = []
train_labels = []
test_images = []
test_labels = []

for train_task in range(0, 10, 2):
    train_x, train_y = make_mnist_subset_categorical_labels(training_data, [train_task, train_task + 1])
    test_x, test_y = make_mnist_subset_categorical_labels(test_data, [train_task, train_task + 1])
    train_images.append(train_x)
    train_labels.append(train_y)
    test_images.append(test_x)
    test_labels.append(test_y)

# Create 5-headed network:
net = Network(5)
net.learning_rate = 0.01
############################################## RUN THE MODEL ########################################################
n_runs = 10

score = np.full([5, 5, n_runs], np.nan)

with tf.Session() as sess:
    for run in range(n_runs):
        sess.run(tf.global_variables_initializer())
        for train_task in range(5):
            net.reset_optimizers(sess)
            net.run_training_cycle(sess, train_task, train_images[train_task], train_labels[train_task])

            for test_task in range(5):
                score[train_task, test_task, run] = net.test_model(test_task,
                                                                   test_images[test_task],
                                                                   test_labels[test_task])


plt.figure()
plot_average_split_mnist_scores(score, 'split_mnist_reset_optimizer.png')
