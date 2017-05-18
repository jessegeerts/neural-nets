import tensorflow as tf
from mnist_loader import load_mnist, make_mnist_subset_categorical_labels
from multi_network import Network
import numpy as np

training_data, validation_data, test_data = load_mnist()

# Create data subsets
train_images = []
train_labels = []
test_images = []
test_labels = []

for task in range(0, 10, 2):
    train_x, train_y = make_mnist_subset_categorical_labels(training_data, [task, task + 1])
    test_x, test_y = make_mnist_subset_categorical_labels(test_data, [task, task + 1])
    train_images.append(train_x)
    train_labels.append(train_y)
    test_images.append(test_x)
    test_labels.append(test_y)

# Create 5-headed network:
net = Network(5)

############################################## RUN THE MODEL ########################################################

score = np.full(5,np.nan)
score_task_1 = np.full(5,np.nan)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for task in range(5):
        net.run_training_cycle(sess, task, train_images[task], train_labels[task])
        score[task] = net.test_model(task,test_images[task],test_labels[task])
        score_task_1[task] = net.test_model(0,test_images[0],test_labels[0])
