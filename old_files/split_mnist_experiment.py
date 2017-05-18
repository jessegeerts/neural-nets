import numpy as np
import tensorflow as tf

from mnist_loader import make_mnist_subset, load_mnist
from old_files.network_class import Network
from plotting import plot_split_mnist_scores

training_data, validation_data, test_data = load_mnist()

net = Network()
scores = np.full((5, 5), np.NaN)


mySess = tf.Session()

with mySess as sess:

    sess.run(net.network_initializer)

    for task in [0,1,2,3,4]:
        print('\nTask ', int(task), ':\n')
        start_digit = task * 2

        net.select_head(sess, start_digit, 2)

        # Train:
        train_x, train_y = make_mnist_subset(training_data, [start_digit, start_digit + 1])
        net.run_training_cycle(sess, train_x, train_y)

        for test in range(5):
            sd = test*2
            net.select_head(sess, sd, 2)
            test_x, test_y = make_mnist_subset(test_data, [sd, sd+1])
            scores[task, test] = net.test_model(test_x, test_y)

plot_split_mnist_scores(scores)


net2 = Network()
mySess = tf.Session()

with mySess as sess:
    net2.reset_optimizer(sess)

    net2.reset_optimizer_2(sess)

# TODO: fix the bugs. As a first step, try without loop, just task 1, test task 1. then, task 2, test task 1 and 2


train_x_01, train_y_01 = make_mnist_subset(training_data, [0, 1])
test_x_01, test_y_01 = make_mnist_subset(test_data, [0, 1])
train_x_23, train_y_23 = make_mnist_subset(training_data, [5, 6])
test_x_23, test_y_23 = make_mnist_subset(test_data, [5, 6])

net1 = Network()
sess1 = tf.Session()

MODEL_FILENAME = "nn_model_task1.ckpt"

with sess1 as sess:

    sess.run(net1.network_initializer)
    net1.select_head(sess, 0, 2)
    # TASK 1
    #net.select_head(sess, 0, 2)
    net1.run_training_cycle(sess, train_x_01, train_y_01)
    score_task1 = net1.test_model(test_x_01, test_y_01)

    saveModel(sess,MODEL_FILENAME)


net2 = Network()
sess2 = tf.Session()
with sess2 as sess:

    saver = tf.train.Saver()
    saver.restore(sess, "./saved_models/nn_model_task1.ckpt")

    loadModel(sess, model_filename)
    # TASK 2
    net.select_head(sess, 5, 2)
    net.run_training_cycle(sess, train_x_23, train_y_23)
    score_task2 = net.test_model(test_x_23, test_y_23)

    # RE-TEST TASK 1:
    net.select_head(sess, 0, 2)
    score_task1_after_task2 = net.test_model(test_x_01, test_y_01)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./saved_models/nn_model_task1.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))



#########################################GIVE IT ANOTHER TRY #############################################

train_x_01, train_y_01 = make_mnist_subset(training_data, [0, 1])
test_x_01, test_y_01 = make_mnist_subset(test_data, [0, 1])
train_x_23, train_y_23 = make_mnist_subset(training_data, [2, 3])
test_x_23, test_y_23 = make_mnist_subset(test_data, [2, 3])

net1 = Network()
saver = tf.train.Saver()

# Running first session
print("Starting 1st session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(net1.network_initializer)

    net1.run_training_cycle(sess, train_x_01, train_y_01)
    score_task1 = net1.test_model(test_x_01, test_y_01)

    save_path = saver.save(sess,'task-1.ckpt')


print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(net1.network_initializer)

    # Restore model weights from previously saved model
    saver.restore(sess, './task-1.ckpt')
    print("Model restored from file: %s" % save_path)

    net1.select_head(sess, 2, 2)

    net1.run_training_cycle(sess, train_x_23, train_y_23)
    score_task1 = net1.test_model(test_x_23, test_y_23)

    save_path = saver.save(sess, 'task-2.ckpt')


print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(net1.network_initializer)

    # Restore model weights from previously saved model
    saver.restore(sess, './task-2.ckpt')
    print("Model restored from file: %s" % save_path)

    net1.select_head(sess, 0, 2)

    score_task1 = net1.test_model(test_x_01, test_y_01)


