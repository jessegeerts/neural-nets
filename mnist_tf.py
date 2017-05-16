import numpy as np
import tensorflow as tf

from mnist_loader import make_mnist_subset, load_mnist
from network import multilayer_perceptron, run_training_cycle, test_model

training_data, validation_data, test_data = load_mnist()

# Parameters
learning_rate = 0.001
n_training_epochs = 15
batch_size = 64
display_step = 1

# Network Parameters/home/jesse/Code/neural-networks-and-deep-learning/data/mnist.pkl.gz
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

slice_pred = tf.slice(pred, [0, 0], [-1, 2])
slice_y = tf.slice(y, [0, 0], [-1, 2])

cost_zero_one = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=slice_pred, labels=slice_y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_zero_one)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
task = 0

train_x, train_y = make_mnist_subset(training_data, [task, task + 1])
test_x, test_y = make_mnist_subset(test_data, [task, task + 1])

scores = np.full(5, np.NaN)

mySess = tf.Session()

with mySess as sess:
    sess.run(init)



    for i in range(0, 2, 2):

        run_training_cycle(sess, x, y, train_x, train_y, n_training_epochs, batch_size, optimizer, cost_zero_one)

        test_model(x, y, slice_pred, slice_y, test_x, test_y)

