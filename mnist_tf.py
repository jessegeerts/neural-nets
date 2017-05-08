import gzip
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def load_mnist():
    """

    :return:
    """
    f = gzip.open('/home/jesse/Code/neural-networks-and-deep-learning/data/mnist.pkl.gz')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


training_data, validation_data, test_data = load_mnist()


def make_subset(data, digits):
    """
    Takes a list of numbers and returns the part of the MNIST dataset corresponding to these numbers.

    :param tuple data: e.g. train_data, validation_data or test_data
    :param list digits: list of digits to be included in subset (will be converted into list if int is given)
    :return:
    """
    if isinstance(digits, int):
        digits = [digits]
    idx = [np.squeeze(np.argwhere(data[1] == i)) for i in digits]
    idx = np.sort(np.concatenate(idx, axis=0))
    return data[0][idx], data[1][idx]


# train_x, train_y = make_subset(training_data, [0, 1])
train_x = training_data[0]
train_y = training_data[1]
test_x = test_data[0]
test_y = test_data[1]

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


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

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[:,[0:2]], labels=y[:,[0:2]]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = training_data[1].shape[0] / batch_size
        # Loop over all batches
        batches = [(train_x[k: k + batch_size], train_y[k: k + batch_size]) for k in range(0, len(train_y), batch_size)]

        for batch in batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch[0],
                                                          y: keras.utils.to_categorical(batch[1], num_classes=10)})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_x, y: keras.utils.to_categorical(test_y, num_classes=10)}))
