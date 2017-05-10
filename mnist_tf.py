import numpy as np
import tensorflow as tf
from mnist_loader import make_mnist_subset, load_mnist

training_data, validation_data, test_data = load_mnist()

# Parameters
learning_rate = 0.001
training_epochs = 15
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


# Create model
def multilayer_perceptron(x, weights, biases):
    """
    This function takes in the input placeholder, weights and biases and returns the output tensor of a network with
    two hidden ReLU layers, and an output layer with linear activation.

    :param tf.placeholder x: Placeholder for input
    :param dict weights: Dictionary containing Variables describing weights of each layer
    :param dict biases: Dictionary containing Variables describing biases of each layer
    :return: The activations of the output layer
    """
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

scores = np.full(5,np.NaN)


mySess = tf.Session()

with mySess as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = training_data[1].shape[0] / batch_size
        # Loop over all batches
        batches = [(train_x[k: k + batch_size], train_y[k: k + batch_size]) for k in range(0, len(train_y), batch_size)]

        for batch in batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost_zero_one], feed_dict={x: batch[0],
                                                                   y: batch[1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost))
    print("Optimization finished")

    # Test model
    correct_prediction = tf.equal(tf.argmax(slice_pred, 1), tf.argmax(slice_y, 1))
    prediction = tf.argmax(pred, 1)
    print("prediction:", prediction.eval({x: test_x, y: test_y}))

    # TODO: test with only one head
    # TODO: evaluate with test data

    # print("prediction: ", tf.cast(tf.argmax(pred,1),"float"))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))

    #print("Accuracy (training data):", accuracy.eval({x: train_x[0:batch_size], y: train_y[0:batch_size]}))