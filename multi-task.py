import tensorflow as tf
import numpy as np

from mnist_loader import make_mnist_subset, load_mnist


# Parameters
learning_rate = 0.001
n_training_epochs = 50
batch_size = 64
display_step = 1

# Network Parameters/home/jesse/Code/neural-networks-and-deep-learning/data/mnist.pkl.gz
n_hidden_1 = 30  # 1st layer number of features
n_hidden_2 = 30  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes_total = 10  # MNIST total classes (0-9 digits)
n_classes = int(n_classes_total/5)


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y1 = tf.placeholder("float", [None, n_classes])
y2 = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out1': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
    'out2': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out1': tf.Variable(tf.random_normal([n_classes])),
    'out2': tf.Variable(tf.random_normal([n_classes]))
}


def twoheaded_perceptron(x, weights, biases):
    """
    This function takes in the input placeholder, weights and biases and returns the output tensor of a network with
    two hidden ReLU layers, and an output layer with linear activation.

    :param tf.placeholder x: Placeholder for input
    :param dict weights: Dictionary containing Variables describing weights of each layer
    :param dict biases: Dictionary containing Variables describing biases of each layer
    :return: The activations of the output layer
    """
    # Shared:
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    #not shared
    # Output layer with linear activation
    out_layer1 = tf.matmul(layer_2, weights['out1']) + biases['out1']
    out_layer2 = tf.matmul(layer_2, weights['out2']) + biases['out2']
    return out_layer1, out_layer2

# Create model
pred1, pred2 = twoheaded_perceptron(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Calculate Loss
Y1_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred1, labels=y1))
Y2_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2, labels=y2))
Joint_Loss = Y1_cost + Y2_cost

# optimisers

Optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Joint_Loss)
Y1_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Y1_cost)
Y2_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Y2_cost)



init = tf.global_variables_initializer()

# make data subsets
train_x_1, train_y_1 = make_mnist_subset_categorical(training_data, [0, 1])
test_x_1, test_y_1 = make_mnist_subset_categorical(test_data, [0, 1])
train_x_2, train_y_2 = make_mnist_subset_categorical(training_data, [2, 3])
test_x_2, test_y_2 = make_mnist_subset_categorical(test_data, [2, 3])

############################################## RUN THE MODEL ########################################################



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train task 1
    for epoch in range(n_training_epochs):
        avg_cost = 0.
        total_batch = train_x_1.shape[0] / batch_size
        # Loop over all batches
        batches = [(train_x_1[k: k + batch_size], train_y_1[k: k + batch_size]) for k in range(0, len(train_y_1), batch_size)]

        for batch in batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([Y1_op, Y1_cost], feed_dict={x: batch[0],
                                                          y1: batch[1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization finished")

    #test task 1:

    correct_prediction = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    score1 = accuracy.eval({x: test_x_1, y1: test_y_1})

    """
    # test task 2:

    correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y2, 1))
    print("prediction:", correct_prediction.eval({x: test_x_2, y2: test_y_2}))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    score2 = accuracy.eval({x: test_x_2, y2: test_y_2})
    """

    # TRAIN TASK 2
    for epoch in range(n_training_epochs):
        avg_cost = 0.
        total_batch = train_x_2.shape[0] / batch_size
        # Loop over all batches
        batches = [(train_x_2[k: k + batch_size], train_y_2[k: k + batch_size]) for k in range(0, len(train_y_2), batch_size)]

        for batch in batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([Y2_op, Y2_cost], feed_dict={x: batch[0],
                                                          y2: batch[1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization finished")

    # test task 2:

    correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y2, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    score2 = accuracy.eval({x: test_x_2, y2: test_y_2})

    # test task 1 again:

    correct_prediction = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    score1_after2 = accuracy.eval({x: test_x_1, y1: test_y_1})


print('score1: '+ str(score1) + ', score2: ' + str(score2) + ', score 1 after 2: ' + str(score1_after2))
