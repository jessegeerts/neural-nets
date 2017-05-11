import tensorflow as tf

class Network(object):

    def __init__(self, batch_size):
        """

        """
        # Parameters
        self.batch_size = batch_size
        self.learning_rate = 0.001
        self.n_training_epochs = 15
        self.display_step = 1

        # Network Parameters
        self.n_hidden_1 = 256  # 1st layer number of features
        self.n_hidden_2 = 256  # 2nd layer number of features
        self.n_input = 784  # MNIST data input (img shape: 28*28)
        self.n_classes = 10  # MNIST total classes (0-9 digits)

        # Input and output placeholders
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def multilayer_perceptron(self):
        """
        This function takes in the input placeholder, weights and biases and returns the output tensor of a network with
        two hidden ReLU layers, and an output layer with linear activation.

        :param tf.placeholder x: Placeholder for input
        :param dict weights: Dictionary containing Variables describing weights of each layer
        :param dict biases: Dictionary containing Variables describing biases of each layer
        :return: The activations of the output layer
        """
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer


    def run_training_cycle(self, sess, x, y, images, labels, n_training_epochs, batch_size, optimizer, cost, display_step=1):
        """
        Runs a training cycle for an already opened Tensorflow session, with a user-defined optimizer and cost function.
        The function loops over a user-defined number of epochs, and splits the data set up in batches of size batch_size.

        :param y:
        :param x:
        :param tf.Session sess: The Tensorflow session under which to run this training
        :param images: images to classify
        :param labels: labels belonging to each image
        :param int n_training_epochs: The number of training epochs.
        :param int batch_size: Size of each batch
        :param tf.Operation optimizer: Choose the optimizer to use during training
        :param tf.Tensor cost: Specify which cost function to use
        :param int display_step: Display progress with steps display_step. Default: 1 (display all epochs)
        :return:
        """
        for epoch in range(n_training_epochs):
            avg_cost = 0.
            total_batch = images.shape[0] / batch_size
            # Loop over all batches
            batches = [(images[k: k + batch_size], labels[k: k + batch_size]) for k in range(0, len(labels), batch_size)]

            for batch in batches:
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch[0],
                                                              y: batch[1]})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization finished")


    def test_model(x, y, output, label, test_images, test_labels):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
        print("prediction:", correct_prediction.eval({x: test_images, y: test_labels}))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        score = accuracy.eval({x: test_images, y: test_labels})
        print("Accuracy:", score)
        return score


