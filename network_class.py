import os
import tensorflow as tf


class Network(object):
    def __init__(self):
        """
        Initialises constants and variables in the Network object
        """
        # Parameters
        self.batch_size = 64
        self.learning_rate = 0.001
        self.n_training_epochs = 10
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

        # Create the network:
        self.out_layer = None
        self.multilayer_perceptron()

        # Define (default) loss and optimizer:
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_layer, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.999).minimize(self.cost)

        # For the multi-headed approach, set default to whole output:
        self.head_output = self.out_layer
        self.head_y = self.y

        # Initialize all variables
        self.network_initializer = tf.global_variables_initializer()

    def multilayer_perceptron(self):
        """
        This function takes in the input placeholder, weights and biases and returns the output tensor of a network with
        two hidden ReLU layers, and an output layer with linear activation.

        :return: The activations of the output layer
        """
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        self.out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']

    def run_training_cycle(self, sess, images, labels, display_step=1):
        """
        Runs a training cycle for an already opened Tensorflow session, with a user-defined optimizer and cost function.
        The function loops over a user-defined number of epochs, and splits the data set up in batches of size
        batch_size.

        :param (tf.Session) sess: The Tensorflow session under which to run this training
        :param images: images to classify
        :param labels: labels belonging to each image
        :param (int) display_step: Display progress with steps display_step. Default: 1 (display all epochs)
        :return:
        """
        for epoch in range(self.n_training_epochs):
            avg_cost = 0.
            total_batch = images.shape[0] / self.batch_size
            # Loop over all batches
            batches = [(images[k: k + self.batch_size], labels[k: k + self.batch_size]) for k in
                       range(0, len(labels), self.batch_size)]

            for batch in batches:
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch[0],
                                                                        self.y: batch[1]})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization finished")

    def test_model(self, test_images, test_labels):
        """
        This function evaluates the performance of the network on the test data

        :param (array) test_images:
        :param (array) test_labels:
        :return:
        """
        # TODO: change this into slice:
        correct_prediction = tf.equal(tf.argmax(self.head_output, 1), tf.argmax(self.head_y, 1))
        # print("prediction:", correct_prediction.eval({self.x: test_images, self.y: test_labels}))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        score = accuracy.eval({self.x: test_images, self.y: test_labels})
        print("Accuracy:", score)
        return score

    def select_head(self, sess, start, size):
        """

        :param sess:
        :param start:
        :param size:
        :return: Sets cost and optimizer to
        """
        # TODO: change this into not actually defining head output and head_y as variables
        self.head_output = tf.slice(self.out_layer, [0, start], [-1, size])
        self.head_y = tf.slice(self.y, [0, start], [-1, size])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.head_output, labels=self.head_y))
        self.reset_optimizer(sess)

    def reset_optimizer(self, sess):
        # TODO: make this function neater (explicit variable re-initialisation)
        temp = set(tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.999).minimize(self.cost)
        print(set(tf.global_variables()) - temp)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

    def reset_optimizer_2(self, sess):
        model_variables = set(tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.999)
        self.train_run = self.optimizer.minimize(self.cost)
        print({self.optimizer._beta1_power, self.optimizer._beta2_power})
        sess.run(tf.variables_initializer([self.optimizer._beta1_power, self.optimizer._beta2_power]))


def saveModel(sess, model_filename):
    model_folder = './saved_models/'
    if not os.path.exists(model_folder):
        print('Creating path where to save model: ' + model_folder)
        os.mkdir(model_folder)

    print('Saving model at: ' + model_filename)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_folder, model_filename))
    print('Model succesfully saved.\n')


def loadModel(sess, model_filename):
    if os.path.exists(model_filename):
        print('Loading save model from: ' + model_filename)
        saver = tf.train.Saver()
        saver.restore(sess, model_filename)
        print('Model succesfully loaded.\n')
        return True
    else:
        print('Model file <<' + MODEL_FILENAME + '>> does not exists!')
        return False
