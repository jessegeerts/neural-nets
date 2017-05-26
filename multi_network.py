import os
import tensorflow as tf


class Network(object):
    def __init__(self, n_heads=1):
        """
        Initialises constants and variables in the Network object
        """
        # Parameters
        self.n_heads = n_heads
        self.batch_size = 64
        self.learning_rate = 0.005
        self.n_training_epochs = 50
        self.display_step = 1

        # Network Parameters
        self.n_hidden_1 = 30  # 1st layer number of features
        self.n_hidden_2 = 30  # 2nd layer number of features
        self.n_input = 784  # MNIST data input (img shape: 28*28)
        self.n_classes_total = 10  # MNIST total classes (0-9 digits)
        self.n_classes_per_head = int(self.n_classes_total / self.n_heads)

        # Input and output placeholders
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = []
        for i in range(n_heads):
            self.y.append(tf.placeholder("float", [None, self.n_classes_per_head]))

        # Store layers weight & bias
        layer_1_weights = tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='H1_weights')
        layer_2_weights = tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='H2_weights')
        output_layer_weights = []
        for i in range(self.n_heads):
            varname = 'outweights_head_' + str(i)
            output_layer_weights.append(tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes_per_head]),
                                                    name=varname))

        self.weights = {'h1': layer_1_weights,
                        'h2': layer_2_weights,
                        'out': output_layer_weights}

        layer_1_biases = tf.Variable(tf.random_normal([self.n_hidden_1]), name='H1_bias')
        layer_2_biases = tf.Variable(tf.random_normal([self.n_hidden_2]), name='H2_bias')
        output_layer_biases = []
        for i in range(self.n_heads):
            output_layer_biases.append(tf.Variable(tf.random_normal([self.n_classes_per_head]), name='out_bias_'
                                                                                                     + str(i)))

        self.biases = {
            'b1': layer_1_biases,
            'b2': layer_2_biases,
            'out': output_layer_biases
        }

        # Tensor that keeps track of the number of batches processed:
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        # Create the network:
        self.out_layer = []
        self.multilayer_perceptron()

        # Calculate Loss
        self.cost = []
        for head in range(n_heads):
            cost_per_head = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.out_layer[head],
                labels=self.y[head]))
            self.cost.append(cost_per_head)

        self.Joint_Loss = sum(self.cost)

        # optimisers
        self.joint_optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Joint_Loss)
        self.optimiser = []
        self.train_step = []
        for head in range(n_heads):
            optimiser_per_head = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_step_h = optimiser_per_head.minimize(self.cost[head], global_step=self.global_step_tensor)
            self.optimiser.append(optimiser_per_head)
            self.train_step.append(train_step_h)

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
        for i in range(self.n_heads):
            head_output = tf.matmul(layer_2, self.weights['out'][i]) + self.biases['out'][i]
            self.out_layer.append(head_output)

    def run_training_cycle(self, sess, head, images, labels, display_step=1):
        """
        Runs a training cycle for an already opened Tensorflow session, with a user-defined optimizer and cost function.
        The function loops over a user-defined number of epochs, and splits the data set up in batches of size
        batch_size.

        :param head:
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
                _, c = sess.run([self.train_step[head], self.cost[head]], feed_dict={self.x: batch[0],
                                                                                    self.y[head]: batch[1]})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization finished")

    def test_model(self, head, test_images, test_labels):
        """
        This function evaluates the performance of the network on the test data

        :param head:
        :param (array) test_images:
        :param (array) test_labels:
        :return:
        """
        correct_prediction = tf.equal(tf.argmax(self.out_layer[head], 1), tf.argmax(self.y[head], 1))
        print("prediction:", correct_prediction.eval({self.x: test_images, self.y[head]: test_labels}))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        score = accuracy.eval({self.x: test_images, self.y[head]: test_labels})
        print("Accuracy:", score)
        return score

    def reset_optimizers(self, sess):
        # TODO: make this function neater (explicit variable re-initialisation)

        print('Resetting optimisers...')
        temp = set(tf.global_variables())
        for head in range(self.n_heads):
            self.optimiser[head] = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.999).minimize(self.cost[head])
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

    def compute_omega(self, loss, head):
        weights = [self.weights['h1'], self.weights['h2'], self.weights['out'][head]]
        # Gradient of the loss function with respect to the weights
        weight_gradients = tf.gradients(loss, weights)
        # Parameter update: partial derivative of the parameters wrt time
        parameter_update = self.optimiser[head].compute_gradients(loss,weights)
        self.omega = weight_gradients * parameter_update