from old_files.network_class import Network

net = Network()

training_data, validation_data, test_data = load_mnist()


train_x_01, train_y_01 = make_mnist_subset(training_data, [0, 1])
test_x_01, test_y_01 = make_mnist_subset(test_data, [0, 1])
train_x_23, train_y_23 = make_mnist_subset(training_data, [2, 3])
test_x_23, test_y_23 = make_mnist_subset(test_data, [2, 3])



mySess = tf.Session()

with mySess as sess:

    sess.run(net.network_initializer)

    net.run_training_cycle(sess, train_x_01, train_y_01)
    net.test_model(test_x_01, test_y_01)

    net.run_training_cycle(sess, train_x_23, train_y_23)
    net.test_model(test_x_23, test_y_23)
    net.test_model(test_x_01, test_y_01)
    net.test_model(test_x_23, test_y_23)