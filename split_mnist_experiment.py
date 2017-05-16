

net = Network()
scores_task0 = np.full(5, np.NaN)
scores = np.full(5, np.NaN)


mySess = tf.Session()

with mySess as sess:

    #net.select_head(sess,task, 2)
    sess.run(net.network_initializer)
    #train_x, train_y = make_mnist_subset(training_data, [task, task + 1])
    #net.run_training_cycle(sess, train_x, train_y)

    for task in range(0,10,2):
        print('\nTask ', int(task/2),':\n')

        net.select_head(sess,task, 2)

        # Train:
        train_x, train_y = make_mnist_subset(training_data, [task, task + 1])
        net.run_training_cycle(sess, train_x,train_y)

        # test on current task:
        test_x, test_y = make_mnist_subset(test_data, [task, task + 1])
        #scores[task] = net.test_model
        scores[task] = net.test_model(test_x,test_y)

        # test on task 0:
        net.select_head(sess, 0, 2)
        test_x, test_y = make_mnist_subset(test_data, [0, 0 + 1])
        scores_task0[task] = net.test_model(test_x,test_y)
