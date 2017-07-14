import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd
from tqdm import tqdm
from mnist_loader import TrainingSet, TestSet


# Prepare Datasets

training_data_list = []
test_data_list = []

for task in range(0, 10, 2):
    train = TrainingSet(one_hot=False)
    test = TestSet(one_hot=False)
    train.take_subset([task,task+1])
    test.take_subset([task,task+1])
    train.labels = train.labels - task
    test.labels = test.labels - task
    training_data_list.append(train)
    test_data_list.append(test)


##### SETTING UP THE NEURAL NETWORK ######
ed.set_seed(314159)
N = 100   # number of images in a minibatch.
D = 784   # number of features.
K = 10   # number of classes.
n_heads = 5
head_size = int(K/n_heads)


# define the feedforward neural network function

def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return h

def run_training_cycle(inference,batch_size,training_data,x,y_ph):
    for _ in range(inference.n_iter):
        X_batch, Y_batch = training_data.next_batch(batch_size,shuffle=False)
        info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
        inference.print_progress(info_dict)

def take_posterior_samples(n_samples,X_test,testhead,qW_0,qb_0,qW_1,qb_1):

    prob_lst = []
    samples = []
    w_0_samples = []
    b_0_samples = []
    w_1_samples = []
    b_1_samples = []

    for _ in tqdm(range(n_samples)):
        w0_samp = qW_0.sample()
        b0_samp = qb_0.sample()
        w1_samp = qW_1[testhead].sample()
        b1_samp = qb_1[testhead].sample()

        w_0_samples.append(w0_samp)
        b_0_samples.append(b0_samp)
        w_1_samples.append(w1_samp)
        b_1_samples.append(b1_samp)

        # Also compute the probabiliy of each class for each sample.
        prob = tf.nn.softmax(neural_network(X_test, w0_samp, w1_samp, b0_samp, b1_samp))
        prob_lst.append(prob.eval())
        sample = tf.concat([tf.reshape(w1_samp, [-1]), b1_samp], 0)
        samples.append(sample.eval())
    return prob_lst, samples

# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [N, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
W_0 = Normal(loc=tf.zeros([D, 20]), scale=tf.ones([D, 20]))
b_0 = Normal(loc=tf.zeros(20), scale=tf.ones(20))

W_1 = []
b_1 = []
for head in range(5):
    W_1.append(Normal(loc=tf.zeros([20, head_size]), scale=tf.ones([20, head_size])))
    b_1.append(Normal(loc=tf.zeros(head_size), scale=tf.ones(head_size)))

# Categorical likelihood for classication.
y=[]
for head in range(5):
    y.append(Categorical(neural_network(x, W_0, W_1[head], b_0, b_1[head])))

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 20])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 20]))))
qb_0 = Normal(loc=tf.Variable(tf.random_normal([20])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([20]))))

qW_1 = []
qb_1 = []
for head in range(5):
    qW_1.append(Normal(loc=tf.Variable(tf.random_normal([20, head_size])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([head_size])))))
    qb_1.append(Normal(loc=tf.Variable(tf.random_normal([head_size])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([head_size])))))

# We use a placeholder for the labels in anticipation of the training data.
y_ph = tf.placeholder(tf.int32, [N])
head = 0

accuracies = []
for head in range(5):

    training_data = training_data_list[head]

    # Define the VI inference technique, ie. minimise the KL divergence between q and p.
    inference = ed.KLqp({W_0: qW_0, W_1[head]: qW_1[head],
                         b_0: qb_0, b_1[head]: qb_1[head]}, data={y[head]:y_ph})

    # Initialise the inference variables
    inference.initialize(n_iter=2000, n_print=100, scale={y[head]: float(training_data._num_examples) / N})

    # We will use an interactive session.
    sess = tf.InteractiveSession()
    # Initialise all the vairables in the session.
    tf.global_variables_initializer().run()

    # Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
    run_training_cycle(inference, N, training_data, x, y_ph)

    for testhead in range(5):

        test_data = test_data_list[testhead]

        X_test = test_data.images
        Y_test = test_data.labels

        # Generate samples from the posterior and store them.

        prob_lst, samples = take_posterior_samples(10, X_test, testhead, qW_0, qb_0, qW_1, qb_1)

        # Compute the accuracy of the model.
        accy_test = []
        for prob in prob_lst:
            y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
            acc = (y_trn_prd == Y_test).mean()*100
            accy_test.append(acc)

        """
        plt.hist(accy_test)
        plt.title("Histogram of prediction accuracies in the MNIST test data")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        """

        # Here we compute the mean of probabilties for each class for all the (w,b) samples.
        Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)
        accuracy = (Y_pred == Y_test).mean()*100
        print("accuracy in predicting the test data = ", accuracy)
        accuracies.append(accuracy)

        plt.figure()
        # Create a Pandas DataFrame of posterior samples.
        samples_df = pd.DataFrame(data = samples, index=range(10))
        # Now create a small subset by taking the first 5 weights, labelled as W_0, ... , W_4.
        samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values,columns=["W_0", "W_1", "W_2", "W_3", "W_4"])
        # We use Seaborn PairGrid to make a triale plot to show auto and cross correlations.
        g = sns.PairGrid(samples_5, diag_sharey=False)
        g.map_lower(sns.kdeplot, n_levels = 4,cmap="Blues_d")
        g.map_upper(plt.scatter)
        g.map_diag(sns.kdeplot,legend=False)
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle('Joint posterior distribution of the first 5 weights')


# Load the first image from the test data and its label.
test_image = X_test[1:2]
test_label = Y_test[1]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.figure()
plt.imshow(pixels,cmap='Blues')

# Now the check what the model perdicts for each (w,b) sample from the posterior. This may take a few seconds...
sing_img_probs = []
n_samples = 100
for _ in tqdm(range(n_samples)):
    w0_samp = qW_0.sample()
    b0_samp = qb_0.sample()
    w1_samp = qW_1[head].sample()
    b1_samp = qb_1[head].sample()

    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(neural_network(X_test[1:2], w0_samp, w1_samp, b0_samp, b1_samp))
    sing_img_probs.append(prob.eval())


# Create a histogram of these predictions.
plt.figure()
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,K))
plt.xlim(0,K)
plt.xlabel("Accuracy of the prediction of the test digit")
plt.ylabel("Frequency")