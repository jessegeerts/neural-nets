import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd
from tqdm import tqdm
from mnist_loader import TrainingSet, TestSet


training_data = TrainingSet(one_hot=True)
test_data = TestSet(one_hot=False)
# select specific subset for task
digits = [0,1]
#training_data.take_subset(digits)
#test_data.take_subset(digits)

ed.set_seed(314159)
N = 100   # number of images in a minibatch.
D = 784   # number of features.
K = 10# len(digits)    # number of classes.

# define the feedforward neural network function

def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return h

# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [N, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
W_0 = Normal(loc=tf.zeros([D, 20]), scale=tf.ones([D, 20]))
W_1 = Normal(loc=tf.zeros([20, K]), scale=tf.ones([20, K]))
b_0 = Normal(loc=tf.zeros(20), scale=tf.ones(20))
b_1 = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(neural_network(x, W_0, W_1, b_0, b_1))

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 20])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 20]))))
qW_1 = Normal(loc=tf.Variable(tf.random_normal([20, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([20, K]))))
qb_0 = Normal(loc=tf.Variable(tf.random_normal([20])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([20]))))
qb_1 = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the training data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({W_0: qW_0, W_1: qW_1,
                     b_0: qb_0, b_1: qb_1}, data={y:y_ph})

# Initialise the inference variables
inference.initialize(n_iter=5000, n_print=100, scale={y: float(training_data._num_examples) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.

for _ in range(inference.n_iter):
    X_batch, Y_batch = training_data.next_batch(N,shuffle=False)
    #X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vector format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)

    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

X_test = test_data.images
Y_test = test_data.labels

# Generate samples the posterior and store them.
n_samples = 100
prob_lst = []
samples = []
w_0_samples = []
b_0_samples = []
w_1_samples = []
b_1_samples = []

for _ in tqdm(range(n_samples)):
    w0_samp = qW_0.sample()
    b0_samp = qb_0.sample()
    w1_samp = qW_1.sample()
    b1_samp = qb_1.sample()

    w_0_samples.append(w0_samp)
    b_0_samples.append(b0_samp)
    w_1_samples.append(w1_samp)
    b_1_samples.append(b1_samp)

    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(neural_network(X_test, w0_samp, w1_samp, b0_samp, b1_samp))
    #prob = tf.nn.softmax(tf.matmul( X_test,w_samp ) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w0_samp,[-1]),b0_samp],0)
    samples.append(sample.eval())


# Compute the accuracy of the model.
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum probability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")


# Here we compute the mean of probabilties for each class for all the (w,b) samples.
# We then use the class with maximum of the mean proabilities as the prediction.
# In other words, we have used (w,b) samples to construct a set of models and
# used their combined outputs to make the predcitions.
Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)


# Create a Pandas DataFrame of posterior samples.
samples_df = pd.DataFrame(data = samples, index=range(n_samples))
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
    w1_samp = qW_1.sample()
    b1_samp = qb_1.sample()

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