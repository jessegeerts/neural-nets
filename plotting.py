import matplotlib.pyplot as plt

def plot_split_mnist_scores(scores):
    plt.figure()
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.plot(scores[:,i])
        plt.ylabel('Classification accuracy')
        plt.xlabel('Task number')
        plt.ylim([0.4,1.1])