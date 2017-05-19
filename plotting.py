import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def plot_split_mnist_scores(scores):
    # plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.plot(scores[:, i])
        plt.ylabel('Classification accuracy')
        plt.xlabel('Task number')
        plt.ylim([0.4, 1.1])


def plot_average_split_mnist_scores(scores, filename=None):
    """

    :param scores:
    :return:
    """
    for task in range(scores.shape[0]):

        plt.subplot(1, 5, task + 1)

        score_task = pd.Series(scores[:, task, :].flatten())
        taskn = pd.Series(np.matlib.repmat([1, 2, 3, 4, 5], 10, 1).T.flatten())

        df = pd.concat([taskn, score_task], axis=1)
        df.rename(columns={0: 'Task', 1: 'Score'}, inplace=True)

        sns.pointplot(x=df['Task'], y=df['Score'])
        plt.title('Task ' + str(task + 1))
        plt.ylabel('Accuracy')
        plt.ylim([0.4, 1.05])

        if filename is not None:
            figure_folder = './figs'
            plt.savefig(os.path.join(figure_folder, filename))
