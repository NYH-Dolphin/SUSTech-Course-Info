from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from sklearn import datasets
import numpy as np
import os
from mlp_numpy import MLP, DataSet
from modules import CrossEntropy
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20,12,6,5'  # 隐藏层
LEARNING_RATE_DEFAULT = 1e-2  # 学习率
MAX_EPOCHS_DEFAULT = 500  # epoch
EVAL_FREQ_DEFAULT = 10  # 每多少个epoch进行一次输出
BATCH = 10  # 梯度下降批处理

FLAGS = None


def get_data():
    samples, labels = datasets.make_moons(2000, shuffle=True, random_state=20)
    train_samples = samples[0:1400]
    train_labels = labels[0:1400]
    test_samples = samples[1400:]
    test_labels = labels[1400:]
    return DataSet(train_samples, train_labels, test_samples, test_labels)


def accuracy(model: MLP):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    x = np.arange(0, model.epoch)
    y = model.accuracy
    plt.plot(x, y)
    plt.title("Accuracy of train set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    x = np.arange(0, model.epoch)
    y = model.loss
    plt.plot(x, y)
    plt.title("Loss of train set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("Loss(Cross Entropy)")
    plt.show()

    x = np.arange(0, model.epoch / model.eval_freq)
    y = model.test_accuracy
    plt.plot(x, y)
    plt.title("Accuracy of test set in each eval frequency")
    plt.xlabel("eval frequency")
    plt.ylabel("accuracy")
    plt.show()

    x = np.arange(0, model.epoch / model.eval_freq)
    y = model.test_loss
    plt.plot(x, y)
    plt.title("Loss of test set in each eval frequency")
    plt.xlabel("epoch")
    plt.ylabel("Loss(Cross Entropy)")
    plt.show()
    return accuracy


def train(model: MLP):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    model.train()


def main(FLAGS=None,n_hidden=list(map(int, DNN_HIDDEN_UNITS_DEFAULT.split(','))),lr=LEARNING_RATE_DEFAULT, epoch=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT, batch=BATCH):
    """
    Main function
    """
    datSet = get_data()
    if FLAGS != None:
        n_hidden = list(map(int, FLAGS.dnn_hidden_units.split(',')))
        lr = FLAGS.learning_rate
        epoch = FLAGS.max_steps
        eval_freq = FLAGS.eval_freq
        batch = FLAGS.batch
    model = MLP(2, n_hidden, 2, datSet, lr, epoch, eval_freq, batch)

    train(model)
    accuracy(model)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch', type=int, default=BATCH, help='Batch used for gradient descent')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
