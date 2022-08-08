from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import random

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20,12,6,5'  # 隐藏层
LEARNING_RATE_DEFAULT = 0.1  # 学习率
MAX_EPOCHS_DEFAULT = 500  # epoch
EVAL_FREQ_DEFAULT = 10  # 每多少个batch进行一次输出

FLAGS = None


# 修改 label 的维度
def change_labels(labels):
    m_labels = list()
    for i in range(len(labels)):
        m_label = np.zeros(2)
        c = labels[i]
        m_label[c] = 1
        m_labels.append(m_label)
    return np.array(m_labels)


# 获取数据集
def get_data():
    samples, labels = datasets.make_moons(2000, shuffle=True, random_state=20)
    train_samples = samples[0:1400]
    train_labels = change_labels(labels[0:1400])
    test_samples = samples[1400:]
    test_labels = change_labels(labels[1400:])

    # 转换成tensor张量
    x_tra = Variable(torch.from_numpy(train_samples))
    x_tra = x_tra.float()
    y_tra = Variable(torch.from_numpy(train_labels))
    y_tra = y_tra.float()

    x_val = Variable(torch.from_numpy(test_samples))
    x_val = x_val.float()
    y_val = Variable(torch.from_numpy(test_labels))
    y_val = y_val.float()

    train_dataset = torch.utils.data.TensorDataset(x_tra, y_tra)
    train_loader = DataLoader(dataset=train_dataset)
    test_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_loader = DataLoader(dataset=test_dataset)
    return train_loader, test_loader


def accuracy(train_accuracy, test_accuracy):
    x = np.arange(0, len(train_accuracy))
    y = train_accuracy
    plt.plot(x, y)
    plt.title("Accuracy of train set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    x = np.arange(0, len(test_accuracy))
    y = test_accuracy
    plt.plot(x, y)
    plt.title("Accuracy of test set in each eval frequency")
    plt.xlabel("eval frequency")
    plt.ylabel("accuracy")
    plt.show()


    return accuracy


def train(net: MLP, epoch, eval_freq, criterion, optimizer):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    train_loader, test_loader = get_data()
    train_accuracy = list() # 训练集正确率
    test_accuracy = list()  # 测试集正确率
    for e in range(epoch):
        running_loss = 0.0
        train_acc = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            # loss function
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            c_labels = copy.copy(labels)
            c_outputs = copy.copy(outputs)
            if c_labels.detach().numpy()[0][0] == 1 and c_outputs.detach().numpy()[0][0] > c_outputs.detach().numpy()[0][1]:
                train_acc += 1
            elif c_labels.detach().numpy()[0][1] == 1 and c_outputs.detach().numpy()[0][1] > c_outputs.detach().numpy()[0][0]:
                train_acc += 1


            running_loss += loss.item()
            if i % eval_freq == eval_freq - 1:  # print every eval freq
                # 评估测试集效果
                # test_acc = 0  # 预测正确的个数
                # for j, data_test in enumerate(test_loader, 0):
                #     test_inputs, test_labels = data_test
                #     test_outputs = net(test_inputs)
                #     # 计算和转换
                #     c_test_labels = copy.copy(test_labels)
                #     c_test_outputs = copy.copy(test_outputs)
                #     if c_test_labels.detach().numpy()[0][0] == 1 and c_test_outputs.detach().numpy()[0][0] > c_test_outputs.detach().numpy()[0][1]:
                #         test_acc += 1
                #     elif c_test_labels.detach().numpy()[0][1] == 1 and c_test_outputs.detach().numpy()[0][1] > c_test_outputs.detach().numpy()[0][0]:
                #         test_acc += 1
                # test_accuracy.append(test_acc/600)
                # print('accuracy: %.3f' % (test_acc/600))
                # print('[%d, %5d] loss: %.3f' %
                #       (e + 1, i + 1, running_loss))
                running_loss = 0.0
        train_accuracy.append(train_acc/1400)
        # 评估测试集效果
        if e % eval_freq == eval_freq - 1:
            test_acc = 0  # 预测正确的个数
            for j, data_test in enumerate(test_loader, 0):
                test_inputs, test_labels = data_test
                test_outputs = net(test_inputs)
                # 计算和转换
                c_test_labels = copy.copy(test_labels)
                c_test_outputs = copy.copy(test_outputs)
                if c_test_labels.detach().numpy()[0][0] == 1 and c_test_outputs.detach().numpy()[0][0] > \
                        c_test_outputs.detach().numpy()[0][1]:
                    test_acc += 1
                elif c_test_labels.detach().numpy()[0][1] == 1 and c_test_outputs.detach().numpy()[0][1] > \
                        c_test_outputs.detach().numpy()[0][0]:
                    test_acc += 1
            test_accuracy.append(test_acc / 600)
            #print('accuracy: %.3f' % (test_acc / 600))
    return train_accuracy, test_accuracy

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(FLAGS=None, n_hidden=list(map(int, DNN_HIDDEN_UNITS_DEFAULT.split(','))), lr=LEARNING_RATE_DEFAULT,
         epoch=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT):
    """
    Main function
    """
    setup_seed(42)
    if FLAGS != None:
        n_hidden = list(map(int, FLAGS.dnn_hidden_units.split(',')))
        lr = FLAGS.learning_rate
        epoch = FLAGS.max_steps
        eval_freq = FLAGS.eval_freq

    net = MLP(2, n_hidden, 2)  # MLP网络
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.09)  # 优化器

    # 训练
    train_accuracy, test_accuracy = train(net, epoch, eval_freq, criterion, optimizer)
    # 获得 accuracy
    accuracy(train_accuracy, test_accuracy)


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
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
