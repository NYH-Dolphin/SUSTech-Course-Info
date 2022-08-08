from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import mean
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from lstm import  LSTM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def print_diagram(train_accuracy, test_accuracy, train_loss, test_loss):
    x = np.arange(0, len(train_accuracy))
    y = train_accuracy
    plt.plot(x, y)
    plt.title("Accuracy of train set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    x = np.arange(0, len(train_loss))
    y = train_loss
    plt.plot(x, y)
    plt.title("Loss of train set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    x = np.arange(0, len(test_accuracy))
    y = test_accuracy
    plt.plot(x, y)
    plt.title("Accuracy of test set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    x = np.arange(0, len(test_loss))
    y = train_loss
    plt.plot(x, y)
    plt.title("Loss of test set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def main(config=None, input_length=10, input_dim=1, num_classes=10,
         num_hidden=16, batch_size=128, lr=0.02, train_steps=100,
         device=torch.device('cpu'), max_norm=10.0, epoch=100, to_print=True):
    setup_seed(42)
    # 导入配置
    if config != None:
        input_length = config.input_length
        input_dim = config.input_dim
        num_classes = config.num_classes
        num_hidden = config.num_hidden
        batch_size = config.batch_size
        lr = config.learning_rate
        train_steps = config.train_steps
        max_norm = config.max_norm
        device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = LSTM(seq_length=input_length, input_dim=input_dim, hidden_dim=num_hidden, output_dim=num_classes,
                       batch_size=batch_size).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(input_length + 1)
    train_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    test_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=lr)

    train_acc = list()
    train_loss = list()
    test_acc = list()
    test_loss = list()
    for e in range(epoch):
        # print('epoch{}'.format(e))
        model.train()  # 进入训练模式
        running_loss = 0.0
        correct_num = 0
        sum_num = 0
        for step, data in enumerate(train_data_loader):
            optimizer.zero_grad()
            batch_inputs, batch_targets = data[0].to(device), data[1].to(device)
            batch_outputs = model(batch_inputs)

            y_hat = F.softmax(batch_outputs, dim=1)
            for b in range(len(y_hat)):
                output = y_hat[b]
                predict = torch.argmax(output)
                if predict.item() == batch_targets[b].item():
                    correct_num += 1
                sum_num += 1

            loss = criterion(batch_outputs, batch_targets)
            running_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            loss.backward()
            optimizer.step()

            if step == train_steps:
                # print acuracy/loss here
                # print('train loss = ' + str(running_loss))
                train_loss.append(running_loss)
                # print('train accuracy = ' + str(correct_num / sum_num))
                train_acc.append(correct_num / sum_num)
                break

        model.eval()
        running_loss = 0.0
        correct_num = 0
        sum_num = 0
        for step, data in enumerate(test_data_loader):
            batch_inputs, batch_targets = data[0].to(device), data[1].to(device)
            batch_outputs = model(batch_inputs)

            y_hat = F.softmax(batch_outputs, dim=1)
            for b in range(len(y_hat)):
                output = y_hat[b]
                predict = torch.argmax(output)
                if predict.item() == batch_targets[b].item():
                    correct_num += 1
                sum_num += 1

            loss = criterion(batch_outputs, batch_targets)
            running_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            if step == train_steps:
                # print acuracy/loss here
                # print('test loss = ' + str(running_loss))
                test_loss.append(running_loss)
                # print('test accuracy = ' + str(correct_num / sum_num))
                test_acc.append(correct_num / sum_num)
                break

    if to_print:
        print_diagram(train_acc, test_acc, train_loss, test_loss)
        print('Done training.')
    return train_acc


def printT():
    T_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    acc_list = list()
    for T in T_list:
        acc = main(input_dim=T, to_print=False, epoch=20)
        acc_list.append(mean(acc))
    plt.plot(T_list, acc_list)
    plt.title("Accuracy versus Palindrome length")
    plt.xlabel("Palindrome length")
    plt.ylabel("Accuracy versus")
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=16, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--device', type=str, default='cpu', help='Training Device')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    main(config)
