from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import random
import argparse
import numpy as np
import os
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './CIFAR10data'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    return accuracy


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 一个 epoch 的训练
def train(net, train_loader, optimizer, loss_function, epoch, batch, device):
    net.main()  # 进入训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data[0], data[1]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % batch == batch - 1:
            # print('[%d, %5d] loss = %.3f' % (epoch + 1, i + 1, running_loss / batch))
            running_loss = 0.0
    acc = correct / total
    print('[epoch %d] train accuracy = %.3f' % (epoch + 1, acc))
    return acc, running_loss


def evaluate(net, test_loader, epoch, loss_function, device):
    net.eval()  # 进入模型评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print('[epoch %d] test accuracy = %.3f' % (epoch + 1, acc))
    return acc, running_loss


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


def main(FLAGS=None, lr=LEARNING_RATE_DEFAULT, batch=BATCH_SIZE_DEFAULT, epoch=MAX_EPOCHS_DEFAULT,
         eval_freq=EVAL_FREQ_DEFAULT, optim=OPTIMIZER_DEFAULT, data_dir=DATA_DIR_DEFAULT):
    """
    Main function
    """
    setup_seed(42)
    if FLAGS != None:
        lr = FLAGS.learning_rate
        epoch = FLAGS.max_steps
        batch = FLAGS.batch_size
        eval_freq = FLAGS.eval_freq
        optim = FLAGS.optimizer
        data_dir = FLAGS.data_dir

    # transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 把图像变换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 注意规范化要在ToTensor之后
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=data_transform)
    test_dataset = datasets.CIFAR10(root='./CIFAR10data', train=False, download=False, transform=data_transform)

    # 数据准备
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True)

    net = CNN()
    loss_function = nn.CrossEntropyLoss()
    # 指定训练设备
    device = torch.device('cpu')

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    if optim == 'ADAM':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)

    train_accuracy = list()
    test_accuracy = list()
    train_losslist = list()
    test_losslist = list()
    for e in range(epoch):
        train_acc, train_loss = train(net=net, train_loader=train_loader,
                                      optimizer=optimizer, loss_function=loss_function,
                                      epoch=e, batch=100, device=device)
        train_accuracy.append(train_acc)
        train_losslist.append(train_loss)
        test_acc, test_loss = evaluate(net=net, test_loader=test_loader,
                                       epoch=e, loss_function=loss_function, device=device)
        test_accuracy.append(test_acc)
        test_losslist.append(test_loss)
        print('[epoch %d] train accuracy = %.3f, test accuracy = %.3f' % (e + 1, train_acc, test_acc))
        print('[epoch %d] train loss = %.3f, test loss = %.3f' % (e + 1, train_loss, test_loss))

    print_diagram(train_accuracy, test_accuracy, train_losslist, test_losslist)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        help='Optimizer used in training network')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
