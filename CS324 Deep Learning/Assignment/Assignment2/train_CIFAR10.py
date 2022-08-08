import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import random
from CIFAR10_model import CIFAR10Net


def print_accuracy(train_accuracy, test_accuracy):
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
    plt.title("Accuracy of test set in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    return print_accuracy

# 一个 epoch 的训练
def train(net, train_loader, optimizer, loss_function, epoch, batch):
    net.main()  # 进入训练模式
    running_loss = 0.0
    correct_num = 0
    sum_num = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data[0], data[1]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)
        sum_num += labels.size(0)
        correct_num += (predict == labels).sum().item()

        # if i % batch == batch - 1:
        #     print('[%d, %5d] loss = %.3f' % (epoch + 1, i + 1, running_loss / batch))
        #     running_loss = 0.0
    acc = correct_num / sum_num
    #print('[epoch %d] train accuracy = %.3f' % (epoch + 1, acc))
    return acc


def evaluate(net, test_loader, epoch):
    net.eval()  # 进入模型评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0], data[1]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    #print('[epoch %d] test accuracy = %.3f' % (epoch + 1, acc))
    return acc


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(epoch=100):
    setup_seed(42)
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 把图像变换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 注意规范化要在ToTensor之后
    ])
    dataset = datasets.CIFAR10(root='./CIFAR10data', train=True, download=True, transform=data_transform)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[round(0.8 * len(dataset)), round(0.2 * len(dataset))],
        generator=torch.Generator().manual_seed(42)
    )

    # 数据准备
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    # 指定模型配置
    net = CIFAR10Net()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.003)

    train_accuracy = list()
    test_accuracy = list()
    for e in range(epoch):
        train_acc = train(net=net, train_loader=train_loader,
                          optimizer=optimizer, loss_function=loss_function,
                          epoch=e, batch=100)
        train_accuracy.append(train_acc)
        test_acc = evaluate(net=net, test_loader=test_loader,
                            epoch=e)
        test_accuracy.append(test_acc)
        print('[epoch %d] train accuracy = %.3f, test accuracy = %.3f' % (e + 1, train_acc, test_acc))

    print_accuracy(train_accuracy, test_accuracy)


if __name__ == '__main__':
    main()
