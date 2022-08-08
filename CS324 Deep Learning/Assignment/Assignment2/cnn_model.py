from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


# 28x28x3 -> 10
# conv:(28-3+2)/1+1=28  -> 64 channel
# maxpool:(28-2+2)/2+1=15
# conv:(15-3+2)/1+1=15 -> 128 channel
# maxpool:(15-3+0)/2+1=6
# conv:(6-3+2)/1+1=6 -> 256 channel
# conv:(6-3+2)/1+1=6 -> 512 channel
# maxpool:(6-2+0)/2+1=3 -> 3x3x 512 channel


class CNN(nn.Module):

    def __init__(self):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1))
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.fc1 = nn.Linear(14 * 14 * 6, 20)
        # self.fc2 = nn.Linear(20, 10)
        # -----
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(4 * 4 * 512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Performs forward pass of the input.
    
        Args:
         x: input to the network
        Returns:
        out: outputs of the network
        """
        # return out
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = x.view(-1, 1176)
        # x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x))
        # -----

        # 广义卷积层部分
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool3(F.relu(self.conv4(x)))

        # 使用 torch.Tensor.view 函数，把张量 reshape 成适合全连接层的维度
        x = x.view(-1, 4 * 4 * 512)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x
