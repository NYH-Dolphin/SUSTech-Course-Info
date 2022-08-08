from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        dims = [n_inputs]
        dims.extend(n_hidden)
        self.layers = list()
        # 隐藏层
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(layer)
        # 输出层
        layer = nn.Linear(dims[-1], n_classes)
        self.fcn = layer

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for fc in self.layers:
            x = nn.functional.relu(fc(x))
        out = nn.functional.softmax(self.fcn(x), dim=1)
        return out
