from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from modules import *

# 数据集，存放训练集和测试集的相关信息
class DataSet:
    def __init__(self, train_samples, train_labels, test_samples, test_labels):
        self.train_samples = train_samples
        self.train_labels = train_labels
        self.test_samples = test_samples
        self.test_labels = test_labels
        self.m_train_labels = list()
        self.m_test_labels = list()


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes, data_set: DataSet, n_lr=1e-2, n_epoch=1500, eval_freq=10, batch=1):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
            n_lr: learning rate
            n_epoch: learning epoch
        """
        # dims - 包括输入维度 + 隐藏层维度
        dims = [n_inputs]
        dims.extend(n_hidden)

        # layers - 各层
        layers = list()
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1], n_lr))
            layers.append(ReLU(dims[i + 1], dims[i + 1]))
        layers.append(Linear(dims[-1], n_classes, n_lr))
        layers.append(SoftMax())

        # 神经网络模型
        self.nn = layers  # 网络
        self.classes = n_classes  # 分类个数
        self.lr = n_lr  # 学习率
        self.epoch = n_epoch  # 训练迭代次数
        self.eval_freq = eval_freq  # 训练集验证的平均epoch
        self.CE = CrossEntropy()  # 交叉熵损失函数计算
        self.batch = batch  # True:使用批处理梯度下降 False:使用随机梯度下降
        self.loss = list()  # 损失(每个epoch)
        self.accuracy = list()  # 精度(每个epoch)
        self.test_loss = list()
        self.test_accuracy = list()

        # 数据集处理
        self.data_set = data_set
        self.data_set.m_train_labels = self.label2m_label(self.data_set.train_labels)
        self.data_set.m_test_labels = self.label2m_label(self.data_set.test_labels)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for layer in self.nn:
            x = layer.forward(x)
        out = x
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for layer in reversed(self.nn):
            dout = layer.backward(dout)
        return

    def get_class(self, m_predict):
        val = m_predict.max()
        for i in range(len(m_predict[0])):
            if m_predict[0][i] == val:
                return i

    def label2m_label(self, labels):
        # 修改label的维度
        # eg classes = 2, label = 0, m_label = [[1,0]]
        # eg classes = 2, label = 1, m_label = [[0,1]]
        m_labels = list()
        for i in range(len(labels)):
            m_label = np.zeros((1, self.classes))
            c = labels[i]
            m_label[0][c] = 1
            m_labels.append(m_label)
        return m_labels

    def train(self):
        """
        MLP模型训练
        Args:
            samples: 训练样本
            labels: 训练标签
        """
        samples = self.data_set.train_samples
        labels = self.data_set.train_labels
        m_labels = self.data_set.m_train_labels

        cnt = 0
        for _ in range(self.epoch):
            sum_loss = 0
            sum_accuracy = 0
            sum_dout = list()
            bat = 0
            for i in range(len(samples)):

                # 获取样本和标签
                sample = samples[i]
                m_label = m_labels[i]

                # 前向传播与反向传播
                m_predict = self.forward(np.array([sample]))
                dout = self.lr * self.CE.backward(m_predict, m_label)


                predict = self.get_class(m_predict)
                if predict == labels[i]:
                    sum_accuracy += 1
                else:
                    bat += 1
                    sum_dout.append(dout)

                # 随机梯度下降的情况 -> 每一次都更新梯度
                if bat == self.batch:
                    self.backward(sum(sum_dout) / self.batch)
                    bat = 0
                    sum_dout.clear()

                # 计算训练效果
                loss = self.CE.forward(m_predict, m_label)  # 损失函数
                sum_loss += loss



            # 加入每个 epoch 的结果
            self.loss.append(sum_loss)
            self.accuracy.append(sum_accuracy / len(samples))

            # 执行批次导入测试集进行测试
            cnt += 1
            if cnt == self.eval_freq:
                cnt = 0
                test_loss, test_accuracy = self.predict()
                self.test_loss.append(test_loss)
                self.test_accuracy.append(test_accuracy)

    def predict(self):
        """
        MPL模型预测
        Args:
            samples:
            labels:

        Returns:
            loss: 交叉熵损失函数结果
            accuracy: 预测精度
        """
        samples = self.data_set.test_samples
        labels = self.data_set.test_labels
        m_labels = self.data_set.m_test_labels

        sum_accuracy = 0
        sum_loss = 0
        for i in range(len(samples)):
            sample = samples[i]
            m_label = m_labels[i]
            m_predict = self.forward(np.array([sample]))
            # 计算测试效果
            loss = self.CE.forward(m_predict, m_label)  # 损失函数
            sum_loss += loss
            predict = self.get_class(m_predict)
            if predict == labels[i]:
                sum_accuracy += 1
        return sum_loss, sum_accuracy / len(samples)
