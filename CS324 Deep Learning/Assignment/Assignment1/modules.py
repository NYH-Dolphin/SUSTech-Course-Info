import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features, lr):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        np.random.seed(42)
        self.lr = lr
        self.x = np.zeros((1, in_features))
        self.w = np.random.normal(loc=0, scale=0.1, size=in_features * out_features).reshape(
            (in_features, out_features))
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        # 更新梯度
        #self.b -= self.lr * dout
        self.b -= dout
        w_d = np.dot(self.x.T, dout)
        #self.w -= self.lr * w_d
        self.w -= w_d

        # 继续向后传
        dx = np.dot(dout, self.w.T)
        return dx


class ReLU(object):
    def __init__(self, in_features, out_features):
        self.z_d = np.zeros((1, out_features))  # 输入的倒数

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.z_d = x.copy()
        self.z_d = np.where(self.z_d > 0, 1, 0)
        out = np.maximum(x, 0)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = dout * self.z_d  # 向量与向量逐个元素相乘
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        # axis = 0 -> max of each column
        # axis = 1 -> max of each row
        assert (len(x.shape) == 2)
        row_max = np.max(x, axis=1).reshape(-1, 1)
        x -= row_max
        x_exp = np.exp(x)
        out = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = dout
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """

        # x_log = np.log(x) 存在x为0的情况
        x_log = x.copy()
        for i in range(len(x_log[0])):
            if x_log[0][i] != 0:
                x_log[0][i] = np.log(x_log[0][i])
        out = -(x_log * y).sum()
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = x - y  # cross entropy + softmax' -> predict label - actual label
        return dx
