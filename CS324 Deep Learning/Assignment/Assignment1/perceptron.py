import numpy as np


class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.w = np.zeros(n_inputs + 1)  # weight
        self.epoch = int(max_epochs)
        self.lr = learning_rate

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        x = np.pad(input, (0, 1), 'constant', constant_values=(0, 1))
        predict = np.dot(self.w, x)
        if predict < 0:
            label = -1
        else:
            label = 1
        return label

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        num = len(training_inputs)
        for i in range(self.epoch):
            for j in range(num):
                x_temp = np.array(training_inputs[j])
                x = np.pad(x_temp, (0, 1), 'constant', constant_values=(0, 1))
                y = labels[j]  # true label
                if y * np.dot(self.w, x) <= 0:
                    self.w = self.w + self.lr * (y * x)
