import numpy as np
from layer import Layer


class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)

    def forward(self, input):
        self.input = input
        # Y(op) = W.X(ip) + B
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # dE/dW = dE/dY.X^T
        weight_gradient = np.dot(output_gradient, self.input.T)
        # dE/dX = W^T.dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)

        # updating weights and bias
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
