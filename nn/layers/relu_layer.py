import numpy as np
from copy import deepcopy
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.data = None

    def forward(self, data):
        # TODO
        self.data = data
        output = deepcopy(data)
        output[data <= 0] = 0
        return output

    def backward(self, previous_partial_gradient):
        # TODO
        grad = deepcopy(previous_partial_gradient)
        grad[self.data <= 0] = 0
        return grad

class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        data_flatten = data.flatten()
        for i in range(len(data_flatten)):
            if data_flatten[i] <= 0:
                data_flatten[i] = 0
        return np.reshape(data_flatten, data.shape)

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        data_flatten = data.flatten()
        grad_flatten = grad.flatten()
        for i in range(len(data_flatten)):
            if data_flatten[i] <= 0:
                grad_flatten[i] = 0
        return np.reshape(grad_flatten, data.shape)

    def backward(self, previous_partial_gradient):
        # TODO
        output = self.backward_numba(self.data, previous_partial_gradient)
        return output
