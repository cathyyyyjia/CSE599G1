import numpy as np
from copy import deepcopy
from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope

    def forward(self, data):
        # TODO
        self.data = data
        output = deepcopy(data)
        output[data <= 0] *= self.slope
        return output

    def backward(self, previous_partial_gradient):
        # TODO
        grad = deepcopy(previous_partial_gradient)
        grad[self.data <= 0] = self.slope
        return grad
