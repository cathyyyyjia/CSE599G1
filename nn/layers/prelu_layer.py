import numpy as np
from copy import deepcopy
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None

    def forward(self, data):
        # TODO
        self.data = data
        slope_dim = len(self.slope.data)
        if slope_dim != 1:
            n = len(data.flatten()) // slope_dim
            output = np.reshape(data, (n, slope_dim))
            for i in range(len(output)):
                if output[i].all() <= 0:
                    output[i] *= self.slope.data
            output = np.reshape(output, data.shape)
        else:
            output = deepcopy(data)
            output[data <= 0] *= self.slope.data
        return output

    def backward(self, previous_partial_gradient):
        # TODO
        # Compute input gradient
        slope_dim = len(self.slope.data)
        if slope_dim != 1:
            n = len(self.data.flatten()) // slope_dim
            data_reshape = np.reshape(self.data, (n, slope_dim))
            grad = np.reshape(previous_partial_gradient, (n, slope_dim))
            for i in range(len(data_reshape)):
                if data_reshape[i].all() <= 0:
                    grad[i] *= self.slope.data
            grad = np.reshape(grad, previous_partial_gradient.shape)
        else:
            grad = deepcopy(previous_partial_gradient)
            grad[self.data <= 0] = self.slope.data
        # Compute and update slope gradient
        grad_f = np.zeros_like(previous_partial_gradient.flatten())
        data_flatten = self.data.flatten()
        grad_f[data_flatten <= 0] = data_flatten[data_flatten <= 0]
        grad_f = np.reshape(grad_f, previous_partial_gradient.shape)
        self.slope.grad = np.sum(np.multiply(previous_partial_gradient, grad_f))
        return grad
