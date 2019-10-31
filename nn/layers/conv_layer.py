from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias, kernel_size, padding, stride, pad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K

        batch = data.shape[0]
        output_channels = weights.shape[1]
        output_height = (data.shape[2] + padding * 2 - kernel_size) // stride + 1
        output_width = (data.shape[3] + padding * 2 - kernel_size) // stride + 1

        # initialize output with zeros
        output = np.zeros((batch, output_channels, output_height, output_width))

        for b in prange(batch):
            for c in prange(output_channels):
                for h in prange(output_height):
                    for w in prange(output_width):
                        # padding area
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        curr_pad = pad[b, :, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.sum(curr_pad * weights[:, c, :, :]) + bias[c]

        return output

    def forward(self, data):
        # TODO
        self.data = data
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        pad = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        return self.forward_numba(data, self.weight.data, self.bias.data, kernel_size, padding, stride, pad)

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, weights, bias, kernel_size, padding, stride, pad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K

        batch = data.shape[0]
        output_channels = weights.shape[1]
        input_height = data.shape[2]
        input_width = data.shape[3]
        output_height = (input_height + padding * 2 - kernel_size) // stride + 1
        output_width = (input_width + padding * 2 - kernel_size) // stride + 1

        # initiate output with zeros
        dw = np.zeros_like(weights)
        db = np.zeros_like(bias)
        dpad = np.zeros_like(pad)

        for b in range(batch):
            for c in range(output_channels):
                db[c] += np.sum(previous_grad[b, c])
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        dw[:, c, :, :] += pad[b, :, h_start:h_end, w_start:w_end] * previous_grad[b, c, h, w]
                        dpad[b, :, h_start:h_end, w_start:w_end] += weights[:, c, :, :] * previous_grad[b, c, h, w]

        dx = dpad[:, :, padding:padding+input_height, padding:padding+input_width]

        return dx, dw, db

    def backward(self, previous_partial_gradient):
        # TODO
        data = self.data
        weights = self.weight.data
        bias = self.bias.data
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        pad = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        dx, dw, db = self.backward_numba(previous_partial_gradient, data, weights, bias, kernel_size, padding, stride, pad)
        self.weight.grad = dw
        self.bias.grad = db
        return dx

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
