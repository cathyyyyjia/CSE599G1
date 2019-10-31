import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, kernel_size, padding, stride, pad):
        # data is N x C x H x W
        # TODO

        # initialize output with zeros
        batch = data.shape[0]
        output_channels = data.shape[1]
        output_height = (data.shape[2] + padding * 2 - kernel_size) // stride + 1
        output_width = (data.shape[3] + padding * 2 - kernel_size) // stride + 1
        output = np.zeros((batch, output_channels, output_height, output_width))

        for b in prange(batch):
            for c in prange(output_channels):
                for h in prange(output_height):
                    for w in prange(output_width):
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        output[b, c, h, w] = np.max(pad[b, c, h_start:h_end, w_start:w_end])

        return output

    def forward(self, data):
        # TODO
        self.data = data
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        pad = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        return self.forward_numba(data, kernel_size, padding, stride, pad)

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel_size, padding, stride, pad):
        # data is N x C x H x W
        # TODO

        batch = data.shape[0]
        output_channels = data.shape[1]
        input_height = data.shape[2]
        input_width = data.shape[3]
        output_height = (input_height + padding * 2 - kernel_size) // stride + 1
        output_width = (input_width + padding * 2 - kernel_size) // stride + 1

        # initiate output with zeros
        dpad = np.zeros_like(pad)

        for b in prange(batch):
            for c in prange(output_channels):
                for h in prange(output_height):
                    for w in range(output_width):
                        # padding area
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        curr_pad = pad[b, c, h_start:h_end, w_start:w_end]
                        check_pad = (curr_pad == np.max(curr_pad))
                        dpad[b, c, h_start:h_end, w_start:w_end] += check_pad * previous_grad[b, c, h, w]

        dx = dpad[:, :, padding:padding+input_height, padding:padding+input_width]

        return dx

    def backward(self, previous_partial_gradient):
        # TODO
        data = self.data
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        pad = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        dx = self.backward_numba(previous_partial_gradient, data, kernel_size, padding, stride, pad)
        return dx

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
