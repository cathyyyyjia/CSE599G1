from .layer import Layer
import numpy as np


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.data = None

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        self.data = data
        batch, channels, height, width = data.shape
        return np.reshape(data, (batch, channels * height * width))

    def backward(self, previous_partial_gradient):
        # TODO
        batch, channels, height, width = self.data.shape
        return np.reshape(previous_partial_gradient, (batch, channels, height, width))
