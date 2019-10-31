from .. import *


class ResNetBlock(LayerUsingLayer):
    def __init__(self, conv_params, parent=None):
        super(ResNetBlock, self).__init__(parent)
        self.conv_layers = SequentialLayer([ConvLayer(*conv_params), ReLULayer(), ConvLayer(*conv_params)], self.parent)
        self.add_layer = AddLayer(self.conv_layers)
        self.relu2 = ReLULayer(self.add_layer)
        assert not any([parent is None for parent in self.conv_layers.parents])
        assert not any([parent is None for parent in self.add_layer.parents])
        assert not any([parent is None for parent in self.relu2.parents])
        self.data = None

    @property
    def final_layer(self):
        # TODO
        # raise NotImplementedError
        return self.relu2

    def forward(self, data):
        # TODO
        self.data = data
        out = self.conv_layers(data)
        out = self.add_layer([out, data])
        out = self.final_layer(out)
        return out
