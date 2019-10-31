import numpy as np
from copy import deepcopy
from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        self.prob = None
        self.label_prob = None
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        # print(logits.shape, targets.shape)
        n = logits.shape[0]
        d = logits.shape[1]
        x = logits - np.max(logits, axis=1, keepdims=True)  # subtract the max of each row
        exp = np.exp(x)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        log_softmax = x - np.log(sum_exp)
        label_prob = np.zeros((n, d))
        label_prob[np.arange(n), targets] = 1
        loss = - np.sum(np.multiply(label_prob, log_softmax))
        if self.reduction == "mean":
            loss /= n
        self.prob = exp / sum_exp
        self.label_prob = label_prob
        return loss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        grad = self.prob - self.label_prob
        if self.reduction == "mean":
            grad /= self.prob.shape[0]
        return grad
