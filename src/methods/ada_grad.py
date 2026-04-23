import numpy as np


class AdaGrad(object):

    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G = None  # accumulated squared gradients

    def penalize(self, W, grad):
        """
        Performs one AdaGrad update step.
        
        Arguments:
            W    : current weights (np.array)
            grad : gradient of the loss w.r.t W (np.array)
        Returns:
            W    : updated weights
        """
        # Initialize accumulator on first call
        if self.G is None:
            self.G = np.zeros_like(W)

        # Accumulate squared gradients
        self.G += grad ** 2

        # Adaptive update
        W -= (self.lr / (np.sqrt(self.G) + self.epsilon)) * grad

        return W