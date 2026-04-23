import numpy as np


class Adam(object):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1      # decay rate for first moment (momentum)
        self.beta2 = beta2      # decay rate for second moment (variance)
        self.epsilon = epsilon
        self.m = None           # first moment (mean of gradients)
        self.v = None           # second moment (variance of gradients)
        self.t = 0              # timestep

    def penalize(self, W, grad):
        """
        Performs one Adam update step.

        Arguments:
            W    : current weights (np.array)
            grad : gradient of the loss w.r.t W (np.array)
        Returns:
            W    : updated weights
        """
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(W)
            self.v = np.zeros_like(W)

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Bias correction (important especially in early iterations)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        W -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return W