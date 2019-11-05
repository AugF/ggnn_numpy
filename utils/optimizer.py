import numpy as np


class Adam:
    """adam optimizer"""
    def __init__(self, weights, learning_rate=0.001):
        """params, init"""
        self.learning_rate = learning_rate
        self.theta_t = weights
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.m_t = np.zeros(weights.shape)
        self.v_t = np.zeros(weights.shape)
        self.t = 0
        # self.grad_function = grad_function

    def minimize(self, g_t):
        """more efficient"""
        self.t += 1
        # g_t = self.grad_function(self.theta_t)
        alpha_t = self.learning_rate * ((1 - self.beta_2 ** self.t) ** 0.5) / (1 - self.beta_1 ** self.t)
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * np.multiply(g_t, g_t)
        self.theta_t -= alpha_t * self.m_t / (np.power(self.v_t, 0.5) + self.epsilon)
        return self.theta_t


if __name__ == '__main__':
    weights = np.random.random((3, 4, 3))
    print(weights)
    adam = Adam(weights=weights)
    grad_weights = np.random.random(weights.shape)
    print(adam.minimize(grad_weights))