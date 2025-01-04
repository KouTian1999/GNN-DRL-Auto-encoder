import numpy as np


# 暂时搁置
class GreedyGaussianNoise():
    def __init__(self, exploration, decay_rate, min_eps, mu, sigma):
        self.epsilon = exploration
        self.decay = decay_rate
        self.min_eps = min_eps
        self.mu = mu
        self.sigma = sigma

    def __call__(self, input_shape):
        if np.random.random() < self.epsilon:
            noise = np.random.normal(self.mu, self.sigma, size=input_shape)
        else:
            noise = np.zeros(shape=input_shape)
        self.update()
        return noise

    def update(self):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon, self.min_eps)


class OUActionNoise():
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)