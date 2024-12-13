import numpy as np
import matplotlib.pyplot as plt

class LogNormal:

    def __init__(self, S0, mu, sigma, T, steps):
        """
        Initialize the LogNormalPathSimulator.

        Parameters:
        S0 : float - Initial value
        mu : float - Mean of the underlying normal distribution
        sigma : float - Volatility (standard deviation) of the underlying normal distribution
        T : float - Total time period
        steps : int - Number of steps
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = T / steps

    def simulate(self, seed=None):
        """
        Simulate the log-normal random path.

        Returns:
        S : ndarray - Log-normal path
        t : ndarray - Time steps
        """

        if seed is not None:
            np.random.seed(seed)

        # t = np.linspace(0, self.T, self.steps)
        # W = np.random.standard_normal(size=self.steps)
        # W = np.cumsum(W) * np.sqrt(self.dt)  # Brownian motion
        # X = (self.mu - 0.5 * self.sigma**2) * t + self.sigma * W
        # S = self.S0 * np.exp(X)

        S = np.zeros(self.steps)
        S[0] = self.S0
        for i in range(1, self.steps):
            Z = np.random.standard_normal()
            S[i] = S[i - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

        t = np.linspace(0, self.T, self.steps)

        return t, S


    def plot(self, t=None, S=None):
        """
        Generate and plot the log-normal random path.
        """
        if t == None and S == None:
            t, S = self.simulate()

        plt.figure(figsize=(10, 6))
        plt.plot(t, S, label="Log-normal path")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Log-normal Random Path")
        plt.legend()
        plt.show()

    def exact_variance(self, t):

        return (self.S0 ** 2) * np.exp(2. * self.mu * t) * (np.exp(self.sigma ** 2 * t) - 1.)

