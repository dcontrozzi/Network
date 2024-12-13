
import numpy as np
from matplotlib import pyplot as plt
from numpy import random as rn



class MeanRevertingOU:

    def __init__(self, S0, mu, sigma, eta):
        """
        Process
        S(t) = S(0) + eta * ( mu - S(0)) dt + sigma * dW
        """

        G = 1629562571

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        # self.Sm = Sm
        self.eta = eta

    # def __int__(self, ):


    def simulate(self, N, T, seed=None):
        """
        N: number of steps
        T: time horizon
        """

        if seed != None:
            np.random.seed(seed)

        epsilon = rn.randn(N)
        S = self.S0 * np.ones((N + 1))
        dt = T / N

        for i in range(0, N -1):
            S[i + 1] = S[i] + self.eta * (self.mu - S[i]) * dt + self.sigma * epsilon[i] * np.sqrt(dt)
            # S[i + 1] = self.sigma * epsilon[i] * np.sqrt(dt)

        return S


    # def simulate(self, T, N):
    #
    #     dt = T / N
    #     t = np.linspace(0, T, N)
    #     X = np.zeros(N)
    #     X[0] = self.X0
    #
    #     for i in range(1, N):
    #         dW = np.random.normal(0, np.sqrt(dt))
    #         X[i] = X[i-1] + self.theta * (self.mu - X[i-1]) * dt + self.sigma * dW
    #
    #     return t, X


    # def plot(self, M):
    #     """
    #     M: Number of simulations
    #     """
    #
    #     plt.figure(figsize=(13,7))
    #     fontsize=15
    #     plt.title('Path-Dependent Monte Carlo Simulation - Mean-Reversion Process with Drift',fontsize=fontsize)
    #     plt.xlabel('Years',fontsize=fontsize)
    #     plt.ylabel('CPI prices (USD)',fontsize=fontsize)
    #     plt.grid(axis='y')
    #     a = [ rn.randint(0,M) for j in range(1,40)]
    #     for runer in a:
    #         plt.plot(np.arange(0,T+dt,dt), S[runer], 'red')

    def mean(self, t):

        return self.mu + (self.S0 - self.mu) * np.exp( - self.eta * t)

    def var(self, t):

        return ( 1. - np.exp( - 2. * self.eta * t) ) * self.sigma * self.sigma / (2. * self.eta)

    def diff_mean(self, t1, t2):

        return (self.S0 - self.mu) * ( np.exp( - self.eta * t2) - np.exp( - self.eta * t1) )

    def diff_var(self, t1, t2):

        return self.sigma * self.sigma / (2. * self.eta) * ( 2. - np.exp( - 2. * self.eta * t1) - np.exp( - 2. * self.eta * t2)
                                                             - 2. * np.exp( - self.eta * ( t2 - t1) ) * (1. - np.exp(-2. * self.eta * t1)) )

if __name__ == "__main__":

    S0 = 1
    mu = 1.
    sigma = 0.25
    Sm = 5
    eta = 0.025

    model = MeanRevertingOU(S0, mu, sigma, eta)

    T = 100
    n_list = [100]
    nb_simulations = 100000

    mean_dict_mean = {}
    mean_dict_std = {}

    var_dict_mean = {}
    var_dict_std = {}

    diff_mean_dict_mean = {}
    diff_mean_dict_std = {}

    diff_var_dict_mean = {}
    diff_var_dict_std = {}

    # Test different values of N
    for j in n_list:
        var = []
        mean = []
        diff_var = []
        diff_mean = []
        # Run nb_simulations and average
        for i in range(nb_simulations):
            S = model.simulate(j, T)
            var.append(np.var(S))
            mean.append(np.mean(S))
            S_diff = np.diff(S)
            diff_var.append(np.var(S_diff))
            diff_mean.append(np.mean(S_diff))

        var_dict_mean[j] = np.mean(var)
        var_dict_std[j] = np.std(var)

        mean_dict_mean[j] = np.mean(mean)
        mean_dict_std[j] = np.std(mean)

        diff_var_dict_mean[j] = np.mean(diff_var)
        diff_var_dict_std[j] = np.std(diff_var)

        diff_mean_dict_mean[j] = np.mean(diff_mean)
        diff_mean_dict_std[j] = np.std(diff_mean)

    print('exact mean ', model.mean(T), ' calculated mean ', mean_dict_mean[n_list[-1]], ' mean std ',  mean_dict_std[n_list[-1]])
    print('exact str ', model.var(T), ' calculated std ', var_dict_mean[n_list[-1]], ' var std ', var_dict_std[n_list[-1]])

    print('diff exact mean ', model.diff_mean(T, T + 1), ' calculated mean ', diff_mean_dict_mean[n_list[-1]], ' mean std ',  diff_mean_dict_std[n_list[-1]])
    print('diff exact str ', model.diff_var(T, T + 1), ' calculated std ', diff_var_dict_mean[n_list[-1]], ' var std ', diff_var_dict_std[n_list[-1]])

    N = 100
    T = 10

    S = model.simulate(N, T)
    S2 = model.simulate(N, T, 100)
    S3 = model.simulate(N, T, 345)

    plt.plot(np.arange(0,T+T/N,T/N), S, 'red')
    plt.plot(np.arange(0,T+T/N,T/N), S2, 'red')
    plt.plot(np.arange(0,T+T/N,T/N), S3, 'red')
    plt.show()


    pass
