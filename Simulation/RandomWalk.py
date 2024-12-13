import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import random as rn

# import Simulation.MeanReverting


# from scipy import stats
# import scipy.stats as si
# import seaborn as sns


class RandomWalk:

    def __init__(self, S0, mu, sigma):
        """
        Process
        S(t) = S(0) + sigma * dW
        dW = d\epsilon * sqrt(t)
        """

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma



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

        for i in range(0, N):
            S[i + 1] = S[i] + self.sigma * epsilon[i] * np.sqrt(dt)

        return S

    def mean(self, t):

        return self.mu + (self.S0 - self.mu) * np.exp( - self.eta * t)

    def var(self, t):

        return ( 1. - np.exp( - 2. * self.eta * t) ) * self.sigma * self.sigma / (2. * self.eta)

if __name__ == "__main__":

    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    S0 = 1
    mu = 0.05
    sigma = 0.25
    Sm = 5
    eta = 0.25

    model = RandomWalk(S0, mu, sigma)

    N = 10
    T = 100

    n_list = [100, 500, 1000, 10000, 1000000]
    nb_simulations = 10
    var_dict_mean = {}
    var_dict_std = {}
    for j in n_list:
        # seeds = np.random.randint(0, 1000, nb_simulations)
        var = []
        for i in range(nb_simulations):
            # S = model.simulate(j, T, seeds[i])
            S = model.simulate(j, T)
            var.append(np.var(S))

        var_dict_mean[j] = np.mean(var)
        var_dict_std[j] = np.std(var)


    S = model.simulate(N, T)
    S2 = model.simulate(N, T, 100)
    S3 = model.simulate(N, T, 345)

    print (np.var(S), np.var(S2), np.var(S3), (np.var(S) + np.var(S2) + np.var(S3))/ 3.)

    plt.plot(np.arange(0,T+T/N,T/N), S, 'red')
    plt.plot(np.arange(0,T+T/N,T/N), S2, 'red')
    plt.plot(np.arange(0,T+T/N,T/N), S3, 'red')
    plt.show()


    pass

    # M = 50000
    #
    # ε = rn.randn(M,N)
    # S = S0*np.ones((M,N+1))
    # dt = T/N
    #
    # start_time = datetime.now()
    # for i in range(0,N):
    #     S[:,i+1] = S[:,i]*(1 + η*(Sm*np.exp(μ*dt)-S[:,i])*dt + μ*dt + σ*ε[:,i]*np.sqrt(dt))