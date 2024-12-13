import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import random as rn

# import Simulation.MeanReverting


# from scipy import stats
# import scipy.stats as si
# import seaborn as sns


class MeanReverting:

    def __init__(self, S0, mu, sigma, Sm, eta):
        """
        Process
        S(t) = S(0)*(1 + η*( Sm * exp(μ*dt)- S(0) )*dt + μ*dt + σ*ε(t) * sqrt(dt)
        """

        G = 1629562571

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.Sm = Sm
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

        for i in range(0, N):
            S[i + 1] = S[i] * (1 + self.eta * (self.Sm * np.exp(self.mu * dt) - S[i]) * dt + self.mu * dt + self.sigma * epsilon[i] * np.sqrt(dt))

        return S

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



if __name__ == "__main__":

    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    S0 = 1
    mu = 0.05
    sigma = 0.25
    Sm = 5
    eta = 0.25

    model = MeanReverting(S0, mu, sigma, Sm, eta)

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

    # M = 50000
    #
    # ε = rn.randn(M,N)
    # S = S0*np.ones((M,N+1))
    # dt = T/N
    #
    # start_time = datetime.now()
    # for i in range(0,N):
    #     S[:,i+1] = S[:,i]*(1 + η*(Sm*np.exp(μ*dt)-S[:,i])*dt + μ*dt + σ*ε[:,i]*np.sqrt(dt))