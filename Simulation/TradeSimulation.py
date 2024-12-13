
import numpy as np

class TradeSimulation:

    def __init__(self, levels, trade_probabilities):
        """
        :param levels: array N x M where N = number of observations, M = number of assets
        :param trade_probabilities: array of lenth M = number of assets
        """

        self.levels = levels
        self.trade_probabilities = trade_probabilities

    def generate_trades(self):

        jumps = []
        for i, p in self.trade_probabilities.enumerate():
            observations_length = len(self.levels[i])
            #  generate observation level using class Observations
            jumps.append(np.random.binomial(1, p, observations_length))

        trade_levels = np.array(jumps) * np.array(self.levels)

        return trade_levels
