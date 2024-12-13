
import numpy as np
from matplotlib import pyplot as plt

from Simulation import MeanReverting

class ExcessSpreadSimulator:

    def __init__(self, base_spread_changes, beta, std):

        self.base_spread_changes = base_spread_changes
        self.beta = beta
        self.std = std


    def simulate(self):

        try:
            epsilon = np.random.normal(0., self.std, len(self.base_spread_changes))
        except Exception as e:
            print (str(e))
            print(self.std, len(self.base_spread_changes))

        return self.beta * self.base_spread_changes + epsilon


if __name__ == "__main__":

    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    S0 = 1
    mu = 0.05
    sigma = 0.25
    Sm = 5
    eta = 0.25

    model = MeanReverting.MeanReverting(S0, mu, sigma, Sm, eta)

    N = 100
    T = 10

    S = model.simulate(N, T)

    market_spread_changes = np.diff(S)

    sector_beta = 2.
    sector_std = 1.3
    issuer_beta = 0.5
    issue_std = 2.3
    bond_beta = 1.5
    bond_std = 10.

    sector_similator = ExcessSpreadSimulator(market_spread_changes, sector_beta, sector_std)
    sector_spread_change = sector_similator.simulate()

    issuer_similator = ExcessSpreadSimulator(sector_spread_change, issuer_beta, issue_std)
    issuer_spread_changes = issuer_similator.simulate()

    bond_similator = ExcessSpreadSimulator(issuer_spread_changes, bond_beta, bond_std)
    bond_spread_changes = bond_similator.simulate()

    plt.plot(sector_spread_change, 'blue')
    plt.plot(market_spread_changes, 'red')
    plt.plot(issuer_spread_changes, 'green')
    plt.plot(bond_spread_changes, 'black')

    plt.show()

    plt.plot(np.cumsum(sector_spread_change), 'blue')
    plt.plot(np.cumsum(market_spread_changes), 'red')
    plt.plot(np.cumsum(issuer_spread_changes), 'green')
    plt.plot(np.cumsum(bond_spread_changes), 'black')

    plt.show()

    pass