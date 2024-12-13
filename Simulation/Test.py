import numpy as np
from matplotlib import pyplot as plt

import Bond.BondsRefDataMap as bond_map
from Simulation import MeanReverting, ExcessSpreadSimulator, Observations


path = '../Tests/data/'
bond_ref_data_LQD = bond_map.BondsRefDataMap()
bond_ref_data_LQD.load_bond_indic(path + 'BondIndicsLQD.csv')

bonds_df = bond_ref_data_LQD.isin_to_indic_df[['isin', 'ticker', 'industry_sector']]

bond_betas = {b: np.random.normal(1., 0.1) for b in bonds_df['isin']}

issuer_betas = {b: np.random.normal(1., 0.5) for b in bonds_df['ticker']}

sector_betas = {b: np.random.normal(1., 0.5) for b in bonds_df['industry_sector']}

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

sector_timeseries = {}
sector_std = 0.5
for sector in bonds_df['industry_sector']:
    sector_beta = sector_betas[sector]
    sector_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(market_spread_changes, sector_beta, sector_std)
    sector_spread_change = sector_similator.simulate()
    sector_timeseries[sector] = sector_spread_change

issuer_timeseries = {}
issuer_std = 0.5
for ticker in bonds_df['ticker']:
    issuer_beta = issuer_betas[ticker]
    sector = bonds_df[bonds_df['ticker'] == ticker]['industry_sector'].to_list()[0]
    sector_timeserie = sector_timeseries[sector]
    issuer_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(sector_timeserie, issuer_beta, issuer_std)
    issuer_spread_change = issuer_similator.simulate()
    issuer_timeseries[ticker] = issuer_spread_change

bond_timeseries = {}
bond_std = 0.2
for bond in bonds_df['isin']:
    bond_beta = bond_betas[bond]
    ticker = bonds_df[bonds_df['isin'] == bond]['ticker'].to_list()[0]
    ticker_timeserie = issuer_timeseries[ticker]
    bond_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(ticker_timeserie, bond_beta, bond_std)
    bond_spread_change = bond_similator.simulate()
    bond_timeseries[bond] = bond_spread_change

test_issuer = 'ABIBB'
test_issuer_ts = issuer_timeseries[test_issuer]
test_sector = bonds_df[bonds_df['ticker'] == test_issuer]['industry_sector'].to_list()[0]
test_sector_ts = sector_timeseries[test_sector]
test_bonds = bonds_df[bonds_df['ticker'] == test_issuer]['isin'].to_list()

plt.plot(np.cumsum(test_sector_ts), 'blue')
plt.plot(np.cumsum(market_spread_changes), 'red')
plt.plot(np.cumsum(test_issuer_ts), 'green')

bond_spread_levels_list = []
for b in test_bonds:
    bond_spread_changes = bond_timeseries[b]
    bond_spread_levels = np.cumsum(bond_spread_changes)
    bond_spread_levels_list.append(bond_spread_levels)
    plt.plot(bond_spread_levels, 'black')

plt.show()

observation_std = 2.5
observations_model = Observations(observation_std)


pass
