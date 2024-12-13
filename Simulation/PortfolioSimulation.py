
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Bond.BondsRefDataMap as bond_map
from Bond.Sectors import Sectors
from Simulation import MeanReverting, ExcessSpreadSimulator, MeanRevertingOU

class PortfolioSimulation:

    def __init__(self, model_name,
                 market_model,
                 sector_model,
                 issuer_model,
                 bond_model,
                 issuer_to_sector_map,
                 bond_to_issuer_map,
                 sector_betas: {},
                 issuer_betas: {},
                 bond_betas: {},
                 sector_std: {},
                     issuer_std: {},
                 bond_std: {},
                 sector_initial_level: {},
                 issuer_initial_level: {},
                 bond_initial_level: {},
                 sector_list=None,
                 issuer_list=None,
                 bond_list=None
                 ):

        self.model_name = model_name
        self.market_model = market_model
        self.sector_model = sector_model
        self.issuer_model = issuer_model
        self.bond_model = bond_model
        self.issuer_to_sector_map = issuer_to_sector_map
        self.bond_to_issuer_map = bond_to_issuer_map

        self.tickers_list = list(set(self.issuer_to_sector_map.keys())) if issuer_list == None else issuer_list
        self.sectors_list = list(set(self.issuer_to_sector_map.values())) if sector_list == None else sector_list
        self.isin_list = list(set(self.bond_to_issuer_map.keys())) if bond_list == None else bond_list


        self.sector_betas = sector_betas
        self.issuer_betas = issuer_betas
        self.bond_betas = bond_betas

        self.sector_std = sector_std
        self.issuer_std = issuer_std
        self.bond_std = bond_std

        self.sector_initial_level = sector_initial_level
        self.issuer_initial_level = issuer_initial_level
        self.bond_initial_level = bond_initial_level

    # def calibrate(self):

    def simulate(self, N, T, plot=False):
        """
        N: number of steps
        T: time horizon
        """

        market_levels_ts = self.market_model.simulate(N, T)
        market_changes_ts = np.diff(market_levels_ts)

        exact_mean = self.market_model.mean(N)
        exact_var = self.market_model.var(N)

        empirical_mean = np.mean(market_levels_ts)
        empirical_var = np.var(market_levels_ts)

        exact_diff_mean = self.market_model.diff_mean(N, N + 1)
        exact_diff_var = self.market_model.diff_var(N, N + 1)

        empirical_diff_mean = np.mean(market_changes_ts)
        empirical_diff_var = np.var(market_changes_ts)


        sector_changes = {}
        sector_levels = {}

        # sectors_str_list = [Sectors.to_string(s) for s in self.sectors_list]
        for sector_str in self.sectors_list:

            sector_beta = self.sector_betas[sector_str]
            sector_std = self.sector_std[sector_str]

            sector_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(market_changes_ts, sector_beta, sector_std)
            sector_spread_change = sector_similator.simulate()

            sector_changes[sector_str] = sector_spread_change
            sector_initial_level = self.sector_initial_level[sector_str]
            sector_levels[sector_str] = sector_initial_level + np.cumsum(sector_spread_change)

            if plot:
                plt.plot(sector_levels[sector_str], label=sector_str)

        if plot:
            plt.plot(market_levels_ts, 'red', label='Market', linewidth=5, markersize=12)
            plt.title('Sector time-series')
            plt.legend()
            plt.savefig('sectors.png')
            plt.show()

        # sector_changes_ts = pd.DataFrame(sector_changes)
        # sector_levels_df = pd.DataFrame(sector_levels)
        # sector_levels_df.to_csv(self.model_name + '_simulated_sector_spreads.csv')

        # n_rows = len(sectors_list)
        # n_columns = len(sectors_list)
        # exact_cov_matrix_sectors = np.zeros((n_rows, n_columns))
        # market_variance = exact_var
        # for i_row in range(n_rows):
        #     row_sector = Sectors.to_string(sectors_list[i_row])
        #     for i_columns in range(n_columns):
        #         col_sector = Sectors.to_string(sectors_list[i_columns])
        #         exact_cov_matrix_sectors[i_row, i_columns] = market_variance * sector_betas[row_sector] * sector_betas[col_sector]
        #         if i_row == i_columns:
        #             exact_cov_matrix_sectors[i_row, i_columns] += (sector_std * sector_std) * T
        #
        # n_rows = len(sectors_list)
        # n_columns = len(sectors_list)
        # exact_cov_matrix_sectors_diff = np.zeros((n_rows, n_columns))
        # market_diff_variance = exact_diff_var
        # for i_row in range(n_rows):
        #     row_sector = Sectors.to_string(sectors_list[i_row])
        #     for i_columns in range(n_columns):
        #         col_sector = Sectors.to_string(sectors_list[i_columns])
        #         exact_cov_matrix_sectors_diff[i_row, i_columns] = market_diff_variance * sector_betas[row_sector] * sector_betas[col_sector]
        #         if i_row == i_columns:
        #             exact_cov_matrix_sectors_diff[i_row, i_columns] += (sector_std * sector_std)
        #

        issuer_changes = {}
        issuer_levels = {}
        for ticker in self.tickers_list:

            issuer_beta = self.issuer_betas[ticker]
            issuer_std = self.issuer_std[ticker]

            sector = self.issuer_to_sector_map[ticker]
            sector_timeserie = sector_changes[sector]

            issuer_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(sector_timeserie, issuer_beta, issuer_std)

            issuer_spread_change = issuer_similator.simulate()
            issuer_changes[ticker] = issuer_spread_change
            issue_initial_level = self.issuer_initial_level[ticker]
            issuer_levels[sector] = issue_initial_level + np.cumsum(issuer_spread_change)

        # issuer_levels_df = pd.DataFrame(issuer_levels)
        # issuer_levels_df.to_csv(self.model_name + '_simulated_issuer_spreads.csv')

        bond_timeseries = {}
        bond_levels = {}
        for bond in self.isin_list:

            bond_beta = self.bond_betas[bond]
            bond_std = self.bond_std[bond]

            ticker = self.bond_to_issuer_map[bond]
            ticker_timeserie = issuer_changes[ticker]
            bond_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(ticker_timeserie, bond_beta, bond_std)
            bond_spread_change = bond_similator.simulate()

            bond_timeseries[bond] = bond_spread_change

            initial_level = self.bond_initial_level[bond]
            bond_levels[bond] = initial_level + np.cumsum(bond_spread_change)

        # bond_ts_df = pd.DataFrame(bond_timeseries)
        # bond_levels_df = pd.DataFrame(bond_levels)

        # bond_levels_df.to_csv(self.model_name + 'simulated_bond_spreads_new.csv')

        n_rows = len(self.isin_list)
        n_columns = len(self.isin_list)
        # exact_cov_matrix_bond = np.zeros((n_rows, n_columns))
        # market_variance = exact_var
        # for i_row in range(n_rows):
        #     row_isin = self.isin_list[i_row]
        #
        #     row_issuer = self.bond_to_issuer_map[row_isin]
        #     row_issuer_beta = self.issuer_betas[row_issuer]
        #
        #     row_sector = self.issuer_to_sector_map[row_issuer]
        #     row_sector_beta = self.sector_betas[row_sector]
        #
        #     row_market_beta = row_sector_beta * row_issuer_beta * self.bond_betas[row_isin]
        #     row_beta_sector_adj = row_issuer_beta * self.bond_betas[row_isin]
        #     for i_columns in range(n_columns):
        #         col_isin = self.isin_list[i_columns]
        #
        #         col_issuer = self.bond_to_issuer_map[col_isin]
        #         col_issuer_beta = self.issuer_betas[col_issuer]
        #
        #         col_sector = self.issuer_to_sector_map[col_issuer]
        #         col_sector_beta = self.sector_betas[col_sector]
        #
        #         col_market_beta = col_sector_beta * col_issuer_beta * self.bond_betas[col_isin]
        #         col_beta_sector_adj = col_issuer_beta * self.bond_betas[col_isin]
        #
        #         exact_cov_matrix_bond[i_row, i_columns] = market_variance * row_market_beta * col_market_beta
        #         if row_sector == col_sector:
        #             exact_cov_matrix_bond[
        #                 i_row, i_columns] += row_beta_sector_adj * col_beta_sector_adj * sector_std * sector_std
        #         if row_issuer == col_issuer:
        #             exact_cov_matrix_bond[i_row, i_columns] += self.bond_betas[row_isin] * self.bond_betas[
        #                 col_isin] * issuer_std * issuer_std
        #         if i_row == i_columns:
        #             exact_cov_matrix_bond[i_row, i_columns] += bond_std * bond_std
        #
        # pd.DataFrame(exact_cov_matrix_bond).to_csv('exact_covariance.csv')

        exact_cov_matrix_diff_bond = np.zeros((n_rows, n_columns))
        market_variance = exact_diff_var
        for i_row in range(n_rows):
            row_isin = self.isin_list[i_row]

            row_issuer = self.bond_to_issuer_map[row_isin]
            row_issuer_beta = self.issuer_betas[row_issuer]
            row_issuer_std = self.issuer_std[row_issuer]

            row_sector = self.issuer_to_sector_map[row_issuer]
            row_sector_beta = self.sector_betas[row_sector]
            row_sector_std = self.sector_std[row_sector]

            row_market_beta = row_sector_beta * row_issuer_beta * self.bond_betas[row_isin]
            row_beta_sector_adj = row_issuer_beta * self.bond_betas[row_isin]
            for i_columns in range(n_columns):
                col_isin = self.isin_list[i_columns]

                col_issuer = self.bond_to_issuer_map[col_isin]
                col_issuer_beta = self.issuer_betas[col_issuer]
                col_issuer_std = self.issuer_std[col_issuer]

                col_sector = self.issuer_to_sector_map[col_issuer]
                col_sector_beta = self.sector_betas[col_sector]
                col_sector_std = self.sector_std[col_sector]

                col_market_beta = col_sector_beta * col_issuer_beta * self.bond_betas[col_isin]
                col_beta_sector_adj = col_issuer_beta * self.bond_betas[col_isin]

                exact_cov_matrix_diff_bond[i_row, i_columns] = market_variance * row_market_beta * col_market_beta
                if row_sector == col_sector:
                    exact_cov_matrix_diff_bond[i_row, i_columns] += row_beta_sector_adj * col_beta_sector_adj * row_sector_std * col_sector_std
                if row_issuer == col_issuer:
                    exact_cov_matrix_diff_bond[i_row, i_columns] += self.bond_betas[row_isin] * self.bond_betas[col_isin] * row_issuer_std * col_issuer_std
                if i_row == i_columns:
                    exact_cov_matrix_diff_bond[i_row, i_columns] += self.bond_std[row_isin] * self.bond_std[col_isin]

        self.exact_cov_matrix_diff_bond = exact_cov_matrix_diff_bond
        self.bond_levels = bond_levels
        self.bon_changes = bond_timeseries

        return bond_timeseries


# if __name__ == "__main__":
#
#     path = '../Tests/data/'
#     bond_ref_data_LQD = bond_map.BondsRefDataMap()
#     bond_ref_data_LQD.load_bond_indic(path + 'BondIndicsLQD.csv')
#
#     real_bond_spreads = pd.read_csv(path + 'agg_ts_plus.csv')
#     bonds_df = bond_ref_data_LQD.isin_to_indic_df[['isin', 'ticker', 'industry_sector']]
#     isin_to_ticker_map = dict(zip(bonds_df['isin'], bonds_df['ticker']))
#     unique_tickers = bonds_df.drop_duplicates(subset = ['ticker'])
#     ticker_to_sector_map = dict(zip( unique_tickers['ticker'], unique_tickers['industry_sector']))
#
#     # compute sector everages
#     sectors_list = list(set(bonds_df['industry_sector']))
#     sector_spreads = real_bond_spreads[sectors_list]
#     initial_levels_sector = sector_spreads.iloc[0].to_dict()
#     # compute issuer averages
#     issuers_list = list(set(bonds_df['ticker']))
#     issuer_spreads = real_bond_spreads[issuers_list]
#     initial_levels_issuer = issuer_spreads.iloc[0].to_dict()
#
#     bond_list = list(set(bonds_df['isin'].to_list()))
#     bond_spreads = real_bond_spreads[bond_list]
#     initial_levels_bond = bond_spreads.iloc[0].to_dict()
#
#     bond_betas = {b: np.random.normal(1., 0.1) for b in bonds_df['isin']}
#     issuer_betas = {b: np.random.normal(1., 0.5) for b in issuers_list}
#     sector_betas = {b: np.random.normal(1., 0.5) for b in sectors_list}
#
#     bond_std = {b: 0.5 for b in bonds_df['isin']}
#     issuer_std = {b: 0.5 for b in issuers_list}
#     sector_std = {b: 0.5 for b in sectors_list}
#
#     S0 = 100.
#     mu = 0.05
#     sigma = 5.
#     Sm = 105.
#     eta = 0.05
#
#     model = MeanRevertingOU.MeanRevertingOU(S0, Sm, sigma, eta)
#     # model = MeanReverting.MeanReverting(S0, mu, sigma, eta)
#     market_model = MeanRevertingOU.MeanRevertingOU(S0, Sm, sigma, eta)
#
#     N = 1000
#     T = 1000
#
#     S = model.simulate(N, T)
#     exact_mean = model.mean(N)
#     exact_var = model.var(N)
#
#     pd.DataFrame(S).to_csv('simulated_market_spreads_new.csv')
#
#     market_spread_changes = np.diff(S)
#
#     print('mean ',  exact_mean, ' ', np.mean(S), 'var ', exact_var, ' ' , np.var(S))
#
#     portfolio_ts =  PortfolioSimulation('model_name', market_model,
#                                         None, None, None,
#                                         ticker_to_sector_map, isin_to_ticker_map,
#                                         sector_betas, issuer_betas, bond_betas,
#                                         sector_std, issuer_std, bond_std,
#                                         initial_levels_sector, initial_levels_issuer, initial_levels_bond)
#
#     portfolio_ts.simulate(T, N)
#
#     sector_timeseries = {}
#     sector_levels = {}
#     sector_std = 0.5
#     sectors_set = list(set(bonds_df['industry_sector']))
#     for sector in sectors_set:
#         sector_beta = sector_betas[sector]
#         sector_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(market_spread_changes, sector_beta, sector_std)
#         sector_spread_change = sector_similator.simulate()
#         sector_timeseries[sector] = sector_spread_change
#         sector_initial_level = real_bond_spreads[sector][0]
#         sector_levels[sector] = sector_initial_level + np.cumsum(sector_spread_change)
#
#         # plt.plot(np.cumsum(sector_spread_change), label=sector)
#         plt.plot(sector_levels[sector], label=sector)
#
#     sector_changes_ts = pd.DataFrame(sector_timeseries)
#     sector_levels_df = pd.DataFrame(sector_levels)
#     sector_levels_df.to_csv('simulated_sector_spreads.csv')
#
#     n_rows = len(set(bonds_df['industry_sector']))
#     n_columns = len(set(bonds_df['industry_sector']))
#     exact_cov_matrix_sectors = np.zeros((n_rows, n_columns))
#     market_variance = exact_var
#     for i_row in range(n_rows):
#         row_sector = sectors_set[i_row]
#         for i_columns in range(n_columns):
#             col_sector = sectors_set[i_columns]
#             exact_cov_matrix_sectors[i_row, i_columns] = market_variance * sector_betas[row_sector] * sector_betas[col_sector]
#             if i_row == i_columns:
#                 exact_cov_matrix_sectors[i_row, i_columns] += (sector_std * sector_std) * T
#
#     plt.plot(S, 'red', label='Market', linewidth=5, markersize=12)
#     plt.title('Sector time-series')
#     plt.legend()
#     plt.savefig('sectors.png')
#     plt.show()
#
#     issuer_timeseries = {}
#     issuer_levels = {}
#     issuer_std = 0.5
#     for ticker in bonds_df['ticker']:
#         issuer_beta = issuer_betas[ticker]
#         sector = bonds_df[bonds_df['ticker'] == ticker]['industry_sector'].to_list()[0]
#         sector_timeserie = sector_timeseries[sector]
#         issuer_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(sector_timeserie, issuer_beta, issuer_std)
#         issuer_spread_change = issuer_similator.simulate()
#         issuer_timeseries[ticker] = issuer_spread_change
#         issue_initial_level = real_bond_spreads[ticker][0]
#         issuer_levels[sector] = issue_initial_level + np.cumsum(issuer_spread_change)
#
#     issuer_levels_df = pd.DataFrame(issuer_levels)
#     issuer_levels_df.to_csv('simulated_issuer_spreads.csv')
#
#     test_sector0 = 'Financial'
#     issuers = set(bonds_df[bonds_df['industry_sector'] == test_sector0]['ticker'])
#     for iss in issuers:
#         test_issuer_ts = issuer_timeseries[iss]
#         plt.plot(np.cumsum(test_issuer_ts), label=iss)
#
#     test_sector_ts = sector_timeseries[test_sector0]
#     plt.plot(np.cumsum(test_sector_ts), label=test_sector0, linewidth=5, markersize=12)
#     plt.title('Issuers time-series')
#     plt.legend()
#     plt.savefig('issuers.png')
#     plt.show()
#
#
#     bond_timeseries = {}
#     bond_std = 0.2
#     bond_levels = {}
#     isin_list = bonds_df['isin'].to_list()
#     for bond in isin_list:
#         bond_beta = bond_betas[bond]
#         ticker = bonds_df[bonds_df['isin'] == bond]['ticker'].to_list()[0]
#         ticker_timeserie = issuer_timeseries[ticker]
#         bond_similator = ExcessSpreadSimulator.ExcessSpreadSimulator(ticker_timeserie, bond_beta, bond_std)
#         bond_spread_change = bond_similator.simulate()
#         bond_timeseries[bond] = bond_spread_change
#         initial_level = real_bond_spreads[bond][0]
#         bond_levels[bond] = initial_level + np.cumsum(bond_spread_change)
#
#     bond_ts_df = pd.DataFrame(bond_timeseries)
#     bond_levels_df = pd.DataFrame(bond_levels)
#
#     bond_levels_df.to_csv('simulated_bond_spreads_new.csv')
#
#     n_rows = len(isin_list)
#     n_columns = len(isin_list)
#     exact_cov_matrix_bond = np.zeros((n_rows, n_columns))
#     market_variance = exact_var
#     for i_row in range(n_rows):
#         row_isin = isin_list[i_row]
#         row_sector = Sectors.to_string(bond_ref_data_LQD.isin_to_sector[row_isin])
#         row_sector_beta = sector_betas[row_sector]
#         row_issuer = bond_ref_data_LQD.isin_to_ticker[row_isin]
#         row_issuer_beta = issuer_betas[row_issuer]
#         row_market_beta = row_sector_beta * row_issuer_beta * bond_betas[row_isin]
#         row_beta_sector_adj = row_issuer_beta * bond_betas[row_isin]
#         for i_columns in range(n_columns):
#             col_isin = isin_list[i_columns]
#             col_sector = Sectors.to_string(bond_ref_data_LQD.isin_to_sector[col_isin])
#             col_sector_beta = sector_betas[col_sector]
#             col_issuer = bond_ref_data_LQD.isin_to_ticker[col_isin]
#             col_issuer_beta = issuer_betas[col_issuer]
#             col_market_beta = col_sector_beta * col_issuer_beta * bond_betas[col_isin]
#             col_beta_sector_adj = col_issuer_beta * bond_betas[col_isin]
#
#             exact_cov_matrix_bond[i_row, i_columns] = market_variance * row_market_beta * col_market_beta
#             if row_sector == col_sector:
#                 exact_cov_matrix_bond[i_row, i_columns] += row_beta_sector_adj * col_beta_sector_adj * sector_std * sector_std * T
#             if row_issuer == col_issuer:
#                 exact_cov_matrix_bond[i_row, i_columns] += bond_betas[row_isin] * bond_betas[col_isin] * issuer_std * issuer_std * T
#             if i_row == i_columns:
#                 exact_cov_matrix_bond[i_row, i_columns] += bond_std * bond_std * T
#
#     pd.DataFrame(exact_cov_matrix_bond).to_csv('exact_covariance.csv')
#
#     test_issuer = 'ABIBB'
#     test_issuer_ts = issuer_timeseries[test_issuer]
#     test_sector = bonds_df[bonds_df['ticker'] == test_issuer]['industry_sector'].to_list()[0]
#     test_sector_ts = sector_timeseries[test_sector]
#     test_bonds = bonds_df[bonds_df['ticker'] == test_issuer]['isin'].to_list()
#
#     plt.plot(np.cumsum(test_sector_ts), 'blue', label=test_sector)
#     plt.plot(np.cumsum(market_spread_changes), 'red', label='Market')
#     plt.plot(np.cumsum(test_issuer_ts), 'green', label=test_issuer)
#
#     for b in test_bonds:
#         bond_spread_changes = bond_timeseries[b]
#         bond_spread_levels = np.cumsum(bond_spread_changes)
#         plt.plot(bond_spread_levels, 'grey')
#
#     plt.savefig('bonds.png')
#     plt.show()
#
#
#     pass
