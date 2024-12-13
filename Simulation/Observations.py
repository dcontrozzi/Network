
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from Simulation import MeanReverting, ExcessSpreadSimulator

class Observations:

    def __init__(self, name, observation_std, trade_probability_mean, trade_probability_std, trade_distr_type='Exponential'):

        self.name = name
        self.std = observation_std
        self.trade_dist_type = trade_distr_type
        self.trade_probability_mean = trade_probability_mean
        self.trade_probability_std = trade_probability_std


    def get_simulation_str(self):

        return self.name + '_'+ str(self.std) + '_'+ str(self.trade_dist_type) + '_'+ str(self.trade_probability_mean) + '_'+ str(self.trade_probability_std)

    def simulate(self, underlying_process, observation_time_index):

        process_level = underlying_process[observation_time_index]

        observation_adjustment = np.random.normal(0., self.std)

        return process_level + observation_adjustment

    def simulate_multi_dim_observation_levels(self, underlying_process_df):
        """
        Add observation noise to the processes
        :param underlying_process_df:
        :return:
        """

        observation_df = pd.DataFrame()
        for i, row in underlying_process_df.T.iterrows():
            obs_row = [process_level + np.random.normal(0., self.std) for process_level in row]
            if observation_df.empty:
                observation_df = pd.DataFrame.from_dict({i:obs_row}, orient='index')
            else:
                observation_df.loc[i] = obs_row
        return observation_df.T

    def generate_trades_from_observations_ts(self, observations, save_results, save_trade_dist):

        observation_array = observations.T.to_numpy()
        trade_prob_var = self.trade_probability_std * self.trade_probability_std

        if self.trade_dist_type == 'Normal':
            probabiliy_of_trade = \
                np.random.normal(self.trade_probability_mean, trade_prob_var, len(observation_array))
        elif self.trade_dist_type == 'Exponential':
            probabiliy_of_trade = np.random.exponential(scale=self.trade_probability_mean, size=len(observation_array))
        else:
            raise Exception('trade_dist_type not recognised')

        probabiliy_of_trade = [round(max(0., min(p, 1.)), 1) for p in probabiliy_of_trade]

        a = []
        i = 0
        for levels in observation_array:
            number_of_trades_per_day = np.random.poisson(lam=probabiliy_of_trade[i], size=len(levels))
            nb_days = len(levels)
            for day, nb_trades in enumerate(number_of_trades_per_day):
                trade_times = [0] * nb_trades + [i + random.uniform(0, 1)]
                trade_level = [levels[day] + np.random.normal(0., self.std) for t in trade_times]

            a.append(levels * np.random.binomial(1, probabiliy_of_trade[i], len(levels)))
            # a.append(levels * np.random.poisson(lam=self.trade_probability_mean, size=len(levels)))
            i += 1

        trades_df = pd.DataFrame(np.transpose(a), columns=observations.columns)

        if save_results:
            trades_df.to_csv('trades_' + type + '.csv')

        if save_trade_dist:
            plt.hist(probabiliy_of_trade, density=True, label='Trades')
            plt.title('Trades probability distribution')
            plt.legend()
            plt.savefig('trades_distribution.png')

        return trades_df

    def generate_trades_from_levels_ts(self, levels_df, save_results, save_trade_dist):

        # probabiliy_of_trade per bond
        bonds = levels_df.columns
        number_of_bonds = len(bonds)
        if self.trade_dist_type == 'Normal':
            trade_prob_var = self.trade_probability_std * self.trade_probability_std
            probabiliy_of_trade = \
                np.random.normal(self.trade_probability_mean, trade_prob_var, number_of_bonds)
        elif self.trade_dist_type == 'Exponential':
            probabiliy_of_trade = np.random.exponential(scale=self.trade_probability_mean, size=number_of_bonds)

        probabiliy_of_trade = [round(max(0.01, min(p, 1.)), 2) for p in probabiliy_of_trade]
        probabiliy_of_trade_per_bond = {bonds[i]:p for i, p in enumerate(probabiliy_of_trade) }

        #  loop over bonds
        bonds_in_df = list(levels_df.columns)
        entries = []
        output_columns = bonds_in_df + ['time']
        print('number of bonds ', len(bonds_in_df))
        print('number of entries per bond', len(levels_df))
        bond_i = 0
        for bond in bonds_in_df:
            bond_levels = levels_df[bond]
            bond_index = bonds_in_df.index(bond)
            bond_trade_prob = probabiliy_of_trade_per_bond[bond]
            print('bond index ', bond_i)
            for day, level in enumerate(bond_levels):
                nb_trades = np.random.poisson(lam=bond_trade_prob, size=1)[0]
                trade_times = [day + random.uniform(0, 1) for t in range(nb_trades)]
                trade_levels = [level + np.random.normal(0., self.std) for t in trade_times]
                for t,l in zip(trade_times, trade_levels):
                    entry = [0.] * len(output_columns)
                    entry[bond_index] = l
                    entry[-1] = t
                    entries.append(entry)

            bond_i += 1

        observation_df = pd.DataFrame(entries)
        np.savetxt("array_data.csv", entries, delimiter=",")
        observation_df.columns = output_columns
        trades_df = observation_df.sort_values(by='time')

        if save_results:
            trades_df.to_csv('trades_' + self.get_simulation_str() + '.csv')

        if save_trade_dist:
            plt.hist(probabiliy_of_trade, bins=[i/10 for i in range(0, 10, 1)], density=True, label='Trades')
            plt.title('Trades probability distribution')
            plt.legend()
            plt.savefig('trades_distribution' + self.get_simulation_str() + '.png')

        return trades_df

    # def expand_observation(self, observation_df:pd.DataFrame):
    #
    #     new_df = pd.DataFrame(columns=list(observation_df.columns) + ['time'] )
    #     for i, row in observations.iterrows():
    #         row_length = len(row)
    #         for j in range(row_length):
    #             print(i, j)
    #             observation_value = row.iloc[j]
    #             if observation_value != 0.:
    #                 new_row = [0] * row_length + [i + random.uniform(0, 1)]
    #                 new_row[j] = observation_value
    #                 new_df.loc[len(new_df)] = new_row
    #     new_df = new_df.sort_values(by='time')
    #     return new_df

    def expand_observation(self, observation_df: pd.DataFrame):
        rows = []  # Use a list to collect new rows

        for i, row in observation_df.iterrows():
            row_length = len(row)
            for j in range(row_length):
                observation_value = row.iloc[j]
                print(i,j)
                if observation_value != 0.:
                    new_row = [0] * row_length + [i + random.uniform(0, 1)]
                    new_row[j] = round(observation_value, 3)
                    rows.append(new_row)  # Append the new row to the list

        # Convert the list of rows into a DataFrame
        new_df = pd.DataFrame(rows, columns=list(observation_df.columns) + ['time'])

        # Sort the DataFrame by the 'time' column
        new_df = new_df.sort_values(by='time')

        return new_df.reset_index()

    def generate_observations(self, levels_ts, save_results, save_figure):

        observations = self.simulate_multi_dim_observation_levels(levels_ts)

        trades_df = self.generate_trades_from_observations_ts(observations, save_results, save_figure)


        new_obs = self.expand_observation(trades_df)

        if save_results:
            new_obs.to_csv('obs_' + self.name + '_new.csv')


if __name__ == "__main__":

    # path_simulation = '../Simulation/'
    # type = 'bond'
    # bond_spread_levels = pd.read_csv('simulated_' + type + '_spreads_new.csv')

    bond_ts_data_path = '../Tests/data/'
    bond_spread_levels = pd.read_csv(bond_ts_data_path + 'ts_data/simulated_bond_levels.csv')
    bond_spread_levels = bond_spread_levels[[c for c in bond_spread_levels.columns if c != 'date']]

    observation_undectainty = 0.5 # std
    trade_prob_mean = 0.2
    trade_prob_std = 0.3
    name = 'simulated_levels'
    observation_model = Observations(name, observation_undectainty, trade_prob_mean, trade_prob_std)

    bond_spread_levels = bond_spread_levels.head(len(bond_spread_levels) - 500)

    trades_df = observation_model.generate_trades_from_levels_ts(bond_spread_levels, True, True)

    pass
    # observations = observation_model.simulate_multi_dim(bond_spread_levels)
    #
    # trade_prob = [0.1 for i in range(0,len(observations))]
    # # trade_level = []
    # observation_array = observations.T.to_numpy()
    #
    # probabiliy_of_trade = np.random.normal(0.2, 0.3, len(observation_array))
    # probabiliy_of_trade = [round(max(0., min(p, 1.)), 1) for p in probabiliy_of_trade]
    #
    # a = []
    # i = 0
    # for levels in observation_array:
    #     a.append(levels * np.random.binomial(1, probabiliy_of_trade[i], len(levels)))
    #     i += 1
    #
    # trades_df = pd.DataFrame(np.transpose(a), columns = observations.columns)
    # trades_df.to_csv('trades_' + type + '.csv')
    #
    # # split times
    #
    # observations = pd.read_csv(path_simulation + 'trades_' + type + '.csv', index_col=[0])
    #
    # obs = Observations(1.)
    #
    # new_obs = obs.expand_observation(observations)
    #
    # new_obs.to_csv('obs_' + type + '_new.csv')
    #
    pass
    # type = 'sector'
    # bond_spread_levels = pd.read_csv('simulated_' + type + '_spreads.csv')
    #
    # observation_undectainty = 0.5 # std
    # observation_model = Observations(observation_undectainty)

    # print(bond_spread_levels[10])
    # observerd_sprd = observation_model.simulate(bond_spread_levels, 10)
    # print(observerd_sprd)

    # observations = observation_model.simulate_multi_dim(bond_spread_levels)
    #
    # trade_prob = [0.1 for i in range(0,len(observations))]
    # # trade_level = []
    # observation_array = observations.T.to_numpy()
    #
    # probabiliy_of_trade = np.random.normal(0.2, 0.3, len(observation_array))
    # probabiliy_of_trade = [round(max(0., min(p, 1.)), 1) for p in probabiliy_of_trade]
    #
    # a = []
    # i = 0
    # for levels in observation_array:
    #     a.append(levels * np.random.binomial(1, probabiliy_of_trade[i], len(levels)))
    #     i += 1
    #
    # trades_df = pd.DataFrame(np.transpose(a), columns = observations.columns)
    # trades_df.to_csv('trades_' + type + '.csv')

    pass