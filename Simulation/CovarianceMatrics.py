import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import fast_tmfg as tmfg



class CovarianceMatrix:

    def __init__(self, ts_df):

        self.original_data = ts_df
        self.tmfg_obj = tmfg.TMFG()

    def get_covariance_on_subset(self, number_of_points):

        # get subset of number_of_points points in the ts to be used for covariance calculation
        # X_train, X_test, y_train, y_test = train_test_split(x_series, y_series, test_size=number_of_points,
        #                                                     random_state=random_state, shuffle=shuffle)
        #
        # self.covariance_ts = res
        #
        # if add_noise:
        #     self.covariance_ts += noise
        #
        #
        # self.covarriance = self.covarriance_ts.cov()

        return self.original_data[:number_of_points].cov()

    def get_network_covariance_on_subset(self, number_of_points):

        sub_matrix = self.original_data[:number_of_points]
        corr = np.square(sub_matrix.corr())
        self.tmfg_obj.fit_transform(weights=corr, cov=sub_matrix.cov(), output='logo')
        sub_matrix_inv_cov = self.tmfg_obj.J

        return np.linalg.inv(sub_matrix_inv_cov)

    def get_ensamble_covariance(self, number_of_ensambles, ensamble_size):
        '''

        :param number_of_points:
        :return:
        '''

    def get_randomised_covariance(self, random_process):
        """
        randomise timeseries

        :param random_process:
        :return:
        """

    def get_ensamble_average(self, ensamble_number, ensamble_size):

        avg_cov = []
        last_entry = len(self.original_data) - ensamble_size
        start_points = set([round(r) for r in np.random.uniform(0, last_entry, ensamble_number)])
        ensamble_number = len(start_points)
        for p in start_points:
            sub_matrix_cov = self.original_data[p: p+ensamble_size].cov()
            if len(avg_cov) == 0:
                avg_cov = sub_matrix_cov
            else:
                avg_cov += sub_matrix_cov

        avg_cov /= ensamble_number

        return avg_cov / ensamble_size


    def get_ensamble_network_average(self, ensamble_number, ensamble_size):

        avg_inv_cov = []
        last_entry = len(self.original_data) - ensamble_size
        start_points = set([round(r) for r in np.random.uniform(0, last_entry, ensamble_number)])
        ensamble_number = len(start_points)
        for p in start_points:
            sub_matrix = self.original_data[p: p+ensamble_size]
            corr = np.square(sub_matrix.corr())
            self.tmfg_obj.fit_transform(weights=corr, cov=sub_matrix.cov(), output='logo')
            sub_matrix_inv_cov = self.tmfg_obj.J
            if len(avg_inv_cov) == 0:
                avg_inv_cov = sub_matrix_inv_cov
            else:
                avg_inv_cov += sub_matrix_inv_cov

        avg_inv_cov /= ensamble_number
        avg_inv_cov /= ensamble_size

        return np.linalg.inv(avg_inv_cov)

    import numpy as np

    @staticmethod
    def frobenius_distance(matrix1, matrix2, normalised=True):
        """
        Compute the Frobenius distance between two matrices.

        Parameters:
        matrix1 : ndarray - First matrix.
        matrix2 : ndarray - Second matrix.

        Returns:
        float - Frobenius distance between the two matrices.
        """
        # Ensure both matrices have the same shape
        if matrix1.shape != matrix2.shape:
            raise ValueError("Matrices must have the same shape.")

        # Compute the Frobenius distance
        #  you can also use np.linalg.norm(diff, 'fro')
        diff = matrix1 - matrix2
        return np.sqrt(np.sum(diff ** 2)) if not normalised else np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(matrix1 ** 2))


if __name__ == "__main__":
    path_simulation = './/'

    exact_values_df = pd.read_csv(path_simulation + 'simulated_bond_spreads.csv', index_col=[0])

    df_size = len(exact_values_df)
    cm = CovarianceMatrix(exact_values_df)

    ensamble_size = 10
    c10 = cm.get_ensamble_average(5, ensamble_size)
    c10_net = cm.get_ensamble_network_average(5, ensamble_size)

    ensamble_size = 20
    c20 = cm.get_ensamble_average(5, ensamble_size)
    c20_net = cm.get_ensamble_network_average(5, ensamble_size)

    ensamble_size = 40
    c40 = cm.get_ensamble_average(5, ensamble_size)
    c40_net = cm.get_ensamble_network_average(5, ensamble_size)

    ensamble_size = 80
    c100 = cm.get_ensamble_average(5, ensamble_size)
    c100_net = cm.get_ensamble_network_average(5, ensamble_size)

    print(np.linalg.cond(100*c10), np.linalg.cond(100*c20), np.linalg.cond(100*c40), np.linalg.cond(100*c100))
    print(np.linalg.cond(100*c10_net), np.linalg.cond(100*c20_net), np.linalg.cond(100*c40_net), np.linalg.cond(100*c100_net))

    ensamble_size = 10
    c10 = cm.get_covariance_on_subset(ensamble_size)
    c10_net = cm.get_network_covariance_on_subset(ensamble_size)

    ensamble_size = 20
    c20 = cm.get_covariance_on_subset(ensamble_size)
    c20_net = cm.get_network_covariance_on_subset(ensamble_size)

    ensamble_size = 40
    c40 = cm.get_covariance_on_subset(ensamble_size)
    c40_net = cm.get_network_covariance_on_subset(ensamble_size)

    ensamble_size = 80
    c100 = cm.get_covariance_on_subset(ensamble_size)
    c100_net = cm.get_network_covariance_on_subset(ensamble_size)

    print(np.linalg.cond(100*c10), np.linalg.cond(100*c20), np.linalg.cond(100*c40), np.linalg.cond(100*c100))
    print(np.linalg.cond(100*c10_net), np.linalg.cond(100*c20_net), np.linalg.cond(100*c40_net), np.linalg.cond(100*c100_net))

    pass