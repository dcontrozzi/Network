
import pandas as pd
import numpy as np
import time
import pylab
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
import cProfile, pstats, io
from pstats import SortKey
from scipy.stats import chi2
import scipy.stats as stats
import fast_tmfg as tmfg
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from Analysis.CovarianceEstimation.Utils import estimate_covariance, effective_sparsity, normed_l1_sparsity, gaussian_log_likelihood, replace_outliers_with_mean, calibrate_lasso, precision_to_adjacency, kk_divergence_gaussian
from mfcf import MFCF
import Bond.BondsRefDataMap as bond_map



path = '../../Tests/data/'

bond_ref_data_LQD = bond_map.BondsRefDataMap()
bond_ref_data_LQD.load_bond_indic(path + 'BondIndicsLQD.csv')

# path_simulation = '../../Simulation/'
# exact_values_df = pd.read_csv(path_simulation +  'simulated_bond_spreads_new.csv',index_col=[0])
exact_values_df = pd.read_csv(path +  'ts_data/LQD_with_factors_ts.csv',index_col=[0])
exact_values_changes_df = exact_values_df.diff()



exact_values = exact_values_df.to_numpy()

# auto-correlation as measure of liquidity - not used
# ac = df_autocorr(exact_values_df, 5)
# rac = df_rolling_autocorr(exact_values_df, 10)
isins= bond_ref_data_LQD.isin_to_ticker.keys()
exact_values_changes_df = exact_values_changes_df[isins]

Q = exact_values_changes_df.cov()
# Q = exact_covariance_df
# print('Q condition number', np.linalg.cond(Q))

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store cross-validation results


net_errors = []
ridge_errors = []
lasso_errors = []
sample_errors = []
shrinkage_errors = []
oas_errors = []

net_log_likelyhood_list = []
net_dense_log_likelyhood_list = []
net_sparse_log_likelyhood_list = []
ridge_log_likelyhood_list = []
lasso_log_likelyhood_list = []
sample_log_likelyhood_list = []
shrinkage_log_likelyhood_list = []
oas_log_likelyhood_list = []

tmfg_obj = tmfg.TMFG()

exact_values_changes_df = exact_values_changes_df.dropna()
count_duplicates = exact_values_changes_df.apply(lambda col: col.duplicated().sum())
# exact_values_changes_df = exact_values_changes_df.loc[:, count_duplicates < 10]
X = exact_values_changes_df.to_numpy()
# reduce size of data
# X = X[100:]

calibration_done = False
lasso_alpha = 0.8

# test-training split
test_size = 50
number_of_days = len(X)
test_set = X[number_of_days - test_size: number_of_days -1]
test_index = list(range(number_of_days - test_size, number_of_days -1))
training_set_size = [10, 20, 50, 80, 100, 150]

# training_sets =[list(range(number_of_days - test_size - s, number_of_days - test_size)) for s in training_set_size]
training_sets =[list(range(number_of_days - test_size - 50, number_of_days - test_size)),
                list(range(number_of_days - test_size - 100, number_of_days - test_size - 50)),
                list(range(number_of_days - test_size - 150, number_of_days - test_size - 100)),
                list(range(number_of_days - test_size - 200, number_of_days - test_size - 150)),
                list(range(number_of_days - test_size - 250, number_of_days - test_size - 200))]

test_sets =[list(range(number_of_days - 50, number_of_days)),
                list(range(number_of_days - 100, number_of_days - 50)),
                list(range(number_of_days - 150, number_of_days - 100)),
                list(range(number_of_days - 200, number_of_days - 150)),
                list(range(number_of_days - 250, number_of_days - 200))]
train_set_size = []
use_quick_lasso = False
# Cross-validate
# for train_index, test_index in kf.split(X):
# for train_index in training_sets:
for train_index, test_index in zip(training_sets, test_sets):

    print(train_index)

    X_train, X_test = X[train_index], X[test_index]
    train_set_size.append(len(train_index))

    oas_simulated_cov, _ = estimate_covariance(pd.DataFrame(X_train).to_numpy(), 'OAS')

    print('OAS')

    ridgr_est_cov, _ = estimate_covariance(pd.DataFrame(X_train).to_numpy(), 'ridge')

    print('ridge')

    corr = np.square(pd.DataFrame(ridgr_est_cov).corr())
    tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridgr_est_cov), output='logo')
    inverse_Q = tmfg_obj.J
    network_Q = np.linalg.inv(tmfg_obj.J)

    print('tmfg')
    # inverse_Q_mfcf_sparse = get_precision_matrix(ridgr_est_cov, min_clique_size=2, max_clique_size=2, threshold=0, coordination_number=20)
    # network_Q_sparse = np.linalg.inv(inverse_Q_mfcf_sparse.J)
    #
    # inverse_Q_mfcf_dense = get_precision_matrix(ridgr_est_cov, min_clique_size=5, max_clique_size=5, threshold=0, coordination_number=20)
    # network_Q_dense = np.linalg.inv(inverse_Q_mfcf_dense.J)

    sample_covariance = pd.DataFrame(X_train).cov().to_numpy()

    # if not calibration_done:
    lasso_data = X_train[:, :100]
    lasso_data = X_train
    lasso_alpha, fitted_lasso = calibrate_lasso(lasso_data, use_quick_lasso, n_cross_val=3)
    # # # lasso_alpha = 1.
    print(lasso_alpha)
    # calibration_done  = True

    lasso_type = 'skggm' if use_quick_lasso else 'lasso'
    try:
        lasso_est_cov, _ = estimate_covariance(pd.DataFrame(X_train), lasso_type, {'alpha': lasso_alpha})
        print('lasso')
    except:
        lasso_est_cov = np.array([])
        print('lasso failed')

    # log-likelood out of sample
    mu = pd.DataFrame(X_test).mean().to_numpy()
    net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, network_Q)
    # dense_net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, network_Q_dense)
    # sparse_net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, network_Q_sparse)
    ridge_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, ridgr_est_cov)

    try:
        sample_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, sample_covariance)
    except:
        sample_log_likelyhood = np.nan
    # shrinkage_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, shrinkage_est_cov)
    oas_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, oas_simulated_cov)

    net_log_likelyhood_list.append(net_log_likelyhood)
    # net_dense_log_likelyhood_list.append(dense_net_log_likelyhood)
    # net_sparse_log_likelyhood_list.append(sparse_net_log_likelyhood)
    ridge_log_likelyhood_list.append(ridge_log_likelyhood)
    if len(lasso_est_cov) > 0:
        lasso_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(X_test).to_numpy(), mu, lasso_est_cov)
        lasso_log_likelyhood_list.append(lasso_log_likelyhood)
    sample_log_likelyhood_list.append(sample_log_likelyhood)
    # shrinkage_log_likelyhood_list.append(shrinkage_log_likelyhood)
    oas_log_likelyhood_list.append(oas_log_likelyhood)


    # Calculate MSE between estimated and true test covariances for both methods
    # sample_cov_test = np.cov(X_test, rowvar=False)

    # net_mse = mean_squared_error(sample_cov_test, network_Q)
    # ridge_mse = mean_squared_error(sample_cov_test, ridgr_est_cov)
    # lasso_mse = mean_squared_error(sample_cov_test, lasso_est_cov)
    # sample_mse = mean_squared_error(sample_cov_test, sample_covariance)
    # ?shrinkage_mse = mean_squared_error(sample_cov_test, shrinkage_est_cov)
    # oas_mse = mean_squared_error(sample_cov_test, oas_simulated_cov)

    # net_errors.append(net_mse)
    # ridge_errors.append(ridge_mse)
    # lasso_errors.append(lasso_mse)
    # sample_errors.append(sample_mse)
    # shrinkage_errors.append(shrinkage_mse)
    # oas_errors.append(oas_mse)


# Calculate average MSE across all folds
# mean_net_error = np.mean(net_errors), np.std(net_errors)
# mean_ridge_error = np.mean(ridge_errors), np.std(ridge_errors)
# mean_lasso_error = np.mean(lasso_errors), np.std(lasso_errors)
# mean_sample_error = np.mean(sample_errors), np.std(sample_errors)
# mean_shrinkage_error = np.mean(shrinkage_errors), np.std(shrinkage_errors)
# mean_oas_error = np.mean(oas_errors), np.std(oas_errors)

print(train_set_size)
print(net_log_likelyhood_list)
print(net_sparse_log_likelyhood_list)
print(net_dense_log_likelyhood_list)
print(ridge_log_likelyhood_list)
print(lasso_log_likelyhood_list)
print(sample_log_likelyhood_list)
print(oas_log_likelyhood_list)

net_log_likelyhood = np.mean(net_log_likelyhood_list), np.std(net_log_likelyhood_list)
net_sparse_log_likelyhood = np.mean(net_sparse_log_likelyhood_list), np.std(net_sparse_log_likelyhood_list)
net_dense_log_likelyhood = np.mean(net_dense_log_likelyhood_list), np.std(net_dense_log_likelyhood_list)
ridge_log_likelyhood = np.mean(ridge_log_likelyhood_list), np.std(ridge_log_likelyhood_list)
lasso_log_likelyhood = np.mean(lasso_log_likelyhood_list), np.std(lasso_log_likelyhood_list)
sample_log_likelyhood = np.mean(sample_log_likelyhood_list), np.std(sample_log_likelyhood_list)
# shrinkage_log_likelyhood = np.mean(shrinkage_log_likelyhood_list), np.std(shrinkage_log_likelyhood_list)
oas_log_likelyhood = np.mean(oas_log_likelyhood_list), np.std(oas_log_likelyhood_list)

log_likelyhood_df = pd.DataFrame(columns=['TMFG', 'Dense', 'Sparse', 'Ridge', 'Lasso', 'OAS'])
log_likelyhood_df.loc[len(log_likelyhood_df)] = [net_log_likelyhood[0], net_dense_log_likelyhood[0], net_sparse_log_likelyhood[0], ridge_log_likelyhood[0], lasso_log_likelyhood[0], oas_log_likelyhood[0]]
log_likelyhood_df.loc[len(log_likelyhood_df)] = [net_log_likelyhood[1], net_dense_log_likelyhood[1], net_sparse_log_likelyhood[1], ridge_log_likelyhood[1], lasso_log_likelyhood[1], oas_log_likelyhood[1]]
log_likelyhood_df.to_csv('real_data_loglikelyhood.csv')

pass