import pandas as pd
import numpy as np
import fast_tmfg as tmfg
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as spla
from sklearn.covariance import GraphicalLasso
from gglasso.problem import glasso_problem
import time


from Simulation.PortfolioSimulation import PortfolioSimulation
from Simulation.CovarianceMatrics import CovarianceMatrix
from Simulation.MeanRevertingOU import MeanRevertingOU
import Bond.BondsRefDataMap as bond_map
from Bond.Sectors import Sectors
from Analysis.CovarianceEstimation.Utils import compute_inv_via_lu_decomposition
from Analysis.CovarianceEstimation.Utils import estimate_covariance, effective_sparsity, normed_l1_sparsity, gaussian_log_likelihood, replace_outliers_with_mean, calibrate_lasso, precision_to_adjacency


path = '../../Tests/data/'
bond_ref_data_LQD = bond_map.BondsRefDataMap()
bond_ref_data_LQD.load_bond_indic(path + 'BondIndicsLQD.csv')

real_bond_spreads = pd.read_csv(path + 'agg_ts_plus.csv')
bonds_df = bond_ref_data_LQD.isin_to_indic_df[['isin', 'ticker', 'industry_sector']]
bonds_df.drop_duplicates(subset=['isin'], inplace=True)
bonds_df.sort_values(by=['industry_sector', 'ticker'], inplace=True)
isin_to_ticker_map = dict(zip(bonds_df['isin'], bonds_df['ticker']))
unique_tickers = bonds_df.drop_duplicates(subset = ['ticker'])
ticker_to_sector_map = dict(zip( unique_tickers['ticker'], unique_tickers['industry_sector']))

sectors_list = list(set(bonds_df['industry_sector']))
sector_spreads = real_bond_spreads[sectors_list]
initial_levels_sector = sector_spreads.iloc[0].to_dict()

issuers_list = list(set(bonds_df['ticker']))
issuer_spreads = real_bond_spreads[issuers_list]
initial_levels_issuer = issuer_spreads.iloc[0].to_dict()

bond_list = list(bonds_df['isin'].to_list())
bond_spreads = real_bond_spreads[bond_list]
initial_levels_bond = bond_spreads.iloc[0].to_dict()

print('nb sectors ', len(sectors_list), ' nb issuers ', len(issuers_list), ' nb bonds ', len(bond_list))

# bond_betas = {b: np.random.normal(1., 0.2) for b in bonds_df['isin']}
# issuer_betas = {b: np.random.normal(0.8, 0.1) for b in issuers_list}
# sector_betas = {b: np.random.normal(0.6, 0.1) for b in sectors_list}
# #
# x = [list(bond_betas.values()), list(issuer_betas.values()), list(sector_betas.values())]
# plt.hist(x, 30, density=True, label=['Bond', 'Issuer', 'Sector'], histtype='bar')
# plt.legend()
# plt.savefig('betas.png')
# plt.show()

bond_betas = {b: 0. for b in bonds_df['isin']}
issuer_betas = {b: 0. for b in issuers_list}
sector_betas = {b: 0. for b in sectors_list}

bond_std = {b: max(0., np.random.normal(5., 0.2)) for b in bonds_df['isin']}
issuer_std = {b: max(0., np.random.normal(2., 0.2)) for b in issuers_list}
sector_std = {b: max(0, np.random.normal(1., 0.2)) for b in sectors_list}

# bond_std = {b: 5. for b in bonds_df['isin']}
# issuer_std = {b: 5. for b in issuers_list}
# sector_std = {b: 5. for b in sectors_list}

S0 = 100.
mu = 102.
sigma = 5.
eta = 0.05

market_model = MeanRevertingOU(S0, mu, sigma, eta)

portfolio_ts = PortfolioSimulation('model_name',
                                   market_model, None, None, None,
                                   ticker_to_sector_map, isin_to_ticker_map,
                                   sector_betas, issuer_betas, bond_betas,
                                   sector_std, issuer_std, bond_std,
                                   initial_levels_sector, initial_levels_issuer, initial_levels_bond,
                                   sectors_list, issuers_list, bond_list)


# Calibrate lasso alpha
# single_simulation_N = 5
# lasso_data = portfolio_ts.simulate(single_simulation_N, single_simulation_N)
# lasso_data_df = pd.DataFrame(lasso_data)
# lasso_alpha, fitted_lasso = calibrate_lasso(lasso_data_df.to_numpy())
lasso_alpha = 3.

# sample_size_list = [50]
sample_size_list = [10, 20, 50, 100, 500]
nb_similations = 2

results_df = pd.DataFrame(columns=['n_sim', 'dist_list', 'dist_list_std', 'net_dist_list', 'net_dist_list_std', 'sample_mean', 'sample_std'])
inv_results_df = pd.DataFrame(columns=['n_sim', 'dist_list', 'dist_list_std', 'net_dist_list', 'net_dist_list_std', 'sample_mean', 'sample_std'])
# cond_nb_df = pd.DataFrame(columns=['n_sim', 'cond_num', 'cond_num_std', 'net_cond_num', 'net_cond_num_std', 'sample_mean', 'sample_std', 'exact'])
cond_nb1_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
sparsity_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
eff_sparsity_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
dist_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
dist_std_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
inv_dist_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])
inv_dist_std_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'Shrinkage', 'TMFG'])

gglasso_lambda1 = 1.9

for n_sim in sample_size_list:

    dist_list = []
    inv_dist_list = []
    # cond_nb_list = []

    ridge_dist_list = []
    ridge_inv_dist_list = []
    # ridge_cond_nb_list = []

    OAS_dist_list = []
    OAS_inv_dist_list = []
    # OAS_cond_nb_list = []

    lasso_dist_list = []
    lasso_inv_dist_list = []
    # lasso_cond_nb_list = []

    shrinkage_dist_list = []
    shrinkage_inv_dist_list = []
    # shrinkage_cond_nb_list = []

    dist_net_list = []
    inv_dist_net_list = []
    # cond_nb_net_list = []

    avg_matrix = []
    for i_sim in range(nb_similations):

        bond_changes = portfolio_ts.simulate(n_sim, n_sim)

        # Exact Covariance
        exact_covariance = portfolio_ts.exact_cov_matrix_diff_bond
        exact_inverse = np.linalg.inv(exact_covariance)
        cov_diag = np.diag(exact_covariance)
        cov_diag = np.diag(np.full(len(exact_covariance), cov_diag))

        print('exact sparsity ', normed_l1_sparsity(portfolio_ts.exact_cov_matrix_diff_bond), effective_sparsity(portfolio_ts.exact_cov_matrix_diff_bond, 0.001))

        dist_diag = CovarianceMatrix.frobenius_distance(cov_diag, portfolio_ts.exact_cov_matrix_diff_bond)
        print('dist diag ', dist_diag)

        exact_cond_nb = np.linalg.cond(exact_covariance)

        # Sample
        sample_covariance = pd.DataFrame(bond_changes).cov().to_numpy()
        sample_inv_covariance = np.linalg.inv(sample_covariance)
        dist = CovarianceMatrix.frobenius_distance(sample_covariance, exact_covariance)
        inv_dist = CovarianceMatrix.frobenius_distance(sample_inv_covariance, exact_inverse)
        dist_list.append(dist)
        inv_dist_list.append(inv_dist)
        print(i_sim, 'sample dist ', dist, inv_dist)

        # Ridge
        simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'ridge')
        ridgr_est_cov = simulated_cov
        dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
        ridge_dist_list.append(dist)
        ridge_inv_dist_list.append(inv_dist)
        print(i_sim, 'dist ridge', dist, inv_dist)
        print('sparsity ', normed_l1_sparsity(simulated_cov), effective_sparsity(simulated_cov, 0.001))

        # # GGlasso
        prob = glasso_problem(ridgr_est_cov, n_sim)
        if gglasso_lambda1 == None:
            lambda1_range = sorted([l / 10. for l in list(range(1, 30, 6))], reverse=True)
            modelselect_params = {'lambda1_range': lambda1_range}
            prob.model_selection(modelselect_params=modelselect_params, method='eBIC', gamma=0.1)
            gglasso_lambda1 = prob.reg_params['lambda1']
            print('calibrated lambda1 ', gglasso_lambda1)
        else:
            prob.set_reg_params({'lambda1': gglasso_lambda1})
        # prob.set_reg_params(lambda1=0.1)
        start_time = time.time()  # Start time
        prob.solve()
        end_time = time.time()  # End time
        execution_time = end_time - start_time
        print(f"gglasso execution time: {execution_time / 60:.2f} minutes")
        gglasso_precision_matrix = prob.solution.precision_
        gglasso_covariance_matrix = np.linalg.inv(gglasso_precision_matrix)

        dist = CovarianceMatrix.frobenius_distance(gglasso_covariance_matrix, portfolio_ts.exact_cov_matrix_diff_bond)
        inv_dist = CovarianceMatrix.frobenius_distance(gglasso_precision_matrix, exact_inverse)
        print(i_sim, 'dist gglasso', dist, inv_dist)
        print(i_sim, 'sparsity gglasso', normed_l1_sparsity(gglasso_covariance_matrix), effective_sparsity(gglasso_covariance_matrix, 0.001))

        # Lasso
        simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'lasso', {'alpha': lasso_alpha})
        lasso_est_cov = simulated_cov
        dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
        lasso_dist_list.append(dist)
        # lasso_dist_list.append(0.)
        lasso_inv_dist_list.append(inv_dist)
        # lasso_inv_dist_list.append(0.)
        print(i_sim, 'dist lasso', lasso_est_cov, inv_dist)
        print('sparsity ', normed_l1_sparsity(lasso_est_cov), effective_sparsity(lasso_est_cov, 0.001))

        # if len(avg_matrix) == 0:
        #     avg_matrix = ridgr_est_cov
        # else:
        #     avg_matrix += ridgr_est_cov
        #
        # #  Shrinkage
        # simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'shrink')
        # dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        # inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
        # shrinkage_est_cov = simulated_cov
        # shrinkage_dist_list.append(dist)
        # shrinkage_inv_dist_list.append(inv_dist)
        # print(i_sim, 'dist shrik', dist, inv_dist)
        #
        # # OAS
        # simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'OAS')
        # oas_simulated_cov =simulated_cov
        # dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        # inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
        # OAS_dist_list.append(dist)
        # OAS_inv_dist_list.append(inv_dist)
        # print(i_sim, 'dist OAS', dist, inv_dist)
        #
        # # Network
        # tmfg_obj = tmfg.TMFG()
        # corr = np.square(pd.DataFrame(ridgr_est_cov).corr())
        # start_time = time.time()
        # tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridgr_est_cov), output='logo')
        # end_time = time.time()  # End time
        # execution_time = end_time - start_time
        # print(f"Execution time tmfg : {execution_time / 60:.2f} minutes")
        #
        # inverse_Q = tmfg_obj.J
        # network_Q = np.linalg.inv(tmfg_obj.J)
        # print('sparsity net', normed_l1_sparsity(network_Q), effective_sparsity(network_Q, 0.001))
        #
        # dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
        # dist_net_list.append(dist_net)
        #
        # inv_dist_net = CovarianceMatrix.frobenius_distance(tmfg_obj.J, exact_inverse)
        # inv_dist_net_list.append(inv_dist_net)
        #
        # print(i_sim, 'dist net ', dist_net)
        # print(i_sim, 'inv dist net ', inv_dist_net)

    # avg_matrix /= nb_similations
    # corr = np.square(pd.DataFrame(avg_matrix).corr())
    # tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(avg_matrix), output='logo')
    # inverse_Q = tmfg_obj.J
    # network_Q = np.linalg.inv(tmfg_obj.J)
    # dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
    # inv_dist_net = CovarianceMatrix.frobenius_distance(inverse_Q, exact_inverse)
    # print('net on avg matrix ', dist_net, inv_dist_net)

    print('dist: mean ', np.mean(dist_list), ' std ', np.std(dist_list))
    print('dist net: mean ', np.mean(dist_net_list), ' std ', np.std(dist_net_list))
    print('inv dist: mean ', np.mean(inv_dist_list), ' std ', np.std(inv_dist_list))

    print('inv dist net: mean ', np.mean(inv_dist_net_list), ' std ', np.std(inv_dist_net_list))

    print('dist ridge: mean ', np.mean(ridge_dist_list), ' std ', np.std(ridge_dist_list))
    print('dist shrinkage: mean ', np.mean(shrinkage_dist_list), ' std ', np.std(shrinkage_dist_list))
    print('dist OAS: mean ', np.mean(OAS_dist_list), ' std ', np.std(OAS_dist_list))

    print('inv dist ridge: mean ', np.mean(ridge_inv_dist_list), ' std ', np.std(ridge_inv_dist_list))
    print('inv dist shrinkage: mean ', np.mean(shrinkage_inv_dist_list), ' std ', np.std(shrinkage_inv_dist_list))
    print('inv dist OAS: mean ', np.mean(OAS_inv_dist_list), ' std ', np.std(OAS_inv_dist_list))

    results_df.loc[len(results_df)] = [n_sim, np.mean(ridge_dist_list), np.std(ridge_dist_list), np.mean(dist_net_list), np.std(dist_net_list), np.mean(dist_list), np.std(dist_list)]
    inv_results_df.loc[len(results_df)] = [n_sim, np.mean(ridge_inv_dist_list), np.std(ridge_inv_dist_list), np.mean(inv_dist_net_list), np.std(inv_dist_net_list), np.mean(inv_dist_list), np.std(inv_dist_list)]

    exact_sparsity = normed_l1_sparsity(portfolio_ts.exact_cov_matrix_diff_bond)
    lasso_sparsity = 0.0 # normed_l1_sparsity(lasso_est_cov)
    sample_sparsity = normed_l1_sparsity(sample_covariance)
    # ridge_sparsity = normed_l1_sparsity(ridgr_est_cov)
    # oas_sparsity = normed_l1_sparsity(oas_simulated_cov)
    # shrinkage_sparsity = normed_l1_sparsity(shrinkage_est_cov)
    # networs_sparsity = normed_l1_sparsity(network_Q)

    # sparsity_df.loc[len(sparsity_df)] = [n_sim, exact_sparsity, sample_sparsity, lasso_sparsity, ridge_sparsity, oas_sparsity, shrinkage_sparsity, networs_sparsity]
    # exact_eff_sparsity = effective_sparsity(portfolio_ts.exact_cov_matrix_diff_bond, 0.001)
    # sample_eff_sparsity = effective_sparsity(sample_covariance, 0.001)
    # lasso_eff_sparsity = 0.0 # effective_sparsity(lasso_est_cov, 0.001)
    # ridge_eff_sparsity = effective_sparsity(ridgr_est_cov, 0.001)
    # oas_eff_sparsity = effective_sparsity(oas_simulated_cov, 0.001)
    # shrinkage_eff_sparsity = effective_sparsity(shrinkage_est_cov, 0.001)
    # networs_eff_sparsity = effective_sparsity(network_Q, 0.001)
    # eff_sparsity_df.loc[len(eff_sparsity_df)] = [n_sim, exact_eff_sparsity, sample_eff_sparsity, lasso_eff_sparsity, ridge_eff_sparsity, oas_eff_sparsity, shrinkage_eff_sparsity, networs_eff_sparsity]
    #
    # exact_cond_numb = np.linalg.cond(exact_covariance)
    # sample_cond_numb = np.linalg.cond(sample_covariance)
    # lasso_cond_numb = 0.0 # np.linalg.cond(lasso_est_cov)
    # ridge_cond_numb = np.linalg.cond(ridgr_est_cov)
    # oas_cond_numb = np.linalg.cond(oas_simulated_cov)
    # shrinkage_cond_numb = np.linalg.cond(shrinkage_est_cov)
    # networl_cond_numb = np.linalg.cond(network_Q)
    #
    # cond_nb1_df.loc[len(cond_nb1_df)] = [n_sim, exact_cond_numb, sample_cond_numb, lasso_cond_numb, ridge_cond_numb, oas_cond_numb, shrinkage_cond_numb, networl_cond_numb]
    # dist_df.loc[len(dist_df)] = [n_sim, dist_list[-1], lasso_dist_list[-1], ridge_dist_list[-1], OAS_dist_list[-1], shrinkage_dist_list[-1], dist_net_list[-1]]
    # # dist_std_df.loc[len(dist_std_df)] = [n_sim, std[-1], lasso_inv_dist_list[-1], ridge_inv_dist_list[-1], OAS_inv_dist_list[-1], shrinkage_inv_dist_list[-1], inv_dist_net_list[-1]]
    # inv_dist_df.loc[len(inv_dist_df)] = [n_sim, inv_dist_list[-1], lasso_inv_dist_list[-1], ridge_inv_dist_list[-1], OAS_inv_dist_list[-1], shrinkage_inv_dist_list[-1], inv_dist_net_list[-1]]


cond_nb1_df.to_csv('cond_nb1_df.csv')
dist_df.to_csv('dist_df.csv')
# dist_std_df.to_csv(('dist_std_df.csv'))
inv_dist_df.to_csv('inv_dist_df.csv')
results_df.to_csv('results_df.csv')
inv_results_df.to_csv('inv_results_df.csv')
# cond_nb_df.to_csv('cond_nb_df.csv')
sparsity_df.to_csv('sparsity_df.csv')
eff_sparsity_df.to_csv('eff_sparsity_df.csv')
