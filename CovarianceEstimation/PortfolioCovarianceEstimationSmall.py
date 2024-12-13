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
from mfcf import MFCF


from Simulation.PortfolioSimulation import PortfolioSimulation
from Simulation.CovarianceMatrics import CovarianceMatrix
from Simulation.MeanRevertingOU import MeanRevertingOU
import Bond.BondsRefDataMap as bond_map
from Bond.Sectors import Sectors
from Analysis.CovarianceEstimation.Utils import compute_inv_via_lu_decomposition
from Analysis.CovarianceEstimation.Utils import eigenvalue_distribution_similarity, estimate_covariance, effective_sparsity, normed_l1_sparsity, gaussian_log_likelihood, replace_outliers_with_mean, calibrate_lasso, precision_to_adjacency, kk_divergence_gaussian


# Model set-up
sectors_list = ['Financial', 'Energy', 'Consumer']
initial_levels_sector = [101, 200., 160.]
initial_levels_sector = {k: v for k, v in zip(sectors_list, initial_levels_sector)}
# compute issuer averages
issuers_list = ['F1', 'F2', 'F3', 'E1', 'E2', 'C1', 'C2', 'C3', 'C4', 'C5']
initial_levels_issuer = list(range(99, 102, 1)) + list(range(198, 202, 2)) + list(range(150, 155, 1))
initial_levels_issuer = {k: v for k, v in zip(issuers_list, initial_levels_issuer)}

bond_list = ['F1b1', 'F2b1', 'F2b2', 'F3b1', 'F3b2', 'F3b3', 'E1b1', 'E2b1', 'E2b2', 'E2b3',
             'C1b1', 'C1b2', 'C2b1', 'C2b2', 'C2b3', 'C2b4', 'C3b1', 'C4b1', 'C5b1', 'C5b2']

initial_levels_bond = list(range(99, 105, 1)) + list(range(198, 206, 2)) + list(range(140, 170, 3))
initial_levels_bond = {k: v for k, v in zip(bond_list, initial_levels_bond)}

beta_b = 0.8
beta_i = 0.6
beta_s = 0.4

bond_betas = {b: beta_b for b in bond_list}
issuer_betas = {b: beta_i for b in issuers_list}
sector_betas = {'Financial': beta_s, 'Energy': beta_s, 'Consumer': beta_s}

bond_std = {b: 5.5 for b in bond_list}

issuer_std = {'F1': 2., 'F2': 2.5, 'F3': 2.3, 'E1': 5., 'E2': 5.2, 'C1': 3., 'C2': 3.2, 'C3': 3.2, 'C4': 3.5, 'C5': 3.4}
sector_std = {b: 2.5 for b in sectors_list}


S0 = 100.
mu = 102.
sigma = 5.
eta = 0.05

market_model = MeanRevertingOU(S0, mu, sigma, eta)

issuer_to_sector_map = {'F1': 'Financial', 'F2': 'Financial', 'F3': 'Financial', 'E1': 'Energy', 'E2': 'Energy',
                        'C1': 'Consumer', 'C2': 'Consumer', 'C3': 'Consumer', 'C4': 'Consumer', 'C5': 'Consumer'}
bond_to_issuer_map = {'F1b1': 'F1', 'F2b1': 'F2', 'F2b2': 'F2', 'F3b1': 'F3', 'F3b2': 'F3', 'F3b3': 'F3',
                      'E1b1': 'E1', 'E2b1': 'E2', 'E2b2': 'E2', 'E2b3': 'E2',
                      'C1b1': 'C1', 'C1b2': 'C1', 'C2b1': 'C2', 'C2b2': 'C2', 'C2b3': 'C2', 'C2b4': 'C2', 'C3b1': 'C3','C4b1': 'C4', 'C5b1': 'C5', 'C5b2': 'C5'}

portfolio_ts = PortfolioSimulation('model_name',
                                   market_model, None, None, None,
                                   issuer_to_sector_map, bond_to_issuer_map,
                                   sector_betas, issuer_betas, bond_betas,
                                   sector_std, issuer_std, bond_std,
                                   initial_levels_sector, initial_levels_issuer, initial_levels_bond,
                                   sectors_list, issuers_list, bond_list)

data = portfolio_ts.simulate(100, 100)
alpha, fitted_lasso = calibrate_lasso(pd.DataFrame(data).to_numpy(), use_quick=True)

N = 5
T = 5
nb_similations = 10

tmfg_obj = tmfg.TMFG()

sample_size_list0 = [10, 20, 50, 100, 500, 1000]

distance0 = []
distance_std0 = []
inv_distance0 = []
inv_distance_std0 = []
sparsity0 = []
sparsity1 = []
cond_number0 = []
log_likelyhood0 = []
kk_divergence = []
kk_divergence_std = []
for n_sim0 in sample_size_list0:

    dist_list = []
    dist_net_list = []
    dist_net_list_dense = []
    dist_net_list_sparse = []
    dist_ridge_list = []
    dist_lasso_list = []
    dist_oas_list = []
    inv_dist_list = []
    inv_dist_net_list = []
    inv_dist_net_list_dense = []
    inv_dist_net_list_sparse = []
    inv_dist_ridge_list = []
    inv_dist_lasso_list = []
    inv_dist_oas_list = []

    kk_sample_list = []
    kk_tmfg_list = []
    kk_net_list_dense = []
    kk_net_list_sparse = []
    kk_ridge_list = []
    kk_lasso_list = []
    kk_oas_list = []
    kk_sparse_list = []
    kk_dense_list = []

    for i_sim in range(nb_similations):

        bond_changes = portfolio_ts.simulate(n_sim0, n_sim0)
        exact_cov = portfolio_ts.exact_cov_matrix_diff_bond
        inv_exact = np.linalg.inv(exact_cov)

        # Sample Covariance
        simulated_cov = pd.DataFrame(bond_changes).cov().to_numpy()
        simulated_inv_cov = np.linalg.inv(simulated_cov)
        dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_list.append(dist)
        inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, inv_exact)
        inv_dist_list.append(inv_dist)

        # Ridge
        ridge_simulated_cov, adv_simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'ridge')
        dist = CovarianceMatrix.frobenius_distance(ridge_simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_ridge_list.append(dist)
        inv_dist = CovarianceMatrix.frobenius_distance(adv_simulated_inv_cov, inv_exact)
        inv_dist_ridge_list.append(inv_dist)
        pd.DataFrame(ridge_simulated_cov).to_csv(f'ridgs_{beta_b}_{beta_i}_{beta_s}.csv')

        # OAS
        oas_simulated_cov, oas_simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'OAS')
        dist = CovarianceMatrix.frobenius_distance(oas_simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_oas_list.append(dist)
        inv_dist = CovarianceMatrix.frobenius_distance(oas_simulated_inv_cov, inv_exact)
        inv_dist_oas_list.append(inv_dist)

        # Lasso
        lasso_simulated_cov, lasso_simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'skggm', {'alpha': alpha, 'covariance_matrix_input': simulated_cov})
        # lasso_simulated_cov, lasso_simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'gglasso', {'alpha': alpha, 'covariance_matrix_input': simulated_cov})
        # lasso_simulated_cov, lasso_simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'lasso', {'alpha': alpha})
        dist = CovarianceMatrix.frobenius_distance(lasso_simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_lasso_list.append(dist)
        inv_dist = CovarianceMatrix.frobenius_distance(lasso_simulated_inv_cov, inv_exact)
        inv_dist_lasso_list.append(inv_dist)

        # Network
        corr = np.square(pd.DataFrame(ridge_simulated_cov).corr())
        tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridge_simulated_cov), output='logo')
        inverse_Q = tmfg_obj.J
        network_Q = np.linalg.inv(tmfg_obj.J)

        dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_net_list.append((dist_net))

        inv_dist_net = CovarianceMatrix.frobenius_distance(inverse_Q, inv_exact)
        inv_dist_net_list.append((inv_dist_net))

        J_MFCF_8_ridgs_08_06_04_inv = pd.read_csv('J_MFCF_8_ridgs_0.8_0.6_0.4.csv')
        del J_MFCF_8_ridgs_08_06_04_inv['Unnamed: 0']
        J_MFCF_8_ridgs_08_06_04 = np.linalg.inv(J_MFCF_8_ridgs_08_06_04_inv)
        dist_net = CovarianceMatrix.frobenius_distance(J_MFCF_8_ridgs_08_06_04, portfolio_ts.exact_cov_matrix_diff_bond)
        print(dist_net)
        J_MFCF_20_ridgs_08_06_04_inv = pd.read_csv('J_MFCF_20_ridgs_0.8_0.6_0.4.csv')
        del J_MFCF_20_ridgs_08_06_04_inv['Unnamed: 0']
        J_MFCF_20_ridgs_08_06_04 = np.linalg.inv(J_MFCF_20_ridgs_08_06_04_inv)
        dist_net = CovarianceMatrix.frobenius_distance(J_MFCF_20_ridgs_08_06_04, portfolio_ts.exact_cov_matrix_diff_bond)
        print(dist_net)
        J_MST_ridgs_08_06_04_inv = pd.read_csv('J_MST_ridgs_0.8_0.6_0.4.csv')
        del J_MST_ridgs_08_06_04_inv['Unnamed: 0']
        J_MST_ridgs_08_06_04 = np.linalg.inv(J_MST_ridgs_08_06_04_inv)
        dist_net = CovarianceMatrix.frobenius_distance(J_MST_ridgs_08_06_04, portfolio_ts.exact_cov_matrix_diff_bond)
        print(dist_net)
        J_TMFG_ridgs_08_06_04_inv = pd.read_csv('J_TMFG_ridgs_0.8_0.6_0.4.csv')
        del J_TMFG_ridgs_08_06_04_inv['Unnamed: 0']
        J_TMFG_ridgs_08_06_04 = np.linalg.inv(J_TMFG_ridgs_08_06_04_inv)
        dist_net = CovarianceMatrix.frobenius_distance(J_TMFG_ridgs_08_06_04, portfolio_ts.exact_cov_matrix_diff_bond)
        print(dist_net)


        #  Dense Network
        mfcf_dense = MFCF(ridge_simulated_cov, min_clique_size = 5, max_clique_size = 5, coordination_number = 20, threshold=0)
        inverse_Q_mfcf_dense = mfcf_dense.get_precision_matrix()
        network_Q_mfcf_dense = np.linalg.inv(inverse_Q_mfcf_dense)

        dist_net_dense = CovarianceMatrix.frobenius_distance(network_Q_mfcf_dense, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_net_list_dense.append(dist_net_dense)
        inv_dist_net_dense = CovarianceMatrix.frobenius_distance(inverse_Q_mfcf_dense, inv_exact)
        inv_dist_net_list_dense.append(inv_dist_net_dense)

        #  Sparse
        mfcf_sparse = MFCF(ridge_simulated_cov, min_clique_size = 2, max_clique_size = 2, coordination_number = 20, threshold=0)
        inverse_Q_mfcf_sparse = mfcf_sparse.get_precision_matrix()
        network_Q_mfcf_sparse = np.linalg.inv(inverse_Q_mfcf_sparse)

        dist_net_sparse = CovarianceMatrix.frobenius_distance(network_Q_mfcf_sparse, portfolio_ts.exact_cov_matrix_diff_bond)
        dist_net_list_sparse.append(dist_net_sparse)
        inv_dist_net_sparse = CovarianceMatrix.frobenius_distance(inverse_Q_mfcf_sparse, inv_exact)
        inv_dist_net_list_sparse.append(inv_dist_net_sparse)

        # a,b,c = eigenvalue_distribution_similarity(exact_cov, oas_simulated_cov)

        sample_kk_divergence = kk_divergence_gaussian(exact_cov, simulated_cov)
        kk_sample_list.append(sample_kk_divergence)
        lasso_kk_divergence = kk_divergence_gaussian(exact_cov, lasso_simulated_cov)
        kk_lasso_list.append(lasso_kk_divergence)
        ridge_kk_divergence = kk_divergence_gaussian(exact_cov, ridge_simulated_cov)
        kk_ridge_list.append(ridge_kk_divergence)
        oas_kk_divergence = kk_divergence_gaussian(exact_cov, oas_simulated_cov)
        kk_oas_list.append(oas_kk_divergence)
        tmfg_kk_divergence = kk_divergence_gaussian(exact_cov, network_Q)
        kk_tmfg_list.append(tmfg_kk_divergence)
        dense_kk_divergence = kk_divergence_gaussian(exact_cov, network_Q_mfcf_dense)
        kk_dense_list.append(dense_kk_divergence)
        sparse_kk_divergence = kk_divergence_gaussian(exact_cov, network_Q_mfcf_sparse)
        kk_sparse_list.append(sparse_kk_divergence)

    # avg_matrix /= nb_similations #net on avg matrix  32.454103357567874 0.06853360292423909 net on avg matrix  35.05078858192285 0.062230654347088944
    # corr = np.square(pd.DataFrame(avg_matrix).corr())
    # tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(avg_matrix), output='logo')
    # inverse_Q = tmfg_obj.J
    # network_Q = np.linalg.inv(tmfg_obj.J)
    # dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
    # inv_dist_net = CovarianceMatrix.frobenius_distance(inverse_Q, inv_exact)
    # print('net on avg matrix ',dist_net, inv_dist_net )

    # log-likelood out of sample
    out_of_sample_bond_changes = portfolio_ts.simulate(1000, 1000)
    mu = pd.DataFrame(out_of_sample_bond_changes).mean().to_numpy()
    exact_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, portfolio_ts.exact_cov_matrix_diff_bond)
    net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, network_Q)
    dense_log_likelihood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, network_Q_mfcf_dense)
    sparse_log_likelihood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, network_Q_mfcf_sparse)
    adv_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, ridge_simulated_cov)
    lasso_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, lasso_simulated_cov)
    simple_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, simulated_cov)

    print('log-likelyhood ', exact_log_likelyhood, ',', net_log_likelyhood, ',', adv_log_likelyhood,  ',', lasso_log_likelyhood, ',', simple_log_likelyhood)

    exact_cond_number = np.linalg.cond(portfolio_ts.exact_cov_matrix_diff_bond)
    net_cond_number = np.linalg.cond(network_Q)
    sim_cond_number = np.linalg.cond(simulated_cov)
    ridge_cond_number = np.linalg.cond(ridge_simulated_cov)
    lasso_cond_number = np.linalg.cond(lasso_simulated_cov)
    print('cond number', exact_cond_number,
          net_cond_number,
          ridge_cond_number,
          lasso_cond_number,
          sim_cond_number)


    exact_l1_sparsity = normed_l1_sparsity(portfolio_ts.exact_cov_matrix_diff_bond)
    net_l1_sparsity = normed_l1_sparsity(network_Q)
    sim_l1_sparsity = normed_l1_sparsity(simulated_cov)
    ridge_l1_sparsity = normed_l1_sparsity(ridge_simulated_cov)
    lasso_l1_sparsity = normed_l1_sparsity(lasso_simulated_cov)
    print('l1 sparsity', exact_l1_sparsity,
          net_l1_sparsity,
          ridge_l1_sparsity,
          lasso_l1_sparsity,
          sim_l1_sparsity)

    inv_exact = np.linalg.inv(portfolio_ts.exact_cov_matrix_diff_bond)
    inv_exact_l1_sparsity = normed_l1_sparsity(inv_exact)
    inv_net_l1_sparsity = normed_l1_sparsity(inverse_Q)
    inv_sim = np.linalg.inv(simulated_cov)
    inv_sim_l1_sparsity = normed_l1_sparsity(inv_sim)
    inv_ridge_l1_sparsity = normed_l1_sparsity(adv_simulated_inv_cov)
    inv_lasso_l1_sparsity = normed_l1_sparsity(lasso_simulated_inv_cov)
    print('l1 sparsity inv', inv_exact_l1_sparsity,
          inv_net_l1_sparsity,
          inv_ridge_l1_sparsity,
          inv_lasso_l1_sparsity,
          inv_sim_l1_sparsity)

    exact_effective_sparsity = effective_sparsity(portfolio_ts.exact_cov_matrix_diff_bond, 0.01)
    net_effective_sparsity = effective_sparsity(network_Q, 0.01)
    sim_effective_sparsity = effective_sparsity(simulated_cov, 0.01)
    ridge_effective_sparsity = effective_sparsity(ridge_simulated_cov, 0.01)
    lasso_effective_sparsity = effective_sparsity(lasso_simulated_cov, 0.01)
    print('threshold sparsity', exact_effective_sparsity,
          net_effective_sparsity,
          ridge_effective_sparsity,
          lasso_effective_sparsity,
          sim_effective_sparsity)

    inv_exact_effective_sparsity = effective_sparsity(inv_exact, 0.01)
    inv_net_effective_sparsity = effective_sparsity(inverse_Q, 0.01)
    inv_sim_effective_sparsity = effective_sparsity(inv_sim, 0.01)
    inv_ridge_effective_sparsity = effective_sparsity(adv_simulated_inv_cov, 0.01)
    inv_lasso_effective_sparsity = effective_sparsity(lasso_simulated_inv_cov, 0.01)
    print('inv threshold sparsity', inv_exact_effective_sparsity,
          inv_net_effective_sparsity,
          inv_ridge_effective_sparsity,
          inv_lasso_effective_sparsity,
          inv_sim_effective_sparsity)

    plot = False
    if plot:
        exact_cov_df = pd.DataFrame(portfolio_ts.exact_cov_matrix_diff_bond, columns=bond_list)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns_heathmap = sns.heatmap(exact_cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True,  xticklabels=False, yticklabels=False)
        sns_heathmap.get_figure()
        ax.set_title('Exact covariance matrix, '+
                     'L1 sparsity: ' +
                     str(round(exact_l1_sparsity, 3)) +
                     ', effective sparsity: ' +
                     str(round(exact_effective_sparsity, 3)), weight='bold')
        fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "exact.png")
        plt.show()

        # exact_inv_cov_df = pd.DataFrame(inv_exact, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(exact_inv_cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True,  xticklabels=False, yticklabels=False)
        # sns_heathmap.get_figure()
        # ax.set_title('Exact Precision, '+
        #              'L1 sparsity: ' +
        #              str(round(inv_exact_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(inv_exact_effective_sparsity, 3)), weight='bold')
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "inv_exact.png")
        # plt.show()
        #
        # cov_df = pd.DataFrame(network_Q, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        # ax.set_title('Method: Network, '+
        #              'L1 sparsity: ' +
        #              str(round(net_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(net_effective_sparsity, 3)), weight='bold')
        # fig = sns_heathmap.get_figure()
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "network.png")
        # plt.show()
        #
        # inv_cov_df = pd.DataFrame(inverse_Q, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(inv_cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        # ax.set_title('Inverse matrix: Network, '+
        #              'L1 sparsity: ' +
        #              str(round(net_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(net_effective_sparsity, 3)), weight='bold')
        # fig = sns_heathmap.get_figure()
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "inv_network.png")
        # plt.show()
        #
        # cov_df = pd.DataFrame(simulated_cov, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        # ax.set_title('Method: Person, '+
        #              'L1 sparsity: ' +
        #              str(round(sim_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(sim_effective_sparsity, 3)), weight='bold')
        # fig = sns_heathmap.get_figure()
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "sim.png")
        # plt.show()
        #
        # cov_df = pd.DataFrame(ridge_simulated_cov, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        # ax.set_title('Method: Ridge, '+
        #              'L1 sparsity: ' +
        #              str(round(ridge_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(ridge_effective_sparsity, 3)), weight='bold')
        # fig = sns_heathmap.get_figure()
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "adv.png")
        # plt.show()
        #
        # inv_cov_df = pd.DataFrame(adv_simulated_inv_cov, columns=bond_list)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # sns_heathmap = sns.heatmap(inv_cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        # ax.set_title('Inverse Matrix: Ridge, '+
        #              'L1 sparsity: ' +
        #              str(round(ridge_l1_sparsity, 3)) +
        #              ', effective sparsity: ' +
        #              str(round(ridge_effective_sparsity, 3)), weight='bold')
        # fig = sns_heathmap.get_figure()
        # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "inv_adv.png")
        # plt.show()

    # sns.heatmap(np.linalg.inv(portfolio_ts.exact_cov_matrix_diff_bond), cmap ='RdYlGn', linewidths = 0.30, annot = True)
    # sns.heatmap(inverse_Q, cmap ='RdYlGn', linewidths = 0.30, annot = True)
    # sns.heatmap(np.linalg.inv(simulated_cov), cmap ='RdYlGn', linewidths = 0.30, annot = True)
    # sns.heatmap(adv_simulated_inv_cov, cmap ='RdYlGn', linewidths = 0.30, annot = True)

    # print('dist: mean ', np.mean(dist_list), ' std ', np.std(dist_list))
    # print('dist net: mean ', np.mean(dist_net_list), ' std ', np.std(dist_net_list))
    # print('dist adv: mean ', np.mean(dist_ridge_list), ' std ', np.std(dist_ridge_list))

    print ('dist,', n_sim0, ',', np.mean(dist_net_list) ,',',np.mean(dist_ridge_list),',',np.mean(dist_lasso_list),',',np.mean(dist_list))
    print ('std,',  n_sim0, ',', np.std(dist_net_list) ,',',np.std(dist_ridge_list),',',np.std(dist_lasso_list),',',np.std(dist_list))

    # print('dist inv : mean ', CovarianceMatrix.frobenius_distance(np.linalg.inv(simulated_cov), np.linalg.inv(portfolio_ts.exact_cov_matrix_diff_bond)))
    # print('dist inv net: mean ', CovarianceMatrix.frobenius_distance(inverse_Q, np.linalg.inv(portfolio_ts.exact_cov_matrix_diff_bond)))

    print ('dist inv,', n_sim0, ',', np.mean(inv_dist_net_list),',',np.mean(inv_dist_ridge_list),',',np.mean(inv_dist_lasso_list),',',np.mean(inv_dist_list))
    print ('std inv ,',  n_sim0, ',', np.std(inv_dist_net_list),',',np.std(inv_dist_ridge_list),',',np.std(inv_dist_lasso_list),',',np.std(inv_dist_list))


    distance0.append([n_sim0, np.mean(dist_net_list), np.mean(dist_net_list_dense), np.mean(dist_net_list_sparse), np.mean(dist_ridge_list), np.mean(dist_oas_list), np.mean(dist_lasso_list), np.mean(dist_list)])
    distance_std0.append([n_sim0, np.std(dist_net_list), np.std(dist_ridge_list), np.std(dist_oas_list), np.std(dist_lasso_list), np.std(dist_list)])

    kk_divergence.append([n_sim0, np.mean(kk_tmfg_list), np.mean(kk_dense_list), np.mean(kk_sparse_list), np.mean(kk_ridge_list), np.mean(kk_oas_list), np.mean(kk_lasso_list), np.mean(kk_sample_list)])
    kk_divergence_std.append([n_sim0, np.std(kk_tmfg_list), np.std(kk_dense_list), np.std(kk_sparse_list), np.std(kk_ridge_list), np.std(kk_oas_list),np.std(kk_lasso_list), np.std(kk_sample_list)])

    inv_distance0.append([n_sim0, np.mean(inv_dist_net_list), np.mean(inv_dist_ridge_list), np.mean(inv_dist_lasso_list), np.mean(inv_dist_list)])
    inv_distance_std0.append([n_sim0, np.std(inv_dist_net_list), np.std(inv_dist_ridge_list), np.std(inv_dist_lasso_list), np.std(inv_dist_list)])

    sparsity0.append([n_sim0, exact_l1_sparsity,
          net_l1_sparsity,
          ridge_l1_sparsity,
          lasso_l1_sparsity,
          sim_l1_sparsity])
    sparsity1.append([n_sim0, exact_effective_sparsity,
          net_effective_sparsity,
          ridge_effective_sparsity,
          lasso_effective_sparsity,
          sim_effective_sparsity])
    cond_number0.append([n_sim0, exact_cond_number,
          net_cond_number,
          ridge_cond_number,
          lasso_cond_number,
          sim_cond_number])

    log_likelyhood0.append([n_sim0, exact_log_likelyhood, net_log_likelyhood, dense_log_likelihood, sparse_log_likelihood, adv_log_likelyhood, lasso_log_likelyhood, simple_log_likelyhood])

distance0_df = pd.DataFrame(distance0, columns=['Number of Simulations', 'TMFG', 'Dense', 'Sparse', 'Ridge', 'OAS',  'Lasso', 'Sample']).round(3).to_csv('distance0.csv')
distance_std0_df = pd.DataFrame(distance_std0, columns=['Number of Simulations', 'TMFG', 'Ridge', 'OAS', 'Lasso', 'Sample']).round(4).to_csv('distance_std0.csv')
kk_df = pd.DataFrame(kk_divergence, columns=['Number of Simulations', 'TMFG', 'Dense', 'Sparse', 'Ridge', 'OAS', 'Lasso', 'Sample']).round(4).to_csv('kk_df.csv')
kk_std_df = pd.DataFrame(kk_divergence_std, columns=['Number of Simulations', 'TMFG', 'Dense', 'Sparse', 'Ridge', 'OAS', 'Lasso', 'Sample']).round(5).to_csv('kk_df_std0.csv')

inv_distance_0_df = pd.DataFrame(inv_distance0, columns=['Number of Simulations', 'TMFG', 'Ridge', 'Lasso', 'Sample']).to_csv('inv_distance0.csv')
inv_distance_std0_df = pd.DataFrame(inv_distance_std0, columns=['Number of Simulations', 'TMFG', 'Ridge', 'Lasso', 'Sample']).to_csv('inv_distance_std0.csv')


sparsity0_df = pd.DataFrame(sparsity0, columns=['Number of Simulations', ' Exact', 'TMFG', 'Ridge', 'Lasso', 'Sample']).round(5).to_csv('sparsity0.csv')
sparsity1_df = pd.DataFrame(sparsity1, columns=['Number of Simulations', ' Exact', 'TMFG', 'Ridge', 'Lasso', 'Sample']).round(5).to_csv('sparsity1.csv')
cond_number0_df = pd.DataFrame(cond_number0, columns=['Number of Simulations', ' Exact', 'TMFG', 'Ridge', 'Lasso', 'Sample']).to_csv('cond_number0.csv')
log_likelyhood0_df = pd.DataFrame(log_likelyhood0, columns=['Number of Simulations', ' Exact', 'TMFG', 'Dense', 'Sparse', 'Ridge', 'Lasso', 'Sample']).round(0).to_csv('log_likelyhood0.csv')



###########################################################

single_simulation_N = 100

lasso_data = portfolio_ts.simulate(single_simulation_N, single_simulation_N)
corr_exact = portfolio_ts.exact_cov_matrix_diff_bond
inv_exact = np.linalg.inv(corr_exact)
exact_adj = precision_to_adjacency(inv_exact)

lasso_data_df = pd.DataFrame(lasso_data)

use_quick = False
start_time = time.time()  # Start time
lasso_alpha, fitted_lasso = calibrate_lasso(lasso_data_df.to_numpy(), use_quick=use_quick)
end_time = time.time()    # End time
execution_time = end_time - start_time
print(f"Execution time scikitlearn : {execution_time / 60:.2f} minutes")
print('lasso alpha', lasso_alpha) #lasso alpha 4.702686745128865/1.307270283963246

lasso_cov1, lasso_inv1 = estimate_covariance(pd.DataFrame(lasso_data).to_numpy(), 'lasso', {'alpha': lasso_alpha})
lasso_adj = precision_to_adjacency(lasso_inv1)

dist_lasso = CovarianceMatrix.frobenius_distance(lasso_cov1, corr_exact)
inv_dist_lasso = CovarianceMatrix.frobenius_distance(inv_exact, lasso_inv1)

ridge_cov, inv_ridge = estimate_covariance(pd.DataFrame(lasso_data).to_numpy(), 'ridge')
ridge_adj = precision_to_adjacency(inv_ridge)

dist_ridge = CovarianceMatrix.frobenius_distance(ridge_cov, corr_exact)
inv_dist_ridge = CovarianceMatrix.frobenius_distance(inv_ridge, inv_exact)

# lasso_data = pd.DataFrame(lasso_data)[list(pd.DataFrame(lasso_data).columns)[:1000]]
sample_covariance = lasso_data_df.cov().to_numpy()
sample_inverse_cov = np.linalg.inv(sample_covariance)
sample_adj = precision_to_adjacency(sample_inverse_cov)

N = lasso_data_df.to_numpy().shape[0]  # nb samples

prob = glasso_problem(sample_covariance, N, reg_params={'lambda1': 0.5})

start_time = time.time()  # Start time
lambda1_range = sorted([l/ 10. for l in list(range(1, 50, 1)) ], reverse=True)
modelselect_params = {'lambda1_range': lambda1_range}
prob.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)
end_time = time.time()    # End time
execution_time = end_time - start_time
print(f"Execution time model selection: {execution_time / 60:.2f} minutes")
print('gglasso regularization params', prob.reg_params)
# prob.set_reg_params({'lambda1': prob.reg_params})
start_time = time.time()  # Start time
prob.solve()
end_time = time.time()    # End time
execution_time = end_time - start_time
print(f"Execution time: {execution_time / 60:.2f} minutes")
gglasso_precision_matrix = prob.solution.precision_

dist_lasso2 = CovarianceMatrix.frobenius_distance(np.linalg.inv(gglasso_precision_matrix), corr_exact)
inv_dist_lasso2 = CovarianceMatrix.frobenius_distance(gglasso_precision_matrix, inv_exact)

corr = np.square(pd.DataFrame(lasso_data).corr())
start_time = time.time()  # Start time
tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridge_cov), output='logo')
end_time = time.time()    # End time
execution_time = end_time - start_time
print(f"Execution time tmfg: {execution_time / 60:.2f} minutes")
inverse_Q = tmfg_obj.J
network_Q = np.linalg.inv(tmfg_obj.J)
precision_to_adjacency(network_Q)

dist_net = CovarianceMatrix.frobenius_distance(network_Q, corr_exact)
inv_dist_net = CovarianceMatrix.frobenius_distance(inverse_Q, inv_exact)

print('dist lasso1: ', dist_lasso, inv_dist_lasso)
print('dist lasso2: ', dist_lasso2, inv_dist_lasso2)
print('dist net: ', dist_net, inv_dist_net)
print('dist ridge: ', dist_ridge, inv_dist_ridge)

print(effective_sparsity(inv_exact, 0.0),
      effective_sparsity(lasso_inv1, 0.0),
      effective_sparsity(inv_ridge, 0.0),
      effective_sparsity(inverse_Q, 0.0))

print(normed_l1_sparsity(inv_exact),
      normed_l1_sparsity(lasso_inv1),
      normed_l1_sparsity(inv_ridge),
      normed_l1_sparsity(inverse_Q))

print(eigenvalue_distribution_similarity(exact_cov, ridge_cov),
      eigenvalue_distribution_similarity(exact_cov, lasso_cov1),
      eigenvalue_distribution_similarity(exact_cov, network_Q))

import networkx as nx

fig, axs = plt.subplots(2,5, figsize=(10,8))
node_size = 50
font_size = 9

G1 = nx.from_numpy_array(prob.solution.adjacency_)
G0 = nx.from_numpy_array(lasso_adj)
G2 = nx.from_numpy_array(precision_to_adjacency(inverse_Q))
G3 = nx.from_numpy_array(exact_adj)
G4 = nx.from_numpy_array(sample_adj)
G5 = nx.from_numpy_array(ridge_adj)

pos = nx.drawing.layout.spring_layout(G1, seed = 1234)

vmin = min(inv_exact.min(), lasso_inv1.min(), inverse_Q.min(), inv_ridge.min(), sample_inverse_cov.min())
vmax = max(inv_exact.max(), lasso_inv1.max(), inverse_Q.max(), inv_ridge.max(), sample_inverse_cov.max())

# exact
nx.draw_networkx(G3, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
                 font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 0])
axs[0, 0].axis('off')
axs[0, 0].set_title("Exact graph")

sns.heatmap(inv_exact, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
            xticklabels=[], yticklabels=[], ax=axs[1, 0], vmin=vmin, vmax=vmax)
axs[1, 0].set_title("Exact precision matrix")

nx.draw_networkx(G0, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
                 font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 1])
axs[0, 1].axis('off')
axs[0, 1].set_title("Glasso")

sns.heatmap(lasso_inv1, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
            xticklabels=[], yticklabels=[], ax=axs[1, 1], vmin=vmin, vmax=vmax)
axs[1, 1].set_title("Glasso")

nx.draw_networkx(G2, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
                 font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 2])
axs[0, 2].axis('off')
axs[0, 2].set_title("TMFG")

sns.heatmap(inverse_Q, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
            xticklabels=[], yticklabels=[], ax=axs[1, 2], vmin=vmin, vmax=vmax)
axs[1, 2].set_title("TMFG")


# nx.draw_networkx(G0, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
#                  font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 3])
# axs[0, 3].axis('off')
# axs[0, 3].set_title("Recovered scikitlearn graph")
#
# sns.heatmap(simulated_inv_cov, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
#             xticklabels=[], yticklabels=[], ax=axs[1, 3])
# axs[1, 3].set_title("Recovered scikitlearn precision matrix")
#


nx.draw_networkx(G5, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
                 font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 3])
axs[0, 3].axis('off')
axs[0, 3].set_title("Ridge")

sns.heatmap(inv_ridge, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
            xticklabels=[], yticklabels=[], ax=axs[1, 3], vmin=vmin, vmax=vmax)
axs[1, 3].set_title("Ridge")

nx.draw_networkx(G4, pos=pos, node_size=node_size, node_color="peru", edge_color="peru", \
                 font_size=font_size, font_color='white', with_labels=True, ax=axs[0, 4])
axs[0, 4].axis('off')
axs[0, 4].set_title("Sample")

sns.heatmap(sample_inverse_cov, cmap="RdYlGn", linewidth=.5, square=True, cbar=False, \
            xticklabels=[], yticklabels=[], ax=axs[1, 4], vmin=vmin, vmax=vmax)
axs[1, 4].set_title("Sample")

fig.savefig(str(single_simulation_N) + str(beta_s) + str(beta_i) + str(beta_b) + "aggregated_res.png")

plt.show()

#########################################


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

bond_betas = {b: np.random.normal(1., 0.2) for b in bonds_df['isin']}
issuer_betas = {b: np.random.normal(0.8, 0.2) for b in issuers_list}
sector_betas = {b: np.random.normal(0., 0.1) for b in sectors_list}

# bond_betas = {b: 1. for b in bonds_df['isin']}
# issuer_betas = {b: 1. for b in issuers_list}
# sector_betas = {b: 1. for b in sectors_list}

bond_std = {b: max(0., np.random.normal(1., 0.2)) for b in bonds_df['isin']}
issuer_std = {b: max(0., np.random.normal(1., 0.2)) for b in issuers_list}
sector_std = {b: max(0, np.random.normal(1., 0.2)) for b in sectors_list}

# bond_std = {b: 5. for b in bonds_df['isin']}
# issuer_std = {b: 5. for b in issuers_list}
# sector_std = {b: 5. for b in sectors_list}

portfolio_ts = PortfolioSimulation('model_name',
                                   market_model, None, None, None,
                                   ticker_to_sector_map, isin_to_ticker_map,
                                   sector_betas, issuer_betas, bond_betas,
                                   sector_std, issuer_std, bond_std,
                                   initial_levels_sector, initial_levels_issuer, initial_levels_bond,
                                   sectors_list, issuers_list, bond_list)

# bond_changes = portfolio_ts.simulate(500, 500)


# Calculate and print execution time
# prob.set_reg_params(lambda1=0.1)

# gglasso_covariance_matrix = solution.get_covariance()
# gglasso_precision_matrix = solution.get_precision()
# dist = CovarianceMatrix.frobenius_distance(gglasso_covariance_matrix, portfolio_ts.exact_cov_matrix_diff_bond)
# inv_dist = CovarianceMatrix.frobenius_distance(gglasso_precision_matrix, exact_inverse)
# print(i_sim, 'dist gglasso', dist, inv_dist)

# print('large sample')
#
# # sample_size_list = [50]
# sample_size_list = [10, 20, 50, 100, 500]
# results_df = pd.DataFrame(columns=['n_sim', 'dist_list', 'dist_list_std', 'net_dist_list', 'net_dist_list_std', 'sample_mean', 'sample_std'])
# inv_results_df = pd.DataFrame(columns=['n_sim', 'dist_list', 'dist_list_std', 'net_dist_list', 'net_dist_list_std', 'sample_mean', 'sample_std'])
# cond_nb_df = pd.DataFrame(columns=['n_sim', 'cond_num', 'cond_num_std', 'net_cond_num', 'net_cond_num_std', 'sample_mean', 'sample_std', 'exact'])
# gglasso_lambda1 = 2.5
# for n_sim in sample_size_list:
#
#     dist_list = []
#     inv_dist_list = []
#     cond_nb_list = []
#
#     ridge_dist_list = []
#     ridge_inv_dist_list = []
#     ridge_cond_nb_list = []
#
#     OAS_dist_list = []
#     OAS_inv_dist_list = []
#     OAS_cond_nb_list = []
#
#     lasso_dist_list = []
#     lasso_inv_dist_list = []
#     lasso_cond_nb_list = []
#
#     shrinkage_dist_list = []
#     shrinkage_inv_dist_list = []
#     shrinkage_cond_nb_list = []
#
#     dist_net_list = []
#     inv_dist_net_list = []
#     cond_nb_net_list = []
#
#     avg_matrix = []
#     for i_sim in range(nb_similations):
#         bond_changes = portfolio_ts.simulate(n_sim, n_sim)
#         exact_covariance = portfolio_ts.exact_cov_matrix_diff_bond
#         exact_inverse = np.linalg.inv(exact_covariance)
#         cov_diag = np.diag(exact_covariance)
#         cov_diag = np.diag(np.full(len(exact_covariance), cov_diag))
#         print('exact sparsity ', normed_l1_sparsity(portfolio_ts.exact_cov_matrix_diff_bond), effective_sparsity(portfolio_ts.exact_cov_matrix_diff_bond, 0.001))
#         dist_diag = CovarianceMatrix.frobenius_distance(cov_diag, portfolio_ts.exact_cov_matrix_diff_bond)
#         print('dist diag ', dist_diag)
#         exact_cond_nb = np.linalg.cond(exact_covariance)
#
#         sample_covariance = pd.DataFrame(bond_changes).cov().to_numpy()
#         sample_inv_covariance = np.linalg.inv(sample_covariance)
#         dist = CovarianceMatrix.frobenius_distance(sample_covariance, exact_covariance)
#         inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         dist_list.append(dist)
#         inv_dist_list.append(inv_dist)
#         print(i_sim, 'sample dist ', dist, inv_dist)
#
#         # cond_nb = np.linalg.cond(simulated_cov)
#         # print(i_sim, 'cond number', cond_nb, exact_cond_nb)
#         # cond_nb_list.append(cond_nb)
#
#         # exact_cov_df = pd.DataFrame(portfolio_ts.exact_cov_matrix_diff_bond, columns=bond_list)
#         # fig, ax = plt.subplots(figsize=(14, 6))
#         # sns_heathmap = sns.heatmap(exact_cov_df, cmap='RdYlGn', linewidths=0.30, annot=True, xticklabels=True,
#         #                            yticklabels=False)
#         # sns_heathmap.get_figure()
#         # ax.set_title('Method: Exact, ' +
#         #              'L1 sparsity: ' +
#         #              str(round(exact_l1_sparsity, 3)) +
#         #              ', effective sparsity: ' +
#         #              str(round(exact_effective_sparsity, 3)), weight='bold')
#         # fig.savefig(str(N) + str(beta_s) + str(beta_i) + str(beta_b) + "exact.png")
#         # plt.show()
#
#         simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'ridge')
#         ridgr_est_cov = simulated_cov
#         dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#         inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         ridge_dist_list.append(dist)
#         ridge_inv_dist_list.append(inv_dist)
#         print(i_sim, 'dist ridge', dist, inv_dist)
#         print('sparsity ', normed_l1_sparsity(simulated_cov), effective_sparsity(simulated_cov, 0.001))
#
#         prob = glasso_problem(ridgr_est_cov, n_sim)
#         if gglasso_lambda1 == None:
#             lambda1_range = sorted([l / 10. for l in list(range(1, 30, 6))], reverse=True)
#             modelselect_params = {'lambda1_range': lambda1_range}
#             prob.model_selection(modelselect_params=modelselect_params, method='eBIC', gamma=0.1)
#             gglasso_lambda1 = prob.reg_params['lambda1']
#             print('calibrated lambda1 ', gglasso_lambda1)
#         else:
#             prob.set_reg_params({'lambda1': gglasso_lambda1})
#         # prob.set_reg_params(lambda1=0.1)
#         start_time = time.time()  # Start time
#         prob.solve()
#         end_time = time.time()  # End time
#         execution_time = end_time - start_time
#         print(f"gglasso execution time: {execution_time / 60:.2f} minutes")
#         gglasso_precision_matrix = prob.solution.precision_
#         gglasso_covariance_matrix = np.linalg.inv(gglasso_precision_matrix)
#
#         dist = CovarianceMatrix.frobenius_distance(gglasso_covariance_matrix, portfolio_ts.exact_cov_matrix_diff_bond)
#         inv_dist = CovarianceMatrix.frobenius_distance(gglasso_precision_matrix, exact_inverse)
#         print(i_sim, 'dist gglasso', dist, inv_dist)
#         print(i_sim, 'sparsity gglasso', normed_l1_sparsity(gglasso_covariance_matrix), gglasso_covariance_matrix(dist, 0.001))
#
#         # simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'lasso', {'alpha': lasso_alpha})
#         # lasso_est_cov = simulated_cov
#         # dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#         # inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         # lasso_dist_list.append(dist)
#         # lasso_inv_dist_list.append(inv_dist)
#         # print(i_sim, 'dist lasso', lasso_est_cov, inv_dist)
#         # print('sparsity ', normed_l1_sparsity(lasso_est_cov), effective_sparsity(lasso_est_cov, 0.001))
#
#         if len(avg_matrix) == 0:
#             avg_matrix = ridgr_est_cov
#         else:
#             avg_matrix += ridgr_est_cov
#
#         cond_nb = np.linalg.cond(simulated_cov)
#         print(i_sim, 'cond number ridge', cond_nb, exact_cond_nb)
#         ridge_cond_nb_list.append(cond_nb)
#
#         # simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'shrink')
#         # dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#         # inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         # shrinkage_dist_list.append(dist)
#         # shrinkage_inv_dist_list.append(inv_dist)
#         # print(i_sim, 'dist shrik', dist, inv_dist)
#         #
#         # cond_nb = np.linalg.cond(simulated_cov)
#         # print(i_sim, 'cond number shrink', cond_nb, exact_cond_nb)
#         # shrinkage_cond_nb_list.append(cond_nb)
#
#         oas_simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'OAS')
#         dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#         inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         OAS_dist_list.append(dist)
#         OAS_inv_dist_list.append(inv_dist)
#         print(i_sim, 'dist OAS', dist, inv_dist)
#
#         cond_nb = np.linalg.cond(simulated_cov)
#         print(i_sim, 'cond number oas', cond_nb, exact_cond_nb)
#         OAS_cond_nb_list.append(cond_nb)
#
#         # try:
#         #     lsimulated_cov, lsimulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'lasso')
#         #     dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#         #     inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         #     lasso_dist_list.append(dist)
#         #     lasso_inv_dist_list.append(inv_dist)
#         #     print('dist lasso', dist, inv_dist)
#         #     cond_nb = np.linalg.cond(simulated_cov)
#         #     print(i_sim, 'cond number lasso', cond_nb, exact_cond_nb)
#         #     lasso_cond_nb_list.append(cond_nb)
#         #
#         # except Exception as e:
#         #     print('lasso failed ', str(e))
#         # dist = CovarianceMatrix.frobenius_distance(simulated_cov, portfolio_ts.exact_cov_matrix_diff_bond)
#
#         # inv_dist = CovarianceMatrix.frobenius_distance(simulated_inv_cov, exact_inverse)
#         # print(i_sim, 'inv dist ', inv_dist)
#
#         tmfg_obj = tmfg.TMFG()
#         corr = np.square(pd.DataFrame(ridgr_est_cov).corr())
#         start_time = time.time()
#         tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridgr_est_cov), output='logo')
#         end_time = time.time()  # End time
#         execution_time = end_time - start_time
#         print(f"Execution time tmfg : {execution_time / 60:.2f} minutes")
#
#         inverse_Q = tmfg_obj.J
#         network_Q = np.linalg.inv(tmfg_obj.J)
#         print('sparsity net', normed_l1_sparsity(network_Q), effective_sparsity(network_Q, 0.001))
#
#         cond_nb = np.linalg.cond(network_Q)
#         print(i_sim, 'cond number net', cond_nb, exact_cond_nb)
#         cond_nb_net_list.append(cond_nb)
#
#         #
#         # P, L, U = sp.linalg.lu(inverse_Q)
#         # A_inv = sp.linalg.inv(U).dot(sp.linalg.inv(L)).dot(P)
#         #
#         # print('inv diff ', CovarianceMatrix.frobenius_distance(A_inv, portfolio_ts.exact_cov_matrix_diff_bond),
#         #       CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond))
#         #
#         #
#         dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
#         dist_net_list.append(dist_net)
#
#         inv_dist_net = CovarianceMatrix.frobenius_distance(tmfg_obj.J, exact_inverse)
#         inv_dist_net_list.append(inv_dist_net)
#
#         print(i_sim, 'dist net ', dist_net)
#         print(i_sim, 'inv dist net ', inv_dist_net)
#
#         cond_nb_net = np.linalg.cond(network_Q)
#         print(i_sim, 'cond number net', cond_nb_net, exact_cond_nb)
#         cond_nb_net_list.append(cond_nb_net)
#
#     avg_matrix /= nb_similations
#     corr = np.square(pd.DataFrame(avg_matrix).corr())
#     tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(avg_matrix), output='logo')
#     inverse_Q = tmfg_obj.J
#     network_Q = np.linalg.inv(tmfg_obj.J)
#     dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
#     inv_dist_net = CovarianceMatrix.frobenius_distance(inverse_Q, exact_inverse)
#     print('net on avg matrix ', dist_net, inv_dist_net)
#
#     print('dist: mean ', np.mean(dist_list), ' std ', np.std(dist_list))
#     print('dist net: mean ', np.mean(dist_net_list), ' std ', np.std(dist_net_list))
#     print('cond nb: mean ', np.mean(cond_nb_list), ' std ', np.std(cond_nb_list))
#     print('inv dist: mean ', np.mean(inv_dist_list), ' std ', np.std(inv_dist_list))
#
#     print('inv dist net: mean ', np.mean(inv_dist_net_list), ' std ', np.std(inv_dist_net_list))
#     print('cond nb net: mean ', np.mean(cond_nb_net_list), ' std ', np.std(cond_nb_net_list))
#
#     print('dist ridge: mean ', np.mean(ridge_dist_list), ' std ', np.std(ridge_dist_list))
#     print('dist shrinkage: mean ', np.mean(shrinkage_dist_list), ' std ', np.std(shrinkage_dist_list))
#     print('dist OAS: mean ', np.mean(OAS_dist_list), ' std ', np.std(OAS_dist_list))
#
#     print('inv dist ridge: mean ', np.mean(ridge_inv_dist_list), ' std ', np.std(ridge_inv_dist_list))
#     print('inv dist shrinkage: mean ', np.mean(shrinkage_inv_dist_list), ' std ', np.std(shrinkage_inv_dist_list))
#     print('inv dist OAS: mean ', np.mean(OAS_inv_dist_list), ' std ', np.std(OAS_inv_dist_list))
#
#     # results_df.loc[len(results_df)] = [n_sim, np.mean(dist_list), np.std(dist_list), np.mean(dist_net_list), np.std(dist_net_list)]
#     results_df.loc[len(results_df)] = [n_sim, np.mean(ridge_dist_list), np.std(ridge_dist_list), np.mean(dist_net_list), np.std(dist_net_list), np.mean(dist_list), np.std(dist_list)]
#     # inv_results_df.loc[len(results_df)] = [n_sim, np.mean(inv_dist_list), np.std(inv_dist_list), np.mean(inv_dist_net_list), np.std(inv_dist_net_list)]
#     inv_results_df.loc[len(results_df)] = [n_sim, np.mean(ridge_inv_dist_list), np.std(ridge_inv_dist_list), np.mean(inv_dist_net_list), np.std(inv_dist_net_list), np.mean(inv_dist_list), np.std(inv_dist_list)]
#     cond_nb_df.loc[len(results_df)] = [n_sim, np.mean(ridge_cond_nb_list), np.std(ridge_cond_nb_list), np.mean(cond_nb_net_list), np.std(cond_nb_net_list), np.mean(cond_nb_list), np.std(cond_nb_list), exact_cond_nb]
#
#     # out_of_sample_bond_changes = portfolio_ts.simulate(1000, 1000)
#     # mu = pd.DataFrame(out_of_sample_bond_changes).mean().to_numpy()
#     # exact_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, exact_covariance)
#     # net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, network_Q)
#     # adv_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, ridgr_est_cov)
#     # simple_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, simulated_cov)
#     # diag_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, cov_diag)
#     #
#     # print('log-likelyhood ', exact_log_likelyhood, ',', net_log_likelyhood, ',', adv_log_likelyhood,  ',', simple_log_likelyhood, ',', diag_log_likelyhood)
#
# results_df.to_csv('results_df.csv')
# inv_results_df.to_csv('inv_results_df.csv')
# cond_nb_df.to_csv('cond_nb_df.csv')
#
#
# # exact_cov_df = pd.DataFrame(portfolio_ts.exact_cov_matrix_diff_bond, columns=bond_list)
# # fig, ax = plt.subplots(figsize=(14, 6))
# # sns_heathmap = sns.heatmap(exact_cov_df, cmap ='RdYlGn', linewidths = 0.30, annot = True,  xticklabels=True, yticklabels=False)
# # sns_heathmap.get_figure()
# # ax.set_title('Method: Exact, '+
# #              'L1 sparsity: ' +
# #              str(round(exact_l1_sparsity, 3)) +
# #              ', effective sparsity: ' +
# #              str(round(exact_effective_sparsity, 3)), weight='bold')
# # fig.savefig(str(N) + str(beta_s)  + str(beta_i) + str(beta_b) + "exact.png")
# # plt.show()
# #
# # exact_inv_cov_df = pd.DataFrame(exact_inverse, columns=bond_list)
# # fig, ax = plt.subplots(figsize=(14, 6))
# # sns_heathmap = sns.heatmap(exact_inv_cov_df, cmap='RdYlGn', linewidths=0.30, annot=False, xticklabels=False,
# #                            yticklabels=False)
# # sns_heathmap.get_figure()
# # ax.set_title('Method: Inverse Exact, ' +
# #              'L1 sparsity: ' +
# #              str(round(inv_exact_l1_sparsity, 3)) +
# #              ', effective sparsity: ' +
# #              str(round(inv_exact_effective_sparsity, 3)), weight='bold')
# # fig.savefig(str(N) + str(beta_s) + str(beta_i) + str(beta_b) + "inv_exact.png")
# # plt.show()
#
# # tmfg_obj = tmfg.TMFG()
# # corr = np.square(pd.DataFrame(simulated_cov).corr())
# # tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(simulated_cov), output='logo')
# # inverse_Q = tmfg_obj.J
# # network_Q = np.linalg.inv(tmfg_obj.J)
# #
# # dist = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)
# # print('network', 'dist ', dist)
# # print('network', 'cond number', np.linalg.cond(network_Q), np.linalg.cond(portfolio_ts.exact_cov_matrix_diff_bond))
