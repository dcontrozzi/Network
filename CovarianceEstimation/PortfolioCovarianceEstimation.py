import pandas as pd
import numpy as np
import fast_tmfg as tmfg
import seaborn as sns
import matplotlib.pyplot as plt
import time


from Simulation.PortfolioSimulation import PortfolioSimulation
from Simulation.CovarianceMatrics import CovarianceMatrix
from Simulation.MeanRevertingOU import MeanRevertingOU
import Bond.BondsRefDataMap as bond_map
from CovarianceEstimation.Utils import eigenvalue_distribution_similarity, estimate_covariance, effective_sparsity, normed_l1_sparsity, gaussian_log_likelihood, replace_outliers_with_mean, calibrate_lasso, precision_to_adjacency, kk_divergence_gaussian


path = '../Tests/data/'
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

beta_b = 0.8
beta_i = 0.
beta_s = 0.
bond_betas = {b: np.random.normal(beta_b, 0.2) for b in bonds_df['isin']}
issuer_betas = {b: np.random.normal(beta_i, 0.2) for b in issuers_list}
sector_betas = {b: np.random.normal(beta_s, 0.2) for b in sectors_list}

sector_betas['Energy'] = 0.
sector_betas['Consumer, Cyclical'] = 0.
sector_betas['Basic Materials'] = 0.
sector_betas['Financial'] = 0.8 if beta_s > 0 else 0.
sector_betas['Technology'] = 0.6 if beta_s > 0 else 0.

x = [list(bond_betas.values()), list(issuer_betas.values()), list(sector_betas.values())]
# plt.hist(x, 30, density=True, label=['Level 1', 'Leval 2', 'Level 3'], histtype='bar')
# plt.legend()
# plt.savefig(f'betas_{beta_b}_{beta_i}_{beta_s}.png')
# plt.show()

print('sector_betas', sector_betas)

# bond_betas = {b: 1. for b in bonds_df['isin']}
# issuer_betas = {b: 0.8 for b in issuers_list}
# sector_betas = {b: 0.6 for b in sectors_list}

bond_std = {b: max(0., np.random.normal(5., 0.2)) for b in bonds_df['isin']}
issuer_std = {b: max(0., np.random.normal(2., 0.2)) for b in issuers_list}
sector_std = {b: max(0, np.random.normal(1., 0.2)) for b in sectors_list}

# bond_std = {b: 5. for b in bonds_df['isin']}
# issuer_std = {b: 5. for b in issuers_list}
# sector_std = {b: 5. for b in sectors_list}

x = [list(bond_std.values()), list(issuer_std.values()), list(sector_std.values())]
# plt.hist(x, 30, density=True, label=['Level 1', 'Leval 2', 'Level 3'], histtype='bar')
# plt.legend()
# plt.savefig('sigma.png')
# plt.show()

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


cond_nb1_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

eigen_dist_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])
eigen_std_dist_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

sparsity_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])
eff_sparsity_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

dist_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])
dist_std_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

log_likelyhood_df = pd.DataFrame(columns=['n_sim', 'Exact', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

kk_divergence_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])
kk_divergence_std_df = pd.DataFrame(columns=['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG', 'MFCF_8', 'MFCF_20', 'MST'])

# Calibrate lasso alpha
use_quick = True
# print('lasso cal start')
# single_simulation_N = 100
# lasso_data = portfolio_ts.simulate(single_simulation_N, single_simulation_N)
# lasso_data_df = pd.DataFrame(lasso_data)
# lasso_data_np = lasso_data_df.to_numpy()[:, :100]
# lasso_alpha, fitted_lasso = calibrate_lasso(lasso_data_np, use_quick=use_quick)
# print(lasso_alpha)
lasso_alpha = 0.33113112148259105

# sample_size_list = [50]
sample_size_list = [10, 50, 100, 500]
nb_similations = 10

for n_sim in sample_size_list:

    print('sample size', n_sim)

    dist_list = []
    ridge_dist_list = []
    OAS_dist_list = []
    lasso_dist_list = []
    tmfg_dist_list = []
    J_MFCF_8_dist_list = []
    J_MFCF_20_dist_list = []
    J_MST_dist_list = []

    eigen_dist_list = []
    eigen_ridge_dist_list = []
    eigen_OAS_dist_list = []
    eigen_lasso_dist_list = []
    eigen_tmfg_dist_list = []
    eigen_J_MFCF_8_dist_list = []
    eigen_J_MFCF_20_dist_list = []
    eigen_J_MST_dist_list = []

    sample_kk_list = []
    ridge_kk_list = []
    OAS_kk_list = []
    lasso_kk_list = []
    tmfg_kk_list = []
    J_MFCF_8_kk_list = []
    J_MFCF_20_kk_list = []
    J_MST_kk_list = []

    for i_sim in range(nb_similations):

        print('sim', i_sim)

        bond_changes = portfolio_ts.simulate(n_sim, n_sim)

        # Exact Covariance
        exact_covariance = portfolio_ts.exact_cov_matrix_diff_bond

        cov_diag = np.diag(exact_covariance)
        cov_diag = np.diag(np.full(len(exact_covariance), cov_diag))

        dist_diag = CovarianceMatrix.frobenius_distance(cov_diag, exact_covariance)

        begin = time.time()

        # Sample
        sample_covariance = pd.DataFrame(bond_changes).cov().to_numpy()
        dist = CovarianceMatrix.frobenius_distance(sample_covariance, exact_covariance)
        dist_list.append(dist)
        sample_kk_list.append(kk_divergence_gaussian(exact_covariance, sample_covariance))

        end = time.time()
        print(f"sample: Total runtime of the program is {end - begin}")
        begin = time.time()

        # Graphs
        # filename = f'./pre_calculated_network_matrices/J_MFCF_8_ridge_large_{n_sim}_{beta_b}_{beta_i}_{beta_s}.csv'
        # J_MFCF_8_inv = pd.read_csv(filename)
        # del J_MFCF_8_inv['Unnamed: 0']
        # J_MFCF_8 = np.linalg.inv(J_MFCF_8_inv)
        # dist_J_MFCF_8 = CovarianceMatrix.frobenius_distance(J_MFCF_8, exact_covariance)
        # J_MFCF_8_dist_list.append(dist_J_MFCF_8)
        # J_MFCF_8_kk_list.append(kk_divergence_gaussian(exact_covariance, J_MFCF_8))
        #
        # filename = f'./pre_calculated_network_matrices/J_MFCF_20_ridge_large_{n_sim}_{beta_b}_{beta_i}_{beta_s}.csv'
        # J_MFCF_20_inv = pd.read_csv(filename)
        # del J_MFCF_20_inv['Unnamed: 0']
        # J_MFCF_20 = np.linalg.inv(J_MFCF_20_inv)
        # dist_J_MFCF_20 = CovarianceMatrix.frobenius_distance(J_MFCF_20, exact_covariance)
        # J_MFCF_20_dist_list.append(dist_J_MFCF_20)
        # J_MFCF_20_kk_list.append(kk_divergence_gaussian(exact_covariance, J_MFCF_20))
        #
        # filename = f'./pre_calculated_network_matrices/J_MST_ridge_large_{n_sim}_{beta_b}_{beta_i}_{beta_s}.csv'
        # J_MST_inv = pd.read_csv(filename)
        # del J_MST_inv['Unnamed: 0']
        # J_MST = np.linalg.inv(J_MST_inv)
        # dist_J_MST = CovarianceMatrix.frobenius_distance(J_MST, exact_covariance)
        # J_MST_dist_list.append(dist_J_MST)
        # J_MST_kk_list.append(kk_divergence_gaussian(exact_covariance, J_MST))
        #
        # end = time.time()
        # print(f"graphs: Total runtime of the program is {end - begin}")
        # begin = time.time()
        #
        # # Ridge
        # ridgr_est_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'ridge')
        # dist = CovarianceMatrix.frobenius_distance(ridgr_est_cov, exact_covariance)
        # ridge_dist_list.append(dist)
        # ridge_kk_list.append(kk_divergence_gaussian(exact_covariance, ridgr_est_cov))
        #
        # end = time.time()
        # print(f"ridge: Total runtime of the program is {end - begin}")
        # begin = time.time()

        # pd.DataFrame(ridgr_est_cov).to_csv(f'ridge_large_{n_sim}_{beta_b}_{beta_i}_{beta_s}.csv')

        # Lasso
        if use_quick:
            lasso_est_cov, simulated_inv_cov = estimate_covariance(bond_changes, 'skggm', {'alpha': lasso_alpha})
        else:
            lasso_est_cov, simulated_inv_cov = estimate_covariance(bond_changes, 'lasso', {'alpha': lasso_alpha})
        dist = CovarianceMatrix.frobenius_distance(lasso_est_cov, exact_covariance)
        lasso_dist_list.append(dist)
        lasso_kk_list.append(kk_divergence_gaussian(exact_covariance, lasso_est_cov))

        end = time.time()
        print(f"lasso: Total runtime of the program is {end - begin}")
        begin = time.time()

        # if len(avg_matrix) == 0:
        #     avg_matrix = ridgr_est_cov
        # else:
        #     avg_matrix += ridgr_est_cov

        # OAS
        oas_simulated_cov, simulated_inv_cov = estimate_covariance(pd.DataFrame(bond_changes).to_numpy(), 'OAS')
        dist = CovarianceMatrix.frobenius_distance(oas_simulated_cov, exact_covariance)
        OAS_dist_list.append(dist)
        OAS_kk_list.append(kk_divergence_gaussian(exact_covariance, oas_simulated_cov))

        end = time.time()
        print(f"oas: Total runtime of the program is {end - begin}")
        begin = time.time()

        # Network
        tmfg_obj = tmfg.TMFG()
        corr = np.square(pd.DataFrame(ridgr_est_cov).corr())
        tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(ridgr_est_cov), output='logo')
        inverse_Q = tmfg_obj.J
        network_Q = np.linalg.inv(tmfg_obj.J)
        dist_net = CovarianceMatrix.frobenius_distance(network_Q, exact_covariance)
        tmfg_dist_list.append(dist_net)
        tmfg_kk_list.append(kk_divergence_gaussian(exact_covariance, network_Q))


        end = time.time()
        print(f"net: Total runtime of the program is {end - begin}")
        begin = time.time()

        eigen_similarity_sample, exact_cond_numb, sample_cond_numb = eigenvalue_distribution_similarity(
            exact_covariance, sample_covariance)
        eigen_similarity_lasso, _, lasso_cond_numb = eigenvalue_distribution_similarity(exact_covariance, lasso_est_cov)
        eigen_similarity_ridge, _, ridge_cond_numb = eigenvalue_distribution_similarity(exact_covariance, ridgr_est_cov)
        eigen_similarity_oas, _, oas_cond_numb = eigenvalue_distribution_similarity(exact_covariance, oas_simulated_cov)
        eigen_similarity_net, _, networl_cond_numb = eigenvalue_distribution_similarity(exact_covariance, network_Q)
        eigen_similarity_MFCF_8, _, MFCF_8_cond_numb = eigenvalue_distribution_similarity(exact_covariance, J_MFCF_8)
        eigen_similarity_MFCF_20, _, MFCF_20_cond_numb = eigenvalue_distribution_similarity(exact_covariance, J_MFCF_20)
        eigen_similarity_MST, _, MST_cond_numb = eigenvalue_distribution_similarity(exact_covariance, J_MST)

        eigen_dist_list.append(eigen_similarity_sample)
        eigen_ridge_dist_list.append(eigen_similarity_ridge)
        eigen_OAS_dist_list.append(eigen_similarity_oas)
        eigen_lasso_dist_list.append(eigen_similarity_lasso)
        eigen_tmfg_dist_list.append(eigen_similarity_net)
        eigen_J_MFCF_8_dist_list.append(eigen_similarity_MFCF_8)
        eigen_J_MFCF_20_dist_list.append(eigen_similarity_MFCF_20)
        eigen_J_MST_dist_list.append(eigen_similarity_MST)

        # mfcf = MFCF(ridgr_est_cov, min_clique_size=2, max_clique_size=2, threshold=0, coordination_number=20)
        # inverse_Q_mfcf = mfcf.get_precision_matrix()
        # network_Q_mfcf = np.linalg.inv(inverse_Q_mfcf)
        # dist_net2 = CovarianceMatrix.frobenius_distance(network_Q_mfcf, exact_covariance)
        # dist_net_list2.append(dist_net2)


    # avg_matrix /= nb_similations
    # corr = np.square(pd.DataFrame(avg_matrix).corr())
    # tmfg_obj.fit_transform(weights=corr, cov=pd.DataFrame(avg_matrix), output='logo')
    # inverse_Q = tmfg_obj.J
    # network_Q = np.linalg.inv(tmfg_obj.J)
    # dist_net = CovarianceMatrix.frobenius_distance(network_Q, portfolio_ts.exact_cov_matrix_diff_bond)

    exact_sparsity = normed_l1_sparsity(exact_covariance)
    lasso_sparsity = normed_l1_sparsity(lasso_est_cov)
    sample_sparsity = normed_l1_sparsity(sample_covariance)
    ridge_sparsity = normed_l1_sparsity(ridgr_est_cov)
    oas_sparsity = normed_l1_sparsity(oas_simulated_cov)
    networs_sparsity = normed_l1_sparsity(network_Q)
    MFCF_8_sparsity = normed_l1_sparsity(J_MFCF_8)
    MFCF_20_sparsity = normed_l1_sparsity(J_MFCF_20)
    MST_sparsity = normed_l1_sparsity(J_MST)
    sparsity_df.loc[len(sparsity_df)] = [n_sim, exact_sparsity, sample_sparsity, lasso_sparsity, ridge_sparsity, oas_sparsity, networs_sparsity, MFCF_8_sparsity, MFCF_20_sparsity, MST_sparsity]

    exact_eff_sparsity = effective_sparsity(exact_covariance, 0.001)
    sample_eff_sparsity = effective_sparsity(sample_covariance, 0.001)
    lasso_eff_sparsity = effective_sparsity(lasso_est_cov, 0.001)
    ridge_eff_sparsity = effective_sparsity(ridgr_est_cov, 0.001)
    oas_eff_sparsity = effective_sparsity(oas_simulated_cov, 0.001)
    networs_eff_sparsity = effective_sparsity(network_Q, 0.001)
    MFCF_8_eff_sparsity = effective_sparsity(J_MFCF_8, 0.001)
    MFCF_20_eff_sparsity = effective_sparsity(J_MFCF_20, 0.001)
    MST_eff_sparsity = effective_sparsity(J_MST, 0.001)
    eff_sparsity_df.loc[len(eff_sparsity_df)] = [n_sim, exact_eff_sparsity, sample_eff_sparsity, lasso_eff_sparsity, ridge_eff_sparsity, oas_eff_sparsity, networs_eff_sparsity, MFCF_8_eff_sparsity, MFCF_20_eff_sparsity, MST_eff_sparsity]

    eigen_similarity_sample = np.mean(eigen_dist_list)
    eigen_similarity_lasso = np.mean(eigen_lasso_dist_list)
    eigen_similarity_ridge = np.mean(eigen_ridge_dist_list)
    eigen_similarity_oas = np.mean(eigen_OAS_dist_list)
    eigen_similarity_net = np.mean(eigen_tmfg_dist_list)
    eigen_similarity_MFCF_8 = np.mean(eigen_J_MFCF_8_dist_list)
    eigen_similarity_MFCF_20 = np.mean(eigen_J_MFCF_20_dist_list)
    eigen_similarity_MST = np.mean(eigen_J_MST_dist_list)

    # kk divergence
    sample_kk_divergence = np.mean(sample_kk_list)
    lasso_kk_divergence = np.mean(lasso_kk_list)
    ridge_kk_divergence = np.mean(ridge_kk_list)
    oas_kk_divergence = np.mean(OAS_kk_list)
    networl_kk_divergence = np.mean(tmfg_kk_list)
    MFCF_8_kk_divergence = np.mean(J_MFCF_8_kk_list)
    MFCF_20_kk_divergence = np.mean(J_MFCF_20_kk_list)
    MST_kk_divergence = np.mean(J_MST_kk_list)

    kk_divergence_df.loc[len(kk_divergence_df)] = [n_sim, sample_kk_divergence, lasso_kk_divergence, ridge_kk_divergence, oas_kk_divergence, networl_kk_divergence, MFCF_8_kk_divergence, MFCF_20_kk_divergence, MST_kk_divergence]
    kk_divergence_std_df.loc[len(kk_divergence_std_df)] = [n_sim,
                                                           np.std(sample_kk_list),
                                                           np.std(lasso_kk_list),
                                                           np.std(ridge_kk_list),
                                                           np.std(OAS_kk_list),
                                                           np.std(tmfg_kk_list),
                                                           np.std(J_MFCF_8_kk_list),
                                                           np.std(J_MFCF_20_kk_list),
                                                           np.std(J_MST_kk_list)]

    cond_nb1_df.loc[len(cond_nb1_df)] = [n_sim, exact_cond_numb, sample_cond_numb, lasso_cond_numb, ridge_cond_numb, oas_cond_numb, networl_cond_numb, MFCF_8_cond_numb, MFCF_20_cond_numb, MST_cond_numb]

    eigen_dist_df.loc[len(eigen_dist_df)] = [n_sim, eigen_similarity_sample, eigen_similarity_lasso, eigen_similarity_ridge, eigen_similarity_oas, eigen_similarity_net, eigen_similarity_MFCF_8, eigen_similarity_MFCF_20, eigen_similarity_MST]
    eigen_std_dist_df.loc[len(eigen_std_dist_df)] = [n_sim, np.std(eigen_dist_list), np.std(eigen_lasso_dist_list), np.std(eigen_ridge_dist_list), np.std(eigen_OAS_dist_list), np.std(eigen_tmfg_dist_list), np.std(eigen_J_MFCF_8_dist_list), np.std(eigen_J_MFCF_20_dist_list), np.std(eigen_J_MST_dist_list)]

    dist_df.loc[len(dist_df)] = [n_sim, np.mean(dist_list), np.mean(lasso_dist_list), np.mean(ridge_dist_list), np.mean(OAS_dist_list), np.mean(tmfg_dist_list), np.mean(J_MFCF_8_dist_list), np.mean(J_MFCF_20_dist_list), np.mean(J_MST_dist_list)]
    dist_std_df.loc[len(dist_std_df)] = [n_sim, np.std(dist_list), np.std(lasso_dist_list), np.std(ridge_dist_list), np.std(OAS_dist_list), np.std(tmfg_dist_list), np.std(J_MFCF_8_dist_list), np.std(J_MFCF_20_dist_list), np.std(J_MST_dist_list)]

    # log-likelood out of sample
    out_of_sample_bond_changes = portfolio_ts.simulate(1000, 1000)
    mu = pd.DataFrame(out_of_sample_bond_changes).mean().to_numpy()
    exact_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, exact_covariance)
    net_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, network_Q)
    MFCF_8_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, J_MFCF_8)
    MFCF_20_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, J_MFCF_20)
    MST_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, J_MST)
    ridge_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, ridgr_est_cov)
    lasso_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, lasso_est_cov)
    sample_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, sample_covariance)
    oas_log_likelyhood = gaussian_log_likelihood(pd.DataFrame(out_of_sample_bond_changes).to_numpy(), mu, oas_simulated_cov)

    log_likelyhood_df.loc[len(log_likelyhood_df)] = [n_sim, exact_log_likelyhood, sample_log_likelyhood, lasso_log_likelyhood, ridge_log_likelyhood, oas_log_likelyhood, net_log_likelyhood, MFCF_8_log_likelyhood, MFCF_20_log_likelyhood, MST_log_likelyhood]


kk_divergence_df.to_csv('kk_divergence_df.csv')
kk_divergence_std_df.to_csv('kk_divergence_std_df.csv')
log_likelyhood_df.to_csv('log_likelyhood_df.csv')
cond_nb1_df.to_csv('cond_nb1_df.csv')
eigen_dist_df.to_csv('eigen_dist_df.csv')
eigen_std_dist_df.to_csv('eigen_dist_std_df.csv')
dist_df.to_csv('dist_df.csv')
dist_std_df.to_csv(('dist_std_df.csv'))
sparsity_df.to_csv('sparsity_df.csv')
eff_sparsity_df.to_csv('eff_sparsity_df.csv')
