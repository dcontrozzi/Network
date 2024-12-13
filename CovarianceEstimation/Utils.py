
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from sklearn.covariance import GraphicalLasso, ShrunkCovariance, LedoitWolf, OAS, GraphicalLassoCV
import numpy as np
import pandas as pd
from gglasso.problem import glasso_problem
import plotly.figure_factory as ff
import numpy as np
# from inverse_covariance import QuicGraphicalLassoCV, AdaptiveGraphicalLasso
# from inverse_covariance import QuicGraphicalLasso
from numpy.linalg import inv, slogdet
from scipy.linalg import solve


# from MFCF_Example import get_precision_matrix
# np.random.seed(1)
#
# X = np.random.rand(15, 12)  # 15 samples, with 12 dimensions each
# fig = ff.create_dendrogram(X)
# fig.update_layout(width=800, height=500)
# fig.show()
#
# def plot_dendrogram():
#
#     np.random.seed(1)
#
#     X = np.random.rand(15, 12)  # 15 samples, with 12 dimensions each
#     fig = ff.create_dendrogram(X)
#     fig.update_layout(width=800, height=500)
#     fig.show()


def compute_inv_via_lu_decomposition(matrix):
    rng = np.random.default_rng()
    S = sp.random(3, 4, density=0.25, random_state=rng)


    lu = spla.splu(sp.csr_matrix(matrix).tocsc())
    size = np.size(matrix)
    # Solve for inverse matrix using LU factors
    I = sp.eye(size)  # Identity matrix
    A_inv = lu.solve(I.toarray())

    return A_inv

def normed_l1_sparsity(matrix):
    """Calculate the normed l1-norm sparsity metric."""
    total_elements = matrix.size
    l1_norm = np.sum(np.abs(matrix))
    return l1_norm / total_elements


def effective_sparsity(matrix, epsilon=1e-5):
    """Calculate the effective sparsity, considering small values as zero."""
    total_elements = matrix.size
    small_values_count = np.sum(np.abs(matrix) < epsilon)
    return small_values_count / total_elements

def signal_to_noise_ratio(exact_inv_covariance, exact_mean, est_inv_cov, est_mean):

    exact = np.dot(exact_mean.T, np.dot(exact_inv_covariance, exact_mean))
    est = np.dot(est_mean.T, np.dot(est_inv_cov, est_mean))

    return exact - est


def calibrate_lasso(data, use_quick=False, n_cross_val=5):

    # Optimal alpha found through cross-validation
    if use_quick:
        glasso_cv = QuicGraphicalLassoCV(cv=n_cross_val)
        glasso_cv.fit(data)
        optimal_alpha = glasso_cv.lam_

    else:
        glasso_cv = GraphicalLassoCV(cv=n_cross_val)
        glasso_cv.fit(data)
        optimal_alpha = glasso_cv.alpha_

    return optimal_alpha, glasso_cv

def estimate_covariance(ts_dict, method, parameters={}):

    ts = pd.DataFrame(ts_dict)
    if method == 'pearson':
        estimated_cov = ts.cov().to_numpy()
        estimated_inv_cov = np.linalg.inv(estimated_cov)
    elif method == 'gglasso':
        covariance_matrix_input = parameters.get('covariance_matrix_input', np.array([]))
        N = len(covariance_matrix_input)
        if N == 0:
            covariance_matrix_input = ts.cov()
            N = ts.to_numpy().shape[0]
        alpha = parameters['alpha']
        prob = glasso_problem(covariance_matrix_input, N, reg_params={'lambda1': alpha})
        prob.solve()
        estimated_inv_cov = prob.solution.precision_
        estimated_cov = np.linalg.inv(estimated_inv_cov)
    elif method == 'lasso':
        alpha = parameters['alpha']
        max_iter = parameters.get('max_iter', 100)
        tol = parameters.get('tol', 1.e-2)
        cov = parameters.get('cov', np.array([]))
        if len(cov) > 0:
            model = GraphicalLasso(covariance='precomputed', alpha=alpha, max_iter=max_iter, tol=tol)  # alpha controls the amount of regularization
            model.fit(cov)
        else:
            model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol)  # alpha controls the amount of regularization
            model.fit(ts.to_numpy())
        estimated_cov = model.covariance_
        estimated_inv_cov = model.precision_
    elif method == 'skggm':
        model = QuicGraphicalLasso(lam=parameters['alpha'], init_method='cov')
        # Fit the model to the data
        cov = parameters.get('cov', np.array([]))
        if len(cov) > 0:
            model.fit(ts.to_numpy(), covariance=cov)
        else:
            model.fit(ts.to_numpy())
        estimated_cov = model.covariance_
        estimated_inv_cov = model.precision_
    elif method == 'skggm_adaptive':
        model = AdaptiveGraphicalLasso(estimator=QuicGraphicalLasso(), method='binary')
        # Fit the model to the data
        model.fit(ts.to_numpy())
        estimated_cov = model.covariance_
        estimated_inv_cov = model.precision_
    elif method == 'ridge':
        lw = LedoitWolf()
        lw.fit(ts.to_numpy())
        estimated_cov = lw.covariance_
        estimated_inv_cov = lw.precision_
    elif method == 'shrink':
        # Shrunk covariance estimator with fixed shrinkage
        shrunk_cov = ShrunkCovariance(shrinkage=0.1)
        shrunk_cov.fit(ts.to_numpy())
        estimated_cov = shrunk_cov.covariance_
        estimated_inv_cov = shrunk_cov.precision_
    elif method == 'OAS':
        # Oracle Approximating Shrinkage (OAS)
        oas = OAS()
        oas.fit(ts.to_numpy())
        estimated_cov = oas.covariance_
        estimated_inv_cov = oas.precision_
    # elif method == 'MSCF':
    #
    #     get_precision_matrix()


    return estimated_cov, estimated_inv_cov


def   gaussian_log_likelihood(observations, mean, cov, inv_covariance=np.array([])):
    """
    Computes the log-likelihood of x under a Gaussian distribution with given mean and covariance matrix.

    Parameters:
    observations: list of numpy array
                  list of data points (1D arrays) for which to compute the log-likelihood.
    mean : numpy array
        The mean vector of the Gaussian distribution.
    cov : numpy array
        The covariance matrix of the Gaussian distribution.

    Returns:
    log_likelihood : float
        The log-likelihood of x.
    """
    # Dimensionality of the distribution
    k = len(mean)
    # number of observations
    nb_observations = len(observations)

    # Calculate the determinant and inverse of the covariance matrix
    # cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov) if len(inv_covariance) == 0 else inv_covariance

    # Compute the term (x - mean)
    exponent_term_sum = 0.
    for x in observations:
        diff = x - mean

        # Compute the quadratic form for the exponent
        exponent_term = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        exponent_term_sum += exponent_term

    # Log of the normalization constant

    sign, log_det = np.linalg.slogdet(cov)

    # log_norm_const = -0.5 * nb_observations * (k * np.log(2 * np.pi) + np.log(cov_det))
    log_norm_const = -0.5 * nb_observations * (k * np.log(2 * np.pi) + log_det)

    # Log-likelihood
    log_likelihood = log_norm_const + exponent_term_sum

    return log_likelihood


def kk_divergence_gaussian(true_precision, estimated_precision):
    """
    Calculate KL divergence between two multivariate Gaussian distributions
    defined by their precision (inverse covariance) matrices.

    Parameters:
    - true_precision (np.ndarray): The true precision matrix (p x p).
    - estimated_precision (np.ndarray): The estimated precision matrix (p x p).

    Returns:
    - float: The KL divergence between the two distributions.
    """
    # Calculate dimensions

    p = true_precision.shape[0]

    # Calculate trace of (Theta * hat(Theta)^{-1})
    trace_term = np.trace(true_precision @ inv(estimated_precision))

    # Calculate log determinant of (Theta * hat(Theta)^{-1})
    sign, log_det_term = slogdet(true_precision @ inv(estimated_precision))

    # Compute KL divergence
    kl_divergence = 0.5 * (trace_term - log_det_term - p)

    return kl_divergence


def kl_divergence_gaussian2(true_precision, estimated_precision):
    """
    Calculate KL divergence between two multivariate Gaussian distributions
    defined by their precision (inverse covariance) matrices.

    Parameters:
    - true_precision (np.ndarray): The true precision matrix (p x p).
    - estimated_precision (np.ndarray): The estimated precision matrix (p x p).

    Returns:
    - float: The KL divergence between the two distributions.
    """
    # Solve estimated_precision @ X = true_precision for X, avoiding explicit inversion
    solved_matrix = solve(estimated_precision, true_precision, assume_a='pos')  # More efficient than inv()

    # Calculate trace of the product
    trace_term = np.trace(solved_matrix)

    # Calculate log determinant using slogdet for numerical stability
    sign, log_det_solved = slogdet(solved_matrix)

    if sign <= 0:
        raise ValueError("Log determinant computation resulted in a non-positive sign.")

    # Dimensions of the matrices
    p = true_precision.shape[0]

    # Compute KL divergence
    kl_divergence = 0.5 * (trace_term - log_det_solved - p)

    return kl_divergence

def replace_outliers_with_mean(df):
    # Iterate over each column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()

            # Replace outliers (outside 2 standard deviations) with the mean
            df[col] = np.where(np.abs(df[col] - mean) > 2 * std, mean, df[col])

    return df


def precision_to_adjacency(precision_matrix, threshold=1e-6):
    # Create an adjacency matrix where non-zero elements correspond to edges
    adjacency_matrix = (np.abs(precision_matrix) > threshold).astype(int)

    # Remove diagonal elements (no self-loops in the adjacency matrix)
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix


import numpy as np


def eigenvalue_distribution_similarity(cov1, cov2):
    """
    Compute the eigenvalue distribution similarity (L2 norm) between two covariance matrices.

    Parameters:
    cov1 (numpy.ndarray): The first covariance matrix (true covariance matrix).
    cov2 (numpy.ndarray): The second covariance matrix (estimated covariance matrix).

    Returns:
    float: The L2 norm of the difference between the eigenvalue distributions of cov1 and cov2.
    """
    # Compute the eigenvalues of each covariance matrix
    eigenvalues1 = np.linalg.eigvalsh(cov1)
    eigenvalues2 = np.linalg.eigvalsh(cov2)

    # Compute the L2 norm of the eigenvalue difference
    similarity = np.linalg.norm(eigenvalues1 - eigenvalues2, ord=2)

    cond_number1, cond_number2 = max(eigenvalues1) / min(eigenvalues1), max(eigenvalues2) / min(eigenvalues2)

    return similarity, cond_number1, cond_number2

def csv_to_latex_table(top, csv_file, output_tex_file=None, columns=[], table_caption="Table", table_label="tab:table"):
    """
    Convert a CSV file to a LaTeX table.

    Parameters:
    - csv_file: Path to the input CSV file.
    - output_tex_file: Path to save the output LaTeX table code (optional).
    - table_caption: Caption for the table.
    - table_label: Label for referencing the table in LaTeX.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    del df['Unnamed: 0']
    if len(columns) > 0:
        df = df[columns]
    df.rename(columns={'n_sim': 'Samples'}, inplace=True)
    df['Samples'] = df['Samples'].round(0)
    # Start building the LaTeX table
    latex_code = r"\begin{table}[h!]" + "\n" \
                 r"\centering" + "\n" \
                 r"\begin{tabular}{|" + " | ".join(["c"] * len(df.columns)) + r"|}" + "\n" \
                 r"\multicolumn { "+ str(len(df.columns)) + "}{ c }{" + top + "}" + r" \\" + "\n" \
                 r"\hline" + "\n"

    # Add the column headers
    latex_code += " & ".join(df.columns) + r" \\" + "\n" + r"\hline" + "\n"

    # Add the rows
    for _, row in df.iterrows():
        latex_code += " & ".join(map(str, row)) + r" \\" + "\n"

    # Close the table
    latex_code += r"\hline" + "\n" \
                  r"\end{tabular}" + "\n" \
                  r"\end{table}"

    # Save the LaTeX code to a file, if specified
    if output_tex_file:
        with open(output_tex_file, "w") as f:
            f.write(latex_code)

    return latex_code

def csv_to_latex_table1(top, csv_file, csv_file_2, output_tex_file=None, columns=[], table_caption="Table", table_label="tab:table", include_begin_table=False):
    """
    Convert a CSV file to a LaTeX table.

    Parameters:
    - csv_file: Path to the input CSV file.
    - output_tex_file: Path to save the output LaTeX table code (optional).
    - table_caption: Caption for the table.
    - table_label: Label for referencing the table in LaTeX.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    del df['Unnamed: 0']
    if len(columns) > 0:
        df = df[columns]
    if 'n_sim' in df.columns:
        df['n_sim'] = df['n_sim'].astype(int)
    df.rename(columns={'n_sim': 'Samples'}, inplace=True)
    df = df.applymap(lambda x: f"{x:.3f}")
    # df = df.astype(str)

    df1 = pd.read_csv(csv_file_2)
    del df1['Unnamed: 0']
    if len(columns) > 0:
        df1 = df1[columns]
    df1.rename(columns={'n_sim': 'Samples'}, inplace=True)
    df1 = df1.applymap(lambda x: f"{x:.3f}")

    df = df + ' (' + df1 + ')'

    # Start building the LaTeX table
    if include_begin_table:
        latex_code = r"\begin{table}[h!]" + "\n" \
                                            r"\centering" + "\n"
    else:
        latex_code = ''
    latex_code += r"\begin{tabular}{|" + " | ".join(["c"] * len(df.columns)) + r"|}" + "\n" \
                 r"\multicolumn { "+ str(len(df.columns)) + "}{ c }{" + top + "}" + r" \\" + "\n" \
                 r"\hline" + "\n"

    # Add the column headers
    latex_code += " & ".join(df.columns) + r" \\" + "\n" + r"\hline" + "\n"

    # Add the rows
    for _, row in df.iterrows():
        latex_code += " & ".join(map(str, row)) + r"  \\" + "\n"

    # Close the table
    latex_code += r"\hline" + "\n" \
                  r"\end{tabular}" + "\n"
    if include_begin_table:
        latex_code += r"\end{table}"

    # Save the LaTeX code to a file, if specified
    if output_tex_file:
        with open(output_tex_file, "w") as f:
            f.write(latex_code)

    return latex_code

if __name__ == '__main__':

    cols =['n_sim', 'Sample', 'Lasso', 'Ridge', 'OAS', 'TMFG']

    latex0 = csv_to_latex_table('Independent', './log_likelyhood_dflarge_0.0_0.0_0.0.csv', columns=cols)
    latex1 = csv_to_latex_table1('Independent', './eigen_dist_dflarge_0.0_0.0_0.0.csv', './eigen_dist_std_dflarge_0.0_0.0_0.0.csv', columns=cols)
    latex2 = csv_to_latex_table1('1-factor', './sparsity0_s2.csv', './sparsity1_s22.csv')
    latex3 = csv_to_latex_table1('2-factors', './sparsity0_s3.csv', './sparsity1_s33.csv')
    latex4 = csv_to_latex_table1('3-factors','./sparsity0_s4.csv', './sparsity1_s4.csv')


    pass