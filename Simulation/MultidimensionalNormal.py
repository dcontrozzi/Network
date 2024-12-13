
from Simulation.CovarianceMatrics import CovarianceMatrix

import numpy as np

def generate_multidimensional_normal(mean, cov, size=1):
    """
    Generate multi-dimensional normal random variables.

    Parameters:
    mean : ndarray - Mean vector of the distribution.
    cov : ndarray - Covariance matrix of the distribution.
    size : int - Number of samples to generate.

    Returns:
    ndarray - Generated samples from the multivariate normal distribution.
    """
    return np.random.multivariate_normal(mean, cov, size)

# Example usage:
mean = [0, 0, 0]
cov = [[1, 0.5, 0.01], [0.5, 1, 0.2], [0.01, 0.2, 1.]]  # Covariance matrix
samples = generate_multidimensional_normal(mean, cov, size=100)

cov_matrix = np.cov(samples.T)

cov_diff = cov - cov_matrix
frobenious_distance = np.sqrt(np.sum(cov_diff ** 2))
print (CovarianceMatrix.frobenius_distance(np.array(cov), cov_matrix))
print(np.linalg.cond(cov_matrix))

pass
