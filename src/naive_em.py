"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = len(mixture.p)  # Number of clusters
    
    # Step 1: Calculate probabilities for each point under each Gaussian
    post = np.zeros((n, K))
    
    for j in range(K):
        # Calculate Gaussian probability for cluster j
        # Using multivariate Gaussian formula
        diff = X - mixture.mu[j]
        exponent = -0.5 * np.sum(diff**2, axis=1) / mixture.var[j]
        normalizer = 1.0 / np.sqrt((2 * np.pi * mixture.var[j]) ** d)
        prob = normalizer * np.exp(exponent)
        
        # Multiply by mixing proportion
        post[:, j] = mixture.p[j] * prob
    
    # Step 2: Normalize so each row sums to 1
    row_sums = post.sum(axis=1, keepdims=True)
    post = post / row_sums
    
    # Step 3: Calculate log-likelihood
    log_likelihood = np.sum(np.log(row_sums))
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> Tuple[GaussianMixture, float]:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
       of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    # Calculate effective number of points in each cluster
    n_j = post.sum(axis=0)  # (K,) array
    
    # Update mixing proportions
    p = n_j / n
    
    # Update means
    mu = np.zeros((K, d))
    for j in range(K):
        # Weighted average: sum(soft_assignments * data) / total_soft_assignments
        mu[j] = np.sum(post[:, j:j+1] * X, axis=0) / n_j[j]
    
    # Update variances
    var = np.zeros(K)
    for j in range(K):
        # Calculate squared distances from mean
        diff = X - mu[j]  # (n, d)
        squared_dist = np.sum(diff**2, axis=1)  # (n,)
        
        # Weighted average of squared distances
        var[j] = np.sum(post[:, j] * squared_dist) / (d * n_j[j])
    
    return GaussianMixture(mu, var, p)

    
def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        mixture: initial Gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    current_LL = None
    
    # Start with initial E-step to ensure post matches current mixture
    post, current_LL = estep(X, mixture)
    
    # EM iteration loop
    while True:
        # Store previous LL for convergence check
        prev_LL = current_LL
        
        # M-step: update mixture parameters
        mixture = mstep(X, post)
        
        # E-step: update posterior probabilities
        post, current_LL = estep(X, mixture)
        
        # Check convergence
        if prev_LL is not None:
            improvement = current_LL - prev_LL
            if improvement <= 1e-6 * abs(current_LL):
                break
    
    return mixture, post, current_LL
    raise NotImplementedError


