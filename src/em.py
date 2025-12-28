"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    n, d = X.shape
    K = len(mixture.p)
    mask = (X != 0)  # missing data encoded 0
    
    log_post = np.zeros((n, K))
    
    for j in range(K):
        for u in range(n):
            rated_movies = mask[u]
            if np.sum(rated_movies) > 0:
                available_ratings = X[u, rated_movies]
                mu_available = mixture.mu[j, rated_movies]
                
                diff = available_ratings - mu_available
                log_prob = -0.5 * np.sum(diff**2) / mixture.var[j]
                log_prob -= 0.5 * np.sum(rated_movies) * np.log(2 * np.pi * mixture.var[j])
                
                log_post[u, j] = np.log(mixture.p[j]) + log_prob
            else:
                log_post[u, j] = np.log(mixture.p[j])  # No rating information
    
    # LogSumExp for normalization
    max_log = np.max(log_post, axis=1, keepdims=True)
    log_post_normalized = log_post - max_log
    post = np.exp(log_post_normalized)
    
    # Normalize to get probabilities
    row_sums = post.sum(axis=1, keepdims=True)
    post = post / row_sums
    
    # Compute log-likelihood
    log_likelihood = np.sum(max_log + np.log(row_sums))
    
    return post, log_likelihood
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    K = post.shape[1]
    mask = (X != 0)
    
    # Mixing proportions
    n_j = post.sum(axis=0)
    p = n_j / n
    
    # Update means with support check
    mu = mixture.mu.copy()
    
    for j in range(K):
        for i in range(d):
            users_rated_i = mask[:, i]
            support = np.sum(post[users_rated_i, j])
            if support >= 1:
                mu[j, i] = np.sum(post[users_rated_i, j] * X[users_rated_i, i]) / support
    
    # Update variances
    var = np.zeros(K)
    for j in range(K):
        # For each user, compute SSE for their rated movies
        squared_errors = np.zeros(n)
        rating_counts = np.zeros(n)
        
        for u in range(n):
            rated = mask[u]
            if np.any(rated):
                diff = X[u, rated] - mu[j, rated]
                squared_errors[u] = np.sum(diff**2)
                rating_counts[u] = np.sum(rated)
        
        total_sse = np.sum(post[:, j] * squared_errors)
        total_ratings = np.sum(post[:, j] * rating_counts)
        
        var[j] = max(total_sse / total_ratings, min_variance) if total_ratings > 0 else min_variance
    
    return GaussianMixture(mu, var, p)
    raise NotImplementedError

def run(X: np.ndarray, mixture: GaussianMixture,
       post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # Deep copy X to avoid altering original data
    X_copy = X.copy()
    
    prev_LL = None
    current_LL = None
    
    # Start with initial E-step to ensure post matches current mixture
    post, current_LL = estep(X_copy, mixture)
    
    # EM iteration loop
    while True:
        # Store previous LL for convergence check
        prev_LL = current_LL
        
        # M-step: update mixture parameters
        mixture = mstep(X_copy, post, mixture)
        
        # E-step: update posterior probabilities
        post, current_LL = estep(X_copy, mixture)
        
        # Check convergence
        if prev_LL is not None:
            improvement = current_LL - prev_LL
            if improvement <= 1e-6 * abs(current_LL):
                break
    
    return mixture, post, current_LL

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    n, d = X.shape
    X_pred = X.copy()
    K = len(mixture.p)
    
    # Mask for missing entries
    mask = (X == 0)
    
    # Calculate posterior probabilities using log-space
    log_post = np.zeros((n, K))
    
    for j in range(K):
        for u in range(n):
            rated_movies = ~mask[u]
            if np.any(rated_movies):
                available_X = X[u, rated_movies]
                available_mu = mixture.mu[j, rated_movies]
                
                diff = available_X - available_mu
                log_prob = -0.5 * np.sum(diff**2) / mixture.var[j]
                log_prob -= 0.5 * np.sum(rated_movies) * np.log(2 * np.pi * mixture.var[j])
                log_post[u, j] = np.log(mixture.p[j]) + log_prob
            else:
                log_post[u, j] = np.log(mixture.p[j])
    
    # Normalize to get probabilities
    max_log = np.max(log_post, axis=1, keepdims=True)
    log_post_normalized = np.clip(log_post - max_log, -700, None)  # Prevent underflow
    post = np.exp(log_post_normalized)
    post = post / post.sum(axis=1, keepdims=True)
    
    # Fill missing entries - vectorized version
    # For each missing position, take weighted average of cluster means
    for u in range(n):
        missing_indices = mask[u]
        if np.any(missing_indices):
            # post[u] is (K,), mixture.mu[:, missing_indices] is (K, num_missing)
            # Weighted average: post[u] @ mixture.mu[:, missing_indices]
            X_pred[u, missing_indices] = post[u] @ mixture.mu[:, missing_indices]
    
    return X_pred