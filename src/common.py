"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Simplified plot function - shows points and cluster means"""
    n, K = post.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    
    # Colors for clusters
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    
    # For each point, find which cluster it belongs to (hard assignment)
    assignments = np.zeros(n, dtype=int)
    for i in range(n):
        assignments[i] = np.argmax(post[i, :])
    
    # Plot points with colors based on cluster
    for j in range(K):
        # Get points in this cluster
        points_in_cluster = X[assignments == j]
        
        if len(points_in_cluster) > 0:
            # Plot the points
            ax.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1],
                      color=colors[j % len(colors)], s=30, alpha=0.6,
                      label=f'Cluster {j}', edgecolor='black', linewidth=0.5)
    
    # Plot cluster centers (means)
    for j in range(K):
        mu = mixture.mu[j]  # Center of cluster
        sigma = np.sqrt(mixture.var[j])  # Standard deviation
        
        # Plot the center as a big 'X'
        ax.scatter(mu[0], mu[1], color=colors[j % len(colors)],
                  s=200, marker='X', linewidths=3, edgecolor='black',
                  zorder=10)  # zorder=10 makes it appear on top
        
        # Draw a circle showing the spread (standard deviation)
        circle = plt.Circle(mu, sigma, color=colors[j % len(colors)],
                           fill=False, linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(circle)
    
    # Add labels and grid
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Make sure circles don't get cut off
    ax.set_xlim([X[:, 0].min() - 2, X[:, 0].max() + 2])
    ax.set_ylim([X[:, 1].min() - 2, X[:, 1].max() + 2])
    
    # Show the plot
    plt.tight_layout()
    plt.show()
def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    n,d = X.shape
    K = len(mixture.p)
    p = K*d+K+(K-1)
    bic_value = log_likelihood - 0.5 * p * np.log(n)
    return bic_value
    raise NotImplementedError
