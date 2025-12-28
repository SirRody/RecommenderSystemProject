try:
    from advanced_cf import run_advanced_methods
    ADVANCED_METHODS_AVAILABLE = True
except ImportError:
    ADVANCED_METHODS_AVAILABLE = False

###
import numpy as np
import matplotlib.pyplot as plt
from common import init, plot, bic
from kmeans import run as kmeans_run
from naive_em import run as em_run

def main():
    # 1. Load data
    X = np.loadtxt('toy_data.txt')
    
    # 2. Test values
    K_values = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]
    
    # 3. Run K-means (Part 1)
    print("=" * 50)
    print("K-MEANS RESULTS")
    print("=" * 50)
    
    kmeans_best_costs = {}
    
    for K in K_values:
        print(f"\nK = {K}:")
        best_cost = float('inf')
        best_kmeans_mixture = None
        best_kmeans_post = None
        
        for seed in seeds:
            mixture, post = init(X, K, seed)
            mixture, post, cost = kmeans_run(X, mixture, post)
            print(f"  Seed {seed}: cost = {cost:.4f}")
            
            if cost < best_cost:
                best_cost = cost
                best_kmeans_mixture = mixture
                best_kmeans_post = post
        
        kmeans_best_costs[K] = best_cost
        print(f"  Best cost: {best_cost:.4f}")
        
        # Plot K-means result
        plot(X, best_kmeans_mixture, best_kmeans_post, 
             title=f"K-means, K={K}, Cost={best_cost:.1f}")
        plt.savefig(f'kmeans_K={K}.png')
        plt.close()
    
    # 4. Run EM (Part 2)
    print("\n" + "=" * 50)
    print("EM ALGORITHM RESULTS")
    print("=" * 50)
    
    em_best_ll = {}
    
    for K in K_values:
        print(f"\nK = {K}:")
        best_ll = float('-inf')
        best_em_mixture = None
        best_em_post = None
        
        for seed in seeds:
            mixture, post = init(X, K, seed)
            mixture, post, ll = em_run(X, mixture, post)
            print(f"  Seed {seed}: log-likelihood = {ll:.4f}")
            
            if ll > best_ll:
                best_ll = ll
                best_em_mixture = mixture
                best_em_post = post
        
        em_best_ll[K] = best_ll
        print(f"  Best log-likelihood: {best_ll:.4f}")
        
        # Plot EM result
        plot(X, best_em_mixture, best_em_post,
             title=f"EM Algorithm, K={K}, LL={best_ll:.1f}")
        plt.savefig(f'em_K={K}.png')
        plt.close()
    
    # 5. Print final comparison
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    print("\nK | K-means (Cost) | EM (Log-Likelihood)")
    print("-" * 40)
    for K in K_values:
        print(f"{K} | {kmeans_best_costs[K]:14.4f} | {em_best_ll[K]:18.4f}")
    # 6. BIC Analysis (Part 5)
    print("\n" + "=" * 50)
    print("BIC ANALYSIS")
    print("=" * 50)
    
    n, d = X.shape
    print(f"Data: n = {n} points, d = {d} features")
    
    # We need to store the best EM mixtures, not just LL
    # Let's create a dictionary to store them
    em_best_mixtures = {}
    em_best_posts = {}
    
    # Re-run or store from earlier - let's store during EM run
    # Actually, we already have them! Let's modify the EM section slightly
    
    # For now, let's re-run EM quickly to store mixtures
    print("\nCalculating BIC for each K:")
    
    best_bic = float('-inf')
    best_K_bic = None
    
    for K in K_values:
        print(f"\nK = {K}:")
        
        # Find best EM result for this K (we already did this)
        # Let's find it again to get the mixture
        best_ll_for_K = float('-inf')
        best_mixture_for_K = None
        
        for seed in seeds:
            mixture, post = init(X, K, seed)
            mixture, post, ll = em_run(X, mixture, post)
            
            if ll > best_ll_for_K:
                best_ll_for_K = ll
                best_mixture_for_K = mixture
        
        # Calculate BIC
        bic_score = bic(X, best_mixture_for_K, best_ll_for_K)
        
        # Calculate number of parameters
        p = K * d + K + (K - 1)
        
        print(f"  Log-likelihood: {best_ll_for_K:.4f}")
        print(f"  Parameters (p): {p}")
        print(f"  BIC score: {bic_score:.4f}")
        
        # Track best BIC
        if bic_score > best_bic:
            best_bic = bic_score
            best_K_bic = K
    
    print("\n" + "=" * 50)
    print(f"OPTIMAL K FROM BIC: {best_K_bic}")
    print(f"BIC SCORE: {best_bic:.4f}")
    print("=" * 50)
    
    # Answer the question
    print(f"\nANSWER FOR PART 5:")
    print(f"Best K: {best_K_bic}")
    print(f"Corresponding BIC: {best_bic:.4f}")
    
    # Does BIC select correct number? (For toy data, usually K=2 or 3)
    if best_K_bic in [2, 3]:
        print("Yes, BIC selects the correct number of clusters for the toy data.")
    else:
        print("BIC might not select the optimal number for this toy data.")


from scipy.special import logsumexp
from em import run as em_run_complete

def run_netflix_em(K_values=[1, 12], seeds=[0, 1, 2, 3, 4], max_iter=100, tol=1e-6):
    """
    Run EM on Netflix incomplete data for given K values and seeds.
    Returns dict of best log likelihoods per K.
    """
    # Load Netflix data
    print("Loading Netflix incomplete data...")
    netflix_data = np.loadtxt('netflix_incomplete.txt')
    N, M = netflix_data.shape
    print(f"Data shape: {N} users × {M} movies")
    
    best_loglikelihoods = {}
    
    for K in K_values:
        print(f"\nRunning EM for K = {K}")
        best_ll = -np.inf
        best_mixture = None
        
        for seed in seeds:
            print(f"  Seed {seed}:", end=" ", flush=True)
            
            # Initialize using k-means init from common.py
            from common import init
            mixture, post = init(netflix_data, K, seed)
            
            # Run EM using the completed EM algorithm (from em.py)
            mixture, post, log_likelihood = em_run_complete(netflix_data, mixture, post)
            
            print(f"LL: {log_likelihood:.4f}")
            
            # Update best LL for this K
            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_mixture = mixture
        
        best_loglikelihoods[K] = best_ll
        print(f"Best log likelihood for K={K}: {best_ll:.6f}")
    
    return best_loglikelihoods


if __name__ == "__main__":
    main()
    
    # Run Netflix EM experiment
    print("\n" + "="*50)
    print("Netflix Data EM Experiment")
    print("="*50)
    
    netflix_results = run_netflix_em(K_values=[1, 12], seeds=[0, 1, 2, 3, 4])
    
    print("\nResults to report:")
    for K, ll in sorted(netflix_results.items()):
        print(f"K = {K}: Best log likelihood = {ll:.6f}")
#######
        # ========== Advanced Methods ==========
    if ADVANCED_METHODS_AVAILABLE:
        print("\n" + "="*50)
        print("Advanced Collaborative Filtering Methods")
        print("="*50)
        
        print("\nNote: These methods may take several minutes to run...")
        
        # Ask user if they want to run advanced methods
        response = input("Run advanced methods? (y/n): ").lower().strip()
        if response == 'y':
            run_advanced_methods()
    else:
        print("\n" + "="*50)
        print("Advanced Methods Not Available")
        print("="*50)
        print("\nTo run advanced methods, install:")
        print("pip install numpy scikit-learn matplotlib torch")
#####


        # ========== Test Predictions Against Gold Targets ==========
    print("\n" + "="*50)
    print("Testing Predictions Against Gold Targets")
    print("="*50)
    
    # Load the gold complete matrix
    X_gold = np.loadtxt('netflix_complete.txt')
    
    # Load the incomplete matrix
    X_incomplete = np.loadtxt('netflix_incomplete.txt')
    
    # Get the best mixture for K=12 (re-run or retrieve)
    # We need to run EM again for K=12 with the best seed (seed 1)
    print("\nRunning EM for K=12 with best seed (seed 1)...")
    from common import init
    from em import run as em_run_complete, fill_matrix
    from common import rmse
    
    # Initialize with seed 1 (which gave the best result)
    best_mixture, best_post = init(X_incomplete, K=12, seed=1)
    best_mixture, best_post, best_ll = em_run_complete(X_incomplete, best_mixture, best_post)
    print(f"Log-likelihood: {best_ll:.4f}")
    
    # Fill the matrix using the best mixture
    print("Filling missing entries in the incomplete matrix...")
    X_pred = fill_matrix(X_incomplete, best_mixture)
    
    # Calculate RMSE between predictions and gold targets
    error = rmse(X_gold, X_pred)
    print(f"\nRoot Mean Squared Error (RMSE): {error:.6f}")
    
    # Additional detail: RMSE on just the missing entries
    # (Since the observed entries should match exactly)
    mask_missing = (X_incomplete == 0)
    if np.any(mask_missing):
        rmse_missing = np.sqrt(np.mean((X_gold[mask_missing] - X_pred[mask_missing])**2))
        print(f"RMSE on missing entries only: {rmse_missing:.6f}")
        
        # Count how many entries were missing
        n_missing = np.sum(mask_missing)
        n_total = X_gold.size
        print(f"Missing entries: {n_missing:,} / {n_total:,} ({n_missing/n_total*100:.1f}%)")


    