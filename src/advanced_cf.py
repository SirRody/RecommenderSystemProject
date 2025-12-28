"""
Advanced Collaborative Filtering Methods
- Matrix Factorization
- Neural Network with Embeddings
- Neural Network with Additional Features
"""
import numpy as np
import time
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ========== 1. MATRIX FACTORIZATION ==========
class MatrixFactorizationRecommender:
    """Matrix Factorization using Alternating Least Squares"""
    
    def __init__(self, n_factors=20, reg=0.1, n_epochs=50):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, X, mask):
        """Fit matrix factorization using ALS"""
        n_users, n_items = X.shape
        
        # Initialize factors
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        
        # Convert to float32
        X = X.astype(np.float32)
        
        print("Training Matrix Factorization...")
        for epoch in range(self.n_epochs):
            # Update user factors
            for u in range(n_users):
                # Get items rated by user u
                rated_items = np.where(mask[u])[0]
                if len(rated_items) == 0:
                    continue
                
                # Solve for user factors: (V^T V + λI) u = V^T r
                V = self.item_factors[rated_items]
                R = X[u, rated_items]
                
                A = V.T @ V + self.reg * np.eye(self.n_factors)
                b = V.T @ R
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Update item factors
            for i in range(n_items):
                # Get users who rated item i
                rated_users = np.where(mask[:, i])[0]
                if len(rated_users) == 0:
                    continue
                
                # Solve for item factors: (U^T U + λI) v = U^T r
                U = self.user_factors[rated_users]
                R = X[rated_users, i]
                
                A = U.T @ U + self.reg * np.eye(self.n_factors)
                b = U.T @ R
                self.item_factors[i] = np.linalg.solve(A, b)
            
            # Compute loss
            if epoch % 10 == 0:
                pred = self.predict_all()
                train_loss = np.sqrt(mean_squared_error(X[mask], pred[mask]))
                print(f"  Epoch {epoch}: Train RMSE = {train_loss:.4f}")
    
    def predict_all(self):
        """Predict all ratings"""
        return self.user_factors @ self.item_factors.T
    
    def predict_missing(self, mask):
        """Predict only missing entries"""
        pred = self.predict_all()
        # Set observed entries to original values
        pred[mask] = X[mask]
        return pred

# ========== 2. NEURAL NETWORK WITH EMBEDDINGS ==========
class RatingDataset(Dataset):
    """Dataset for neural network training"""
    
    def __init__(self, X, mask):
        self.X = X
        self.mask = mask
        
        # Create list of (user, item, rating) tuples for observed ratings
        self.samples = []
        for u in range(X.shape[0]):
            for i in range(X.shape[1]):
                if mask[u, i]:
                    self.samples.append((u, i, X[u, i]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        u, i, r = self.samples[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(r, dtype=torch.float32)

class NeuralMF(nn.Module):
    """Neural Matrix Factorization"""
    
    def __init__(self, n_users, n_items, n_factors=20, hidden_dims=[64, 32]):
        super().__init__()
        
        # Embedding layers
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
        
        # Fully connected layers
        layers = []
        input_dim = n_factors * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
    
    def forward(self, user, item):
        user_emb = self.user_embed(user)
        item_emb = self.item_embed(item)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(x).squeeze()

class NeuralRecommender:
    """Neural Network Recommender"""
    
    def __init__(self, n_users, n_items, n_factors=20, lr=0.001, batch_size=256):
        self.model = NeuralMF(n_users, n_items, n_factors)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        
    def fit(self, X, mask, n_epochs=20):
        """Train neural network"""
        dataset = RatingDataset(X, mask)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        print("Training Neural Network...")
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in dataloader:
                user, item, rating = batch
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                
                self.optimizer.zero_grad()
                pred = self.model(user, item)
                loss = self.criterion(pred, rating)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(user)
            
            avg_loss = np.sqrt(total_loss / len(dataset))
            print(f"  Epoch {epoch}: Train RMSE = {avg_loss:.4f}")
    
    def predict_all(self, X):
        """Predict all ratings"""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        n_users, n_items = X.shape
        predictions = np.zeros((n_users, n_items))
        
        with torch.no_grad():
            # Batch predictions
            batch_size = 1024
            for u in range(0, n_users, batch_size):
                for i in range(0, n_items, batch_size):
                    u_end = min(u + batch_size, n_users)
                    i_end = min(i + batch_size, n_items)
                    
                    # Create grid of user-item pairs
                    users = torch.arange(u, u_end).repeat_interleave(i_end - i)
                    items = torch.arange(i, i_end).repeat(u_end - u)
                    
                    users, items = users.to(device), items.to(device)
                    pred = self.model(users, items)
                    
                    predictions[u:u_end, i:i_end] = pred.cpu().numpy().reshape(u_end - u, i_end - i)
        
        return predictions

# ========== 3. SKLEARN NMF (SIMPLE BASELINE) ==========
def sklearn_nmf_predict(X, mask, n_components=20):
    """Simple NMF using scikit-learn"""
    print("Running scikit-learn NMF...")
    
    # Fill missing values with column means for initialization
    X_filled = X.copy()
    col_means = np.nanmean(np.where(mask, X, np.nan), axis=0)
    X_filled[~mask] = np.take(col_means, np.where(~mask)[1])
    
    # Apply NMF
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=50)
    W = model.fit_transform(X_filled)
    H = model.components_
    
    # Reconstruct matrix
    X_pred = W @ H
    
    # Keep observed values
    X_pred[mask] = X[mask]
    
    return X_pred

# ========== MAIN EXPERIMENT FUNCTION ==========
def run_advanced_methods():
    """Run all advanced collaborative filtering methods"""
    
    # Load data
    print("Loading Netflix data...")
    X_incomplete = np.loadtxt('netflix_incomplete.txt')
    X_complete = np.loadtxt('netflix_complete.txt')
    
    n_users, n_items = X_incomplete.shape
    mask = (X_incomplete != 0)  # True for observed ratings
    
    print(f"Data: {n_users} users × {n_items} movies")
    print(f"Observed ratings: {mask.sum():,} / {mask.size:,} ({mask.sum()/mask.size*100:.1f}%)")
    
    results = {}
    
    # Method 1: Matrix Factorization (ALS)
    print("\n" + "="*50)
    print("METHOD 1: Matrix Factorization (ALS)")
    print("="*50)
    
    start_time = time.time()
    mf_model = MatrixFactorizationRecommender(n_factors=20, n_epochs=30)
    mf_model.fit(X_incomplete, mask)
    X_pred_mf = mf_model.predict_all()
    X_pred_mf[mask] = X_incomplete[mask]  # Keep observed values
    
    mf_rmse = np.sqrt(mean_squared_error(X_complete, X_pred_mf))
    mf_time = time.time() - start_time
    
    results['MF (ALS)'] = {
        'RMSE': mf_rmse,
        'Time': mf_time,
        'Predictions': X_pred_mf
    }
    
    print(f"Matrix Factorization RMSE: {mf_rmse:.6f}")
    print(f"Training time: {mf_time:.1f} seconds")
    
    # Method 2: Neural Network
    print("\n" + "="*50)
    print("METHOD 2: Neural Matrix Factorization")
    print("="*50)
    
    start_time = time.time()
    nn_model = NeuralRecommender(n_users, n_items, n_factors=20, lr=0.001)
    nn_model.fit(X_incomplete, mask, n_epochs=15)
    X_pred_nn = nn_model.predict_all(X_incomplete)
    X_pred_nn[mask] = X_incomplete[mask]  # Keep observed values
    
    nn_rmse = np.sqrt(mean_squared_error(X_complete, X_pred_nn))
    nn_time = time.time() - start_time
    
    results['Neural MF'] = {
        'RMSE': nn_rmse,
        'Time': nn_time,
        'Predictions': X_pred_nn
    }
    
    print(f"Neural Network RMSE: {nn_rmse:.6f}")
    print(f"Training time: {nn_time:.1f} seconds")
    
    # Method 3: scikit-learn NMF (simple baseline)
    print("\n" + "="*50)
    print("METHOD 3: scikit-learn NMF (Baseline)")
    print("="*50)
    
    start_time = time.time()
    X_pred_nmf = sklearn_nmf_predict(X_incomplete, mask, n_components=20)
    
    nmf_rmse = np.sqrt(mean_squared_error(X_complete, X_pred_nmf))
    nmf_time = time.time() - start_time
    
    results['NMF (sklearn)'] = {
        'RMSE': nmf_rmse,
        'Time': nmf_time,
        'Predictions': X_pred_nmf
    }
    
    print(f"scikit-learn NMF RMSE: {nmf_rmse:.6f}")
    print(f"Training time: {nmf_time:.1f} seconds")
    
    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON OF ALL METHODS")
    print("="*50)
    
    print("\nMethod               | RMSE      | Time (s)  ")
    print("-" * 40)
    for method_name, result in results.items():
        print(f"{method_name:20} | {result['RMSE']:.6f} | {result['Time']:.1f}")
    
    # Compare with EM result (from previous experiment)
    print(f"\nEM (K=12)             | ~1.006371 | ~120.0   ")
    
    # Plot comparison
    plot_comparison(results, X_incomplete, X_complete)
    
    return results

def plot_comparison(results, X_incomplete, X_complete):
    """Plot comparison of different methods"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Original incomplete matrix
    axes[0, 0].imshow(X_incomplete, aspect='auto', cmap='viridis', vmin=1, vmax=5)
    axes[0, 0].set_title('Original (Incomplete)')
    axes[0, 0].set_xlabel('Movies')
    axes[0, 0].set_ylabel('Users')
    
    # Plot 2-4: Predictions from each method
    methods = list(results.keys())
    for idx, method in enumerate(methods):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(results[method]['Predictions'], aspect='auto', 
                             cmap='viridis', vmin=1, vmax=5)
        axes[row, col].set_title(f'{method}\nRMSE: {results[method]["RMSE"]:.4f}')
        axes[row, col].set_xlabel('Movies')
        axes[row, col].set_ylabel('Users')
    
    # Plot 5: Complete matrix (ground truth)
    axes[1, 2].imshow(X_complete, aspect='auto', cmap='viridis', vmin=1, vmax=5)
    axes[1, 2].set_title('Ground Truth (Complete)')
    axes[1, 2].set_xlabel('Movies')
    axes[1, 2].set_ylabel('Users')
    
    plt.tight_layout()
    plt.savefig('advanced_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    rmses = [results[m]['RMSE'] for m in methods]
    
    bars = plt.bar(methods, rmses, color=['blue', 'green', 'orange'])
    
    # Add EM result for comparison
    plt.axhline(y=1.006371, color='red', linestyle='--', label='EM (K=12)')
    
    plt.xlabel('Method')
    plt.ylabel('RMSE (lower is better)')
    plt.title('RMSE Comparison of Collaborative Filtering Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add values on top of bars
    for bar, rmse in zip(bars, rmses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rmse:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rmse_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Install required packages if needed
    required = ['numpy', 'scikit-learn', 'matplotlib', 'torch']
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install", " ".join(missing))
        print("\nIf you don't have PyTorch, install it from: https://pytorch.org/get-started/locally/")
    else:
        results = run_advanced_methods()