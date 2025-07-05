import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from scipy.stats import zscore
from sklearn.decomposition import PCA
import random
import time
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class StockMarketSimulator:
    """
    Simulates a stock market with characteristic-based factor structure
    as described in the paper.
    """
    def __init__(self, N=100, T=120, n_characteristics=10, n_factors=3, noise_level=0.1):
        """
        Parameters:
        - N: Number of stocks
        - T: Number of time periods
        - n_characteristics: Number of stock characteristics
        - n_factors: Number of latent factors driving returns
        - noise_level: Standard deviation of idiosyncratic noise
        """
        self.N = N
        self.T = T
        self.n_characteristics = n_characteristics
        self.n_factors = n_factors
        self.noise_level = noise_level
        
        # Initialize parameters
        self.initialize_market()
        
    def initialize_market(self):
        """Initialize stock characteristics and factor loadings"""
        # Generate stock characteristics: N stocks × n_characteristics
        self.characteristics = np.random.randn(self.N, self.n_characteristics)
        
        # Generate factor loadings: n_characteristics × n_factors (random non-linear mapping)
        self.factor_loadings = np.random.randn(self.n_characteristics, self.n_factors)
        
        # Create some non-linearities in how characteristics map to factors
        self.non_linear_params = {
            'weights': np.random.randn(self.n_characteristics, self.n_factors),
            'biases': np.random.randn(self.n_factors)
        }
        
        # Generate time-varying factor returns
        self.factor_returns = np.random.randn(self.T, self.n_factors) * 0.1
        
        # True SDF coefficients (market prices of risk)
        self.true_sdf_coef = np.random.randn(self.n_factors)
        self.true_sdf_coef = self.true_sdf_coef / np.linalg.norm(self.true_sdf_coef)  # Normalize
        
    def non_linear_factor_mapping(self, X):
        """Apply non-linear transformation to map characteristics to factor loadings"""
        # Simple non-linear transformation: tanh(X @ weights + biases)
        return np.tanh(X @ self.non_linear_params['weights'] + self.non_linear_params['biases'])
    
    def generate_returns(self):
        """Generate stock returns based on factor structure"""
        # Compute factor betas using non-linear mapping of characteristics
        factor_betas = self.non_linear_factor_mapping(self.characteristics)
        
        # Expected returns: betas × factor risk premia
        expected_returns = factor_betas @ self.true_sdf_coef
        
        # Generate actual returns: expected returns + factor exposures + idiosyncratic noise
        returns = np.zeros((self.T, self.N))
        for t in range(self.T):
            # Factor component
            factor_component = factor_betas @ self.factor_returns[t]
            
            # Idiosyncratic noise
            idiosyncratic_noise = np.random.randn(self.N) * self.noise_level
            
            # Combine components
            returns[t] = expected_returns + factor_component + idiosyncratic_noise
        
        return returns, factor_betas, expected_returns

class ShallowNetworkSDF(nn.Module):
    """
    A shallow neural network for estimating the SDF.
    This represents a simple one-hidden-layer network.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super(ShallowNetworkSDF, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class DeepNetworkSDF(nn.Module):
    """
    A deep neural network for estimating the SDF.
    This can have an arbitrary number of hidden layers.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=10):
        super(DeepNetworkSDF, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class NTKEstimator:
    """
    Implementation of the Neural Tangent Kernel approach for SDF estimation
    as described in the paper.
    """
    def __init__(self, kernel_type='rbf', gamma=1.0, depth=1):
        """
        Parameters:
        - kernel_type: Type of kernel to use ('rbf', 'polynomial', 'linear')
        - gamma: Kernel parameter
        - depth: Approximation of neural network depth for the kernel
        """
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.depth = depth
        self.weights = None
    
    def compute_kernel(self, X, Y=None):
        """Compute the kernel matrix between X and Y"""
        if Y is None:
            Y = X
        
        if self.kernel_type == 'rbf':
            # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            K = np.exp(-self.gamma * (X_norm + Y_norm - 2 * np.dot(X, Y.T)))
        elif self.kernel_type == 'polynomial':
            # Polynomial kernel: K(x, y) = (gamma * x^T y + 1)^depth
            K = (self.gamma * np.dot(X, Y.T) + 1) ** self.depth
        elif self.kernel_type == 'linear':
            # Linear kernel: K(x, y) = x^T y
            K = np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def compute_portfolio_kernel(self, X, Y, R_X, R_Y=None):
        """
        Compute the Portfolio Tangent Kernel (PTK) as described in the paper
        
        Parameters:
        - X, Y: Characteristics matrices
        - R_X, R_Y: Return matrices
        """
        if R_Y is None:
            R_Y = R_X
            
        # Compute the kernel between characteristics
        K_X_Y = self.compute_kernel(X, Y)
        
        # Compute the Portfolio Tangent Kernel
        PTK = R_X.T @ K_X_Y @ R_Y
        
        return PTK
    
    def fit(self, X, R, ridge_penalty=1e-5):
        """
        Fit the NTK model to maximize Sharpe ratio
        
        Parameters:
        - X: Stock characteristics (T x N x d)
        - R: Stock returns (T x N)
        - ridge_penalty: Regularization parameter
        """
        T = X.shape[0]
        
        # Reshape X to (T*N, d)
        X_flat = X.reshape(-1, X.shape[-1])
        
        # Compute the Portfolio Tangent Kernel matrix
        K = np.zeros((T, T))
        for t1 in range(T):
            for t2 in range(T):
                K[t1, t2] = self.compute_portfolio_kernel(
                    X[t1], X[t2], R[t1].reshape(-1, 1), R[t2].reshape(-1, 1)
                )
        
        # Add ridge penalty
        K_reg = K + ridge_penalty * np.eye(T)
        
        # Solve for weights using ridge regression
        ones = np.ones(T)
        self.weights = np.linalg.solve(K_reg, ones)
        
        return self
    
    def predict(self, X_new, R_new, X_train, R_train):
        """
        Predict using the fitted NTK model
        
        Parameters:
        - X_new: New stock characteristics (N x d)
        - R_new: New stock returns (N)
        - X_train: Training stock characteristics (T x N x d)
        - R_train: Training stock returns (T x N)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
        
        T = X_train.shape[0]
        
        # Compute the cross-kernel between new data and training data
        K_cross = np.zeros(T)
        for t in range(T):
            K_cross[t] = self.compute_portfolio_kernel(
                X_new, X_train[t], R_new.reshape(-1, 1), R_train[t].reshape(-1, 1)
            )
        
        # Predict using the kernel weights
        prediction = K_cross @ self.weights
        
        return prediction

def train_neural_network_sdf(model, X_train, R_train, epochs=1000, lr=0.01, batch_size=32):
    """
    Train a neural network to maximize the Sharpe ratio of the SDF
    
    Parameters:
    - model: PyTorch neural network model
    - X_train: Stock characteristics (T x N x d)
    - R_train: Stock returns (T x N)
    - epochs: Number of training epochs
    - lr: Learning rate
    - batch_size: Batch size
    
    Returns:
    - trained model
    - training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Reshape inputs for batching
    T, N, d = X_train.shape
    X_flat = X_train.reshape(T*N, d)
    R_flat = R_train.reshape(T*N, 1)
    
    # Create tensor dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_flat),
        torch.FloatTensor(R_flat)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    history = {
        'loss': [],
        'sharpe_ratio': []
    }
    
    for epoch in range(epochs):
        epoch_losses = []
        sdf_values = []
        
        for X_batch, R_batch in dataloader:
            # Forward pass
            sdf_batch = model(X_batch)
            
            # Store SDF values for Sharpe ratio calculation
            sdf_values.append(sdf_batch.detach().numpy())
            
            # Compute loss: negative of Sharpe ratio
            # Maximize Sharpe ratio = minimize negative Sharpe ratio
            mean_sdf = torch.mean(1 - sdf_batch * R_batch)
            std_sdf = torch.std(1 - sdf_batch * R_batch)
            loss = -mean_sdf / (std_sdf + 1e-8)  # Add small constant to avoid division by zero
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Compute Sharpe ratio for the epoch
        all_sdf_values = np.concatenate(sdf_values)
        sdf_mean = np.mean(all_sdf_values)
        sdf_std = np.std(all_sdf_values)
        sharpe_ratio = sdf_mean / (sdf_std + 1e-8)
        
        # Store metrics
        history['loss'].append(np.mean(epoch_losses))
        history['sharpe_ratio'].append(sharpe_ratio)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}, Sharpe Ratio = {sharpe_ratio:.4f}")
    
    return model, history

def linear_sdf(X, beta):
    """
    Linear SDF model: SDF = X @ beta
    
    Parameters:
    - X: Stock characteristics (T x N x d)
    - beta: Linear coefficients (d)
    
    Returns:
    - SDF values (T x N)
    """
    return X @ beta

def evaluate_sdf(sdf_values, returns, true_sdf=None):
    """
    Evaluate the performance of an SDF
    
    Parameters:
    - sdf_values: Estimated SDF values (T x N)
    - returns: Stock returns (T x N)
    - true_sdf: True SDF values (T x N) if available
    
    Returns:
    - Dictionary of performance metrics
    """
    # Compute pricing errors
    pricing_errors = 1 - sdf_values * returns
    
    # Compute Sharpe ratio
    sharpe_ratio = np.mean(pricing_errors) / np.std(pricing_errors)
    
    # Compute correlation with true SDF if available
    if true_sdf is not None:
        sdf_correlation = np.corrcoef(sdf_values.flatten(), true_sdf.flatten())[0, 1]
    else:
        sdf_correlation = None
    
    # Compute R-squared for returns
    y_mean = np.mean(returns)
    ss_tot = np.sum((returns - y_mean) ** 2)
    ss_res = np.sum(pricing_errors ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'mean_pricing_error': np.mean(np.abs(pricing_errors)),
        'std_pricing_error': np.std(pricing_errors),
        'sdf_correlation': sdf_correlation,
        'r_squared': r_squared
    }

def run_simulation(n_stocks=100, n_time_periods=120, n_characteristics=10, n_factors=3, 
                   test_size=24, hidden_dim=32, deep_layers=10, epochs=500):
    """
    Run a complete simulation to compare different SDF models
    
    Parameters:
    - n_stocks: Number of stocks
    - n_time_periods: Number of time periods
    - n_characteristics: Number of stock characteristics
    - n_factors: Number of latent factors
    - test_size: Number of periods to use for testing
    - hidden_dim: Hidden dimension for neural networks
    - deep_layers: Number of hidden layers in deep network
    - epochs: Training epochs for neural networks
    
    Returns:
    - Dictionary of results
    """
    # Initialize simulator
    simulator = StockMarketSimulator(
        N=n_stocks, 
        T=n_time_periods, 
        n_characteristics=n_characteristics,
        n_factors=n_factors
    )
    
    # Generate data
    returns, factor_betas, expected_returns = simulator.generate_returns()
    
    # Split into train/test
    train_returns = returns[:n_time_periods-test_size]
    test_returns = returns[n_time_periods-test_size:]
    
    # Create 3D array of characteristics for each time period
    # In a real setting, characteristics would be time-varying
    # For simplicity, we use the same characteristics but add small time-varying noise
    characteristics = np.zeros((n_time_periods, n_stocks, n_characteristics))
    for t in range(n_time_periods):
        noise = np.random.randn(n_stocks, n_characteristics) * 0.05
        characteristics[t] = simulator.characteristics + noise
    
    train_characteristics = characteristics[:n_time_periods-test_size]
    test_characteristics = characteristics[n_time_periods-test_size:]
    
    # Compute true SDF values
    true_sdf = np.zeros((n_time_periods, n_stocks))
    for t in range(n_time_periods):
        factor_loadings_t = simulator.non_linear_factor_mapping(characteristics[t])
        true_sdf[t] = factor_loadings_t @ simulator.true_sdf_coef
    
    train_true_sdf = true_sdf[:n_time_periods-test_size]
    test_true_sdf = true_sdf[n_time_periods-test_size:]
    
    # Model 1: Linear SDF using PCA on characteristics
    pca = PCA(n_components=min(n_factors*2, n_characteristics))
    X_flat = train_characteristics.reshape(-1, n_characteristics)
    X_pca = pca.fit_transform(X_flat)
    X_pca = X_pca.reshape(n_time_periods-test_size, n_stocks, -1)
    
    # Compute linear SDF coefficients
    X_pca_flat = X_pca.reshape(-1, X_pca.shape[-1])
    R_flat = train_returns.reshape(-1, 1)
    
    # Use ridge regression to estimate linear coefficients
    ridge_lambda = 0.1
    XtX = X_pca_flat.T @ X_pca_flat + ridge_lambda * np.eye(X_pca.shape[-1])
    XtY = X_pca_flat.T @ R_flat
    linear_coef = np.linalg.solve(XtX, XtY).flatten()
    
    # Evaluate linear model on test set
    X_test_pca = pca.transform(test_characteristics.reshape(-1, n_characteristics))
    X_test_pca = X_test_pca.reshape(test_size, n_stocks, -1)
    linear_sdf_values = np.zeros((test_size, n_stocks))
    for t in range(test_size):
        linear_sdf_values[t] = X_test_pca[t] @ linear_coef
    
    linear_results = evaluate_sdf(linear_sdf_values, test_returns, test_true_sdf)
    
    # Model 2: Shallow Neural Network
    shallow_model = ShallowNetworkSDF(input_dim=n_characteristics, hidden_dim=hidden_dim)
    shallow_model, shallow_history = train_neural_network_sdf(
        shallow_model, 
        train_characteristics, 
        train_returns, 
        epochs=epochs
    )
    
    # Evaluate shallow model on test set
    shallow_sdf_values = np.zeros((test_size, n_stocks))
    for t in range(test_size):
        X_tensor = torch.FloatTensor(test_characteristics[t])
        with torch.no_grad():
            sdf_t = shallow_model(X_tensor).numpy()
        shallow_sdf_values[t] = sdf_t.flatten()
    
    shallow_results = evaluate_sdf(shallow_sdf_values, test_returns, test_true_sdf)
    
    # Model 3: Deep Neural Network
    deep_model = DeepNetworkSDF(
        input_dim=n_characteristics, 
        hidden_dim=hidden_dim, 
        num_layers=deep_layers
    )
    deep_model, deep_history = train_neural_network_sdf(
        deep_model, 
        train_characteristics, 
        train_returns, 
        epochs=epochs
    )
    
    # Evaluate deep model on test set
    deep_sdf_values = np.zeros((test_size, n_stocks))
    for t in range(test_size):
        X_tensor = torch.FloatTensor(test_characteristics[t])
        with torch.no_grad():
            sdf_t = deep_model(X_tensor).numpy()
        deep_sdf_values[t] = sdf_t.flatten()
    
    deep_results = evaluate_sdf(deep_sdf_values, test_returns, test_true_sdf)
    
    # Model 4: NTK-based SDF
    ntk_model = NTKEstimator(kernel_type='rbf', gamma=0.1, depth=deep_layers)
    ntk_model.fit(train_characteristics, train_returns, ridge_penalty=0.01)
    
    # Evaluate NTK model on test set
    ntk_sdf_values = np.zeros((test_size, n_stocks))
    for t in range(test_size):
        ntk_sdf_values[t] = ntk_model.predict(
            test_characteristics[t], 
            test_returns[t], 
            train_characteristics, 
            train_returns
        )
    
    ntk_results = evaluate_sdf(ntk_sdf_values, test_returns, test_true_sdf)
    
    # Compile results
    results = {
        'linear': linear_results,
        'shallow_nn': shallow_results,
        'deep_nn': deep_results,
        'ntk': ntk_results,
        'shallow_history': shallow_history,
        'deep_history': deep_history
    }
    
    return results

def plot_results(results):
    """
    Plot comparison of different SDF models
    
    Parameters:
    - results: Dictionary of results from run_simulation
    """
    # Plot Sharpe ratios
    metrics = ['sharpe_ratio', 'mean_pricing_error', 'sdf_correlation', 'r_squared']
    models = ['linear', 'shallow_nn', 'deep_nn', 'ntk']
    model_names = ['Linear', 'Shallow NN', 'Deep NN', 'NTK']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] if results[model][metric] is not None else 0 for model in models]
        axs[i].bar(model_names, values)
        axs[i].set_title(metric.replace('_', ' ').title())
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sdf_model_comparison.png')
    plt.show()
    
    # Plot training history
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    axs[0].plot(results['shallow_history']['loss'], label='Shallow NN')
    axs[0].plot(results['deep_history']['loss'], label='Deep NN')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(linestyle='--', alpha=0.7)
    
    # Plot Sharpe ratio
    axs[1].plot(results['shallow_history']['sharpe_ratio'], label='Shallow NN')
    axs[1].plot(results['deep_history']['sharpe_ratio'], label='Deep NN')
    axs[1].set_title('Training Sharpe Ratio')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Sharpe Ratio')
    axs[1].legend()
    axs[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def explore_depth_impact(max_depth=10, n_characteristics=10, n_factors=3):
    """
    Explore the impact of neural network depth on SDF performance
    
    Parameters:
    - max_depth: Maximum number of hidden layers to test
    - n_characteristics: Number of stock characteristics
    - n_factors: Number of latent factors
    
    Returns:
    - Results dictionary
    """
    depths = [1, 2, 4, 8, max_depth]  # Testing powers of 2 as in the paper
    sharpe_ratios = []
    correlations = []
    
    for depth in depths:
        print(f"Testing depth: {depth}")
        
        # Run simulation with specified depth
        results = run_simulation(
            n_characteristics=n_characteristics,
            n_factors=n_factors,
            deep_layers=depth,
            epochs=300  # Reduced epochs for faster execution
        )
        
        # Store metrics
        sharpe_ratios.append(results['deep_nn']['sharpe_ratio'])
        correlations.append(results['deep_nn']['sdf_correlation'])
    
    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].plot(depths, sharpe_ratios, 'o-')
    axs[0].set_title('Sharpe Ratio vs. Network Depth')
    axs[0].set_xlabel('Number of Hidden Layers')
    axs[0].set_ylabel('Sharpe Ratio')
    axs[0].grid(linestyle='--', alpha=0.7)
    
    axs[1].plot(depths, correlations, 'o-')
    axs[1].set_title('SDF Correlation vs. Network Depth')
    axs[1].set_xlabel('Number of Hidden Layers')
    axs[1].set_ylabel('Correlation with True SDF')
    axs[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('depth_impact.png')
    plt.show()
    
    return {
        'depths': depths,
        'sharpe_ratios': sharpe_ratios,
        'correlations': correlations
    }

def main():
    """Main function to run the simulation and display results"""
    print("Running SDF model comparison simulation...")
    
    # Run simulation
    results = run_simulation(
        n_stocks=100,
        n_time_periods=120,
        n_characteristics=10,
        n_factors=3,
        test_size=24,
        hidden_dim=32,
        deep_layers=8,
        epochs=500
    )
    
    # Display results
    print("\nSimulation Results:")
    for model, metrics in results.items():
        if model not in ['shallow_history', 'deep_history']:
            print(f"\n{model.upper()} Model:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
    
    # Plot results
    plot_results(results)
    
    # Explore impact of network depth
    print("\nExploring impact of network depth...")
    depth_results = explore_depth_impact(max_depth=16)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()


def test_portfolio_tangent_kernel():
    """
    Test the Portfolio Tangent Kernel (PTK) concept from the paper
    
    This function demonstrates how the PTK can be used to construct an SDF
    and compares it with other approaches.
    """
    print("Testing Portfolio Tangent Kernel (PTK) concept...")
    
    # Initialize simulator
    n_stocks = 100
    n_time_periods = 120
    n_characteristics = 10
    n_factors = 3
    test_size = 24
    
    simulator = StockMarketSimulator(
        N=n_stocks, 
        T=n_time_periods, 
        n_characteristics=n_characteristics,
        n_factors=n_factors
    )
    
    # Generate data
    returns, factor_betas, expected_returns = simulator.generate_returns()
    
    # Split into train/test
    train_returns = returns[:n_time_periods-test_size]
    test_returns = returns[n_time_periods-test_size:]
    
    # Create 3D array of characteristics for each time period
    characteristics = np.zeros((n_time_periods, n_stocks, n_characteristics))
    for t in range(n_time_periods):
        noise = np.random.randn(n_stocks, n_characteristics) * 0.05
        characteristics[t] = simulator.characteristics + noise
    
    train_characteristics = characteristics[:n_time_periods-test_size]
    test_characteristics = characteristics[n_time_periods-test_size:]
    
    # Compute true SDF values
    true_sdf = np.zeros((n_time_periods, n_stocks))
    for t in range(n_time_periods):
        factor_loadings_t = simulator.non_linear_factor_mapping(characteristics[t])
        true_sdf[t] = factor_loadings_t @ simulator.true_sdf_coef
    
    train_true_sdf = true_sdf[:n_time_periods-test_size]
    test_true_sdf = true_sdf[n_time_periods-test_size:]
    
    # Implement PTK approach with different kernel types
    kernels = ['linear', 'rbf', 'polynomial']
    ridge_penalties = [1e-5, 1e-3, 1e-1, 1.0, 10.0]
    
    # Store results
    kernel_results = {}
    
    for kernel_type in kernels:
        print(f"\nTesting {kernel_type} kernel...")
        penalty_results = []
        
        for penalty in ridge_penalties:
            # Create NTK estimator with the specified kernel
            ntk_model = NTKEstimator(
                kernel_type=kernel_type,
                gamma=0.1 if kernel_type == 'rbf' else 1.0,
                depth=3 if kernel_type == 'polynomial' else 1
            )
            
            # Fit model
            ntk_model.fit(train_characteristics, train_returns, ridge_penalty=penalty)
            
            # Evaluate on test set
            ntk_sdf_values = np.zeros((test_size, n_stocks))
            for t in range(test_size):
                ntk_sdf_values[t] = ntk_model.predict(
                    test_characteristics[t], 
                    test_returns[t], 
                    train_characteristics, 
                    train_returns
                )
            
            # Evaluate performance
            results = evaluate_sdf(ntk_sdf_values, test_returns, test_true_sdf)
            penalty_results.append({
                'penalty': penalty,
                'sharpe_ratio': results['sharpe_ratio'],
                'mean_pricing_error': results['mean_pricing_error'],
                'sdf_correlation': results['sdf_correlation'],
                'r_squared': results['r_squared']
            })
            
            print(f"  Ridge penalty: {penalty:.5f}, Sharpe ratio: {results['sharpe_ratio']:.4f}, "
                  f"Correlation: {results['sdf_correlation']:.4f}")
        
        kernel_results[kernel_type] = penalty_results
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Sharpe ratio vs. ridge penalty
    for kernel_type in kernels:
        penalties = [result['penalty'] for result in kernel_results[kernel_type]]
        sharpe_ratios = [result['sharpe_ratio'] for result in kernel_results[kernel_type]]
        axs[0, 0].plot(penalties, sharpe_ratios, 'o-', label=kernel_type)
    
    axs[0, 0].set_title('Sharpe Ratio vs. Ridge Penalty')
    axs[0, 0].set_xlabel('Ridge Penalty')
    axs[0, 0].set_ylabel('Sharpe Ratio')
    axs[0, 0].set_xscale('log')
    axs[0, 0].legend()
    axs[0, 0].grid(linestyle='--', alpha=0.7)
    
    # Plot SDF correlation vs. ridge penalty
    for kernel_type in kernels:
        penalties = [result['penalty'] for result in kernel_results[kernel_type]]
        correlations = [result['sdf_correlation'] for result in kernel_results[kernel_type]]
        axs[0, 1].plot(penalties, correlations, 'o-', label=kernel_type)
    
    axs[0, 1].set_title('SDF Correlation vs. Ridge Penalty')
    axs[0, 1].set_xlabel('Ridge Penalty')
    axs[0, 1].set_ylabel('Correlation with True SDF')
    axs[0, 1].set_xscale('log')
    axs[0, 1].legend()
    axs[0, 1].grid(linestyle='--', alpha=0.7)
    
    # Plot mean pricing error vs. ridge penalty
    for kernel_type in kernels:
        penalties = [result['penalty'] for result in kernel_results[kernel_type]]
        errors = [result['mean_pricing_error'] for result in kernel_results[kernel_type]]
        axs[1, 0].plot(penalties, errors, 'o-', label=kernel_type)
    
    axs[1, 0].set_title('Mean Pricing Error vs. Ridge Penalty')
    axs[1, 0].set_xlabel('Ridge Penalty')
    axs[1, 0].set_ylabel('Mean Pricing Error')
    axs[1, 0].set_xscale('log')
    axs[1, 0].legend()
    axs[1, 0].grid(linestyle='--', alpha=0.7)
    
    # Plot R-squared vs. ridge penalty
    for kernel_type in kernels:
        penalties = [result['penalty'] for result in kernel_results[kernel_type]]
        r_squared = [result['r_squared'] for result in kernel_results[kernel_type]]
        axs[1, 1].plot(penalties, r_squared, 'o-', label=kernel_type)
    
    axs[1, 1].set_title('R-squared vs. Ridge Penalty')
    axs[1, 1].set_xlabel('Ridge Penalty')
    axs[1, 1].set_ylabel('R-squared')
    axs[1, 1].set_xscale('log')
    axs[1, 1].legend()
    axs[1, 1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ptk_analysis.png')
    plt.show()
    
    return kernel_results

# Run the PTK test
ptk_results = test_portfolio_tangent_kernel()


def test_depth_complexity(max_depth=128):
    """
    Test the virtue of depth complexity concept from the paper
    
    This function evaluates how the performance of neural network SDFs
    changes with increasing depth, using different training window sizes.
    """
    print("Testing virtue of depth complexity...")
    
    # Test different training window sizes
    window_sizes = [12, 60, 120]  # As in the paper
    depths = [1, 2, 4, 8, 16, 32, 64, max_depth]  # Powers of 2 as in the paper
    
    # Fixed parameters
    n_stocks = 100
    n_time_periods = 180  # Ensure we have enough data for longest window
    n_characteristics = 10
    n_factors = 3
    test_size = 24
    
    # Initialize simulator
    simulator = StockMarketSimulator(
        N=n_stocks, 
        T=n_time_periods, 
        n_characteristics=n_characteristics,
        n_factors=n_factors
    )
    
    # Generate data
    returns, factor_betas, expected_returns = simulator.generate_returns()
    
    # Create 3D array of characteristics for each time period
    characteristics = np.zeros((n_time_periods, n_stocks, n_characteristics))
    for t in range(n_time_periods):
        noise = np.random.randn(n_stocks, n_characteristics) * 0.05
        characteristics[t] = simulator.characteristics + noise
    
    # Compute true SDF values
    true_sdf = np.zeros((n_time_periods, n_stocks))
    for t in range(n_time_periods):
        factor_loadings_t = simulator.non_linear_factor_mapping(characteristics[t])
        true_sdf[t] = factor_loadings_t @ simulator.true_sdf_coef
    
    # Store results
    window_results = {}
    
    for window_size in window_sizes:
        print(f"\nTesting window size: {window_size}")
        depth_results = []
        
        # Use the last window_size + test_size periods
        start_idx = n_time_periods - (window_size + test_size)
        train_returns = returns[start_idx:start_idx+window_size]
        test_returns = returns[start_idx+window_size:start_idx+window_size+test_size]
        
        train_characteristics = characteristics[start_idx:start_idx+window_size]
        test_characteristics = characteristics[start_idx+window_size:start_idx+window_size+test_size]
        
        train_true_sdf = true_sdf[start_idx:start_idx+window_size]
        test_true_sdf = true_sdf[start_idx+window_size:start_idx+window_size+test_size]
        
        for depth in depths:
            print(f"  Testing depth: {depth}")
            
            # Train deep neural network
            deep_model = DeepNetworkSDF(
                input_dim=n_characteristics, 
                hidden_dim=32, 
                num_layers=depth
            )
            deep_model, _ = train_neural_network_sdf(
                deep_model, 
                train_characteristics, 
                train_returns, 
                epochs=300  # Reduced for faster execution
            )
            
            # Evaluate on test set
            deep_sdf_values = np.zeros((test_size, n_stocks))
            for t in range(test_size):
                X_tensor = torch.FloatTensor(test_characteristics[t])
                with torch.no_grad():
                    sdf_t = deep_model(X_tensor).numpy()
                deep_sdf_values[t] = sdf_t.flatten()
            
            # Evaluate performance
            results = evaluate_sdf(deep_sdf_values, test_returns, test_true_sdf)
            depth_results.append({
                'depth': depth,
                'sharpe_ratio': results['sharpe_ratio'],
                'mean_pricing_error': results['mean_pricing_error'],
                'sdf_correlation': results['sdf_correlation'],
                'r_squared': results['r_squared']
            })
            
            print(f"    Sharpe ratio: {results['sharpe_ratio']:.4f}, "
                  f"Correlation: {results['sdf_correlation']:.4f}")
        
        window_results[window_size] = depth_results
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Sharpe ratio vs. depth
    for window_size in window_sizes:
        depths = [result['depth'] for result in window_results[window_size]]
        sharpe_ratios = [result['sharpe_ratio'] for result in window_results[window_size]]
        axs[0, 0].plot(depths, sharpe_ratios, 'o-', label=f'Window={window_size}')
    
    axs[0, 0].set_title('Sharpe Ratio vs. Network Depth')
    axs[0, 0].set_xlabel('Number of Hidden Layers')
    axs[0, 0].set_ylabel('Sharpe Ratio')
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].legend()
    axs[0, 0].grid(linestyle='--', alpha=0.7)
    
    # Plot SDF correlation vs. depth
    for window_size in window_sizes:
        depths = [result['depth'] for result in window_results[window_size]]
        correlations = [result['sdf_correlation'] for result in window_results[window_size]]
        axs[0, 1].plot(depths, correlations, 'o-', label=f'Window={window_size}')
    
    axs[0, 1].set_title('SDF Correlation vs. Network Depth')
    axs[0, 1].set_xlabel('Number of Hidden Layers')
    axs[0, 1].set_ylabel('Correlation with True SDF')
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].legend()
    axs[0, 1].grid(linestyle='--', alpha=0.7)
    
    # Plot mean pricing error vs. depth
    for window_size in window_sizes:
        depths = [result['depth'] for result in window_results[window_size]]
        errors = [result['mean_pricing_error'] for result in window_results[window_size]]
        axs[1, 0].plot(depths, errors, 'o-', label=f'Window={window_size}')
    
    axs[1, 0].set_title('Mean Pricing Error vs. Network Depth')
    axs[1, 0].set_xlabel('Number of Hidden Layers')
    axs[1, 0].set_ylabel('Mean Pricing Error')
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].legend()
    axs[1, 0].grid(linestyle='--', alpha=0.7)
    
    # Plot R-squared vs. depth
    for window_size in window_sizes:
        depths = [result['depth'] for result in window_results[window_size]]
        r_squared = [result['r_squared'] for result in window_results[window_size]]
        axs[1, 1].plot(depths, r_squared, 'o-', label=f'Window={window_size}')
    
    axs[1, 1].set_title('R-squared vs. Network Depth')
    axs[1, 1].set_xlabel('Number of Hidden Layers')
    axs[1, 1].set_ylabel('R-squared')
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].legend()
    axs[1, 1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('depth_complexity.png')
    plt.show()
    
    return window_results

# Run the depth complexity test
depth_results = test_depth_complexity(max_depth=64)  # Reduced max_depth for computational efficiency