import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BlackScholes:
    """Black-Scholes model for European options pricing and Greeks"""
    
    def __init__(self, r=0.0, sigma=0.2):
        """
        Initialize the Black-Scholes model
        
        Parameters:
        -----------
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of the underlying asset
        """
        self.r = r
        self.sigma = sigma
    
    def d1(self, S, K, T):
        """Calculate d1 term in Black-Scholes formula"""
        return (np.log(S/K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
    
    def d2(self, S, K, T):
        """Calculate d2 term in Black-Scholes formula"""
        return self.d1(S, K, T) - self.sigma * np.sqrt(T)
    
    def call_price(self, S, K, T):
        """Calculate European call option price"""
        d1 = self.d1(S, K, T)
        d2 = self.d2(S, K, T)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
    
    def call_delta(self, S, K, T):
        """Calculate European call option delta (first derivative w.r.t. S)"""
        d1 = self.d1(S, K, T)
        return norm.cdf(d1)
    
    def call_gamma(self, S, K, T):
        """Calculate European call option gamma (second derivative w.r.t. S)"""
        d1 = self.d1(S, K, T)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(T))
    
    def call_vega(self, S, K, T):
        """Calculate European call option vega (derivative w.r.t. sigma)"""
        d1 = self.d1(S, K, T)
        return S * np.sqrt(T) * norm.pdf(d1)

class AssetSimulator:
    """Simulate asset price paths using Geometric Brownian Motion"""
    
    def __init__(self, S0=100.0, r=0.0, sigma=0.2):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        r : float
            Risk-free interest rate (drift under risk-neutral measure)
        sigma : float
            Volatility of the asset
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    def simulate_paths(self, T=1.0, n_steps=52, n_paths=10000):
        """
        Simulate asset price paths using Euler discretization
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        n_steps : int
            Number of time steps
        n_paths : int
            Number of paths to simulate
        
        Returns:
        --------
        paths : ndarray
            Simulated asset price paths of shape (n_paths, n_steps+1)
        times : ndarray
            Time points of shape (n_steps+1,)
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate random increments for all paths at once
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Simulate paths using Euler discretization
        for t in range(n_steps):
            paths[:, t+1] = paths[:, t] * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                                self.sigma * dW[:, t])
        
        return paths, times
    
    def get_training_data(self, T=1.0, K=110.0, n_steps=52, n_paths=10000):
        """
        Generate training data for machine learning models
        
        Parameters:
        -----------
        T : float
            Time to maturity
        K : float
            Strike price
        n_steps : int
            Number of time steps
        n_paths : int
            Number of paths
        
        Returns:
        --------
        X : ndarray
            Input features (asset prices)
        y : ndarray
            Target values (option payoffs)
        delta_labels : ndarray
            Delta values for differential training
        """
        # Simulate paths
        paths, times = self.simulate_paths(T, n_steps, n_paths)
        
        # Extract spot prices at t=0
        spot_prices = paths[:, 0].reshape(-1, 1)
        
        # Calculate option payoffs at maturity
        payoffs = np.maximum(paths[:, -1] - K, 0)
        
        # Create differential labels (deltas)
        # For a call option, delta is 1 if S_T > K, 0 otherwise
        # We need to calculate the pathwise derivative as described in the paper
        delta_labels = np.zeros_like(spot_prices)
        in_the_money = paths[:, -1] > K
        
        # Calculate dS_T/dS_0 for each path
        # In Black-Scholes, this is S_T/S_0
        dS_T_dS_0 = paths[:, -1] / paths[:, 0]
        
        # Delta labels as per equation (33) in the paper
        delta_labels[in_the_money] = dS_T_dS_0[in_the_money].reshape(-1, 1)
        
        return spot_prices, payoffs.reshape(-1, 1), delta_labels

class LSMCModel:
    """Least Squares Monte Carlo model for option pricing"""
    
    def __init__(self, basis_type='polynomial', degree=5, nn_layers=None):
        """
        Initialize the LSMC model
        
        Parameters:
        -----------
        basis_type : str
            Type of basis functions ('polynomial' or 'neural_network')
        degree : int
            Degree of polynomial basis
        nn_layers : list
            Architecture of neural network if basis_type is 'neural_network'
        """
        self.basis_type = basis_type
        self.degree = degree
        self.nn_layers = nn_layers if nn_layers else [20, 20, 20]
        self.model = None
    
    def _create_polynomial_features(self, X):
        """Create polynomial features up to specified degree"""
        features = np.ones((X.shape[0], self.degree + 1))
        for d in range(1, self.degree + 1):
            features[:, d] = X[:, 0] ** d
        return features
    
    def _build_neural_network(self, input_dim):
        """Build a neural network model"""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Add hidden layers
        for units in self.nn_layers:
            model.add(keras.layers.Dense(units, activation='softplus'))
        
        # Output layer
        model.add(keras.layers.Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, y, epochs=100, batch_size=64, verbose=0):
        """
        Fit the model to the data
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        y : ndarray
            Target values (option payoffs)
        epochs : int
            Number of training epochs for neural network
        batch_size : int
            Batch size for neural network training
        verbose : int
            Verbosity level
        """
        if self.basis_type == 'polynomial':
            # Create polynomial features
            X_poly = self._create_polynomial_features(X)
            
            # Fit using OLS
            self.coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]
            self.model = lambda X: self._create_polynomial_features(X) @ self.coeffs
            
        elif self.basis_type == 'neural_network':
            # Build and train neural network
            self.model = self._build_neural_network(X.shape[1])
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def predict(self, X):
        """
        Predict option prices
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        
        Returns:
        --------
        y_pred : ndarray
            Predicted option prices
        """
        if self.basis_type == 'polynomial':
            return self.model(X)
        else:
            return self.model.predict(X, verbose=0)
    
    def predict_delta(self, X, h=0.01):
        """
        Predict option deltas using finite differences
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        h : float
            Step size for finite differences
        
        Returns:
        --------
        deltas : ndarray
            Predicted option deltas
        """
        X_up = X.copy()
        X_up[:, 0] += h
        X_down = X.copy()
        X_down[:, 0] -= h
        
        price_up = self.predict(X_up)
        price_down = self.predict(X_down)
        
        return (price_up - price_down) / (2 * h)

class DifferentialMLModel:
    """Differential Machine Learning model for option pricing and hedging"""
    
    def __init__(self, lambda_reg=1.0, nn_layers=None):
        """
        Initialize the Differential ML model
        
        Parameters:
        -----------
        lambda_reg : float
            Regularization parameter for the differential term in the loss function
        nn_layers : list
            Architecture of the neural network
        """
        self.lambda_reg = lambda_reg
        self.nn_layers = nn_layers if nn_layers else [32, 32, 16]
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def _build_twin_network(self, input_dim):
        """
        Build the twin network for differential machine learning
        
        This network will output both the price and its derivative w.r.t. input
        """
        # Define the input layer
        inputs = keras.layers.Input(shape=(input_dim,))
        
        # First tower - pricing
        x = inputs
        for i, units in enumerate(self.nn_layers):
            x = keras.layers.Dense(units, activation='softplus', name=f'hidden_{i+1}')(x)
        
        price_output = keras.layers.Dense(1, name='price_output')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=price_output)
        
        # Custom loss function that includes the differential part
        def differential_loss(y_true, y_pred):
            # Extract price and delta labels from y_true
            price_labels = y_true[:, 0:1]
            delta_labels = y_true[:, 1:2]
            
            # Compute the gradients of the output with respect to the inputs
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                predictions = model(inputs)
            
            gradients = tape.gradient(predictions, inputs)
            
            # Calculate the two components of the loss
            price_loss = tf.reduce_mean(tf.square(price_labels - y_pred))
            delta_loss = tf.reduce_mean(tf.square(delta_labels - gradients))
            
            # Combined loss with regularization parameter
            total_loss = price_loss + self.lambda_reg * delta_loss
            
            return total_loss
        
        # Create a custom training model with the differential loss
        training_model = keras.Model(inputs=inputs, outputs=price_output)
        training_model.compile(optimizer='adam', loss=differential_loss)
        
        return training_model, model
    
    def fit(self, X, y, delta_labels, epochs=100, batch_size=64, verbose=0):
        """
        Fit the model to the data
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        y : ndarray
            Target values (option payoffs)
        delta_labels : ndarray
            Delta values for differential training
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        """
        # Standardize inputs and outputs
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Combine price and delta labels for the custom loss function
        y_combined = np.hstack([y_scaled, delta_labels])
        
        # Build the twin network
        self.training_model, self.model = self._build_twin_network(X.shape[1])
        
        # Train the model
        self.training_model.fit(
            X_scaled, y_combined, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose
        )
    
    def predict(self, X):
        """
        Predict option prices
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        
        Returns:
        --------
        y_pred : ndarray
            Predicted option prices
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def predict_delta(self, X):
        """
        Predict option deltas using automatic differentiation
        
        Parameters:
        -----------
        X : ndarray
            Input features (asset prices)
        
        Returns:
        --------
        deltas : ndarray
            Predicted option deltas
        """
        X_scaled = self.scaler_X.transform(X)
        X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            y_pred = self.model(X_tensor)
        
        gradients = tape.gradient(y_pred, X_tensor)
        
        # Adjust gradients for scaling
        scale_factor = self.scaler_y.scale_[0] / self.scaler_X.scale_[0]
        return gradients.numpy() * scale_factor

def run_hedging_experiment(model_type, asset_simulator, bs_model, K=110.0, T=1.0, 
                          n_paths=1000, n_steps=52, rebalance_freq=1):
    """
    Run a hedging experiment comparing different pricing/hedging methods
    
    Parameters:
    -----------
    model_type : str
        Type of model ('bs', 'lsmc_poly', 'lsmc_nn', 'diff_ml')
    asset_simulator : AssetSimulator
        Simulator for asset price paths
    bs_model : BlackScholes
        Black-Scholes model for benchmark
    K : float
        Strike price
    T : float
        Time to maturity
    n_paths : int
        Number of paths for testing
    n_steps : int
        Number of time steps
    rebalance_freq : int
        Frequency of rebalancing in time steps
    
    Returns:
    --------
    pnl : ndarray
        PnL values for each path
    relative_error : float
        Relative hedging error (standard deviation of PnL)
    """
    # Generate training data (more paths for training)
    train_spots, train_payoffs, train_deltas = asset_simulator.get_training_data(
        T=T, K=K, n_steps=n_steps, n_paths=5000
    )
    
    # Fit model if not Black-Scholes
    if model_type == 'lsmc_poly':
        model = LSMCModel(basis_type='polynomial', degree=5)
        model.fit(train_spots, train_payoffs)
    elif model_type == 'lsmc_nn':
        model = LSMCModel(basis_type='neural_network', nn_layers=[32, 32, 16])
        model.fit(train_spots, train_payoffs, epochs=200)
    elif model_type == 'diff_ml':
        model = DifferentialMLModel(lambda_reg=1.0, nn_layers=[32, 32, 16])
        model.fit(train_spots, train_payoffs, train_deltas, epochs=200)
    
    # Generate test paths
    test_paths, times = asset_simulator.simulate_paths(T=T, n_steps=n_steps, n_paths=n_paths)
    dt = T / n_steps
    
    # Initialize PnL array
    pnl = np.zeros(n_paths)
    
    for i in range(n_paths):
        path = test_paths[i]
        
        # Initial option price and delta
        if model_type == 'bs':
            price_0 = bs_model.call_price(path[0], K, T)
            delta_0 = bs_model.call_delta(path[0], K, T)
        elif model_type in ['lsmc_poly', 'lsmc_nn']:
            price_0 = model.predict(path[0].reshape(1, -1))[0, 0]
            delta_0 = model.predict_delta(path[0].reshape(1, -1))[0, 0]
        else:  # diff_ml
            price_0 = model.predict(path[0].reshape(1, -1))[0, 0]
            delta_0 = model.predict_delta(path[0].reshape(1, -1))[0, 0]
        
        # Initialize portfolio: short option, long delta*underlying
        portfolio_value = -price_0 + delta_0 * path[0]
        
        # Hedging simulation
        for t in range(0, n_steps, rebalance_freq):
            if t + rebalance_freq >= n_steps:
                break
                
            # Current position in underlying
            current_delta = delta_0
            
            # New spot price
            new_spot = path[t + rebalance_freq]
            
            # Calculate new delta
            time_to_maturity = T - (t + rebalance_freq) * dt
            if model_type == 'bs':
                new_delta = bs_model.call_delta(new_spot, K, time_to_maturity)
            elif model_type in ['lsmc_poly', 'lsmc_nn']:
                new_delta = model.predict_delta(new_spot.reshape(1, -1))[0, 0]
            else:  # diff_ml
                new_delta = model.predict_delta(new_spot.reshape(1, -1))[0, 0]
            
            # Rebalance portfolio
            portfolio_value = portfolio_value + (new_spot - path[t]) * current_delta
            
            # Update delta
            delta_0 = new_delta
        
        # Final payoff and PnL
        final_spot = path[-1]
        option_payoff = max(final_spot - K, 0)
        
        # Final rebalancing: close the option position and sell delta*underlying
        pnl[i] = portfolio_value + option_payoff - delta_0 * final_spot
    
    # Calculate relative hedging error (std of PnL / initial option price)
    if model_type == 'bs':
        initial_price = bs_model.call_price(asset_simulator.S0, K, T)
    elif model_type in ['lsmc_poly', 'lsmc_nn']:
        initial_price = model.predict(np.array([[asset_simulator.S0]]))[0, 0]
    else:  # diff_ml
        initial_price = model.predict(np.array([[asset_simulator.S0]]))[0, 0]
    
    relative_error = np.std(pnl) / initial_price
    
    return pnl, relative_error

def plot_price_delta_comparison(models, bs_model, S_range=(80, 120), K=110.0, T=1.0):
    """
    Plot price and delta comparison for different models
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare
    bs_model : BlackScholes
        Black-Scholes model for benchmark
    S_range : tuple
        Range of spot prices to plot
    K : float
        Strike price
    T : float
        Time to maturity
    """
    # Generate spot prices
    spot_prices = np.linspace(S_range[0], S_range[1], 100)
    spot_prices_2d = spot_prices.reshape(-1, 1)
    
    # Calculate Black-Scholes prices and deltas
    bs_prices = np.array([bs_model.call_price(S, K, T) for S in spot_prices])
    bs_deltas = np.array([bs_model.call_delta(S, K, T) for S in spot_prices])
    
    # Calculate prices and deltas for each model
    model_prices = {}
    model_deltas = {}
    
    for name, model in models.items():
        model_prices[name] = model.predict(spot_prices_2d).flatten()
        model_deltas[name] = model.predict_delta(spot_prices_2d).flatten()
    
    # Plot prices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(spot_prices, bs_prices, 'k-', label='Black-Scholes')
    
    for name, prices in model_prices.items():
        plt.plot(spot_prices, prices, label=name)
    
    plt.xlabel('Spot Price')
    plt.ylabel('Option Price')
    plt.title('Price Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot deltas
    plt.subplot(1, 2, 2)
    plt.plot(spot_prices, bs_deltas, 'k-', label='Black-Scholes')
    
    for name, deltas in model_deltas.items():
        plt.plot(spot_prices, deltas, label=name)
    
    plt.xlabel('Spot Price')
    plt.ylabel('Option Delta')
    plt.title('Delta Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_pnl_distributions(pnls, model_names):
    """
    Plot PnL distributions for different models
    
    Parameters:
    -----------
    pnls : dict
        Dictionary of PnL arrays for each model
    model_names : list
        Names of the models
    """
    plt.figure(figsize=(12, 6))
    
    for name in model_names:
        sns.kdeplot(pnls[name], label=name)
    
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.title('PnL Distribution Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_relative_errors(rel_errors, model_names, sample_sizes):
    """
    Compare relative hedging errors for different models and sample sizes
    
    Parameters:
    -----------
    rel_errors : dict
        Dictionary of relative errors for each model and sample size
    model_names : list
        Names of the models
    sample_sizes : list
        List of sample sizes
    """
    # Create DataFrame for plotting
    data = []
    for size in sample_sizes:
        for name in model_names:
            data.append({
                'Sample Size': size,
                'Model': name,
                'Relative Error': rel_errors[name][size]
            })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sample Size', y='Relative Error', hue='Model', data=df)
    plt.title('Relative Hedging Error Comparison')
    plt.grid(True, axis='y')
    plt.show()

# Main experiment
def main():
    # Parameters
    S0 = 100.0
    K = 110.0
    r = 0.0
    sigma = 0.2
    T = 1.0
    
    # Initialize models
    bs_model = BlackScholes(r=r, sigma=sigma)
    asset_simulator = AssetSimulator(S0=S0, r=r, sigma=sigma)
    
    # Generate training data
    train_spots, train_payoffs, train_deltas = asset_simulator.get_training_data(
        T=T, K=K, n_steps=52, n_paths=5000
    )
    
    # Train models
    print("Training LSMC with polynomial basis...")
    lsmc_poly = LSMCModel(basis_type='polynomial', degree=5)
    lsmc_poly.fit(train_spots, train_payoffs)
    
    print("Training LSMC with neural network basis...")
    lsmc_nn = LSMCModel(basis_type='neural_network', nn_layers=[32, 32, 16])
    lsmc_nn.fit(train_spots, train_payoffs, epochs=200)
    
    print("Training Differential ML model...")
    diff_ml = DifferentialMLModel(lambda_reg=1.0, nn_layers=[32, 32, 16])
    diff_ml.fit(train_spots, train_payoffs, train_deltas, epochs=200)
    
    # Plot price and delta comparison
    plot_price_delta_comparison(
        {'LSMC Polynomial': lsmc_poly, 'LSMC Neural': lsmc_nn, 'Differential ML': diff_ml},
        bs_model,
        S_range=(80, 120),
        K=K,
        T=T
    )
    
    # Run hedging experiments with different sample sizes
    sample_sizes = [1000, 3000, 5000, 7000]
    model_types = ['bs', 'lsmc_poly', 'lsmc_nn', 'diff_ml']
    model_names = ['Black-Scholes', 'LSMC Polynomial', 'LSMC Neural', 'Differential ML']
    
    all_pnls = {}
    all_errors = {name: {} for name in model_names}
    
    for size in tqdm(sample_sizes, desc="Testing sample sizes"):
        for i, model_type in enumerate(model_types):
            pnl, rel_error = run_hedging_experiment(
                model_type, asset_simulator, bs_model, K=K, T=T, 
                n_paths=size, n_steps=52, rebalance_freq=1
            )
            
            if size == 5000:  # Store PnL for mid-size sample
                all_pnls[model_names[i]] = pnl
            
            all_errors[model_names[i]][size] = rel_error
    
    # Plot PnL distributions
    plot_pnl_distributions(all_pnls, model_names)
    
    # Compare relative errors
    compare_relative_errors(all_errors, model_names, sample_sizes)
    
    # Print relative hedging errors table
    print("\nRelative Hedging Errors:")
    print("------------------------")
    print(f"{'Sample Size':<15}", end="")
    for name in model_names:
        print(f"{name:<20}", end="")
    print()
    
    for size in sample_sizes:
        print(f"{size:<15}", end="")
        for name in model_names:
            print(f"{all_errors[name][size]:.4f}{'':16}", end="")
        print()

if __name__ == "__main__":
    main()