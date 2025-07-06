import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class OptimalLinearSignal:
    """
    Implements the Optimal Linear Signal (OLS) algorithm as described in the paper.
    This is an unsupervised machine learning framework for optimizing PnL with linear signals.
    """
    
    def __init__(self, training_size=252, regularization=None, reg_params=None, 
                 corrective_factor=False, pca_components=None, beta_neutral=False,
                 zscore_threshold=1.0):
        """
        Initialize the Optimal Linear Signal model.
        
        Parameters:
        -----------
        training_size : int
            Number of days to use for training the model
        regularization : str or None
            Type of regularization to use ('l1', 'l2', 'pca', 'significance', None)
        reg_params : dict or None
            Parameters for the regularization
        corrective_factor : bool
            Whether to use the corrective factor described in the paper
        pca_components : int or None
            Number of principal components to use if PCA regularization is selected
        beta_neutral : bool
            Whether to make the signal beta neutral with respect to the asset price
        zscore_threshold : float
            Z-score threshold for signal to trigger a trade
        """
        self.training_size = training_size
        self.regularization = regularization
        self.reg_params = reg_params if reg_params is not None else {}
        self.corrective_factor = corrective_factor
        self.pca_components = pca_components
        self.beta_neutral = beta_neutral
        self.zscore_threshold = zscore_threshold
        self.alpha = None
        self.scaler = StandardScaler()
        self.pca = None
        self.signal_mean = 0
        self.signal_std = 1
        self.last_pnl = 0
        
    def _compute_transformed_variables(self, X, price):
        """
        Compute the transformed variables X̃ = X_{t-1} * (price_t - price_{t-1})
        
        Parameters:
        -----------
        X : ndarray
            Array of exogenous variables (n_samples, n_features)
        price : ndarray
            Array of asset prices (n_samples,)
            
        Returns:
        --------
        X_tilde : ndarray
            Transformed variables (n_samples-1, n_features)
        """
        # Calculate price differences
        price_diff = np.diff(price)
        
        # Shift X by one time step to get X_{t-1}
        X_shifted = X[:-1]
        
        # Compute X_tilde = X_{t-1} * (price_t - price_{t-1})
        X_tilde = X_shifted * price_diff[:, np.newaxis]
        
        return X_tilde
    
    def _compute_optimal_alpha(self, X_tilde):
        """
        Compute the optimal alpha coefficients that maximize the Sharpe ratio.
        
        Parameters:
        -----------
        X_tilde : ndarray
            Transformed variables (n_samples, n_features)
            
        Returns:
        --------
        alpha : ndarray
            Optimal coefficients that maximize the Sharpe ratio
        """
        # Compute mean vector
        mu = np.mean(X_tilde, axis=0)
        
        # Compute covariance matrix
        Sigma = np.cov(X_tilde, rowvar=False)
        
        # Apply regularization if specified
        if self.regularization == 'l2':
            lambda2 = self.reg_params.get('lambda2', 0.01)
            norm_sigma = np.linalg.norm(Sigma) / Sigma.shape[0]
            Sigma_reg = (Sigma + lambda2 * norm_sigma * np.eye(Sigma.shape[0])) / (1 + lambda2)
            Sigma = Sigma_reg
        
        # Apply PCA regularization if specified
        if self.regularization == 'pca':
            if self.pca is None:
                n_components = self.pca_components if self.pca_components is not None else X_tilde.shape[1]
                self.pca = PCA(n_components=n_components)
                self.pca.fit(X_tilde)
                
            # Project data onto principal components
            X_tilde_pca = self.pca.transform(X_tilde)
            
            # Compute mu and Sigma in PCA space
            mu_pca = np.mean(X_tilde_pca, axis=0)
            Sigma_pca = np.cov(X_tilde_pca, rowvar=False)
            
            # Compute optimal alpha in PCA space
            alpha_pca = np.zeros(len(mu_pca))
            for i in range(len(mu_pca)):
                alpha_pca[i] = mu_pca[i] / Sigma_pca[i, i]
            
            # Scale alpha_pca to maximize Sharpe ratio
            norm_factor = np.sqrt(sum((mu_pca[i]**2) / Sigma_pca[i, i] for i in range(len(mu_pca))))
            alpha_pca = alpha_pca / norm_factor
            
            # Convert back to original space
            alpha = self.pca.inverse_transform(alpha_pca)
            return alpha
        
        # Apply beta neutrality if specified
        if self.beta_neutral:
            # This implementation assumes beta is provided in reg_params
            beta = self.reg_params.get('beta', None)
            if beta is not None:
                try:
                    Sigma_inv = np.linalg.inv(Sigma)
                    beta_term = (mu.T @ Sigma_inv @ beta) / (beta.T @ Sigma_inv @ beta) * beta
                    mu = mu - beta_term
                except np.linalg.LinAlgError:
                    print("Covariance matrix is singular. Cannot apply beta neutrality.")
        
        try:
            # Compute Sigma inverse
            Sigma_inv = np.linalg.inv(Sigma)
            
            # Compute optimal alpha
            alpha = Sigma_inv @ mu
            
            # Scale alpha to maximize Sharpe ratio
            norm_factor = np.sqrt(mu.T @ Sigma_inv @ mu)
            alpha = alpha / norm_factor
            
            # Apply L1 regularization if specified
            if self.regularization == 'l1':
                lambda1 = self.reg_params.get('lambda1', 0.01)
                sharpe_max = alpha.T @ mu / np.sqrt(alpha.T @ Sigma @ alpha)
                penalty = lambda1 * sharpe_max * np.sign(alpha)
                # For L1 regularization, we need to solve numerically
                # Here we use a simple approximation by shrinking alpha towards zero
                alpha = alpha * np.maximum(0, 1 - penalty / np.abs(alpha))
            
            # Apply statistical significance regularization if specified
            if self.regularization == 'significance':
                p_threshold = self.reg_params.get('p_threshold', 0.05)
                t_stats = np.sqrt(self.training_size) * alpha * np.sqrt(mu.T @ alpha)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.training_size - 1))
                alpha = np.where(p_values < p_threshold, alpha, 0)
            
            return alpha
            
        except np.linalg.LinAlgError:
            print("Covariance matrix is singular. Using pseudoinverse instead.")
            # Use pseudoinverse as a fallback
            Sigma_inv = np.linalg.pinv(Sigma)
            alpha = Sigma_inv @ mu
            norm_factor = np.sqrt(mu.T @ Sigma_inv @ mu)
            return alpha / norm_factor
    
    def fit(self, X, price):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : ndarray or pandas DataFrame
            Array of exogenous variables (n_samples, n_features)
        price : ndarray or pandas Series
            Array of asset prices (n_samples,)
            
        Returns:
        --------
        self : OptimalLinearSignal
            Fitted model
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(price, pd.Series):
            price = price.values
        
        # Use only the last training_size data points for training
        if len(price) > self.training_size:
            X = X[-self.training_size:]
            price = price[-self.training_size:]
        
        # Standardize the exogenous variables to ensure stationarity and homoscedasticity
        X_std = self.scaler.fit_transform(X)
        
        # Compute transformed variables
        X_tilde = self._compute_transformed_variables(X_std, price)
        
        # Compute optimal alpha
        self.alpha = self._compute_optimal_alpha(X_tilde)
        
        # Calculate the mean and std of the signal for z-scoring
        signals = X_std @ self.alpha
        self.signal_mean = np.mean(signals)
        self.signal_std = np.std(signals)
        
        return self
    
    def predict(self, X, price=None, apply_zscore=True):
        """
        Generate trading signals using the fitted model.
        
        Parameters:
        -----------
        X : ndarray or pandas DataFrame
            Array of exogenous variables (n_samples, n_features)
        price : ndarray or pandas Series or None
            Array of asset prices (n_samples,)
        apply_zscore : bool
            Whether to apply z-score normalization to the signal
            
        Returns:
        --------
        signal : ndarray
            Trading signals (n_samples,)
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(price, pd.Series):
            price = price.values if price is not None else None
        
        # Standardize the exogenous variables
        X_std = self.scaler.transform(X)
        
        # Generate raw signal
        signal = X_std @ self.alpha
        
        # Apply corrective factor if specified
        if self.corrective_factor and self.last_pnl is not None:
            signal = signal * np.sign(self.last_pnl) if self.last_pnl != 0 else signal
        
        # Apply z-score normalization if specified
        if apply_zscore:
            signal = (signal - self.signal_mean) / self.signal_std
        
        return signal
    
    def generate_positions(self, X, price, apply_zscore=True):
        """
        Generate trading positions based on the signals.
        
        Parameters:
        -----------
        X : ndarray or pandas DataFrame
            Array of exogenous variables (n_samples, n_features)
        price : ndarray or pandas Series
            Array of asset prices (n_samples,)
        apply_zscore : bool
            Whether to apply z-score normalization to the signal
            
        Returns:
        --------
        positions : ndarray
            Trading positions (n_samples,)
        """
        # Generate signals
        signal = self.predict(X, price, apply_zscore=apply_zscore)
        
        # Generate positions based on z-score threshold
        if apply_zscore:
            positions = np.where(np.abs(signal) > self.zscore_threshold, signal * price, 0)
        else:
            positions = signal * price
        
        return positions
    
    def calculate_pnl(self, positions, price):
        """
        Calculate the PnL based on positions and price.
        
        Parameters:
        -----------
        positions : ndarray
            Trading positions (n_samples,)
        price : ndarray
            Asset prices (n_samples,)
            
        Returns:
        --------
        pnl : ndarray
            Profit and Loss (n_samples-1,)
        """
        # Calculate price differences
        price_diff = np.diff(price)
        
        # Calculate PnL: position_{t-1} * (price_t - price_{t-1})
        pnl = positions[:-1] / price[:-1] * price_diff
        
        # Store the last PnL for corrective factor
        if len(pnl) > 0:
            self.last_pnl = pnl[-1]
        
        return pnl
    
    def backtest(self, X, price, window_size=None, update_freq=1, plot=True):
        """
        Backtest the strategy by rolling forward in time.
        
        Parameters:
        -----------
        X : ndarray or pandas DataFrame
            Array of exogenous variables (n_samples, n_features)
        price : ndarray or pandas Series
            Array of asset prices (n_samples,)
        window_size : int or None
            Size of the rolling window for training, if None use training_size
        update_freq : int
            Frequency of model updates in days
        plot : bool
            Whether to plot the results
            
        Returns:
        --------
        results : dict
            Dictionary containing backtest results
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(price, pd.Series):
            price = price.values
        
        # Set window size if not specified
        if window_size is None:
            window_size = self.training_size
        
        # Initialize arrays for results
        n_samples = len(price)
        positions = np.zeros(n_samples)
        signals = np.zeros(n_samples)
        pnl = np.zeros(n_samples-1)
        
        # Run backtest
        for i in range(window_size, n_samples, update_freq):
            # Define training and testing periods
            train_end = i
            train_start = train_end - window_size
            test_end = min(i + update_freq, n_samples)
            
            # Extract training data
            X_train = X[train_start:train_end]
            price_train = price[train_start:train_end]
            
            # Fit model
            self.fit(X_train, price_train)
            
            # Extract testing data
            X_test = X[i:test_end]
            price_test = price[i:test_end]
            
            # Generate signals and positions
            test_signals = self.predict(X_test, price_test)
            test_positions = self.generate_positions(X_test, price_test)
            
            # Store results
            signals[i:test_end] = test_signals
            positions[i:test_end] = test_positions
            
            # Calculate PnL for this period
            if i > 0 and i < n_samples - 1:
                pnl[i-1] = positions[i-1] / price[i-1] * (price[i] - price[i-1])
            
            if test_end < n_samples:
                pnl[test_end-2] = positions[test_end-2] / price[test_end-2] * (price[test_end-1] - price[test_end-2])
        
        # Calculate cumulative PnL
        cumulative_pnl = np.cumsum(pnl)
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(pnl) / np.std(pnl) * np.sqrt(252)  # Annualized
        
        # Calculate effective Sharpe ratio (only for days with positions)
        effective_pnl = pnl[positions[:-1] != 0]
        effective_sharpe = np.mean(effective_pnl) / np.std(effective_pnl) * np.sqrt(252) if len(effective_pnl) > 0 else 0
        
        # Calculate turnover
        turnover = np.sum(np.abs(np.diff(positions))) / np.sum(np.abs(positions[:-1])) if np.sum(np.abs(positions[:-1])) > 0 else 0
        
        # Calculate effective turnover (only for days with positions)
        effective_positions = positions[positions != 0]
        effective_turnover = np.sum(np.abs(np.diff(effective_positions))) / np.sum(np.abs(effective_positions[:-1])) if np.sum(np.abs(effective_positions[:-1])) > 0 else 0
        
        # Calculate bips (basis points of return)
        bips = np.mean(pnl) / np.mean(np.abs(positions[:-1])) * 10000 if np.mean(np.abs(positions[:-1])) > 0 else 0
        
        # Calculate effective bips
        effective_bips = np.mean(effective_pnl) / np.mean(np.abs(effective_positions[:-1])) * 10000 if np.mean(np.abs(effective_positions[:-1])) > 0 else 0
        
        # Plot results if specified
        if plot:
            self._plot_backtest_results(price, positions, signals, pnl, cumulative_pnl)
        
        # Store results
        results = {
            'positions': positions,
            'signals': signals,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl,
            'sharpe_ratio': sharpe_ratio,
            'effective_sharpe': effective_sharpe,
            'turnover': turnover,
            'effective_turnover': effective_turnover,
            'bips': bips,
            'effective_bips': effective_bips
        }
        
        return results
    
    def _plot_backtest_results(self, price, positions, signals, pnl, cumulative_pnl):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        price : ndarray
            Asset prices
        positions : ndarray
            Trading positions
        signals : ndarray
            Trading signals
        pnl : ndarray
            Profit and Loss
        cumulative_pnl : ndarray
            Cumulative Profit and Loss
        """
        plt.figure(figsize=(16, 12))
        
        # Plot prices
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(price, label='Asset Price')
        ax1.set_title('Asset Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot signals
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)
        ax2.plot(signals, label='Signal')
        ax2.axhline(y=self.zscore_threshold, color='r', linestyle='--', label='Threshold')
        ax2.axhline(y=-self.zscore_threshold, color='r', linestyle='--')
        ax2.set_title('Trading Signal')
        ax2.legend()
        ax2.grid(True)
        
        # Plot positions
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        ax3.plot(positions, label='Position')
        ax3.set_title('Trading Position')
        ax3.legend()
        ax3.grid(True)
        
        # Plot PnL
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)
        ax4.plot(np.arange(len(pnl)), pnl, label='PnL', alpha=0.5)
        ax4.plot(np.arange(len(cumulative_pnl)), cumulative_pnl, label='Cumulative PnL')
        ax4.set_title('Profit and Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot signal strength vs PnL correlation
        plt.figure(figsize=(10, 6))
        valid_idx = (positions[:-1] != 0)
        if np.sum(valid_idx) > 0:
            plt.scatter(signals[:-1][valid_idx], pnl[valid_idx], alpha=0.5)
            plt.xlabel('Signal Strength')
            plt.ylabel('PnL')
            plt.title('Signal Strength vs PnL')
            plt.grid(True)
            plt.show()


def generate_synthetic_data(n_samples=2000, n_features=10, frequency='D', start_date='2010-01-01'):
    """
    Generate synthetic financial data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features to generate
    frequency : str
        Frequency of the data ('D' for daily, 'M' for monthly, etc.)
    start_date : str
        Start date for the time index
    
    Returns:
    --------
    X : pandas DataFrame
        Synthetic features
    price : pandas Series
        Synthetic price series
    """
    # Generate time index
    dates = pd.date_range(start=start_date, periods=n_samples, freq=frequency)
    
    # Generate synthetic features with some correlation structure
    cov_matrix = np.random.randn(n_features, n_features)
    cov_matrix = cov_matrix.T @ cov_matrix
    
    # Add some autocorrelation to make features more realistic
    X_raw = np.zeros((n_samples, n_features))
    X_raw[0] = np.random.multivariate_normal(np.zeros(n_features), cov_matrix)
    
    for i in range(1, n_samples):
        # AR(1) process with correlation of 0.7
        X_raw[i] = 0.7 * X_raw[i-1] + 0.3 * np.random.multivariate_normal(np.zeros(n_features), cov_matrix)
    
    # Create DataFrame with column names
    X = pd.DataFrame(X_raw, index=dates, columns=[f'Feature_{i+1}' for i in range(n_features)])
    
    # Generate price with some dependency on features
    # We'll use a subset of features to impact the price, making it partially predictable
    feature_weights = np.zeros(n_features)
    feature_weights[:3] = np.random.uniform(-1, 1, 3)  # Only first 3 features affect price
    
    # Generate price as a random walk with some predictable component
    price_raw = np.zeros(n_samples)
    price_raw[0] = 100  # Start at 100
    
    for i in range(1, n_samples):
        # Price change has a small predictable component based on features
        predictable_component = 0.1 * np.dot(X_raw[i-1], feature_weights)
        
        # Random component (larger than predictable to make it challenging)
        random_component = np.random.normal(0, 0.5)
        
        # Price follows a random walk with drift
        price_raw[i] = price_raw[i-1] * (1 + 0.0002 + predictable_component + random_component)
    
    # Create Series with time index
    price = pd.Series(price_raw, index=dates, name='Price')
    
    return X, price


def evaluate_regularization_methods(X, price, training_sizes=[63, 126, 252, 504, 1008], 
                                    regularization_methods=['none', 'l1', 'l2', 'pca', 'significance']):
    """
    Evaluate different regularization methods and training sizes.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features
    price : pandas Series
        Price series
    training_sizes : list
        List of training sizes to evaluate
    regularization_methods : list
        List of regularization methods to evaluate
    
    Returns:
    --------
    results : pandas DataFrame
        Results of the evaluation
    """
    results = []
    
    for training_size in training_sizes:
        for reg_method in regularization_methods:
            # Skip if not enough data for training
            if len(price) <= training_size:
                continue
            
            # Set regularization parameters
            if reg_method == 'l1':
                reg_params = {'lambda1': 0.01}
            elif reg_method == 'l2':
                reg_params = {'lambda2': 0.01}
            elif reg_method == 'pca':
                reg_params = {}
                pca_components = min(5, X.shape[1])
            elif reg_method == 'significance':
                reg_params = {'p_threshold': 0.05}
            else:
                reg_method = None
                reg_params = {}
            
            # Create and train model
            model = OptimalLinearSignal(
                training_size=training_size,
                regularization=reg_method,
                reg_params=reg_params,
                pca_components=pca_components if reg_method == 'pca' else None
            )
            
            # Run backtest
            backtest_results = model.backtest(X, price, plot=False)
            
            # Store results
            results.append({
                'Training Size': training_size,
                'Regularization': reg_method if reg_method else 'none',
                'Sharpe Ratio': backtest_results['sharpe_ratio'],
                'Effective Sharpe': backtest_results['effective_sharpe'],
                'Turnover': backtest_results['turnover'],
                'Effective Turnover': backtest_results['effective_turnover'],
                'Bips': backtest_results['bips'],
                'Effective Bips': backtest_results['effective_bips']
            })
    
    return pd.DataFrame(results)


def evaluate_corrective_factor(X, price, training_size=252):
    """
    Evaluate the effect of the corrective factor.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features
    price : pandas Series
        Price series
    training_size : int
        Training size to use
    
    Returns:
    --------
    results : tuple
        Results with and without corrective factor
    """
    # Model without corrective factor
    model_without = OptimalLinearSignal(
        training_size=training_size,
        regularization='significance',
        reg_params={'p_threshold': 0.05},
        corrective_factor=False
    )
    
    # Model with corrective factor
    model_with = OptimalLinearSignal(
        training_size=training_size,
        regularization='significance',
        reg_params={'p_threshold': 0.05},
        corrective_factor=True
    )
    
    # Run backtests
    results_without = model_without.backtest(X, price, plot=False)
    results_with = model_with.backtest(X, price, plot=False)
    
    # Compare results
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_without['cumulative_pnl'], label='Without Corrective Factor')
    plt.plot(results_with['cumulative_pnl'], label='With Corrective Factor')
    plt.title('Cumulative PnL')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    data = {
        'Without CF': [results_without['sharpe_ratio'], results_without['effective_sharpe']],
        'With CF': [results_with['sharpe_ratio'], results_with['effective_sharpe']]
    }
    index = ['Sharpe Ratio', 'Effective Sharpe']
    df = pd.DataFrame(data, index=index)
    df.plot(kind='bar', ax=plt.gca())
    plt.title('Performance Metrics')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_without, results_with


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    X, price = generate_synthetic_data(n_samples=2500, n_features=10)
    
    # Plot the synthetic data
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(price)
    plt.title('Synthetic Asset Price')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for col in X.columns[:5]:  # Plot first 5 features
        plt.plot(X[col], label=col)
    plt.title('Synthetic Features (First 5)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Basic model demonstration
    print("\nTesting basic model...")
    model = OptimalLinearSignal(training_size=252)
    results = model.backtest(X, price)
    
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Effective Sharpe: {results['effective_sharpe']:.2f}")
    print(f"Turnover: {results['turnover']:.2%}")
    print(f"Effective Turnover: {results['effective_turnover']:.2%}")
    print(f"Bips: {results['bips']:.2f}")
    print(f"Effective Bips: {results['effective_bips']:.2f}")
    
    # Evaluate different regularization methods
    print("\nEvaluating regularization methods...")
    reg_results = evaluate_regularization_methods(
        X, price, 
        training_sizes=[126, 252, 504],
        regularization_methods=['none', 'l1', 'l2', 'pca', 'significance']
    )
    
    # Plot results
    plt.figure(figsize=(16, 12))
    
    # Sharpe Ratio by training size and regularization
    plt.subplot(2, 2, 1)
    sns.barplot(x='Training Size', y='Sharpe Ratio', hue='Regularization', data=reg_results)
    plt.title('Sharpe Ratio by Training Size and Regularization')
    plt.grid(True)
    
    # Effective Sharpe by training size and regularization
    plt.subplot(2, 2, 2)
    sns.barplot(x='Training Size', y='Effective Sharpe', hue='Regularization', data=reg_results)
    plt.title('Effective Sharpe by Training Size and Regularization')
    plt.grid(True)
    
    # Turnover by training size and regularization
    plt.subplot(2, 2, 3)
    sns.barplot(x='Training Size', y='Turnover', hue='Regularization', data=reg_results)
    plt.title('Turnover by Training Size and Regularization')
    plt.grid(True)
    
    # Bips by training size and regularization
    plt.subplot(2, 2, 4)
    sns.barplot(x='Training Size', y='Bips', hue='Regularization', data=reg_results)
    plt.title('Bips by Training Size and Regularization')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate corrective factor
    print("\nEvaluating corrective factor...")
    results_without_cf, results_with_cf = evaluate_corrective_factor(X, price)
    
    # Best model demonstration
    print("\nDemonstrating best model configuration...")
    best_model = OptimalLinearSignal(
        training_size=252,
        regularization='significance',
        reg_params={'p_threshold': 0.05},
        corrective_factor=True
    )
    best_results = best_model.backtest(X, price)
    
    print(f"Best Model - Sharpe Ratio: {best_results['sharpe_ratio']:.2f}")
    print(f"Best Model - Effective Sharpe: {best_results['effective_sharpe']:.2f}")
    print(f"Best Model - Turnover: {best_results['turnover']:.2%}")
    print(f"Best Model - Effective Turnover: {best_results['effective_turnover']:.2%}")
    print(f"Best Model - Bips: {best_results['bips']:.2f}")
    print(f"Best Model - Effective Bips: {best_results['effective_bips']:.2f}")


if __name__ == "__main__":
    main()