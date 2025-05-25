import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)

# For reproducibility
np.random.seed(42)

#########################################################################
# Bloomberg Data Connection
#########################################################################

class BloombergDataManager:
    """Class for handling Bloomberg data connection and retrieval"""
    
    def __init__(self):
        """Initialize Bloomberg connection"""
        self.connection = None
        
    def connect(self):
        """Connect to Bloomberg terminal"""
        try:
            print("Connecting to Bloomberg...")
            self.connection = pdblp.BCon(debug=False, port=8194)
            self.connection.start()
            print("Connected to Bloomberg successfully")
            return True
        except Exception as e:
            print(f"Failed to connect to Bloomberg: {e}")
            return False
            
    def get_index_constituents(self, index_ticker):
        """Get index constituents from Bloomberg"""
        if self.connection is None:
            print("Not connected to Bloomberg")
            return None
        
        try:
            print(f"Retrieving constituents for {index_ticker}...")
            members = self.connection.ref(index_ticker, "INDX_MEMBERS")
            constituents = members.iloc[0, 0].split()
            print(f"Retrieved {len(constituents)} constituents")
            return constituents
        except Exception as e:
            print(f"Error retrieving index constituents: {e}")
            return None
    
    def get_historical_data(self, tickers, start_date, end_date, field='PX_LAST'):
        """Get historical price data from Bloomberg"""
        if self.connection is None:
            print("Not connected to Bloomberg")
            return None
        
        try:
            print(f"Retrieving historical data for {len(tickers)} tickers...")
            data = self.connection.bdh(tickers, field, start_date, end_date)
            
            # Reshape if multiple tickers
            if len(tickers) > 1:
                data.columns = [col[0] for col in data.columns]
            
            print(f"Data retrieved successfully with shape {data.shape}")
            return data
        except Exception as e:
            print(f"Error retrieving historical data: {e}")
            return None
    
    def close(self):
        """Close Bloomberg connection"""
        if self.connection is not None:
            try:
                self.connection.stop()
                print("Bloomberg connection closed")
                self.connection = None
            except Exception as e:
                print(f"Error closing Bloomberg connection: {e}")


#########################################################################
# Adaptive Elastic Net Index Tracking
#########################################################################

class AdaptiveElasticNetIndexTracker:
    """
    Implements index tracking using adaptive elastic net regularization
    as described in the paper "High-dimensional index tracking based on 
    the adaptive elastic net"
    """
    
    def __init__(self, lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1):
        """
        Initialize the Adaptive Elastic Net Index Tracker
        
        Parameters:
        -----------
        lambda1 : float
            Regularization parameter for L1 penalty
        lambda2 : float
            Regularization parameter for L2 penalty
        lambda_c : float
            Regularization parameter for turnover penalty
        tau : float
            Power parameter for adaptive weights
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_c = lambda_c
        self.tau = tau
        self.initial_weights = None  # For turnover penalty
        self.optimal_weights = None
        self.active_assets = None
        self.tracking_error = None
    
    def _calculate_adaptive_weights(self, X, y):
        """
        Calculate adaptive weights based on initial OLS solution
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        v_hat : array-like
            Adaptive weights
        """
        # Full investment constraint matrix for OLS
        n, p = X.shape
        Aeq = np.ones((1, p))
        beq = np.ones((1, 1))
        
        # Solve constrained OLS (full investment only)
        A = X.T @ X / n
        B = X.T @ y / n
        
        # Add a small regularization to avoid singularity
        A_reg = A + 1e-8 * np.eye(p)
        
        # Solve for beta using the formula for equality constrained OLS
        # beta = A^(-1)B + c*A^(-1)*e, where c = (1 - e'A^(-1)B)/(e'A^(-1)e)
        A_inv = np.linalg.inv(A_reg)
        ones = np.ones(p)
        c = (1 - ones.T @ A_inv @ B) / (ones.T @ A_inv @ ones)
        beta_init = A_inv @ B + c * A_inv @ ones
        
        # Apply no-short selling constraint
        beta_init = np.maximum(beta_init, 0)
        
        # Normalize to ensure full investment
        if np.sum(beta_init) > 0:
            beta_init = beta_init / np.sum(beta_init)
        else:
            beta_init = np.ones(p) / p  # Equal weights if all weights are zero
        
        # Calculate adaptive weights
        eps = 1e-8  # Small constant to avoid division by zero
        v_hat = 1 / (np.abs(beta_init) + eps) ** self.tau
        
        return v_hat, beta_init
    
    def _coordinate_descent(self, X, y, v_hat, w_prev=None, max_iter=1000, tol=1e-6):
        """
        Implements coordinate descent algorithm for adaptive elastic net
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        v_hat : array-like
            Adaptive weights
        w_prev : array-like, optional
            Previous weights for turnover penalty
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
            
        Returns:
        --------
        w : array-like
            Optimal weights
        """
        n, p = X.shape
        
        # Initialize weights
        w = np.ones(p) / p
        
        # If no previous weights, use equal weights
        if w_prev is None:
            w_prev = np.zeros(p)
        
        # Precompute X'X and X'y
        XtX = X.T @ X / n
        Xty = X.T @ y / n
        
        # Precompute squared sum of each column in X
        col_squared_sum = np.sum(X**2, axis=0) / n
        
        # Initialize gamma0 (Lagrange multiplier for full investment)
        gamma0 = v_hat.max() * self.lambda1 + self.lambda_c + 0.1
        
        converged = False
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # Update each weight
            for j in range(p):
                # Calculate dj (partial residual)
                Xj = X[:, j]
                y_partial = y - X @ w + Xj * w[j]
                dj = Xj.T @ y_partial / n
                
                # Case logic from paper
                if w_prev[j] < (dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2):
                    w[j] = (dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2)
                elif (dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2) <= w_prev[j] <= (dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2):
                    w[j] = w_prev[j]
                elif 0 <= (dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2) < w_prev[j]:
                    w[j] = (dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2)
                else:
                    w[j] = 0
            
            # Identify sets Su, Sm, Sl for gamma0 update
            Su = np.where((w > w_prev) & (w_prev >= 0))[0]
            Sm = np.where((w == w_prev) & (w_prev > 0))[0]
            Sl = np.where((0 < w) & (w < w_prev))[0]
            
            # Update gamma0
            denominator = np.sum(1 / (col_squared_sum[Su] + 2 * self.lambda2)) + np.sum(1 / (col_squared_sum[Sl] + 2 * self.lambda2))
            if denominator > 0:
                gamma0_numerator = 1 - np.sum(w[Sm])
                gamma0_numerator -= np.sum((dj - v_hat[j] * self.lambda1) / (col_squared_sum[j] + 2 * self.lambda2) for j in np.concatenate((Su, Sl)))
                gamma0_numerator -= np.sum(self.lambda_c / (col_squared_sum[j] + 2 * self.lambda2) for j in Sl)
                gamma0_numerator += np.sum(self.lambda_c / (col_squared_sum[j] + 2 * self.lambda2) for j in Su)
                
                gamma0 = gamma0_numerator / denominator
            
            # Project to ensure full investment and no-short selling
            w = np.maximum(w, 0)
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = np.ones(p) / p  # Equal weights if all weights are zero
            
            # Check convergence
            if np.linalg.norm(w - w_old) < tol:
                converged = True
                break
        
        if not converged:
            print(f"Warning: Coordinate descent did not converge after {max_iter} iterations")
        
        return w
    
    def fit(self, X, y, w_prev=None):
        """
        Fit the adaptive elastic net index tracking model
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        w_prev : array-like, optional
            Previous weights for turnover penalty
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Calculate adaptive weights
        v_hat, beta_init = self._calculate_adaptive_weights(X, y)
        
        # Run coordinate descent to find optimal weights
        self.optimal_weights = self._coordinate_descent(X, y, v_hat, w_prev)
        
        # Identify active assets (non-zero weights)
        self.active_assets = np.where(self.optimal_weights > 1e-6)[0]
        
        # Calculate in-sample tracking error
        y_pred = X @ self.optimal_weights
        self.tracking_error = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return self
    
    def select_parameters_cv(self, X, y, lambda1_range, lambda2_range, n_folds=5):
        """
        Select optimal lambda1 and lambda2 using cross-validation
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        lambda1_range : array-like
            Range of lambda1 values to try
        lambda2_range : array-like
            Range of lambda2 values to try
        n_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        best_lambda1, best_lambda2 : float
            Optimal regularization parameters
        """
        print("Selecting optimal regularization parameters via cross-validation...")
        n_samples = X.shape[0]
        
        # Create cross-validation folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        best_te = float('inf')
        best_lambda1 = None
        best_lambda2 = None
        
        # Grid search
        for lambda1 in lambda1_range:
            for lambda2 in lambda2_range:
                cv_te = 0
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Save current parameters and set new ones
                    orig_lambda1, orig_lambda2 = self.lambda1, self.lambda2
                    self.lambda1, self.lambda2 = lambda1, lambda2
                    
                    # Fit on training data
                    self.fit(X_train, y_train)
                    
                    # Predict and calculate validation tracking error
                    y_pred = X_val @ self.optimal_weights
                    fold_te = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    cv_te += fold_te
                    
                    # Restore original parameters
                    self.lambda1, self.lambda2 = orig_lambda1, orig_lambda2
                
                # Average tracking error across folds
                cv_te /= n_folds
                
                if cv_te < best_te:
                    best_te = cv_te
                    best_lambda1 = lambda1
                    best_lambda2 = lambda2
        
        print(f"Optimal parameters: lambda1={best_lambda1}, lambda2={best_lambda2}, CV TE={best_te:.6f}")
        
        # Update parameters to the best values
        self.lambda1 = best_lambda1
        self.lambda2 = best_lambda2
        
        return best_lambda1, best_lambda2
    
    def get_weights(self):
        """Return the optimal weights"""
        return self.optimal_weights
    
    def get_active_assets(self):
        """Return the indices of active assets"""
        return self.active_assets
    
    def get_tracking_error(self):
        """Return the tracking error"""
        return self.tracking_error


#########################################################################
# Index Tracking Strategy
#########################################################################

class IndexTrackingStrategy:
    """
    Implements an index tracking strategy using the Adaptive Elastic Net approach
    """
    
    def __init__(self, index_ticker, lookback_window=250, rebalance_period=21, 
                 lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1):
        """
        Initialize the index tracking strategy
        
        Parameters:
        -----------
        index_ticker : str
            Bloomberg ticker for the index to track
        lookback_window : int
            Number of days to use for training
        rebalance_period : int
            Number of days between rebalances
        lambda1, lambda2, lambda_c, tau : float
            Parameters for the adaptive elastic net model
        """
        self.index_ticker = index_ticker
        self.lookback_window = lookback_window
        self.rebalance_period = rebalance_period
        
        self.bloomberg = BloombergDataManager()
        self.model = AdaptiveElasticNetIndexTracker(lambda1, lambda2, lambda_c, tau)
        
        self.index_data = None
        self.constituent_data = None
        self.constituents = None
        self.weights = None
        self.active_assets = None
        self.tracking_performance = None
    
    def _prepare_data(self, start_date, end_date):
        """
        Prepare data for the strategy
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        """
        # Connect to Bloomberg
        if not self.bloomberg.connect():
            print("Failed to connect to Bloomberg. Exiting.")
            return False
        
        # Get index constituents
        self.constituents = self.bloomberg.get_index_constituents(self.index_ticker)
        if self.constituents is None or len(self.constituents) == 0:
            print("Failed to retrieve index constituents. Exiting.")
            self.bloomberg.close()
            return False
        
        # Get historical data for index and constituents
        tickers = [self.index_ticker] + self.constituents
        price_data = self.bloomberg.get_historical_data(tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            print("Failed to retrieve historical data. Exiting.")
            self.bloomberg.close()
            return False
        
        # Handle missing values
        price_data = price_data.fillna(method='ffill').dropna(axis=1)
        
        # Calculate log returns
        returns_data = np.log(price_data / price_data.shift(1)).dropna()
        
        # Separate index and constituent returns
        self.index_data = returns_data[self.index_ticker]
        self.constituent_data = returns_data.drop(columns=[self.index_ticker])
        
        # Update constituents list to match data columns (some may be missing)
        self.constituents = list(self.constituent_data.columns)
        
        # Close Bloomberg connection
        self.bloomberg.close()
        
        print(f"Prepared data with {len(self.constituents)} constituents and {len(self.index_data)} days")
        return True
    
    def run_backtest(self, start_date, end_date, optimize_params=False):
        """
        Run backtest of the index tracking strategy
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        optimize_params : bool
            Whether to optimize lambda1 and lambda2 via cross-validation
            
        Returns:
        --------
        performance : dict
            Backtest performance metrics
        """
        if not self._prepare_data(start_date, end_date):
            return None
        
        # Convert dates to pandas datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Initialize performance tracking
        dates = self.index_data.index
        train_end_idx = dates.get_loc(dates[dates >= start][0]) + self.lookback_window
        
        # Initialize tracking variables
        performance = {
            'date': [],
            'portfolio_return': [],
            'index_return': [],
            'tracking_error': [],
            'active_return': [],
            'num_assets': [],
            'turnover': []
        }
        
        current_weights = None
        
        # Optimize parameters if requested
        if optimize_params:
            # Use first training window for optimization
            train_data = self.constituent_data.iloc[:train_end_idx]
            train_index = self.index_data.iloc[:train_end_idx]
            
            # Define parameter ranges
            lambda1_range = np.logspace(-7, -3, 10)
            lambda2_range = np.logspace(-5, -1, 10)
            
            # Run cross-validation
            self.model.select_parameters_cv(
                train_data.values, 
                train_index.values, 
                lambda1_range, 
                lambda2_range
            )
            
            print(f"Selected parameters: lambda1={self.model.lambda1}, lambda2={self.model.lambda2}")
        
        # Main backtest loop
        for i in range(train_end_idx, len(dates), self.rebalance_period):
            # Determine training window
            train_start_idx = max(0, i - self.lookback_window)
            train_end_idx = i
            
            # Prepare training data
            X_train = self.constituent_data.iloc[train_start_idx:train_end_idx].values
            y_train = self.index_data.iloc[train_start_idx:train_end_idx].values
            
            # Fit model
            start_time = time.time()
            self.model.fit(X_train, y_train, current_weights)
            end_time = time.time()
            
            print(f"Fitting model took {end_time - start_time:.2f} seconds")
            
            # Get weights and active assets
            new_weights = self.model.get_weights()
            active_assets = self.model.get_active_assets()
            
            # Calculate turnover
            turnover = 0
            if current_weights is not None:
                turnover = np.sum(np.abs(new_weights - current_weights))
            
            # Update current weights
            current_weights = new_weights
            
            # Determine out-of-sample period
            test_start_idx = i
            test_end_idx = min(i + self.rebalance_period, len(dates))
            
            # Calculate out-of-sample returns
            for j in range(test_start_idx, test_end_idx):
                if j >= len(dates):
                    continue
                    
                date = dates[j]
                
                # Get constituent returns for this day
                if j > 0:  # Skip first day (no return calculation possible)
                    const_returns = self.constituent_data.iloc[j]
                    index_return = self.index_data.iloc[j]
                    
                    # Calculate portfolio return
                    portfolio_return = np.sum(const_returns * current_weights)
                    
                    # Calculate tracking error and active return
                    tracking_error = (portfolio_return - index_return) ** 2
                    active_return = portfolio_return - index_return
                    
                    # Store performance
                    performance['date'].append(date)
                    performance['portfolio_return'].append(portfolio_return)
                    performance['index_return'].append(index_return)
                    performance['tracking_error'].append(tracking_error)
                    performance['active_return'].append(active_return)
                    performance['num_assets'].append(len(active_assets))
                    performance['turnover'].append(turnover)
            
            # Print progress
            print(f"Completed period ending {dates[min(test_end_idx-1, len(dates)-1)].strftime('%Y-%m-%d')}")
            print(f"Number of active assets: {len(active_assets)}")
            print(f"Turnover: {turnover:.4f}")
            print("-" * 50)
        
        # Convert performance to DataFrame
        self.tracking_performance = pd.DataFrame(performance)
        
        # Calculate cumulative performance
        self.tracking_performance['cumulative_portfolio'] = (1 + self.tracking_performance['portfolio_return']).cumprod()
        self.tracking_performance['cumulative_index'] = (1 + self.tracking_performance['index_return']).cumprod()
        
        # Calculate additional metrics
        self._calculate_performance_metrics()
        
        return self.tracking_performance
    
    def _calculate_performance_metrics(self):
        """Calculate and print summary performance metrics"""
        if self.tracking_performance is None or len(self.tracking_performance) == 0:
            print("No performance data available")
            return
        
        # Calculate metrics
        annualized_tracking_error = np.sqrt(np.mean(self.tracking_performance['tracking_error'])) * np.sqrt(252)
        annualized_active_return = np.mean(self.tracking_performance['active_return']) * 252
        annualized_portfolio_return = np.mean(self.tracking_performance['portfolio_return']) * 252
        annualized_index_return = np.mean(self.tracking_performance['index_return']) * 252
        
        correlation = np.corrcoef(
            self.tracking_performance['portfolio_return'],
            self.tracking_performance['index_return']
        )[0, 1]
        
        average_num_assets = np.mean(self.tracking_performance['num_assets'])
        average_turnover = np.mean(self.tracking_performance['turnover'])
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Annualized Tracking Error: {annualized_tracking_error:.4f}")
        print(f"Annualized Active Return: {annualized_active_return:.4f}")
        print(f"Annualized Portfolio Return: {annualized_portfolio_return:.4f}")
        print(f"Annualized Index Return: {annualized_index_return:.4f}")
        print(f"Correlation with Index: {correlation:.4f}")
        print(f"Average Number of Active Assets: {average_num_assets:.2f}")
        print(f"Average Turnover: {average_turnover:.4f}")
    
    def plot_performance(self, save_path=None):
        """
        Plot performance of the tracking portfolio
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.tracking_performance is None or len(self.tracking_performance) == 0:
            print("No performance data available")
            return
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot cumulative returns
        axs[0].plot(self.tracking_performance['date'], self.tracking_performance['cumulative_portfolio'], 
                   label='Tracking Portfolio', linewidth=2)
        axs[0].plot(self.tracking_performance['date'], self.tracking_performance['cumulative_index'], 
                   label='Index', linewidth=2, linestyle='--')
        axs[0].set_title('Cumulative Performance')
        axs[0].set_ylabel('Value')
        axs[0].set_xlabel('Date')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot active returns
        axs[1].plot(self.tracking_performance['date'], self.tracking_performance['active_return'], 
                   color='green', label='Active Return')
        axs[1].axhline(y=0, color='r', linestyle='-')
        axs[1].set_title('Active Returns')
        axs[1].set_ylabel('Return')
        axs[1].set_xlabel('Date')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot number of active assets
        axs[2].plot(self.tracking_performance['date'], self.tracking_performance['num_assets'], 
                   color='purple', label='Active Assets')
        axs[2].set_title('Number of Active Assets')
        axs[2].set_ylabel('Count')
        axs[2].set_xlabel('Date')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


#########################################################################
# Main Function to Run the Strategy
#########################################################################

def run_index_tracking_strategy(index_ticker, start_date, end_date, 
                               lookback_window=250, rebalance_period=21,
                               lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1,
                               optimize_params=True):
    """
    Run the index tracking strategy with the specified parameters
    
    Parameters:
    -----------
    index_ticker : str
        Bloomberg ticker for the index to track
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    lookback_window : int
        Number of days to use for training
    rebalance_period : int
        Number of days between rebalances
    lambda1, lambda2, lambda_c, tau : float
        Parameters for the adaptive elastic net model
    optimize_params : bool
        Whether to optimize lambda1 and lambda2 via cross-validation
        
    Returns:
    --------
    strategy : IndexTrackingStrategy
        The executed strategy object
    """
    # Initialize strategy
    strategy = IndexTrackingStrategy(
        index_ticker=index_ticker,
        lookback_window=lookback_window,
        rebalance_period=rebalance_period,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_c=lambda_c,
        tau=tau
    )
    
    # Run backtest
    strategy.run_backtest(start_date, end_date, optimize_params)
    
    # Plot performance
    strategy.plot_performance(save_path=f"index_tracking_{index_ticker.replace(' ', '_')}.png")
    
    return strategy


# Example usage
if __name__ == "__main__":
    # Example with S&P 100 index
    run_index_tracking_strategy(
        index_ticker="SPX Index",
        start_date="2018-01-01",
        end_date="2023-01-01",
        lookback_window=250,
        rebalance_period=21,  # Monthly rebalancing
        lambda1=1.44e-6,
        lambda2=3.79e-3,
        lambda_c=5e-5,  # Small turnover penalty
        tau=1,
        optimize_params=True
    )
    
    # Example with FTSE 100 index
    run_index_tracking_strategy(
        index_ticker="UKX Index",
        start_date="2018-01-01",
        end_date="2023-01-01",
        lookback_window=250,
        rebalance_period=21,
        lambda1=1.44e-6,
        lambda2=2.98e-3,
        lambda_c=5e-5,
        tau=1,
        optimize_params=True
    )
    
    # Example with Nikkei 225 index
    run_index_tracking_strategy(
        index_ticker="NKY Index",
        start_date="2018-01-01",
        end_date="2023-01-01",
        lookback_window=250,
        rebalance_period=21,
        lambda1=1.06e-5,
        lambda2=3.79e-2,
        lambda_c=5e-5,
        tau=1,
        optimize_params=True
    )