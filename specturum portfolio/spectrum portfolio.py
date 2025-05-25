import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class DynamicSpectralPortfolioCuts:
    """
    Implementation of the Dynamic Spectral Portfolio Cuts approach from the paper
    "Dynamic Portfolio Cuts: A Spectral Approach to Graph-Theoretic Diversification"
    """
    
    def __init__(self, frequency_bins=12, window_size=252):
        """
        Initialize the model parameters.
        
        Args:
            frequency_bins: Number of frequency bins for spectral estimation (M in the paper)
            window_size: Number of days to use for covariance estimation
        """
        self.M = frequency_bins
        self.window_size = window_size
        self.omegas = np.linspace(0, 2*np.pi*(1-1/frequency_bins), frequency_bins)
    
    def _create_spectral_basis(self, t, N):
        """
        Create the augmented spectral basis Φ(t, ω) as defined in equation (13)-(14).
        
        Args:
            t: Time index array
            N: Number of assets
        
        Returns:
            Augmented spectral basis matrix
        """
        T = len(t)
        Phi = np.zeros((T, N, 2*self.M*N), dtype=complex)
        
        for i, ti in enumerate(t):
            phi_t = np.zeros((N, self.M*N), dtype=complex)
            for j, omega in enumerate(self.omegas):
                phi_t[:, j*N:(j+1)*N] = np.exp(1j*omega*ti) * np.eye(N) / np.sqrt(2*self.M)
            
            Phi_t = np.hstack([phi_t, np.conj(phi_t)])
            Phi[i] = Phi_t
            
        return Phi
    
    def _estimate_spectral_moments(self, returns, Phi):
        """
        Estimate the augmented spectral moments as per equations (18)-(19).
        
        Args:
            returns: Asset returns data
            Phi: Augmented spectral basis
        
        Returns:
            Estimated spectral mean and covariance
        """
        T, N = returns.shape
        
        # Compute spectral mean
        m_omega = np.zeros((2*self.M*N), dtype=complex)
        for t in range(T):
            m_omega += np.conj(Phi[t].T) @ returns[t]
        m_omega /= T
        
        # Compute time-varying mean
        m_t = np.zeros((T, N))
        for t in range(T):
            m_t[t] = np.real(Phi[t] @ m_omega)
        
        # Center the returns
        s_t = returns - m_t
        
        # Compute spectral covariance
        R_omega = np.zeros((2*self.M*N, 2*self.M*N), dtype=complex)
        for t in range(T):
            R_omega += np.outer(np.conj(Phi[t].T) @ s_t[t], s_t[t].T @ Phi[t])
        R_omega /= T
        
        return m_omega, R_omega
    
    def _compute_time_varying_covariance(self, R_omega, Phi, t_range):
        """
        Compute the time-varying covariance as per equation (10).
        
        Args:
            R_omega: Spectral covariance
            Phi: Augmented spectral basis
            t_range: Time points to compute covariance for
        
        Returns:
            Time-varying covariance matrices
        """
        T_out = len(t_range)
        N = Phi.shape[1]
        R_t = np.zeros((T_out, N, N))
        
        for i, t in enumerate(t_range):
            # Recreate the spectral basis for time t
            phi_t = np.zeros((N, self.M*N), dtype=complex)
            for j, omega in enumerate(self.omegas):
                phi_t[:, j*N:(j+1)*N] = np.exp(1j*omega*t) * np.eye(N) / np.sqrt(2*self.M)
            
            Phi_t = np.hstack([phi_t, np.conj(phi_t)])
            
            # Compute time-varying covariance
            R_t[i] = np.real(Phi_t @ R_omega @ np.conj(Phi_t.T))
            
            # Ensure the matrix is positive semi-definite
            R_t[i] = (R_t[i] + R_t[i].T) / 2  # Ensure symmetry
            min_eig = np.min(np.real(np.linalg.eigvals(R_t[i])))
            if min_eig < 0:
                R_t[i] += np.eye(N) * (abs(min_eig) + 1e-6)
            
        return R_t
    
    def _create_dynamic_market_graph(self, R_t):
        """
        Create a dynamic market graph from time-varying covariance as per equation (29).
        
        Args:
            R_t: Time-varying covariance matrices
        
        Returns:
            Dynamic weight matrices
        """
        T, N, _ = R_t.shape
        W_t = np.zeros((T, N, N))
        
        for t in range(T):
            # Create diagonal matrix V(t) with inverse square root of diagonal elements
            V_t = np.diag(1.0 / np.sqrt(np.diag(R_t[t])))
            
            # Compute weight matrix as per equation (29)
            W_t[t] = V_t @ np.abs(R_t[t]) @ V_t
            
            # Set diagonal to 1 as per equation (23)
            np.fill_diagonal(W_t[t], 1.0)
            
            # Ensure symmetry
            W_t[t] = (W_t[t] + W_t[t].T) / 2
            
        return W_t
    
    def _compute_dynamic_laplacian(self, W_t):
        """
        Compute the time-varying Laplacian matrix as per equation (22).
        
        Args:
            W_t: Dynamic weight matrices
        
        Returns:
            Dynamic Laplacian matrices
        """
        T, N, _ = W_t.shape
        L_t = np.zeros((T, N, N))
        
        for t in range(T):
            # Compute degree matrix
            D_t = np.diag(np.sum(W_t[t], axis=1))
            
            # Compute Laplacian
            L_t[t] = D_t - W_t[t]
            
        return L_t
    
    def _spectral_clustering(self, L_t, n_clusters):
        """
        Perform dynamic spectral clustering based on the Laplacian matrices.
        
        Args:
            L_t: Dynamic Laplacian matrices
            n_clusters: Number of clusters (K in the paper)
        
        Returns:
            Time-varying cluster assignments
        """
        T, N, _ = L_t.shape
        clusters = np.zeros((T, N), dtype=int)
        
        for t in range(T):
            # Compute eigenvectors of the Laplacian
            # We use the k smallest eigenvectors (excluding the first one)
            eigenvalues, eigenvectors = eigsh(L_t[t], k=min(n_clusters+1, N-1), which='SM')
            
            # Sort by eigenvalues
            idx = np.argsort(eigenvalues)[1:min(n_clusters+1, N)]  # Skip the first eigenvector
            vectors = eigenvectors[:, idx]
            
            # Perform KMeans clustering on the eigenvectors
            kmeans = KMeans(n_clusters=min(n_clusters, N), random_state=0)
            clusters[t] = kmeans.fit_predict(vectors)
            
        return clusters
    
    def _generate_portfolio_weights(self, clusters, allocation_type='binary'):
        """
        Generate portfolio weights based on cluster assignments.
        
        Args:
            clusters: Time-varying cluster assignments
            allocation_type: Type of capital allocation:
                - 'binary': 1/(2^Ki) (Case 1 in paper)
                - 'equal': 1/(K+1) (Case 2 in paper)
        
        Returns:
            Time-varying portfolio weights
        """
        T, N = clusters.shape
        K = np.max(clusters) + 1  # Number of clusters
        weights = np.zeros((T, N))
        
        for t in range(T):
            unique_clusters = np.unique(clusters[t])
            
            for cluster in unique_clusters:
                cluster_indices = np.where(clusters[t] == cluster)[0]
                n_assets_in_cluster = len(cluster_indices)
                
                if allocation_type == 'binary':
                    # Case 1: hi = 1/(2^Ki)
                    Ki = int(np.log2(K)) if K > 1 else 1  # Number of cuts to reach this cluster
                    cluster_weight = 1.0 / (2**Ki)
                else:
                    # Case 2: hi = 1/(K+1)
                    cluster_weight = 1.0 / (K + 1)
                
                # Equal weight within each cluster
                asset_weight = cluster_weight / n_assets_in_cluster
                weights[t, cluster_indices] = asset_weight
                
        # Normalize weights to sum to 1
        for t in range(T):
            # Handle case where weights might be all zeros
            if np.sum(weights[t]) > 0:
                weights[t] /= np.sum(weights[t])
            else:
                # Fallback to equal weights if all weights are zero
                weights[t] = np.ones(N) / N
            
        return weights
    
    def fit(self, returns):
        """
        Fit the model to the training data.
        
        Args:
            returns: Asset returns data
        
        Returns:
            self
        """
        T, N = returns.shape
        
        # Create time index
        t = np.arange(T)
        
        # Create spectral basis
        Phi = self._create_spectral_basis(t, N)
        
        # Estimate spectral moments
        self.m_omega, self.R_omega = self._estimate_spectral_moments(returns, Phi)
        
        # Store dimensions
        self.N = N
        self.Phi = Phi
        
        return self
    
    def predict(self, n_clusters=2, allocation_type='binary', t_pred=None):
        """
        Generate portfolio weights for future periods.
        
        Args:
            n_clusters: Number of clusters (K in the paper)
            allocation_type: Type of capital allocation ('binary' or 'equal')
            t_pred: Time points to predict for (if None, use the training time points)
        
        Returns:
            Portfolio weights for each time period
        """
        if t_pred is None:
            t_pred = np.arange(len(self.Phi))
        
        # Compute time-varying covariance
        R_t = self._compute_time_varying_covariance(self.R_omega, self.Phi, t_pred)
        
        # Create dynamic market graph
        W_t = self._create_dynamic_market_graph(R_t)
        
        # Compute dynamic Laplacian
        L_t = self._compute_dynamic_laplacian(W_t)
        
        # Perform spectral clustering
        clusters = self._spectral_clustering(L_t, n_clusters)
        
        # Generate portfolio weights
        weights = self._generate_portfolio_weights(clusters, allocation_type)
        
        return weights
    
    def backtest(self, returns_train, returns_test, n_clusters=2, allocation_type='binary'):
        """
        Backtest the strategy on historical data.
        
        Args:
            returns_train: Training data for model fitting
            returns_test: Test data for evaluation
            n_clusters: Number of clusters (K in the paper)
            allocation_type: Type of capital allocation ('binary' or 'equal')
        
        Returns:
            Portfolio performance metrics
        """
        # Fit the model
        self.fit(returns_train)
        
        # Generate weights for test period
        T_test = len(returns_test)
        t_pred = np.arange(len(self.Phi), len(self.Phi) + T_test)
        weights = self.predict(n_clusters, allocation_type, t_pred)
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros(T_test)
        for t in range(T_test):
            # Portfolio return = weighted sum of asset returns
            portfolio_returns[t] = np.sum(weights[t] * returns_test[t])
        
        # Calculate performance metrics
        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        mean_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        return {
            'weights': weights,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }


def fetch_bloomberg_data(tickers, start_date, end_date):
    """
    Fetch price data from Bloomberg for the given tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
    
    Returns:
        DataFrame of daily returns
    """
    try:
        import blpapi
        print("Bloomberg API available, fetching real data...")
        
        # Initialize Bloomberg session
        session = blpapi.Session()
        if not session.start():
            raise RuntimeError("Failed to start Bloomberg session")
        
        # Open Bloomberg service
        if not session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata service")
        
        # Get the service
        refDataService = session.getService("//blp/refdata")
        
        # Create request
        prices = {}
        
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Set request parameters
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", "DAILY")
            
            # Append fields one by one
            request.append("fields", "PX_LAST")
            
            # Add the ticker
            request.append("securities", ticker)
            
            try:
                # Send request
                session.sendRequest(request)
                
                # Process response
                while True:
                    event = session.nextEvent(500)
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        for msg in event:
                            security_data = msg.getElement("securityData")
                            ticker_name = security_data.getElement("security").getValue()
                            field_data = security_data.getElement("fieldData")
                            
                            # Extract dates and prices
                            dates = []
                            values = []
                            
                            for i in range(field_data.numValues()):
                                field_value = field_data.getValue(i)
                                date = field_value.getElement("date").getValue()
                                px_last = field_value.getElement("PX_LAST").getValue()
                                
                                dates.append(date.strftime("%Y-%m-%d"))
                                values.append(px_last)
                            
                            # Store in dictionary
                            prices[ticker_name] = pd.Series(values, index=pd.DatetimeIndex(dates))
                        
                        break
                    
                    if event.eventType() == blpapi.Event.TIMEOUT:
                        break
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Stop session
        session.stop()
        
        if not prices:
            raise RuntimeError("No data retrieved from Bloomberg")
        
        # Create price DataFrame
        price_df = pd.DataFrame(prices)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        print(f"Successfully retrieved data for {len(prices)} tickers")
        return returns_df
    
    except Exception as e:
        print(f"Error with Bloomberg API: {e}")
        print("Using synthetic data instead.")
        np.random.seed(42)
        
        # Generate synthetic data
        n_assets = len(tickers)
        
        # Create dates
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate random returns with some correlation
        mean_returns = np.random.normal(0.0005, 0.0002, n_assets)
        cov_matrix = np.random.normal(0, 0.0001, (n_assets, n_assets))
        cov_matrix = cov_matrix.T @ cov_matrix  # Ensure positive semi-definite
        np.fill_diagonal(cov_matrix, np.random.uniform(0.0003, 0.0015, n_assets))
        
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
        
        # Create DataFrame
        returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
        
        print(f"Generated synthetic data with {len(dates)} days for {n_assets} assets")
        return returns_df


def compare_strategies(returns_df, train_end_date, test_end_date, n_clusters=2):
    """
    Compare different portfolio strategies.
    
    Args:
        returns_df: DataFrame of asset returns
        train_end_date: End date for training period
        test_end_date: End date for test period
        n_clusters: Number of clusters for spectral methods
    
    Returns:
        Dictionary of strategy performances
    """
    print("Splitting data into train and test sets...")
    # Split data into train and test
    train_df = returns_df.loc[:train_end_date]
    test_df = returns_df.loc[train_end_date:test_end_date]
    
    # Convert to numpy arrays
    returns_train = train_df.values
    returns_test = test_df.values
    
    print(f"Training data shape: {returns_train.shape}")
    print(f"Testing data shape: {returns_test.shape}")
    
    # Initialize strategies
    strategies = {}
    
    print("Running Dynamic Spectral Portfolio Cuts with binary allocation...")
    # 1. Dynamic Spectral Portfolio Cuts with binary allocation
    dpc_binary = DynamicSpectralPortfolioCuts(frequency_bins=12, window_size=min(252, len(returns_train)))
    strategies['SpectralCutN_binary'] = dpc_binary.backtest(
        returns_train, returns_test, n_clusters=n_clusters, allocation_type='binary'
    )
    
    print("Running Dynamic Spectral Portfolio Cuts with equal allocation...")
    # 2. Dynamic Spectral Portfolio Cuts with equal allocation
    dpc_equal = DynamicSpectralPortfolioCuts(frequency_bins=12, window_size=min(252, len(returns_train)))
    strategies['SpectralCutN_equal'] = dpc_equal.backtest(
        returns_train, returns_test, n_clusters=n_clusters, allocation_type='equal'
    )
    
    print("Running static portfolio cut strategies...")
    # 3. Static portfolio cut with binary allocation (using simple covariance)
    cov_matrix = np.cov(returns_train.T)
    
    # Create static graph Laplacian
    N = returns_train.shape[1]
    V_static = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
    W_static = V_static @ np.abs(cov_matrix) @ V_static
    np.fill_diagonal(W_static, 1.0)
    D_static = np.diag(np.sum(W_static, axis=1))
    L_static = D_static - W_static
    
    # Perform static spectral clustering
    eigenvalues, eigenvectors = eigsh(L_static, k=min(n_clusters+1, N-1), which='SM')
    idx = np.argsort(eigenvalues)[1:min(n_clusters+1, N)]  # Skip the first eigenvector
    vectors = eigenvectors[:, idx]
    
    kmeans = KMeans(n_clusters=min(n_clusters, N), random_state=0)
    static_clusters = kmeans.fit_predict(vectors)
    
    # Generate static weights
    static_weights_binary = np.zeros(N)
    static_weights_equal = np.zeros(N)
    
    for cluster in range(min(n_clusters, N)):
        cluster_indices = np.where(static_clusters == cluster)[0]
        n_assets_in_cluster = len(cluster_indices)
        
        # Binary allocation
        Ki = int(np.log2(n_clusters)) if n_clusters > 1 else 1
        cluster_weight_binary = 1.0 / (2**Ki)
        
        # Equal allocation
        cluster_weight_equal = 1.0 / (n_clusters + 1)
        
        # Assign weights
        static_weights_binary[cluster_indices] = cluster_weight_binary / n_assets_in_cluster
        static_weights_equal[cluster_indices] = cluster_weight_equal / n_assets_in_cluster
    
    # Normalize weights
    if np.sum(static_weights_binary) > 0:
        static_weights_binary /= np.sum(static_weights_binary)
    else:
        static_weights_binary = np.ones(N) / N
        
    if np.sum(static_weights_equal) > 0:
        static_weights_equal /= np.sum(static_weights_equal)
    else:
        static_weights_equal = np.ones(N) / N
    
    # Calculate returns for static cut with binary allocation
    static_binary_returns = np.zeros(len(returns_test))
    for t in range(len(returns_test)):
        static_binary_returns[t] = np.sum(static_weights_binary * returns_test[t])
    
    static_binary_cumulative = np.cumprod(1 + static_binary_returns) - 1
    
    # Calculate returns for static cut with equal allocation
    static_equal_returns = np.zeros(len(returns_test))
    for t in range(len(returns_test)):
        static_equal_returns[t] = np.sum(static_weights_equal * returns_test[t])
    
    static_equal_cumulative = np.cumprod(1 + static_equal_returns) - 1
    
    # Store static cut strategies
    strategies['CutN_binary'] = {
        'returns': static_binary_returns,
        'cumulative_returns': static_binary_cumulative,
        'mean_return': np.mean(static_binary_returns),
        'volatility': np.std(static_binary_returns),
        'sharpe_ratio': np.mean(static_binary_returns) / np.std(static_binary_returns) if np.std(static_binary_returns) > 0 else 0
    }
    
    strategies['CutN_equal'] = {
        'returns': static_equal_returns,
        'cumulative_returns': static_equal_cumulative,
        'mean_return': np.mean(static_equal_returns),
        'volatility': np.std(static_equal_returns),
        'sharpe_ratio': np.mean(static_equal_returns) / np.std(static_equal_returns) if np.std(static_equal_returns) > 0 else 0
    }
    
    print("Running equally weighted portfolio...")
    # 4. Equally Weighted Portfolio
    ew_returns = np.mean(returns_test, axis=1)
    ew_cumulative = np.cumprod(1 + ew_returns) - 1
    strategies['EquallyWeighted'] = {
        'returns': ew_returns,
        'cumulative_returns': ew_cumulative,
        'mean_return': np.mean(ew_returns),
        'volatility': np.std(ew_returns),
        'sharpe_ratio': np.mean(ew_returns) / np.std(ew_returns) if np.std(ew_returns) > 0 else 0
    }
    
    print("Running minimum variance portfolio...")
    # 5. Minimum Variance Portfolio
    try:
        cov_matrix = np.cov(returns_train.T)
        # Add small diagonal to ensure positive definite
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-5
        
        ones = np.ones(returns_train.shape[1])
        inv_cov = np.linalg.inv(cov_matrix)
        mv_weights = inv_cov @ ones
        mv_weights = mv_weights / np.sum(mv_weights)
        
        # Apply weights to test data
        mv_returns = np.zeros(len(returns_test))
        for t in range(len(returns_test)):
            mv_returns[t] = np.sum(mv_weights * returns_test[t])
        
        mv_cumulative = np.cumprod(1 + mv_returns) - 1
        strategies['MinVariance'] = {
            'returns': mv_returns,
            'cumulative_returns': mv_cumulative,
            'mean_return': np.mean(mv_returns),
            'volatility': np.std(mv_returns),
            'sharpe_ratio': np.mean(mv_returns) / np.std(mv_returns) if np.std(mv_returns) > 0 else 0
        }
    except Exception as e:
        print(f"Error computing minimum variance portfolio: {e}")
        print("Skipping minimum variance portfolio")
    
    return strategies


def plot_strategy_performance(strategies, title):
    """
    Plot the performance of different strategies.
    
    Args:
        strategies: Dictionary of strategy performances
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    for name, perf in strategies.items():
        plt.plot(100 * perf['cumulative_returns'], label=f"{name} (Sharpe: {perf['sharpe_ratio']:.2f})")
    
    plt.title(title, fontsize=16)
    plt.xlabel('Trading Days', fontsize=14)
    plt.ylabel('Cumulative Return (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('dynamic_portfolio_cuts_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main execution
def main():
    try:
        print("Starting Dynamic Portfolio Cuts strategy analysis...")
        
        # Define parameters
        start_date = pd.Timestamp('2018-01-01')
        train_end_date = pd.Timestamp('2020-01-01')
        test_end_date = pd.Timestamp('2021-05-17')
        
        # Example: use BCOM constituents
        bcom_tickers = [
            "CL1 Comdty",  # WTI Crude Oil
            "CO1 Comdty",  # Brent Crude
            "XB1 Comdty",  # RBOB Gasoline
            "HO1 Comdty",  # Heating Oil
            "NG1 Comdty",  # Natural Gas
            "GC1 Comdty",  # Gold
            "SI1 Comdty",  # Silver
            "HG1 Comdty",  # Copper
            "C 1 Comdty",  # Corn
            "W 1 Comdty",  # Wheat
            "S 1 Comdty",  # Soybeans
            "SM1 Comdty",  # Soybean Meal
            "BO1 Comdty",  # Soybean Oil
            "KC1 Comdty",  # Coffee
            "SB1 Comdty",  # Sugar
            "CT1 Comdty",  # Cotton
            "CC1 Comdty",  # Cocoa
            "LH1 Comdty",  # Lean Hogs
            "LC1 Comdty",  # Live Cattle
            "LA1 Comdty",  # Aluminum
            "LN1 Comdty",  # Nickel
            "LX1 Comdty",  # Zinc
            "LP1 Comdty",  # Lead
        ]
        
        # Fetch data
        returns_df = fetch_bloomberg_data(bcom_tickers, start_date, test_end_date)
        
        # Number of clusters (K in the paper)
        n_clusters = 15
        print(f"Using {n_clusters} clusters for strategy testing...")
        
        # Compare strategies
        strategies = compare_strategies(returns_df, train_end_date, test_end_date, n_clusters=n_clusters)
        
        # Plot results
        plot_strategy_performance(strategies, f"BCOM Portfolio Strategies Comparison (K={n_clusters})")
        
        # Print Sharpe ratios
        print("\nStrategy Sharpe Ratios:")
        for name, perf in strategies.items():
            print(f"{name}: {perf['sharpe_ratio']:.4f}")
        
        # Save results to CSV
        results = {
            'Strategy': [],
            'Mean Return': [],
            'Volatility': [],
            'Sharpe Ratio': [],
            'Final Return': []
        }
        
        for name, perf in strategies.items():
            results['Strategy'].append(name)
            results['Mean Return'].append(perf['mean_return'])
            results['Volatility'].append(perf['volatility'])
            results['Sharpe Ratio'].append(perf['sharpe_ratio'])
            results['Final Return'].append(perf['cumulative_returns'][-1])
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('dynamic_portfolio_cuts_results.csv', index=False)
        print("\nResults saved to 'dynamic_portfolio_cuts_results.csv'")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()