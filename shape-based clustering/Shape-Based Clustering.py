import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import dtw
from tqdm import tqdm
import pdblp
import datetime
import random
import warnings
warnings.filterwarnings('ignore')

class ShapeBasedPortfolioManager:
    def __init__(self, tickers=None):
        """
        Initialize the Shape-Based Portfolio Manager
        
        Parameters:
        -----------
        tickers : list
            List of tickers to analyze
        """
        self.tickers = tickers
        self.prices = None
        self.returns = None
        self.cumulative_returns = None
        self.clusters = None
        self.cluster_labels = None
        self.distance_matrix = None
        
        # Connect to Bloomberg
        try:
            self.bbg = pdblp.BCon(timeout=60000)
            self.bbg.start()
            print("Connected to Bloomberg")
            self.has_bloomberg = True
        except Exception as e:
            print(f"Could not connect to Bloomberg: {e}")
            print("Will use sample data instead")
            self.has_bloomberg = False
    
    def fetch_data(self, start_date, end_date, field='PX_LAST'):
        """
        Fetch price data from Bloomberg
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        field : str
            Bloomberg field (default: PX_LAST)
            
        Returns:
        --------
        DataFrame of prices
        """
        if not self.has_bloomberg:
            print("Bloomberg connection not available. Using sample data.")
            return self._generate_sample_data(start_date, end_date)
        
        if self.tickers is None or len(self.tickers) == 0:
            raise ValueError("No tickers specified")
        
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        # Format dates for Bloomberg query
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        
        try:
            # Fetch data from Bloomberg
            data = self.bbg.bdh(tickers=self.tickers, 
                                flds=[field], 
                                start_date=start_date_fmt, 
                                end_date=end_date_fmt)
            
            # Convert to DataFrame
            prices = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
            
            # Process data for each ticker
            valid_tickers = []
            for ticker in self.tickers:
                try:
                    ticker_data = data.xs(ticker, level=0, axis=1)
                    if not ticker_data.empty:
                        prices[ticker] = ticker_data[field]
                        valid_tickers.append(ticker)
                        print(f"Successfully fetched data for {ticker}")
                    else:
                        print(f"No data found for {ticker}")
                except Exception as e:
                    print(f"Error processing data for {ticker}: {e}")
            
            # Update tickers list to valid tickers
            self.tickers = valid_tickers
            
            # Drop rows with all NaNs
            prices = prices.dropna(how='all')
            
            # Forward fill remaining NaNs (for holidays)
            prices = prices.fillna(method='ffill')
            
            if prices.empty:
                raise ValueError("No valid data retrieved")
                
            # Store the price data
            self.prices = prices
            
            return prices
            
        except Exception as e:
            print(f"Error fetching data from Bloomberg: {e}")
            return self._generate_sample_data(start_date, end_date)
    
    def _generate_sample_data(self, start_date, end_date):
        """
        Generate sample data for testing without Bloomberg
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        DataFrame of prices
        """
        if self.tickers is None or len(self.tickers) == 0:
            # Create sample tickers if none provided
            self.tickers = [f'STOCK_{i}' for i in range(20)]
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Number of tickers
        n_tickers = len(self.tickers)
        
        # Create correlation matrix with block structure to simulate different clusters
        n_blocks = min(4, n_tickers)  # Create up to 4 clusters
        block_size = n_tickers // n_blocks
        
        # Create correlation matrix
        corr_matrix = np.eye(n_tickers)  # Start with identity matrix
        
        # Fill correlation matrix with blocks
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < n_blocks - 1 else n_tickers
            
            # Within block correlation (higher)
            block_size_i = end_idx - start_idx
            # Create a positive definite block
            block = np.random.rand(block_size_i, block_size_i) * 0.3 + 0.7
            # Make it symmetric
            block = (block + block.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(block, 1.0)
            
            corr_matrix[start_idx:end_idx, start_idx:end_idx] = block
            
            # Between block correlation (lower)
            for j in range(i+1, n_blocks):
                start_idx_j = j * block_size
                end_idx_j = start_idx_j + block_size if j < n_blocks - 1 else n_tickers
                
                block_size_j = end_idx_j - start_idx_j
                block_corr = np.random.rand(block_size_i, block_size_j) * 0.3 + 0.1
                
                corr_matrix[start_idx:end_idx, start_idx_j:end_idx_j] = block_corr
                corr_matrix[start_idx_j:end_idx_j, start_idx:end_idx] = block_corr.T
        
        # Ensure the matrix is symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Check positive-definiteness by computing eigenvalues
        try:
            min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
            if min_eig < 0:
                # Add a small positive value to the diagonal
                corr_matrix += (-min_eig + 1e-8) * np.eye(n_tickers)
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use a simpler approach
            corr_matrix = 0.7 * np.eye(n_tickers) + 0.3 * np.ones((n_tickers, n_tickers))
            np.fill_diagonal(corr_matrix, 1.0)
        
        # Compute Cholesky decomposition for generating correlated returns
        try:
            chol = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use a simpler correlation matrix
            print("Warning: Correlation matrix is not positive definite. Using a simplified matrix.")
            corr_matrix = 0.7 * np.eye(n_tickers) + 0.3 * np.ones((n_tickers, n_tickers))
            np.fill_diagonal(corr_matrix, 1.0)
            chol = np.linalg.cholesky(corr_matrix)
        
        # Generate returns
        n_days = len(date_range)
        uncorrelated_returns = np.random.normal(0.0005, 0.01, (n_days, n_tickers))
        daily_returns = np.dot(uncorrelated_returns, chol.T)
        
        # Convert to prices (starting at 100)
        prices = 100 * np.cumprod(1 + daily_returns, axis=0)
        
        # Create DataFrame
        prices_df = pd.DataFrame(
            prices, 
            index=date_range, 
            columns=self.tickers
        )
        
        # Store the price data
        self.prices = prices_df
        
        print(f"Generated sample data for {len(self.tickers)} tickers from {start_date} to {end_date}")
        return prices_df
    
    def calculate_returns(self):
        """
        Calculate daily returns and cumulative returns
        
        Returns:
        --------
        DataFrame of daily returns
        """
        if self.prices is None:
            raise ValueError("No price data available. Call fetch_data() first.")
        
        # Calculate daily returns
        self.returns = self.prices.pct_change().dropna()
        
        # Calculate cumulative returns
        self.cumulative_returns = (1 + self.returns).cumprod()
        
        return self.returns
    
    def perform_clustering(self, n_clusters=None, method='complete'):
        """
        Perform shape-based clustering using DTW distance and hierarchical clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (if None, it will be estimated)
        method : str
            Linkage method for hierarchical clustering
            
        Returns:
        --------
        Dict with cluster labels
        """
        if self.cumulative_returns is None:
            self.calculate_returns()
        
        print("Computing DTW distance matrix...")
        
        # Normalize each series for shape-based comparison
        normalized_series = {}
        for col in self.cumulative_returns.columns:
            series = self.cumulative_returns[col].values
            min_val = np.min(series)
            max_val = np.max(series)
            if max_val > min_val:  # Avoid division by zero
                normalized_series[col] = (series - min_val) / (max_val - min_val)
            else:
                normalized_series[col] = np.zeros_like(series)
        
        # Compute DTW distance matrix
        n = len(self.cumulative_returns.columns)
        distance_matrix = np.zeros((n, n))
        
        for i in tqdm(range(n)):
            ticker_i = self.cumulative_returns.columns[i]
            series_i = normalized_series[ticker_i]
            
            for j in range(i+1, n):
                ticker_j = self.cumulative_returns.columns[j]
                series_j = normalized_series[ticker_j]
                
                # Compute DTW distance
                distance = dtw(series_i, series_j)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        self.distance_matrix = pd.DataFrame(
            distance_matrix,
            index=self.cumulative_returns.columns,
            columns=self.cumulative_returns.columns
        )
        
        print("Performing hierarchical clustering...")
        
        # Perform hierarchical clustering
        if n_clusters is None:
            # Estimate number of clusters
            Z = linkage(squareform(distance_matrix), method=method)
            plt.figure(figsize=(12, 8))
            dendrogram(Z)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Ticker')
            plt.ylabel('Distance')
            plt.show()
            
            # Ask for the number of clusters
            n_clusters = int(input("Enter the number of clusters based on the dendrogram: "))
        
        # Create clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage=method
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Store cluster labels
        self.cluster_labels = dict(zip(self.cumulative_returns.columns, labels))
        
        # Group tickers by cluster
        self.clusters = {}
        for ticker, label in self.cluster_labels.items():
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(ticker)
        
        return self.clusters
    
    def visualize_clusters(self):
        """
        Visualize the clusters by plotting cumulative returns
        """
        if self.clusters is None:
            raise ValueError("No clustering performed. Call perform_clustering() first.")
        
        n_clusters = len(self.clusters)
        fig, axes = plt.subplots(nrows=n_clusters, figsize=(12, 4*n_clusters))
        
        if n_clusters == 1:
            axes = [axes]
        
        for i, (cluster_id, tickers) in enumerate(self.clusters.items()):
            ax = axes[i]
            for ticker in tickers:
                ax.plot(self.cumulative_returns.index, self.cumulative_returns[ticker], label=ticker)
            ax.set_title(f'Cluster {cluster_id} (n={len(tickers)})')
            ax.legend(fontsize=8)
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_shape_diversified_portfolio(self, n_assets=3, risk_free_rate=0.02):
        """
        Create a portfolio diversified by shape clusters
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to include in the portfolio
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with portfolio information
        """
        if self.clusters is None:
            raise ValueError("No clustering performed. Call perform_clustering() first.")
        
        if n_assets > len(self.clusters):
            print(f"Warning: n_assets ({n_assets}) is greater than the number of clusters ({len(self.clusters)})")
            n_assets = len(self.clusters)
        
        # Compute Sharpe ratios for all assets
        ann_returns = (1 + self.returns.mean()) ** 252 - 1
        ann_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (ann_returns - risk_free_rate) / ann_volatility
        
        # Sort clusters by their best Sharpe ratio
        clusters_with_best_sharpe = []
        for cluster_id, tickers in self.clusters.items():
            cluster_sharpes = sharpe_ratios[tickers]
            best_ticker = cluster_sharpes.idxmax()
            best_sharpe = cluster_sharpes[best_ticker]
            clusters_with_best_sharpe.append((cluster_id, best_ticker, best_sharpe))
        
        # Sort clusters by Sharpe ratio in descending order
        clusters_with_best_sharpe.sort(key=lambda x: x[2], reverse=True)
        
        # Select assets from the best clusters
        selected_assets = []
        for i in range(min(n_assets, len(clusters_with_best_sharpe))):
            cluster_id, best_ticker, _ = clusters_with_best_sharpe[i]
            selected_assets.append(best_ticker)
        
        # Compute portfolio statistics with equal weights
        weights = np.array([1.0 / len(selected_assets)] * len(selected_assets))
        portfolio_returns = self.returns[selected_assets].dot(weights)
        
        portfolio_ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
        portfolio_ann_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_vol
        
        # Create portfolio info dictionary
        portfolio = {
            'assets': selected_assets,
            'weights': dict(zip(selected_assets, weights)),
            'clusters': [self.cluster_labels[asset] for asset in selected_assets],
            'return': portfolio_ann_return,
            'volatility': portfolio_ann_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return portfolio
    
    def optimize_portfolio_weights(self, portfolio, risk_free_rate=0.02):
        """
        Optimize the weights of a portfolio to maximize Sharpe ratio
        
        Parameters:
        -----------
        portfolio : dict
            Portfolio information from create_shape_diversified_portfolio
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with optimized portfolio information
        """
        from scipy.optimize import minimize
        
        def negative_sharpe(weights, returns, risk_free_rate):
            weights = np.array(weights)
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe
        
        assets = portfolio['assets']
        asset_returns = self.returns[assets]
        
        # Initial weights (equal)
        initial_weights = np.array([1.0 / len(assets)] * len(assets))
        
        # Constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Optimization
        result = minimize(
            negative_sharpe,
            initial_weights,
            args=(asset_returns, risk_free_rate),
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(len(assets))]  # No short selling
        )
        
        optimized_weights = result['x']
        
        # Compute portfolio statistics with optimized weights
        portfolio_returns = asset_returns.dot(optimized_weights)
        
        portfolio_ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
        portfolio_ann_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_vol
        
        # Create optimized portfolio info
        optimized_portfolio = {
            'assets': assets,
            'weights': dict(zip(assets, optimized_weights)),
            'clusters': [self.cluster_labels[asset] for asset in assets],
            'return': portfolio_ann_return,
            'volatility': portfolio_ann_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return optimized_portfolio
    
    def compare_portfolios(self, portfolios, names=None):
        """
        Compare multiple portfolios
        
        Parameters:
        -----------
        portfolios : list
            List of portfolio dictionaries
        names : list
            List of portfolio names
            
        Returns:
        --------
        DataFrame with comparison results
        """
        if names is None:
            names = [f"Portfolio {i+1}" for i in range(len(portfolios))]
        
        comparison = pd.DataFrame({
            'Portfolio': names,
            'Annual Return': [p['return'] for p in portfolios],
            'Annual Volatility': [p['volatility'] for p in portfolios],
            'Sharpe Ratio': [p['sharpe_ratio'] for p in portfolios],
            'Assets': [', '.join(p['assets']) for p in portfolios]
        })
        
        return comparison
    
    def backtest_portfolios(self, portfolios, names=None, start_date=None, end_date=None):
        """
        Backtest multiple portfolios
        
        Parameters:
        -----------
        portfolios : list
            List of portfolio dictionaries
        names : list
            List of portfolio names
        start_date : str
            Start date for backtest (if None, uses all available data)
        end_date : str
            End date for backtest (if None, uses all available data)
            
        Returns:
        --------
        DataFrame with cumulative returns
        """
        if names is None:
            names = [f"Portfolio {i+1}" for i in range(len(portfolios))]
        
        # Filter returns by date if specified
        if start_date is not None or end_date is not None:
            if start_date is None:
                start_date = self.returns.index[0]
            if end_date is None:
                end_date = self.returns.index[-1]
            
            returns = self.returns.loc[start_date:end_date]
        else:
            returns = self.returns
        
        # Calculate portfolio returns
        portfolio_returns = pd.DataFrame(index=returns.index)
        
        for i, portfolio in enumerate(portfolios):
            assets = portfolio['assets']
            weights = np.array([portfolio['weights'][asset] for asset in assets])
            
            # Calculate portfolio returns
            portfolio_returns[names[i]] = returns[assets].dot(weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Plot results
        plt.figure(figsize=(12, 6))
        for name in names:
            plt.plot(cumulative_returns.index, cumulative_returns[name], label=name)
        
        plt.title('Portfolio Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return cumulative_returns
    
    def create_random_portfolio(self, n_assets=3, risk_free_rate=0.02):
        """
        Create a randomly selected portfolio
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to include in the portfolio
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with portfolio information
        """
        # Select random assets
        selected_assets = random.sample(list(self.returns.columns), n_assets)
        
        # Compute portfolio statistics with equal weights
        weights = np.array([1.0 / len(selected_assets)] * len(selected_assets))
        portfolio_returns = self.returns[selected_assets].dot(weights)
        
        portfolio_ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
        portfolio_ann_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_vol
        
        # Create portfolio info dictionary
        portfolio = {
            'assets': selected_assets,
            'weights': dict(zip(selected_assets, weights)),
            'clusters': [self.cluster_labels.get(asset, -1) for asset in selected_assets],
            'return': portfolio_ann_return,
            'volatility': portfolio_ann_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return portfolio
    
    def create_minimum_variance_portfolio(self, n_assets=3, risk_free_rate=0.02):
        """
        Create a minimum variance portfolio
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to include in the portfolio
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with portfolio information
        """
        from scipy.optimize import minimize
        
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # Find most diversified assets
        # We'll use a greedy approach by selecting assets with lowest average correlation
        avg_corr = corr_matrix.mean()
        sorted_assets = avg_corr.sort_values().index[:n_assets]
        
        asset_returns = self.returns[sorted_assets]
        
        # Define portfolio variance function to minimize
        def portfolio_variance(weights, returns):
            weights = np.array(weights)
            return np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        
        # Initial weights (equal)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Optimization
        result = minimize(
            portfolio_variance,
            initial_weights,
            args=(asset_returns,),
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]  # No short selling
        )
        
        optimized_weights = result['x']
        
        # Compute portfolio statistics with optimized weights
        portfolio_returns = asset_returns.dot(optimized_weights)
        
        portfolio_ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
        portfolio_ann_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_vol
        
        # Create optimized portfolio info
        min_var_portfolio = {
            'assets': list(sorted_assets),
            'weights': dict(zip(sorted_assets, optimized_weights)),
            'clusters': [self.cluster_labels.get(asset, -1) for asset in sorted_assets],
            'return': portfolio_ann_return,
            'volatility': portfolio_ann_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return min_var_portfolio

    def get_industry_classification(self, field='INDUSTRY_SECTOR'):
        """
        Get industry classification for tickers from Bloomberg
        
        Parameters:
        -----------
        field : str
            Bloomberg field for industry classification
            
        Returns:
        --------
        Dict with industry classifications
        """
        if not self.has_bloomberg:
            print("Bloomberg connection not available. Using sample industry data.")
            # Generate random industry classifications for sample data
            industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Utilities']
            return {ticker: random.choice(industries) for ticker in self.tickers}
        
        industries = {}
        
        try:
            print(f"Retrieving industry classification using {field}...")
            
            # Use reference data service to get industry classification
            data = self.bbg.ref(tickers=self.tickers, flds=[field])
            
            for ticker in self.tickers:
                try:
                    industry = data.loc[ticker, field]
                    industries[ticker] = industry
                except:
                    print(f"No industry data for {ticker}")
                    industries[ticker] = 'Unknown'
                    
            return industries
            
        except Exception as e:
            print(f"Error fetching industry data: {e}")
            # Generate random industry classifications as fallback
            industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Utilities']
            return {ticker: random.choice(industries) for ticker in self.tickers}
    
    def create_industry_diversified_portfolio(self, n_assets=3, risk_free_rate=0.02):
        """
        Create a portfolio diversified by industry
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to include in the portfolio
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict with portfolio information
        """
        # Get industry classification
        industries = self.get_industry_classification()
        
        # Group assets by industry
        industry_groups = {}
        for ticker, industry in industries.items():
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(ticker)
        
        # Compute Sharpe ratios for all assets
        ann_returns = (1 + self.returns.mean()) ** 252 - 1
        ann_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (ann_returns - risk_free_rate) / ann_volatility
        
        # Sort industries by their best Sharpe ratio
        industries_with_best_sharpe = []
        for industry, tickers in industry_groups.items():
            # Filter for tickers that we have return data for
            valid_tickers = [t for t in tickers if t in sharpe_ratios.index]
            if not valid_tickers:
                continue
                
            industry_sharpes = sharpe_ratios[valid_tickers]
            best_ticker = industry_sharpes.idxmax()
            best_sharpe = industry_sharpes[best_ticker]
            industries_with_best_sharpe.append((industry, best_ticker, best_sharpe))
        
        # Sort industries by Sharpe ratio in descending order
        industries_with_best_sharpe.sort(key=lambda x: x[2], reverse=True)
        
        # Select assets from the best industries
        selected_assets = []
        for i in range(min(n_assets, len(industries_with_best_sharpe))):
            industry, best_ticker, _ = industries_with_best_sharpe[i]
            selected_assets.append(best_ticker)
        
        # Compute portfolio statistics with equal weights
        weights = np.array([1.0 / len(selected_assets)] * len(selected_assets))
        portfolio_returns = self.returns[selected_assets].dot(weights)
        
        portfolio_ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
        portfolio_ann_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_ann_return - risk_free_rate) / portfolio_ann_vol
        
        # Create portfolio info dictionary
        portfolio = {
            'assets': selected_assets,
            'weights': dict(zip(selected_assets, weights)),
            'industries': [industries[asset] for asset in selected_assets],
            'clusters': [self.cluster_labels.get(asset, -1) for asset in selected_assets],
            'return': portfolio_ann_return,
            'volatility': portfolio_ann_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return portfolio

# Example usage
if __name__ == "__main__":
    # Define Singapore market tickers (can be replaced with any market)
    sg_tickers = [
        'DBS SP Equity',  # DBS Group Holdings
        'OCBC SP Equity',  # Oversea-Chinese Banking Corp
        'UOB SP Equity',   # United Overseas Bank
        'ST SP Equity',    # Singapore Technologies Engineering
        'SIA SP Equity',   # Singapore Airlines
        'SGX SP Equity',   # Singapore Exchange
        'KEP SP Equity',   # Keppel Corporation
        'STE SP Equity',   # Sembcorp Industries
        'SPH SP Equity',   # Singapore Press Holdings
        'CMT SP Equity',   # CapitaLand Mall Trust
        'CCT SP Equity',   # CapitaLand Commercial Trust
        'GENS SP Equity',  # Genting Singapore
        'YZJ SP Equity',   # Yangzijiang Shipbuilding
        'AREIT SP Equity', # Ascendas REIT
        'TL SP Equity',    # ThaiBev
        'CIT SP Equity',   # City Developments
        'MCT SP Equity',   # Mapletree Commercial Trust
        'WIL SP Equity',   # Wilmar International
        'FPL SP Equity',   # Frasers Property
        'MLT SP Equity'    # Mapletree Logistics Trust
    ]
    
    # Initialize portfolio manager with Singapore tickers
    portfolio_manager = ShapeBasedPortfolioManager(tickers=sg_tickers)
    
    # Fetch data for the last 3 years
    today = datetime.datetime.now()
    three_years_ago = today - datetime.timedelta(days=3*365)
    start_date = three_years_ago.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    prices = portfolio_manager.fetch_data(start_date, end_date)
    
    # Calculate returns
    returns = portfolio_manager.calculate_returns()
    
    # Perform clustering
    clusters = portfolio_manager.perform_clustering()
    
    # Visualize clusters
    portfolio_manager.visualize_clusters()
    
    # Create shape-based diversified portfolio
    shape_portfolio = portfolio_manager.create_shape_diversified_portfolio(n_assets=3)
    
    # Create minimum variance portfolio
    min_var_portfolio = portfolio_manager.create_minimum_variance_portfolio(n_assets=3)
    
    # Create industry-diversified portfolio
    industry_portfolio = portfolio_manager.create_industry_diversified_portfolio(n_assets=3)
    
    # Create random portfolio
    random_portfolio = portfolio_manager.create_random_portfolio(n_assets=3)
    
    # Compare portfolios
    comparison = portfolio_manager.compare_portfolios(
        [shape_portfolio, min_var_portfolio, industry_portfolio, random_portfolio],
        ['Shape-Based', 'Min Variance', 'Industry', 'Random']
    )
    
    print("\nPortfolio Comparison:")
    print(comparison)
    
    # Optimize shape-based portfolio
    optimized_shape_portfolio = portfolio_manager.optimize_portfolio_weights(shape_portfolio)
    
    # Optimize minimum variance portfolio
    optimized_min_var_portfolio = portfolio_manager.optimize_portfolio_weights(min_var_portfolio)
    
    # Optimize industry portfolio
    optimized_industry_portfolio = portfolio_manager.optimize_portfolio_weights(industry_portfolio)
    
    # Compare optimized portfolios
    optimized_comparison = portfolio_manager.compare_portfolios(
        [optimized_shape_portfolio, optimized_min_var_portfolio, optimized_industry_portfolio, random_portfolio],
        ['Optimized Shape-Based', 'Optimized Min Variance', 'Optimized Industry', 'Random']
    )
    
    print("\nOptimized Portfolio Comparison:")
    print(optimized_comparison)
    
    # Backtest portfolios
    portfolio_manager.backtest_portfolios(
        [optimized_shape_portfolio, optimized_min_var_portfolio, optimized_industry_portfolio, random_portfolio],
        ['Optimized Shape-Based', 'Optimized Min Variance', 'Optimized Industry', 'Random']
    )