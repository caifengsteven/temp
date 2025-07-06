import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import linprog, minimize
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plot style
plt.style.use('seaborn-v0_8')


class FinancialDataGenerator:
    """
    Class to generate simulated financial return data
    """
    
    def __init__(self, n_assets=500, n_periods=240, mean_return=0.005, 
                 volatility=0.05, correlation=0.2):
        """
        Initialize the financial data generator
        
        Parameters:
        - n_assets: Number of assets to simulate
        - n_periods: Number of time periods to simulate
        - mean_return: Mean monthly return
        - volatility: Monthly volatility
        - correlation: Base correlation between assets
        """
        self.n_assets = n_assets
        self.n_periods = n_periods
        self.mean_return = mean_return
        self.volatility = volatility
        self.correlation = correlation
        
    def generate_sector_returns(self, n_sectors=10, sector_vol_multiplier=1.5):
        """
        Generate returns with sector structure
        
        Parameters:
        - n_sectors: Number of sectors
        - sector_vol_multiplier: Multiplier for sector-specific volatility
        
        Returns:
        - DataFrame with simulated returns
        """
        # Assign assets to sectors
        assets_per_sector = np.ones(n_sectors, dtype=int) * (self.n_assets // n_sectors)
        assets_per_sector[:self.n_assets % n_sectors] += 1
        sector_assignments = np.concatenate([np.ones(n) * i for i, n in enumerate(assets_per_sector)])
        
        # Generate sector returns
        sector_returns = np.random.normal(
            self.mean_return, 
            self.volatility * sector_vol_multiplier, 
            (self.n_periods, n_sectors)
        )
        
        # Generate asset-specific returns
        asset_returns = np.zeros((self.n_periods, self.n_assets))
        
        # Correlation matrix with block structure
        corr_matrix = np.ones((self.n_assets, self.n_assets)) * self.correlation
        np.fill_diagonal(corr_matrix, 1)
        
        # Enhance correlation within sectors
        for i in range(n_sectors):
            sector_indices = np.where(sector_assignments == i)[0]
            for j in sector_indices:
                for k in sector_indices:
                    if j != k:
                        corr_matrix[j, k] = self.correlation * 2
        
        # Ensure correlation matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix += -min_eig * np.eye(self.n_assets) * 1.1
        
        # Convert correlation to covariance matrix
        vol_vector = np.random.uniform(self.volatility * 0.5, self.volatility * 1.5, self.n_assets)
        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix
        
        # Generate multivariate normal returns
        asset_specific_returns = np.random.multivariate_normal(
            np.zeros(self.n_assets), 
            cov_matrix, 
            self.n_periods
        )
        
        # Combine sector and asset-specific returns
        for i in range(self.n_assets):
            sector_idx = int(sector_assignments[i])
            asset_returns[:, i] = sector_returns[:, sector_idx] + asset_specific_returns[:, i]
        
        # Add some skewness and excess kurtosis
        for i in range(self.n_assets):
            # Generate skewness (-0.5 to 0.5)
            skew = np.random.uniform(-0.5, 0.5)
            # Generate excess kurtosis (0 to 2)
            excess_kurt = np.random.uniform(0, 2)
            
            # Apply skewness and kurtosis using Johnson's SU distribution
            asset_returns[:, i] = stats.johnsonsu.ppf(
                stats.norm.cdf(asset_returns[:, i]),
                0,  # shape parameter a
                1,  # shape parameter b
                loc=self.mean_return,  # location parameter
                scale=vol_vector[i]  # scale parameter
            )
        
        # Create DataFrame
        asset_names = [f'Asset_{i}' for i in range(self.n_assets)]
        sector_names = [f'Sector_{int(sector_assignments[i])}' for i in range(self.n_assets)]
        
        df = pd.DataFrame(asset_returns, columns=asset_names)
        
        # Add sector information
        sector_info = pd.DataFrame({
            'Asset': asset_names,
            'Sector': sector_names
        })
        
        return df, sector_info
    
    def generate_correlated_returns(self):
        """
        Generate correlated returns
        
        Returns:
        - DataFrame with simulated returns
        """
        # Create correlation matrix
        corr_matrix = np.ones((self.n_assets, self.n_assets)) * self.correlation
        np.fill_diagonal(corr_matrix, 1)
        
        # Convert correlation to covariance matrix
        vol_vector = np.random.uniform(self.volatility * 0.5, self.volatility * 1.5, self.n_assets)
        mean_vector = np.random.uniform(self.mean_return * 0.5, self.mean_return * 1.5, self.n_assets)
        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix
        
        # Generate multivariate normal returns
        returns = np.random.multivariate_normal(mean_vector, cov_matrix, self.n_periods)
        
        # Create DataFrame
        df = pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(self.n_assets)])
        
        return df
    

class SSDSpanningOptimizer:
    """
    Class to implement sparse second-order stochastic dominance spanning
    """
    
    def __init__(self, returns, trading_fee=0.001):
        """
        Initialize the SSD spanning optimizer
        
        Parameters:
        - returns: DataFrame with returns (T x N)
        - trading_fee: Trading fee as a proportion of trade value
        """
        self.returns = returns.values if isinstance(returns, pd.DataFrame) else returns
        self.asset_names = returns.columns if isinstance(returns, pd.DataFrame) else [f'Asset_{i}' for i in range(returns.shape[1])]
        self.T, self.N = self.returns.shape
        self.trading_fee = trading_fee
        
    def calculate_lpmd(self, z, portfolio1, portfolio2):
        """
        Calculate Lower Partial Moment Differential (LPMD)
        
        Parameters:
        - z: Threshold
        - portfolio1: First portfolio weights
        - portfolio2: Second portfolio weights
        
        Returns:
        - LPMD value
        """
        port1_returns = self.returns @ portfolio1
        port2_returns = self.returns @ portfolio2
        
        lpmd = np.mean(np.maximum(0, z - port1_returns)) - np.mean(np.maximum(0, z - port2_returns))
        
        return lpmd
    
    def optimize_portfolio_lp(self, z, utility_weights):
        """
        Optimize a portfolio for a given utility function using Linear Programming
        
        Parameters:
        - z: Threshold vector
        - utility_weights: Weights for the utility function
        
        Returns:
        - Optimal portfolio weights
        """
        # Number of return observations and assets
        T, N = self.returns.shape
        
        # Number of possible outcomes
        N1 = len(z)
        
        # Linear Programming setup
        # Variables: [w_1, ..., w_N, y_1, ..., y_T]
        # w_i: portfolio weights
        # y_t: auxiliary variables for the utility function
        
        # Objective: maximize sum(y_t)/T
        c = np.zeros(N + T)
        c[N:] = -1.0 / T  # Negative because linprog minimizes
        
        # Constraints:
        # 1. sum(w_i) = 1
        # 2. w_i >= 0 for all i
        # 3. y_t <= c0_n - c1_n * (returns_t * w) for all t, n
        
        # Constraint 1: sum(w_i) = 1
        A_eq = np.zeros((1, N + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])
        
        # Constraint 3: set up utility constraints
        A_ub = []
        b_ub = []
        
        for n in range(N1):
            if utility_weights[n] > 0:  # Only include constraints with positive weight
                for t in range(T):
                    # For each observation and each threshold
                    row = np.zeros(N + T)
                    row[:N] = self.returns[t]  # Returns for this observation
                    row[N + t] = -1.0  # Coefficient for y_t
                    A_ub.append(row)
                    b_ub.append(-z[n])  # Negative because constraint is y_t <= ...
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for variables
        bounds = [(0, None) for _ in range(N)] + [(None, None) for _ in range(T)]
        
        # Solve LP
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            return result.x[:N]
        else:
            print("LP optimization failed:", result.message)
            return np.ones(N) / N  # Equal weight fallback
    
    def forward_stepwise_selection(self, q, max_iterations=None):
        """
        Implement Forward Stepwise Selection algorithm for sparse spanning
        
        Parameters:
        - q: Desired sparsity (number of assets to select)
        - max_iterations: Maximum number of iterations (if None, use q*ln(T))
        
        Returns:
        - Selected assets
        - Diversification loss
        """
        if max_iterations is None:
            max_iterations = min(q * int(np.log(self.T) + 1), self.N)
        
        # Initialize
        selected_assets = []
        current_indices = []
        
        # Discretize the utility function
        N1 = 10  # Number of points
        N2 = 5   # Number of piecewise segments
        
        # Create thresholds
        min_return = np.min(self.returns)
        max_return = np.max(self.returns)
        z = np.linspace(min_return, max_return, N1)
        
        # Create utility weights (convex mixture of "ramp functions")
        utility_weights = np.zeros(N1)
        utility_weights[0] = 1.0 / N2
        for i in range(1, N1):
            utility_weights[i] = (i / (N1 - 1)) / N2
        
        # Normalize
        utility_weights = utility_weights / np.sum(utility_weights)
        
        # Initial diversification loss (with empty set)
        full_set_weights = self.optimize_portfolio_lp(z, utility_weights)
        best_diversification_loss = float('inf')
        
        for iteration in range(max_iterations):
            best_new_asset = None
            best_new_loss = float('inf')
            
            # Try adding each remaining asset
            for i in range(self.N):
                if i not in current_indices:
                    # Create candidate set
                    candidate_indices = current_indices + [i]
                    
                    # Create mask for selected assets
                    mask = np.zeros(self.N, dtype=bool)
                    mask[candidate_indices] = True
                    
                    # Optimize over the candidate set
                    candidate_set_returns = self.returns[:, mask]
                    sparse_optimizer = SSDSpanningOptimizer(candidate_set_returns)
                    
                    # Optimize for sparse portfolio
                    sparse_weights = sparse_optimizer.optimize_portfolio_lp(z, utility_weights)
                    
                    # Convert to full dimension
                    full_sparse_weights = np.zeros(self.N)
                    full_sparse_weights[mask] = sparse_weights
                    
                    # Calculate diversification loss
                    max_loss = -float('inf')
                    for threshold in z:
                        loss = self.calculate_lpmd(threshold, full_sparse_weights, full_set_weights)
                        max_loss = max(max_loss, loss)
                    
                    if max_loss < best_new_loss:
                        best_new_loss = max_loss
                        best_new_asset = i
            
            # Add the best new asset
            if best_new_asset is not None:
                current_indices.append(best_new_asset)
                selected_assets.append(self.asset_names[best_new_asset])
                best_diversification_loss = best_new_loss
                
                # If we've reached the desired sparsity or zero loss, stop
                if len(selected_assets) >= q or best_diversification_loss <= 0:
                    break
        
        return selected_assets, best_diversification_loss
    
    def evaluate_portfolios(self, portfolio_weights, names=None):
        """
        Evaluate portfolios based on various metrics
        
        Parameters:
        - portfolio_weights: List of portfolio weights
        - names: List of portfolio names
        
        Returns:
        - DataFrame with performance metrics
        """
        if names is None:
            names = [f'Portfolio_{i}' for i in range(len(portfolio_weights))]
        
        results = {}
        
        for i, weights in enumerate(portfolio_weights):
            # Calculate portfolio returns
            portfolio_returns = self.returns @ weights
            
            # Calculate metrics
            mean_return = np.mean(portfolio_returns)
            std_dev = np.std(portfolio_returns)
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            downside_sharpe = mean_return / (np.sqrt(2) * downside_std) if downside_std > 0 else 0
            
            # Value at Risk (95%)
            var_95 = np.percentile(portfolio_returns, 5) * (-1)
            
            # Expected Shortfall (95%)
            es_95 = np.mean(portfolio_returns[portfolio_returns <= -var_95]) * (-1)
            
            # Skewness and kurtosis
            skewness = stats.skew(portfolio_returns)
            kurtosis = stats.kurtosis(portfolio_returns)
            
            # Store results
            results[names[i]] = {
                'Mean Return': mean_return,
                'Standard Deviation': std_dev,
                'Sharpe Ratio': sharpe_ratio,
                'Downside Sharpe': downside_sharpe,
                'VaR (95%)': var_95,
                'ES (95%)': es_95,
                'Skewness': skewness,
                'Kurtosis': kurtosis
            }
        
        return pd.DataFrame(results).T
    

class MAXSEROptimizer:
    """
    Implementation of a sparse mean-variance portfolio optimization approach 
    inspired by the MAXSER method (Ao, Li, and Zheng, 2019)
    """
    
    def __init__(self, returns, regularization=0.1):
        """
        Initialize the MAXSER optimizer
        
        Parameters:
        - returns: DataFrame with returns (T x N)
        - regularization: Regularization parameter for sparsity
        """
        self.returns = returns.values if isinstance(returns, pd.DataFrame) else returns
        self.asset_names = returns.columns if isinstance(returns, pd.DataFrame) else [f'Asset_{i}' for i in range(returns.shape[1])]
        self.T, self.N = self.returns.shape
        self.regularization = regularization
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns, rowvar=False)
        
    def optimize(self, q):
        """
        Optimize the portfolio using MAXSER-inspired approach
        
        Parameters:
        - q: Desired sparsity (number of assets to select)
        
        Returns:
        - Selected assets
        - Portfolio weights
        """
        # First, solve the regularized mean-variance problem
        def objective(weights):
            # Mean-variance objective with L1 penalty
            port_return = np.dot(weights, self.mean_returns)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -port_return / port_risk + self.regularization * np.sum(np.abs(weights))
        
        # Initial guess: equal weights
        x0 = np.ones(self.N) / self.N
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: no short selling
        bounds = [(0, 1) for _ in range(self.N)]
        
        # Solve
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            print("MAXSER optimization failed:", result.message)
            return [], np.ones(self.N) / self.N
        
        # Get the sparse weights
        weights = result.x
        
        # Select the top q assets by weight
        indices = np.argsort(weights)[-q:]
        selected_assets = [self.asset_names[i] for i in indices]
        
        # Re-optimize with only the selected assets
        def reoptimize_objective(sparse_weights):
            # Convert to full weights
            full_weights = np.zeros(self.N)
            full_weights[indices] = sparse_weights
            
            # Mean-variance objective
            port_return = np.dot(full_weights, self.mean_returns)
            port_risk = np.sqrt(np.dot(full_weights.T, np.dot(self.cov_matrix, full_weights)))
            return -port_return / port_risk
        
        # Initial guess: equal weights
        x0_sparse = np.ones(q) / q
        
        # Constraints: weights sum to 1
        constraints_sparse = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: no short selling
        bounds_sparse = [(0, 1) for _ in range(q)]
        
        # Solve
        result_sparse = minimize(reoptimize_objective, x0_sparse, method='SLSQP', 
                               bounds=bounds_sparse, constraints=constraints_sparse)
        
        if not result_sparse.success:
            print("MAXSER re-optimization failed:", result_sparse.message)
            return selected_assets, np.ones(q) / q
        
        # Get the optimal weights for selected assets
        sparse_weights = result_sparse.x
        
        # Convert to full weights
        full_weights = np.zeros(self.N)
        full_weights[indices] = sparse_weights
        
        return selected_assets, full_weights
    

def run_experiments():
    """
    Run a series of experiments to test the SSD spanning and MAXSER methods
    """
    # Generate data
    data_gen = FinancialDataGenerator(n_assets=100, n_periods=240)
    returns, sector_info = data_gen.generate_sector_returns(n_sectors=10)
    
    print(f"Generated returns with shape: {returns.shape}")
    
    # Create optimizer
    ssd_optimizer = SSDSpanningOptimizer(returns)
    maxser_optimizer = MAXSEROptimizer(returns)
    
    # Run experiments for different q values
    q_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ssd_results = []
    maxser_results = []
    
    for q in tqdm(q_values, desc="Testing sparsity levels"):
        # Run SSD spanning
        start_time = time.time()
        ssd_assets, ssd_loss = ssd_optimizer.forward_stepwise_selection(q)
        ssd_time = time.time() - start_time
        
        # Create SSD portfolio
        mask = np.zeros(returns.shape[1], dtype=bool)
        for asset in ssd_assets:
            mask[returns.columns.get_loc(asset)] = True
        
        ssd_returns = returns.loc[:, mask]
        sparse_optimizer = SSDSpanningOptimizer(ssd_returns)
        
        # Optimize for SSD portfolio
        N1 = 10  # Number of points
        N2 = 5   # Number of piecewise segments
        min_return = np.min(returns.values)
        max_return = np.max(returns.values)
        z = np.linspace(min_return, max_return, N1)
        utility_weights = np.zeros(N1)
        utility_weights[0] = 1.0 / N2
        for i in range(1, N1):
            utility_weights[i] = (i / (N1 - 1)) / N2
        utility_weights = utility_weights / np.sum(utility_weights)
        
        ssd_weights = sparse_optimizer.optimize_portfolio_lp(z, utility_weights)
        
        # Convert to full dimension
        full_ssd_weights = np.zeros(returns.shape[1])
        full_ssd_weights[mask] = ssd_weights
        
        # Run MAXSER
        start_time = time.time()
        maxser_assets, maxser_weights = maxser_optimizer.optimize(q)
        maxser_time = time.time() - start_time
        
        # Store results
        ssd_results.append({
            'q': q,
            'selected_assets': ssd_assets,
            'weights': full_ssd_weights,
            'diversification_loss': ssd_loss,
            'time': ssd_time
        })
        
        maxser_results.append({
            'q': q,
            'selected_assets': maxser_assets,
            'weights': maxser_weights,
            'time': maxser_time
        })
    
    # Evaluate portfolios
    ssd_weights = [result['weights'] for result in ssd_results]
    maxser_weights = [result['weights'] for result in maxser_results]
    equal_weight = np.ones(returns.shape[1]) / returns.shape[1]
    
    portfolio_weights = ssd_weights + maxser_weights + [equal_weight]
    portfolio_names = [f'SSD_q{q_values[i]}' for i in range(len(ssd_weights))] + \
                      [f'MAXSER_q{q_values[i]}' for i in range(len(maxser_weights))] + \
                      ['1/N']
    
    performance = ssd_optimizer.evaluate_portfolios(portfolio_weights, portfolio_names)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'q': q_values * 2,
        'Method': ['SSD'] * len(q_values) + ['MAXSER'] * len(q_values),
        'Diversification_Loss': [result['diversification_loss'] for result in ssd_results] + [0] * len(maxser_results),
        'Time': [result['time'] for result in ssd_results] + [result['time'] for result in maxser_results],
        'Sharpe_Ratio': performance['Sharpe Ratio'].values,
        'Downside_Sharpe': performance['Downside Sharpe'].values,
        'VaR': performance['VaR (95%)'].values,
        'ES': performance['ES (95%)'].values
    })
    
    # Analyze sector allocation
    sector_analysis = []
    
    for i, result in enumerate(ssd_results):
        selected_indices = [returns.columns.get_loc(asset) for asset in result['selected_assets']]
        selected_sectors = sector_info.iloc[selected_indices]['Sector'].value_counts().to_dict()
        
        # Calculate weights by sector
        sector_weights = {}
        for j, asset in enumerate(result['selected_assets']):
            asset_idx = returns.columns.get_loc(asset)
            sector = sector_info.iloc[asset_idx]['Sector']
            weight = result['weights'][asset_idx]
            
            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight
        
        sector_analysis.append({
            'q': q_values[i],
            'Method': 'SSD',
            'Sector_Count': selected_sectors,
            'Sector_Weights': sector_weights
        })
    
    for i, result in enumerate(maxser_results):
        selected_indices = [returns.columns.get_loc(asset) for asset in result['selected_assets']]
        selected_sectors = sector_info.iloc[selected_indices]['Sector'].value_counts().to_dict()
        
        # Calculate weights by sector
        sector_weights = {}
        for sector in np.unique(sector_info['Sector']):
            sector_indices = sector_info[sector_info['Sector'] == sector].index
            sector_weights[sector] = np.sum(result['weights'][sector_indices])
        
        sector_analysis.append({
            'q': q_values[i],
            'Method': 'MAXSER',
            'Sector_Count': selected_sectors,
            'Sector_Weights': sector_weights
        })
    
    return {
        'returns': returns,
        'sector_info': sector_info,
        'ssd_results': ssd_results,
        'maxser_results': maxser_results,
        'performance': performance,
        'comparison': comparison,
        'sector_analysis': sector_analysis
    }

def plot_results(results):
    """
    Plot the results of the experiments
    
    Parameters:
    - results: Dictionary with experiment results
    """
    comparison = results['comparison']
    performance = results['performance']
    
    # Plot 1: Diversification Loss vs. q
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.lineplot(data=comparison[comparison['Method'] == 'SSD'], 
                x='q', y='Diversification_Loss')
    plt.title('Diversification Loss vs. Number of Assets (SSD)')
    plt.xlabel('Number of Assets (q)')
    plt.ylabel('Diversification Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe Ratio Comparison
    plt.subplot(2, 2, 2)
    sns.lineplot(data=comparison, x='q', y='Sharpe_Ratio', hue='Method')
    plt.axhline(y=performance.loc['1/N', 'Sharpe Ratio'], color='r', linestyle='--', label='1/N')
    plt.title('Sharpe Ratio vs. Number of Assets')
    plt.xlabel('Number of Assets (q)')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: VaR and ES Comparison
    plt.subplot(2, 2, 3)
    sns.lineplot(data=comparison, x='q', y='VaR', hue='Method')
    plt.axhline(y=performance.loc['1/N', 'VaR (95%)'], color='r', linestyle='--', label='1/N')
    plt.title('Value-at-Risk (95%) vs. Number of Assets')
    plt.xlabel('Number of Assets (q)')
    plt.ylabel('VaR (95%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Computation Time
    plt.subplot(2, 2, 4)
    sns.lineplot(data=comparison, x='q', y='Time', hue='Method')
    plt.title('Computation Time vs. Number of Assets')
    plt.xlabel('Number of Assets (q)')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssd_spanning_results.png', dpi=300)
    plt.show()
    
    # Plot sector allocation
    sector_analysis = results['sector_analysis']
    
    # Find optimal q (where diversification loss is minimized)
    optimal_q_ssd = comparison[comparison['Method'] == 'SSD']['q'].iloc[
        comparison[comparison['Method'] == 'SSD']['Diversification_Loss'].argmin()
    ]
    
    # Get sector weights for optimal SSD and MAXSER
    ssd_optimal = next((item for item in sector_analysis 
                      if item['Method'] == 'SSD' and item['q'] == optimal_q_ssd), None)
    maxser_optimal = next((item for item in sector_analysis 
                         if item['Method'] == 'MAXSER' and item['q'] == optimal_q_ssd), None)
    
    if ssd_optimal and maxser_optimal:
        plt.figure(figsize=(14, 7))
        
        # SSD sector weights
        plt.subplot(1, 2, 1)
        sectors = list(ssd_optimal['Sector_Weights'].keys())
        weights = list(ssd_optimal['Sector_Weights'].values())
        plt.pie(weights, labels=sectors, autopct='%1.1f%%')
        plt.title(f'SSD Sector Weights (q={optimal_q_ssd})')
        
        # MAXSER sector weights
        plt.subplot(1, 2, 2)
        sectors = list(maxser_optimal['Sector_Weights'].keys())
        weights = list(maxser_optimal['Sector_Weights'].values())
        plt.pie(weights, labels=sectors, autopct='%1.1f%%')
        plt.title(f'MAXSER Sector Weights (q={optimal_q_ssd})')
        
        plt.savefig('sector_allocation.png', dpi=300)
        plt.show()
    
    # Plot performance metrics
    metrics = ['Mean Return', 'Standard Deviation', 'Sharpe Ratio', 'Downside Sharpe', 
              'VaR (95%)', 'ES (95%)', 'Skewness', 'Kurtosis']
    
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i+1)
        
        # Filter portfolios for plotting
        ssd_portfolios = [col for col in performance.index if col.startswith('SSD')]
        maxser_portfolios = [col for col in performance.index if col.startswith('MAXSER')]
        
        ssd_values = performance.loc[ssd_portfolios, metric]
        maxser_values = performance.loc[maxser_portfolios, metric]
        one_n_value = performance.loc['1/N', metric]
        
        q_values = [int(port.split('q')[1]) for port in ssd_portfolios]
        
        plt.plot(q_values, ssd_values, 'b-', label='SSD')
        plt.plot(q_values, maxser_values, 'g-', label='MAXSER')
        plt.axhline(y=one_n_value, color='r', linestyle='--', label='1/N')
        
        plt.title(metric)
        plt.xlabel('Number of Assets (q)')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300)
    plt.show()
    
    return

def rolling_window_analysis(returns, window_size=120, test_size=120, q=25):
    """
    Perform rolling window analysis
    
    Parameters:
    - returns: DataFrame with returns
    - window_size: Training window size
    - test_size: Number of test periods
    - q: Number of assets to select
    
    Returns:
    - Dictionary with results
    """
    total_periods = len(returns)
    num_windows = total_periods - window_size - test_size + 1
    
    if num_windows <= 0:
        raise ValueError("Not enough data for rolling window analysis")
    
    # Initialize results
    ssd_portfolio_values = np.ones(total_periods)
    maxser_portfolio_values = np.ones(total_periods)
    one_n_portfolio_values = np.ones(total_periods)
    
    # Initialize weights tracking
    ssd_weights_history = []
    maxser_weights_history = []
    
    # Initialize performance tracking
    ssd_performances = []
    maxser_performances = []
    one_n_performances = []
    
    # Initialize asset selection tracking
    ssd_assets_history = []
    maxser_assets_history = []
    
    # Progress bar
    pbar = tqdm(range(num_windows), desc="Running rolling window analysis")
    
    for i in pbar:
        # Define training and test periods
        train_start = i
        train_end = i + window_size
        test_start = train_end
        test_end = min(test_start + test_size, total_periods)
        
        # Extract training and test data
        train_returns = returns.iloc[train_start:train_end]
        test_returns = returns.iloc[test_start:test_end]
        
        # SSD optimization
        ssd_optimizer = SSDSpanningOptimizer(train_returns)
        ssd_assets, _ = ssd_optimizer.forward_stepwise_selection(q)
        
        # Create SSD portfolio
        mask = np.zeros(train_returns.shape[1], dtype=bool)
        for asset in ssd_assets:
            mask[train_returns.columns.get_loc(asset)] = True
        
        ssd_returns = train_returns.loc[:, mask]
        sparse_optimizer = SSDSpanningOptimizer(ssd_returns)
        
        # Optimize for SSD portfolio
        N1 = 10  # Number of points
        N2 = 5   # Number of piecewise segments
        min_return = np.min(train_returns.values)
        max_return = np.max(train_returns.values)
        z = np.linspace(min_return, max_return, N1)
        utility_weights = np.zeros(N1)
        utility_weights[0] = 1.0 / N2
        for i in range(1, N1):
            utility_weights[i] = (i / (N1 - 1)) / N2
        utility_weights = utility_weights / np.sum(utility_weights)
        
        ssd_weights = sparse_optimizer.optimize_portfolio_lp(z, utility_weights)
        
        # Convert to full dimension
        full_ssd_weights = np.zeros(train_returns.shape[1])
        full_ssd_weights[mask] = ssd_weights
        
        # MAXSER optimization
        maxser_optimizer = MAXSEROptimizer(train_returns)
        maxser_assets, maxser_weights = maxser_optimizer.optimize(q)
        
        # Equal weight portfolio
        one_n_weights = np.ones(train_returns.shape[1]) / train_returns.shape[1]
        
        # Calculate test returns
        ssd_test_returns = np.dot(test_returns.values, full_ssd_weights)
        maxser_test_returns = np.dot(test_returns.values, maxser_weights)
        one_n_test_returns = np.dot(test_returns.values, one_n_weights)
        
        # Update portfolio values
        for j in range(test_start, test_end):
            idx = j - test_start
            ssd_portfolio_values[j] = ssd_portfolio_values[test_start-1] * (1 + ssd_test_returns[idx])
            maxser_portfolio_values[j] = maxser_portfolio_values[test_start-1] * (1 + maxser_test_returns[idx])
            one_n_portfolio_values[j] = one_n_portfolio_values[test_start-1] * (1 + one_n_test_returns[idx])
        
        # Track weights
        ssd_weights_history.append(full_ssd_weights)
        maxser_weights_history.append(maxser_weights)
        
        # Track asset selection
        ssd_assets_history.append(ssd_assets)
        maxser_assets_history.append(maxser_assets)
        
        # Calculate performance metrics
        ssd_performance = {
            'Mean Return': np.mean(ssd_test_returns),
            'Standard Deviation': np.std(ssd_test_returns),
            'Sharpe Ratio': np.mean(ssd_test_returns) / np.std(ssd_test_returns) if np.std(ssd_test_returns) > 0 else 0,
            'VaR (95%)': np.percentile(ssd_test_returns, 5) * (-1),
            'Window': i
        }
        
        maxser_performance = {
            'Mean Return': np.mean(maxser_test_returns),
            'Standard Deviation': np.std(maxser_test_returns),
            'Sharpe Ratio': np.mean(maxser_test_returns) / np.std(maxser_test_returns) if np.std(maxser_test_returns) > 0 else 0,
            'VaR (95%)': np.percentile(maxser_test_returns, 5) * (-1),
            'Window': i
        }
        
        one_n_performance = {
            'Mean Return': np.mean(one_n_test_returns),
            'Standard Deviation': np.std(one_n_test_returns),
            'Sharpe Ratio': np.mean(one_n_test_returns) / np.std(one_n_test_returns) if np.std(one_n_test_returns) > 0 else 0,
            'VaR (95%)': np.percentile(one_n_test_returns, 5) * (-1),
            'Window': i
        }
        
        ssd_performances.append(ssd_performance)
        maxser_performances.append(maxser_performance)
        one_n_performances.append(one_n_performance)
    
    # Convert to DataFrames
    ssd_perf_df = pd.DataFrame(ssd_performances)
    maxser_perf_df = pd.DataFrame(maxser_performances)
    one_n_perf_df = pd.DataFrame(one_n_performances)
    
    # Add method column
    ssd_perf_df['Method'] = 'SSD'
    maxser_perf_df['Method'] = 'MAXSER'
    one_n_perf_df['Method'] = '1/N'
    
    # Combine
    performance_df = pd.concat([ssd_perf_df, maxser_perf_df, one_n_perf_df])
    
    return {
        'ssd_portfolio_values': ssd_portfolio_values,
        'maxser_portfolio_values': maxser_portfolio_values,
        'one_n_portfolio_values': one_n_portfolio_values,
        'ssd_weights_history': ssd_weights_history,
        'maxser_weights_history': maxser_weights_history,
        'ssd_assets_history': ssd_assets_history,
        'maxser_assets_history': maxser_assets_history,
        'performance': performance_df
    }

def plot_rolling_window_results(results, returns):
    """
    Plot the results of the rolling window analysis
    
    Parameters:
    - results: Dictionary with rolling window results
    - returns: Original returns DataFrame
    """
    # Extract portfolio values
    ssd_values = results['ssd_portfolio_values']
    maxser_values = results['maxser_portfolio_values']
    one_n_values = results['one_n_portfolio_values']
    
    # Create time index
    time_index = returns.index
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_index, ssd_values, 'b-', label='SSD')
    plt.plot(time_index, maxser_values, 'g-', label='MAXSER')
    plt.plot(time_index, one_n_values, 'r--', label='1/N')
    plt.title('Cumulative Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot rolling performance metrics
    performance = results['performance']
    
    plt.subplot(2, 2, 2)
    sns.lineplot(data=performance, x='Window', y='Sharpe Ratio', hue='Method')
    plt.title('Rolling Sharpe Ratio')
    plt.xlabel('Window')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    sns.lineplot(data=performance, x='Window', y='Mean Return', hue='Method')
    plt.title('Rolling Mean Return')
    plt.xlabel('Window')
    plt.ylabel('Mean Return')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    sns.lineplot(data=performance, x='Window', y='VaR (95%)', hue='Method')
    plt.title('Rolling VaR (95%)')
    plt.xlabel('Window')
    plt.ylabel('VaR (95%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_window_results.png', dpi=300)
    plt.show()
    
    # Analyze asset stability
    ssd_assets = results['ssd_assets_history']
    maxser_assets = results['maxser_assets_history']
    
    # Count asset appearances
    ssd_asset_counts = {}
    maxser_asset_counts = {}
    
    for assets in ssd_assets:
        for asset in assets:
            ssd_asset_counts[asset] = ssd_asset_counts.get(asset, 0) + 1
    
    for assets in maxser_assets:
        for asset in assets:
            maxser_asset_counts[asset] = maxser_asset_counts.get(asset, 0) + 1
    
    # Convert to DataFrames
    ssd_counts = pd.DataFrame({
        'Asset': list(ssd_asset_counts.keys()),
        'Count': list(ssd_asset_counts.values()),
        'Method': 'SSD'
    })
    
    maxser_counts = pd.DataFrame({
        'Asset': list(maxser_asset_counts.keys()),
        'Count': list(maxser_asset_counts.values()),
        'Method': 'MAXSER'
    })
    
    asset_counts = pd.concat([ssd_counts, maxser_counts])
    
    # Plot top 20 most selected assets
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    top_ssd = ssd_counts.sort_values('Count', ascending=False).head(20)
    sns.barplot(data=top_ssd, x='Count', y='Asset')
    plt.title('Top 20 Most Selected Assets (SSD)')
    plt.xlabel('Count')
    plt.ylabel('Asset')
    
    plt.subplot(1, 2, 2)
    top_maxser = maxser_counts.sort_values('Count', ascending=False).head(20)
    sns.barplot(data=top_maxser, x='Count', y='Asset')
    plt.title('Top 20 Most Selected Assets (MAXSER)')
    plt.xlabel('Count')
    plt.ylabel('Asset')
    
    plt.tight_layout()
    plt.savefig('asset_stability.png', dpi=300)
    plt.show()
    
    return


def main():
    """
    Main function to run the experiments
    """
    print("Starting experiments...")
    
    # Run static experiments
    results = run_experiments()
    
    # Plot results
    plot_results(results)
    
    # Run rolling window analysis
    print("\nRunning rolling window analysis...")
    rolling_results = rolling_window_analysis(results['returns'], window_size=120, test_size=60, q=25)
    
    # Plot rolling window results
    plot_rolling_window_results(rolling_results, results['returns'])
    
    print("\nExperiments completed!")

if __name__ == "__main__":
    main()