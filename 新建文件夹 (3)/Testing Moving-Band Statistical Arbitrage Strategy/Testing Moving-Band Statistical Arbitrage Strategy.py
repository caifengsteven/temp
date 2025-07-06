import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import yfinance as yf
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

class StatArb:
    """
    Statistical Arbitrage class implementing both fixed-band and moving-band strategies
    based on the convex-concave optimization approach from the paper.
    """
    
    def __init__(self, leverage_limit=50, band_width=2, moving_band=False, memory=21, 
                 cleanup_threshold=0.05, max_iterations=50):
        """
        Initialize the StatArb strategy.
        
        Parameters:
        leverage_limit (float): Limit on the leverage of the portfolio
        band_width (float): Width of the price band (fixed at 2 in the paper)
        moving_band (bool): Whether to use moving-band (True) or fixed-band (False)
        memory (int): Number of periods for moving average calculation in moving-band
        cleanup_threshold (float): Threshold for removing small weights in cleanup phase
        max_iterations (int): Maximum number of iterations for convex-concave procedure
        """
        self.leverage_limit = leverage_limit
        self.band_width = band_width
        self.moving_band = moving_band
        self.memory = memory
        self.cleanup_threshold = cleanup_threshold
        self.max_iterations = max_iterations
        self.portfolio = None
        self.midpoint = None
        
    def fit(self, prices, init_portfolio=None, verbose=False):
        """
        Find a statistical arbitrage using convex-concave optimization.
        
        Parameters:
        prices (numpy.ndarray): Asset prices, shape (T, n) where T is number of time periods
                                and n is number of assets
        init_portfolio (numpy.ndarray): Initial portfolio weights, shape (n,)
        verbose (bool): Whether to print progress information
        
        Returns:
        portfolio (numpy.ndarray): Portfolio weights
        midpoint (float): Midpoint of the price band (fixed-band only)
        """
        T, n = prices.shape
        
        # Scale prices to improve numerical stability
        self.price_scaler = StandardScaler()
        scaled_prices = self.price_scaler.fit_transform(prices)
        
        # Average price for leverage constraint
        P_bar = np.mean(scaled_prices, axis=0)
        
        # Initialize portfolio
        if init_portfolio is None:
            portfolio = np.random.uniform(0, 1, n)
        else:
            portfolio = init_portfolio.copy()
            
        # Initialize price and midpoint
        price = scaled_prices @ portfolio
        
        if self.moving_band:
            # For moving-band, initialize midpoints as moving averages
            midpoints = np.zeros(T)
            for t in range(T):
                if t < self.memory:
                    midpoints[t] = np.mean(price[:t+1])
                else:
                    midpoints[t] = np.mean(price[t-self.memory+1:t+1])
            midpoint = None
        else:
            # For fixed-band, initialize midpoint as mean price
            midpoint = np.mean(price)
            midpoints = np.full(T, midpoint)
        
        # Convex-concave procedure
        objective_values = []
        
        for iteration in range(self.max_iterations):
            # Compute portfolio price
            price = scaled_prices @ portfolio
            
            if self.moving_band:
                # Update midpoints based on current portfolio
                for t in range(T):
                    if t < self.memory:
                        midpoints[t] = np.mean(price[:t+1])
                    else:
                        midpoints[t] = np.mean(price[t-self.memory+1:t+1])
            
            # Compute current objective value (price variance)
            objective = np.sum(np.diff(price)**2)
            objective_values.append(objective)
            
            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: objective = {objective:.6f}")
            
            # Linearize the objective
            price_grad = np.zeros(T)
            price_grad[0] = 2 * (price[0] - price[1])
            price_grad[1:-1] = 2 * (2 * price[1:-1] - price[:-2] - price[2:])
            price_grad[-1] = 2 * (price[-1] - price[-2])
            
            # Define and solve the convex problem
            s = cp.Variable(n)
            mu = cp.Variable() if not self.moving_band else None
            p = cp.Variable(T)
            
            # Objective: maximize linearized variance
            objective = cp.Maximize(price_grad @ p)
            
            # Constraints
            constraints = [
                p == scaled_prices @ s,
                cp.sum(cp.abs(s) * P_bar) <= self.leverage_limit
            ]
            
            if self.moving_band:
                # Moving-band constraints
                for t in range(T):
                    constraints.append(p[t] - midpoints[t] <= self.band_width / 2)
                    constraints.append(p[t] - midpoints[t] >= -self.band_width / 2)
            else:
                # Fixed-band constraints
                constraints.extend([
                    p - mu <= self.band_width / 2,
                    p - mu >= -self.band_width / 2,
                    mu >= 0
                ])
            
            # Solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS)
            
            # Check convergence
            if iteration > 0 and abs(objective_values[iteration] - objective_values[iteration-1]) < 1e-4:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                break
                
            # Update portfolio and midpoint
            portfolio = s.value
            if not self.moving_band and mu is not None:
                midpoint = mu.value
        
        # Cleanup phase: remove small weights
        if self.cleanup_threshold > 0:
            small_indices = np.where(np.abs(portfolio) * P_bar < self.cleanup_threshold * np.sum(np.abs(portfolio) * P_bar))[0]
            
            if len(small_indices) > 0:
                # Create mask for assets to keep
                mask = np.ones(n, dtype=bool)
                mask[small_indices] = False
                
                # Create reduced problem
                reduced_prices = scaled_prices[:, mask]
                reduced_portfolio = portfolio[mask]
                
                if verbose:
                    print(f"Cleanup: removing {len(small_indices)} assets with small weights")
                
                # Re-solve with reduced assets
                reduced_result = self.fit(reduced_prices, reduced_portfolio, verbose=False)
                
                # Reconstruct full portfolio with zeros for removed assets
                full_portfolio = np.zeros(n)
                full_portfolio[mask] = reduced_result[0]
                portfolio = full_portfolio
        
        self.portfolio = portfolio
        self.midpoint = midpoint
        
        return portfolio, midpoint
    
    def transform(self, prices):
        """
        Apply the fitted portfolio to new price data.
        
        Parameters:
        prices (numpy.ndarray): Asset prices, shape (T, n)
        
        Returns:
        tuple: (portfolio_prices, midpoints, band_upper, band_lower)
        """
        if self.portfolio is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale prices
        scaled_prices = self.price_scaler.transform(prices)
        
        # Compute portfolio price
        price = scaled_prices @ self.portfolio
        
        if self.moving_band:
            # Compute moving midpoints
            midpoints = np.zeros(len(price))
            for t in range(len(price)):
                if t < self.memory:
                    midpoints[t] = np.mean(price[:t+1])
                else:
                    midpoints[t] = np.mean(price[t-self.memory+1:t+1])
        else:
            # Use fixed midpoint
            midpoints = np.full(len(price), self.midpoint)
        
        # Compute band limits
        band_upper = midpoints + self.band_width / 2
        band_lower = midpoints - self.band_width / 2
        
        return price, midpoints, band_upper, band_lower
    
    def trade(self, prices, trading_cost=0.001, shorting_cost=0.005, initial_cash=None, 
              exit_after=63, exit_days=21, stop_loss_pct=0.5):
        """
        Simulate trading the stat-arb strategy.
        
        Parameters:
        prices (numpy.ndarray): Asset prices, shape (T, n)
        trading_cost (float): One-way trading cost as a fraction
        shorting_cost (float): Annual shorting cost as a fraction
        initial_cash (float): Initial cash. If None, set to half the portfolio leverage
        exit_after (int): Number of days after which to start exiting the position
        exit_days (int): Number of days over which to exit the position
        stop_loss_pct (float): Stop loss as percentage of initial cash
        
        Returns:
        dict: Trading results including portfolio values, positions, returns, etc.
        """
        if self.portfolio is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        T = len(prices)
        
        # Scale prices
        scaled_prices = self.price_scaler.transform(prices)
        
        # Compute portfolio price
        price = scaled_prices @ self.portfolio
        
        if self.moving_band:
            # Compute moving midpoints
            midpoints = np.zeros(T)
            for t in range(T):
                if t < self.memory:
                    midpoints[t] = np.mean(price[:t+1])
                else:
                    midpoints[t] = np.mean(price[t-self.memory+1:t+1])
        else:
            # Use fixed midpoint
            midpoints = np.full(T, self.midpoint)
        
        # Initialize trading variables
        daily_shorting_cost = shorting_cost / 252  # Convert annual to daily
        
        # Initial position and cash
        position = np.zeros(T)
        if initial_cash is None:
            # Set initial cash to half the portfolio leverage
            initial_cash = 0.5 * np.sum(np.abs(self.portfolio) * np.mean(prices[0]))
        
        cash = np.zeros(T)
        cash[0] = initial_cash
        
        # Track portfolio value, trades, costs
        portfolio_value = np.zeros(T)
        portfolio_value[0] = cash[0]
        
        trades = np.zeros(T)
        trading_costs = np.zeros(T)
        holding_costs = np.zeros(T)
        
        # Simple linear trading policy
        for t in range(1, T):
            # Determine position based on price relative to midpoint
            if t < exit_after:
                # Normal trading period
                position[t] = midpoints[t] - price[t]
            elif t < exit_after + exit_days:
                # Exit period - linearly reduce position
                exit_factor = 1 - (t - exit_after) / exit_days
                position[t] = exit_factor * (midpoints[t] - price[t])
            else:
                # Fully exited
                position[t] = 0
            
            # Calculate trade
            trades[t] = position[t] - position[t-1]
            
            # Calculate trading cost
            trading_costs[t] = abs(trades[t]) * price[t] * trading_cost
            
            # Calculate holding cost for short positions
            if position[t-1] < 0:
                holding_costs[t] = abs(position[t-1] * price[t-1]) * daily_shorting_cost
            
            # Update cash
            cash[t] = cash[t-1] - trades[t] * price[t] - trading_costs[t] - holding_costs[t]
            
            # Calculate portfolio value
            portfolio_value[t] = cash[t] + position[t] * price[t]
            
            # Check stop loss
            if portfolio_value[t] < initial_cash * (1 - stop_loss_pct):
                # Liquidate position
                trades[t] = -position[t-1]
                trading_costs[t] = abs(trades[t]) * price[t] * trading_cost
                cash[t] = cash[t-1] - trades[t] * price[t] - trading_costs[t] - holding_costs[t]
                position[t] = 0
                portfolio_value[t] = cash[t]
                
                # Stop trading
                position[t+1:] = 0
                cash[t+1:] = cash[t]
                portfolio_value[t+1:] = cash[t]
                break
        
        # Calculate returns and performance metrics
        returns = np.zeros(T)
        returns[1:] = (portfolio_value[1:] - portfolio_value[:-1]) / portfolio_value[:-1]
        
        cum_returns = np.cumprod(1 + returns) - 1
        
        total_return = portfolio_value[-1] / portfolio_value[0] - 1
        annualized_return = (1 + total_return) ** (252 / T) - 1
        
        daily_std = np.std(returns[1:])
        annualized_risk = daily_std * np.sqrt(252)
        
        sharpe_ratio = annualized_return / annualized_risk if annualized_risk > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = np.max(drawdown)
        
        results = {
            'price': price,
            'midpoints': midpoints,
            'position': position,
            'cash': cash,
            'portfolio_value': portfolio_value,
            'trades': trades,
            'trading_costs': trading_costs,
            'holding_costs': holding_costs,
            'returns': returns,
            'cum_returns': cum_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_risk': annualized_risk,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return results

def generate_simulated_data(n_assets=10, n_days=500, mean_reversion=0.1, volatility=0.02, 
                            correlation=0.5, seed=None):
    """
    Generate simulated price data with mean-reverting behavior.
    
    Parameters:
    n_assets (int): Number of assets
    n_days (int): Number of days
    mean_reversion (float): Mean reversion strength
    volatility (float): Asset volatility
    correlation (float): Correlation between assets
    seed (int): Random seed
    
    Returns:
    numpy.ndarray: Simulated prices, shape (n_days, n_assets)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create correlation structure
    corr_matrix = np.ones((n_assets, n_assets)) * correlation
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Cholesky decomposition for correlated random variables
    chol = np.linalg.cholesky(corr_matrix)
    
    # Generate price processes
    prices = np.zeros((n_days, n_assets))
    
    # Initialize with starting prices around 100
    prices[0] = 100 * (1 + 0.1 * np.random.randn(n_assets))
    
    # Simulate price processes with mean reversion
    for t in range(1, n_days):
        # Generate correlated random shocks
        shocks = np.random.randn(n_assets) @ chol.T * volatility
        
        # Apply mean reversion to log prices
        log_prices = np.log(prices[t-1])
        log_mean = np.log(100)  # Mean price level
        
        # Mean-reverting update
        new_log_prices = log_prices + mean_reversion * (log_mean - log_prices) + shocks
        
        # Convert back to prices
        prices[t] = np.exp(new_log_prices)
    
    return prices

def download_real_data(tickers, start_date, end_date):
    """
    Download real market data for a list of tickers.
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    numpy.ndarray: Price data, shape (n_days, n_assets)
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Use adjusted close prices
    prices = data['Adj Close'].values
    
    # Handle missing values
    prices = pd.DataFrame(prices, columns=tickers).fillna(method='ffill').values
    
    return prices

def plot_results(prices, results, title, moving_band=False):
    """
    Plot the results of the stat-arb strategy.
    
    Parameters:
    prices (numpy.ndarray): Asset prices
    results (dict): Trading results
    title (str): Plot title
    moving_band (bool): Whether this is a moving-band strategy
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # Plot 1: Portfolio price and band
    axs[0].plot(results['price'], 'b-', label='Portfolio Price')
    axs[0].plot(results['midpoints'], 'k--', label='Band Midpoint')
    axs[0].fill_between(
        range(len(results['price'])), 
        results['midpoints'] - 1, 
        results['midpoints'] + 1,
        color='gray', alpha=0.2, label='Price Band'
    )
    
    band_type = "Moving" if moving_band else "Fixed"
    axs[0].set_title(f'{title} - {band_type} Band')
    axs[0].set_xlabel('Trading Day')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Position
    axs[1].plot(results['position'], 'g-', label='Position')
    axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1].set_title('Trading Position')
    axs[1].set_xlabel('Trading Day')
    axs[1].set_ylabel('Position')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: Portfolio value and cumulative returns
    axs[2].plot(results['portfolio_value'], 'r-', label='Portfolio Value')
    axs[2].set_title('Portfolio Value')
    axs[2].set_xlabel('Trading Day')
    axs[2].set_ylabel('Value ($)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)
    
    # Add twin axis for cumulative returns
    ax2 = axs[2].twinx()
    ax2.plot(results['cum_returns'] * 100, 'b--', label='Cumulative Return (%)')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='upper right')
    
    # Add performance metrics as text
    metrics_text = (
        f"Annualized Return: {results['annualized_return']*100:.2f}%\n"
        f"Annualized Risk: {results['annualized_risk']*100:.2f}%\n"
        f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {results['max_drawdown']*100:.2f}%"
    )
    axs[2].text(0.02, 0.05, metrics_text, transform=axs[2].transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def test_with_simulated_data():
    """
    Test the StatArb strategy with simulated data.
    """
    print("Testing with simulated data...")
    
    # Generate simulated data
    n_assets = 10
    train_days = 252  # 1 year
    test_days = 126  # 6 months
    total_days = train_days + test_days
    
    prices = generate_simulated_data(n_assets=n_assets, n_days=total_days, 
                                     mean_reversion=0.05, volatility=0.015,
                                     correlation=0.6, seed=42)
    
    # Split into training and testing sets
    train_prices = prices[:train_days]
    test_prices = prices[train_days-21:total_days]  # Include 21 days overlap for moving average
    
    # Test fixed-band strategy
    print("\nTesting fixed-band strategy...")
    fixed_stat_arb = StatArb(leverage_limit=50, moving_band=False)
    fixed_stat_arb.fit(train_prices, verbose=True)
    
    # Get portfolio weights
    portfolio = fixed_stat_arb.portfolio
    print(f"Portfolio weights: {portfolio}")
    print(f"Number of assets in portfolio: {np.sum(np.abs(portfolio) > 1e-5)}")
    
    # Trade on test data
    fixed_results = fixed_stat_arb.trade(test_prices, exit_after=63, exit_days=21)
    
    # Plot results
    plot_results(test_prices, fixed_results, "Fixed-Band Stat-Arb", moving_band=False)
    
    # Test moving-band strategy
    print("\nTesting moving-band strategy...")
    moving_stat_arb = StatArb(leverage_limit=100, moving_band=True, memory=21)
    moving_stat_arb.fit(train_prices, verbose=True)
    
    # Get portfolio weights
    portfolio = moving_stat_arb.portfolio
    print(f"Portfolio weights: {portfolio}")
    print(f"Number of assets in portfolio: {np.sum(np.abs(portfolio) > 1e-5)}")
    
    # Trade on test data
    moving_results = moving_stat_arb.trade(test_prices, exit_after=125, exit_days=21)
    
    # Plot results
    plot_results(test_prices, moving_results, "Moving-Band Stat-Arb", moving_band=True)
    
    # Compare performance
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} {'Fixed-Band':<15} {'Moving-Band':<15}")
    print("-" * 50)
    print(f"{'Annualized Return':<20} {fixed_results['annualized_return']*100:>14.2f}% {moving_results['annualized_return']*100:>14.2f}%")
    print(f"{'Annualized Risk':<20} {fixed_results['annualized_risk']*100:>14.2f}% {moving_results['annualized_risk']*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<20} {fixed_results['sharpe_ratio']:>14.2f} {moving_results['sharpe_ratio']:>14.2f}")
    print(f"{'Max Drawdown':<20} {fixed_results['max_drawdown']*100:>14.2f}% {moving_results['max_drawdown']*100:>14.2f}%")

def test_with_real_data():
    """
    Test the StatArb strategy with real market data.
    """
    print("Testing with real market data...")
    
    # Define tickers (common tech and financial stocks)
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
        'PG', 'JNJ', 'KO', 'PEP', 'WMT', 'DIS'
    ]
    
    # Download data
    train_start = '2020-01-01'
    train_end = '2020-12-31'
    test_start = '2021-01-01'
    test_end = '2021-06-30'
    
    try:
        print(f"Downloading training data ({train_start} to {train_end})...")
        train_prices = download_real_data(tickers, train_start, train_end)
        
        print(f"Downloading test data ({test_start} to {test_end})...")
        test_prices = download_real_data(tickers, test_start, test_end)
        
        # Test fixed-band strategy
        print("\nTesting fixed-band strategy...")
        fixed_stat_arb = StatArb(leverage_limit=50, moving_band=False)
        fixed_stat_arb.fit(train_prices, verbose=True)
        
        # Get portfolio weights
        portfolio = fixed_stat_arb.portfolio
        asset_weights = [(ticker, weight) for ticker, weight in zip(tickers, portfolio) if abs(weight) > 1e-5]
        print("Portfolio composition:")
        for ticker, weight in asset_weights:
            print(f"  {ticker}: {weight:.4f}")
        
        # Trade on test data
        fixed_results = fixed_stat_arb.trade(test_prices, exit_after=63, exit_days=21)
        
        # Plot results
        plot_results(test_prices, fixed_results, "Fixed-Band Stat-Arb (Real Data)", moving_band=False)
        
        # Test moving-band strategy
        print("\nTesting moving-band strategy...")
        moving_stat_arb = StatArb(leverage_limit=100, moving_band=True, memory=21)
        moving_stat_arb.fit(train_prices, verbose=True)
        
        # Get portfolio weights
        portfolio = moving_stat_arb.portfolio
        asset_weights = [(ticker, weight) for ticker, weight in zip(tickers, portfolio) if abs(weight) > 1e-5]
        print("Portfolio composition:")
        for ticker, weight in asset_weights:
            print(f"  {ticker}: {weight:.4f}")
        
        # Trade on test data
        moving_results = moving_stat_arb.trade(test_prices, exit_after=125, exit_days=21)
        
        # Plot results
        plot_results(test_prices, moving_results, "Moving-Band Stat-Arb (Real Data)", moving_band=True)
        
        # Compare performance
        print("\nPerformance Comparison (Real Data):")
        print(f"{'Metric':<20} {'Fixed-Band':<15} {'Moving-Band':<15}")
        print("-" * 50)
        print(f"{'Annualized Return':<20} {fixed_results['annualized_return']*100:>14.2f}% {moving_results['annualized_return']*100:>14.2f}%")
        print(f"{'Annualized Risk':<20} {fixed_results['annualized_risk']*100:>14.2f}% {moving_results['annualized_risk']*100:>14.2f}%")
        print(f"{'Sharpe Ratio':<20} {fixed_results['sharpe_ratio']:>14.2f} {moving_results['sharpe_ratio']:>14.2f}")
        print(f"{'Max Drawdown':<20} {fixed_results['max_drawdown']*100:>14.2f}% {moving_results['max_drawdown']*100:>14.2f}%")
    
    except Exception as e:
        print(f"Error downloading or processing real data: {e}")
        print("Skipping real data test.")

def test_multiple_initializations():
    """
    Test multiple random initializations and analyze the results.
    """
    print("Testing multiple random initializations...")
    
    # Generate simulated data
    n_assets = 15
    train_days = 252  # 1 year
    test_days = 126  # 6 months
    total_days = train_days + test_days
    
    prices = generate_simulated_data(n_assets=n_assets, n_days=total_days, 
                                     mean_reversion=0.05, volatility=0.015,
                                     correlation=0.6, seed=42)
    
    # Split into training and testing sets
    train_prices = prices[:train_days]
    test_prices = prices[train_days-21:total_days]  # Include 21 days overlap for moving average
    
    # Number of random initializations
    n_init = 10
    
    # Store results
    fixed_returns = []
    fixed_sharpes = []
    fixed_assets = []
    
    moving_returns = []
    moving_sharpes = []
    moving_assets = []
    
    # Run multiple initializations
    for i in tqdm(range(n_init), desc="Testing initializations"):
        # Fixed-band strategy
        fixed_stat_arb = StatArb(leverage_limit=50, moving_band=False)
        fixed_stat_arb.fit(train_prices, verbose=False)
        
        # Count assets in portfolio
        portfolio = fixed_stat_arb.portfolio
        n_fixed_assets = np.sum(np.abs(portfolio) > 1e-5)
        fixed_assets.append(n_fixed_assets)
        
        # Trade on test data
        fixed_results = fixed_stat_arb.trade(test_prices, exit_after=63, exit_days=21)
        fixed_returns.append(fixed_results['annualized_return'])
        fixed_sharpes.append(fixed_results['sharpe_ratio'])
        
        # Moving-band strategy
        moving_stat_arb = StatArb(leverage_limit=100, moving_band=True, memory=21)
        moving_stat_arb.fit(train_prices, verbose=False)
        
        # Count assets in portfolio
        portfolio = moving_stat_arb.portfolio
        n_moving_assets = np.sum(np.abs(portfolio) > 1e-5)
        moving_assets.append(n_moving_assets)
        
        # Trade on test data
        moving_results = moving_stat_arb.trade(test_prices, exit_after=125, exit_days=21)
        moving_returns.append(moving_results['annualized_return'])
        moving_sharpes.append(moving_results['sharpe_ratio'])
    
    # Analyze results
    print("\nFixed-Band Statistics:")
    print(f"Average number of assets: {np.mean(fixed_assets):.2f}")
    print(f"Average annualized return: {np.mean(fixed_returns)*100:.2f}%")
    print(f"Average Sharpe ratio: {np.mean(fixed_sharpes):.2f}")
    print(f"Profitable strategies: {np.sum(np.array(fixed_returns) > 0)}/{n_init}")
    
    print("\nMoving-Band Statistics:")
    print(f"Average number of assets: {np.mean(moving_assets):.2f}")
    print(f"Average annualized return: {np.mean(moving_returns)*100:.2f}%")
    print(f"Average Sharpe ratio: {np.mean(moving_sharpes):.2f}")
    print(f"Profitable strategies: {np.sum(np.array(moving_returns) > 0)}/{n_init}")
    
    # Visualize distribution of returns and Sharpe ratios
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Returns distribution
    sns.histplot(np.array(fixed_returns)*100, kde=True, ax=axs[0, 0], color='blue')
    axs[0, 0].set_title('Fixed-Band Annualized Returns (%)')
    axs[0, 0].axvline(x=0, color='red', linestyle='--')
    
    sns.histplot(np.array(moving_returns)*100, kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].set_title('Moving-Band Annualized Returns (%)')
    axs[0, 1].axvline(x=0, color='red', linestyle='--')
    
    # Sharpe ratio distribution
    sns.histplot(fixed_sharpes, kde=True, ax=axs[1, 0], color='blue')
    axs[1, 0].set_title('Fixed-Band Sharpe Ratios')
    axs[1, 0].axvline(x=0, color='red', linestyle='--')
    
    sns.histplot(moving_sharpes, kde=True, ax=axs[1, 1], color='green')
    axs[1, 1].set_title('Moving-Band Sharpe Ratios')
    axs[1, 1].axvline(x=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with simulated data
    test_with_simulated_data()
    
    # Test with real market data
    test_with_real_data()
    
    # Test multiple initializations
    test_multiple_initializations()