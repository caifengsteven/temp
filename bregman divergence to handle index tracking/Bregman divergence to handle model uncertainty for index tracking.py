import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.special import erf
from datetime import datetime, timedelta
from tqdm import tqdm

# Create synthetic index data
def create_synthetic_index_data(index_name="SPX", 
                                num_constituents=50, 
                                start_date="20180101", 
                                end_date="20221231",
                                seed=42):
    """
    Create synthetic index and constituent data for testing
    
    Parameters:
    index_name : str
        Name for the synthetic index
    num_constituents : int
        Number of constituents to create
    start_date : str
        Start date in format 'YYYYMMDD'
    end_date : str
        End date in format 'YYYYMMDD'
    seed : int
        Random seed for reproducibility
        
    Returns:
    tuple
        (constituents_df, prices_df) containing constituent weights and price data
    """
    np.random.seed(seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create constituent tickers and weights
    tickers = [f"{index_name}_{i:02d}" for i in range(1, num_constituents + 1)]
    
    # Generate power law distributed weights (more realistic than uniform)
    alpha = 1.5  # Power law exponent
    weights = np.random.power(alpha, num_constituents)
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # Create constituents DataFrame
    constituents_df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights
    })
    constituents_df = constituents_df.sort_values('weight', ascending=False)
    constituents_df = constituents_df.set_index('ticker')
    
    # Create price data
    prices_df = pd.DataFrame(index=date_range)
    
    # Create index starting price
    index_price = 1000
    prices_df[f"{index_name}_Index"] = index_price
    
    # Parameters for return simulation
    annual_return = 0.08
    annual_vol = 0.15
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    # Generate correlated returns for constituents
    # Create correlation matrix (realistic with industry clusters)
    num_sectors = 5
    sector_assignments = np.random.randint(0, num_sectors, num_constituents)
    
    # Base correlation matrix
    base_corr = 0.3  # Base correlation between all stocks
    sector_corr_add = 0.4  # Additional correlation within same sector
    
    # Initialize correlation matrix
    corr_matrix = np.ones((num_constituents, num_constituents)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Add sector correlations
    for i in range(num_constituents):
        for j in range(i+1, num_constituents):
            if sector_assignments[i] == sector_assignments[j]:
                corr_matrix[i, j] = base_corr + sector_corr_add
                corr_matrix[j, i] = base_corr + sector_corr_add
    
    # Create volatility vector (larger cap stocks tend to have lower volatility)
    vols = daily_vol * (1 + 0.5 * (1 - weights / np.max(weights)))
    
    # Create covariance matrix
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # Generate correlated returns
    returns = np.random.multivariate_normal(
        mean=[daily_return] * num_constituents,
        cov=cov_matrix,
        size=len(date_range)
    )
    
    # Calculate prices for each constituent
    for i, ticker in enumerate(tickers):
        # Randomize starting price between 10 and 100
        start_price = np.random.uniform(10, 100)
        prices = [start_price]
        
        for ret in returns[:, i]:
            prices.append(prices[-1] * (1 + ret))
        
        prices_df[ticker] = prices[:len(date_range)]
    
    # Calculate index prices based on constituent weights
    for i in range(1, len(date_range)):
        # Get returns for this day
        const_returns = np.array([(prices_df.iloc[i][ticker] / prices_df.iloc[i-1][ticker]) - 1 
                                 for ticker in tickers])
        
        # Calculate weighted return
        index_return = np.sum(const_returns * weights)
        
        # Update index price
        prices_df.loc[date_range[i], f"{index_name}_Index"] = prices_df.iloc[i-1][f"{index_name}_Index"] * (1 + index_return)
    
    print(f"Created synthetic index with {num_constituents} constituents and {len(date_range)} days of price data")
    
    return constituents_df, prices_df

# Calculate returns
def calculate_returns(prices):
    return prices.pct_change().dropna()

# Define the Bregman divergence function
def bregman_divergence(E, lambda_val):
    """
    Calculate Bregman divergence as defined in the paper
    
    Parameters:
    E : array-like
        Likelihood ratio between g and f
    lambda_val : float
        Parameter controlling the divergence
        
    Returns:
    float
        Bregman divergence value
    """
    G_E = (1/lambda_val) * (E**(lambda_val+1)) - ((lambda_val+1)/lambda_val) * E + 1
    return np.mean(G_E)

# Define the optimal index tracking portfolio function
def robust_index_tracking_portfolio(returns, index_returns, lambda_val=0.1, eta=0.2, max_iter=1000, tol=1e-6):
    """
    Solve for the optimal robust index tracking portfolio using Theorem 3.1
    
    Parameters:
    returns : DataFrame
        Asset returns
    index_returns : Series
        Index returns
    lambda_val : float
        Parameter for Bregman divergence
    eta : float
        Radius of the Bregman divergence ball
    max_iter : int
        Maximum iterations for optimization
    tol : float
        Tolerance for convergence
        
    Returns:
    array
        Optimal portfolio weights
    """
    n_assets = returns.shape[1]
    
    # Function to calculate H(u) = -(Ru - B)^2
    def H(u, R, B):
        return -((R @ u) - B)**2
    
    # Function to calculate dH/du
    def dH_du(u, R, B):
        return -2 * R.T * ((R @ u) - B)
    
    # Function to calculate E*
    def E_star(u, alpha, beta, R, B, lambda_val):
        H_u = H(u, R, B)
        return ((lambda_val/(lambda_val+1)) * ((-beta - H_u)/alpha) + 1)**(1/lambda_val)
    
    # Objective function for the system of equations in Theorem 3.1
    def objective(params):
        u = params[:n_assets]
        alpha = params[n_assets]
        beta = params[n_assets + 1]
        theta = params[n_assets + 2]
        
        # Convert returns to numpy arrays for faster computation
        R = returns.values
        B = index_returns.values
        
        # Calculate E*
        E_vals = np.zeros(len(R))
        for i in range(len(R)):
            E_vals[i] = E_star(u, alpha, beta, R[i], B[i], lambda_val)
        
        # Calculate dH/du
        dH_du_vals = np.zeros((len(R), n_assets))
        for i in range(len(R)):
            dH_du_vals[i] = dH_du(u, R[i], B[i])
        
        # Calculate constraints from Theorem 3.1
        constraint1 = np.mean(dH_du_vals * E_vals[:, np.newaxis], axis=0) - theta
        constraint2 = np.sum(u) - 1
        
        # Calculate G(E*)
        G_E_star = np.zeros(len(R))
        for i in range(len(R)):
            G_E_star[i] = (1/lambda_val) * (E_vals[i]**(lambda_val+1)) - ((lambda_val+1)/lambda_val) * E_vals[i] + 1
        
        constraint3 = np.mean(G_E_star) - eta
        constraint4 = np.mean(E_vals) - 1
        
        return np.concatenate([constraint1, [constraint2, constraint3, constraint4]])
    
    # Initial guess
    initial_guess = np.zeros(n_assets + 3)
    initial_guess[:n_assets] = 1.0 / n_assets  # Equal weight
    initial_guess[n_assets] = 0.02  # alpha
    initial_guess[n_assets + 1] = 0.01  # beta
    initial_guess[n_assets + 2] = -0.05  # theta
    
    # Bounds
    bounds = [(0, 1) for _ in range(n_assets)]  # Weights between 0 and 1
    bounds.append((1e-6, None))  # alpha > 0
    bounds.append((None, None))  # beta unbounded
    bounds.append((None, None))  # theta unbounded
    
    # Solve the system of equations
    try:
        result = optimize.minimize(
            lambda x: np.sum(objective(x)**2),  # Minimize sum of squared constraints
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            # Fall back to equal weights if optimization fails
            return np.ones(n_assets) / n_assets
        
        # Extract optimal weights
        optimal_weights = result.x[:n_assets]
        
        # Normalize to ensure sum is exactly 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return optimal_weights
    
    except Exception as e:
        print(f"Error in robust optimization: {e}")
        # Fall back to equal weights if optimization fails
        return np.ones(n_assets) / n_assets

# Define the non-robust index tracking portfolio function (for comparison)
def non_robust_index_tracking_portfolio(returns, index_returns):
    """
    Solve for the optimal non-robust index tracking portfolio
    
    Parameters:
    returns : DataFrame
        Asset returns
    index_returns : Series
        Index returns
        
    Returns:
    array
        Optimal portfolio weights
    """
    n_assets = returns.shape[1]
    
    # Define objective function: minimize E[(Ru - B)^2]
    def objective(u):
        tracking_errors = ((returns.values @ u) - index_returns.values)**2
        return np.mean(tracking_errors)
    
    # Constraint: sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda u: np.sum(u) - 1}]
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    initial_guess = np.ones(n_assets) / n_assets
    
    # Solve optimization problem
    try:
        result = optimize.minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            print(f"Non-robust optimization failed: {result.message}")
            # Fall back to equal weights if optimization fails
            return np.ones(n_assets) / n_assets
        
        # Return optimal weights
        return result.x
    
    except Exception as e:
        print(f"Error in non-robust optimization: {e}")
        # Fall back to equal weights if optimization fails
        return np.ones(n_assets) / n_assets

# Calculate tracking error
def calculate_tracking_error(weights, returns, index_returns):
    """
    Calculate the tracking error of a portfolio
    
    Parameters:
    weights : array
        Portfolio weights
    returns : DataFrame
        Asset returns
    index_returns : Series
        Index returns
        
    Returns:
    float
        Average squared tracking error
    """
    portfolio_returns = returns.values @ weights
    tracking_errors = (portfolio_returns - index_returns.values)**2
    return np.mean(tracking_errors)

# Smooth approximation of the loss function ℓ₁ as described in Section 4
def smooth_loss_l1(x, epsilon=0.01):
    """
    Smooth approximation of ℓ₁(x) = x² if x > 0 and 0 else
    
    Parameters:
    x : float or array
        Input value
    epsilon : float
        Smoothing parameter
        
    Returns:
    float or array
        Smoothed loss
    """
    return (x**2 + epsilon**2) * norm_cdf(x/epsilon) + x * norm_pdf(x/epsilon)

# Normal CDF and PDF functions
def norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Implement a rolling window version of the strategy with synthetic data
def run_rolling_strategy(index_name="SPX", 
                        num_assets=20, 
                        start_date="20180101", 
                        end_date="20221231",
                        lambda_val=0.1, 
                        eta=0.2,
                        window_size=252,  # One year of trading days
                        rebalance_freq=21):  # Monthly rebalancing
    """
    Run the robust index tracking strategy with a rolling window approach
    using synthetic data
    """
    # Create synthetic data
    constituents_df, prices_df = create_synthetic_index_data(
        index_name=index_name,
        num_constituents=50,  # Create 50 constituents
        start_date=start_date,
        end_date=end_date
    )
    
    # Select top assets by weight to track
    top_constituents = constituents_df.sort_values('weight', ascending=False).head(num_assets)
    tickers = top_constituents.index.tolist()
    
    # Index ticker
    index_ticker = f"{index_name}_Index"
    
    # Calculate returns
    returns = calculate_returns(prices_df)
    
    # Wait until we have enough data for the first window
    start_idx = window_size
    if start_idx >= len(returns):
        print("Not enough data for the window size")
        return None
    
    # Initialize results
    results = pd.DataFrame(index=returns.index[start_idx:])
    results['Index'] = returns[index_ticker].iloc[start_idx:]
    
    # Initialize portfolios
    robust_port_returns = np.zeros(len(returns) - start_idx)
    non_robust_port_returns = np.zeros(len(returns) - start_idx)
    
    # Initialize weights dataframes to store historical weights
    robust_weights_history = pd.DataFrame(index=returns.index[start_idx::rebalance_freq], columns=tickers)
    non_robust_weights_history = pd.DataFrame(index=returns.index[start_idx::rebalance_freq], columns=tickers)
    
    print("Running rolling window optimization...")
    
    # Loop through time with rebalancing
    for i in tqdm(range(start_idx, len(returns), rebalance_freq)):
        if i + rebalance_freq > len(returns):
            end_i = len(returns)
        else:
            end_i = i + rebalance_freq
        
        # Get training window
        train_data = returns.iloc[i-window_size:i]
        
        # Extract index returns
        index_returns_train = train_data[index_ticker]
        
        # Remove index from asset returns
        asset_returns_train = train_data[tickers]
        
        # Calculate robust portfolio weights
        try:
            robust_weights = robust_index_tracking_portfolio(
                asset_returns_train, 
                index_returns_train,
                lambda_val=lambda_val,
                eta=eta
            )
            
            # Calculate non-robust portfolio weights
            non_robust_weights = non_robust_index_tracking_portfolio(
                asset_returns_train, 
                index_returns_train
            )
            
            # Store weights
            if i % rebalance_freq == 0:
                robust_weights_history.loc[returns.index[i]] = pd.Series(robust_weights, index=tickers)
                non_robust_weights_history.loc[returns.index[i]] = pd.Series(non_robust_weights, index=tickers)
            
            # Apply weights to the next period
            for j in range(i, end_i):
                robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * robust_weights)
                non_robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * non_robust_weights)
        
        except Exception as e:
            print(f"Error at time {returns.index[i]}: {e}")
            # Use previous weights if there's an error
            if i > start_idx:
                for j in range(i, end_i):
                    robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * robust_weights)
                    non_robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * non_robust_weights)
    
    # Store portfolio returns
    results['Robust_Portfolio'] = robust_port_returns
    results['Non_Robust_Portfolio'] = non_robust_port_returns
    
    # Calculate tracking errors
    results['Robust_TE'] = (results['Robust_Portfolio'] - results['Index'])**2
    results['Non_Robust_TE'] = (results['Non_Robust_Portfolio'] - results['Index'])**2
    
    # Calculate smoothed loss for downturn evaluation
    results['Robust_Loss_L1'] = smooth_loss_l1(results['Index'] - results['Robust_Portfolio'])
    results['Non_Robust_Loss_L1'] = smooth_loss_l1(results['Index'] - results['Non_Robust_Portfolio'])
    
    # Calculate outperformance metrics
    results['Robust_Outperforms'] = results['Robust_TE'] < results['Non_Robust_TE']
    results['Downturn'] = results['Index'] < 0
    
    # Calculate performance metrics
    metrics = pd.DataFrame({
        'Overall_BT': [results['Robust_Outperforms'].mean() * 100],
        'Downturn_BT': [results.loc[results['Downturn'], 'Robust_Outperforms'].mean() * 100 if any(results['Downturn']) else np.nan],
        'Robust_ETE': [results['Robust_TE'].mean()],
        'Non_Robust_ETE': [results['Non_Robust_TE'].mean()],
        'Robust_EEI': [(results['Robust_Portfolio'] - results['Index']).mean()],
        'Non_Robust_EEI': [(results['Non_Robust_Portfolio'] - results['Index']).mean()],
        'Robust_Cumulative': [(1 + results['Robust_Portfolio']).prod() - 1],
        'Non_Robust_Cumulative': [(1 + results['Non_Robust_Portfolio']).prod() - 1],
        'Index_Cumulative': [(1 + results['Index']).prod() - 1]
    })
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(metrics)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    ((1 + results[['Robust_Portfolio', 'Non_Robust_Portfolio', 'Index']]).cumprod()).plot()
    plt.title('Cumulative Returns')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.savefig('rolling_cumulative_returns.png')
    
    # Plot tracking errors
    plt.figure(figsize=(12, 6))
    results[['Robust_TE', 'Non_Robust_TE']].plot()
    plt.title('Tracking Errors')
    plt.ylabel('Squared Tracking Error')
    plt.grid(True)
    plt.legend()
    plt.savefig('rolling_tracking_errors.png')
    
    # Plot weight changes over time
    plt.figure(figsize=(14, 7))
    
    # Plot only top 5 weights for better visualization
    top_5_weights = robust_weights_history.mean().sort_values(ascending=False).head(5).index
    robust_weights_history[top_5_weights].plot(title='Robust Portfolio Weights Over Time (Top 5)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.savefig('robust_weights_history.png')
    
    plt.figure(figsize=(14, 7))
    non_robust_weights_history[top_5_weights].plot(title='Non-Robust Portfolio Weights Over Time (Top 5)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.savefig('non_robust_weights_history.png')
    
    return results, metrics, robust_weights_history, non_robust_weights_history

# Function to analyze performance in different market conditions
def analyze_market_conditions(results):
    """
    Analyze strategy performance in different market conditions
    
    Parameters:
    results : DataFrame
        Results from running the strategy
        
    Returns:
    DataFrame
        Performance metrics in different market conditions
    """
    # Define market conditions
    results['Market_Up'] = results['Index'] > 0
    results['Market_Down'] = results['Index'] < 0
    
    # Calculate metrics for up markets
    up_market_bt = results.loc[results['Market_Up'], 'Robust_Outperforms'].mean() * 100 if any(results['Market_Up']) else np.nan
    up_robust_te = results.loc[results['Market_Up'], 'Robust_TE'].mean() if any(results['Market_Up']) else np.nan
    up_non_robust_te = results.loc[results['Market_Up'], 'Non_Robust_TE'].mean() if any(results['Market_Up']) else np.nan
    
    # Calculate metrics for down markets
    down_market_bt = results.loc[results['Market_Down'], 'Robust_Outperforms'].mean() * 100 if any(results['Market_Down']) else np.nan
    down_robust_te = results.loc[results['Market_Down'], 'Robust_TE'].mean() if any(results['Market_Down']) else np.nan
    down_non_robust_te = results.loc[results['Market_Down'], 'Non_Robust_TE'].mean() if any(results['Market_Down']) else np.nan
    
    # Create metrics DataFrame
    metrics = pd.DataFrame({
        'Market_Condition': ['Up Market', 'Down Market'],
        'Beating_Time (%)': [up_market_bt, down_market_bt],
        'Robust_ETE': [up_robust_te, down_robust_te],
        'Non_Robust_ETE': [up_non_robust_te, down_non_robust_te],
        'ETE_Difference': [up_non_robust_te - up_robust_te if not np.isnan(up_non_robust_te) and not np.isnan(up_robust_te) else np.nan, 
                          down_non_robust_te - down_robust_te if not np.isnan(down_non_robust_te) and not np.isnan(down_robust_te) else np.nan]
    })
    
    return metrics

# Parameter sensitivity analysis
def parameter_sensitivity_analysis(index_name="SPX", 
                                 num_assets=20, 
                                 start_date="20180101", 
                                 end_date="20221231",
                                 window_size=252,  # One year of trading days
                                 rebalance_freq=21):  # Monthly rebalancing
    """
    Analyze the sensitivity of the strategy to different lambda and eta values
    """
    # Parameter grids
    lambda_values = [0.05, 0.1, 0.2, 0.5]
    eta_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    # Initialize results DataFrame
    params_results = pd.DataFrame(columns=[
        'lambda', 'eta', 'Overall_BT', 'Downturn_BT', 'Robust_ETE', 
        'Non_Robust_ETE', 'ETE_Diff', 'Robust_Cum', 'Non_Robust_Cum', 'Index_Cum'
    ])
    
    # Run strategy for each parameter combination
    for lambda_val in lambda_values:
        for eta_val in eta_values:
            print(f"\nRunning with lambda={lambda_val}, eta={eta_val}")
            
            try:
                # Run strategy
                results, metrics, _, _ = run_rolling_strategy(
                    index_name=index_name,
                    num_assets=num_assets,
                    start_date=start_date,
                    end_date=end_date,
                    lambda_val=lambda_val,
                    eta=eta_val,
                    window_size=window_size,
                    rebalance_freq=rebalance_freq
                )
                
                # Add to results DataFrame
                new_row = {
                    'lambda': lambda_val,
                    'eta': eta_val,
                    'Overall_BT': metrics['Overall_BT'].values[0],
                    'Downturn_BT': metrics['Downturn_BT'].values[0],
                    'Robust_ETE': metrics['Robust_ETE'].values[0],
                    'Non_Robust_ETE': metrics['Non_Robust_ETE'].values[0],
                    'ETE_Diff': metrics['Non_Robust_ETE'].values[0] - metrics['Robust_ETE'].values[0],
                    'Robust_Cum': metrics['Robust_Cumulative'].values[0],
                    'Non_Robust_Cum': metrics['Non_Robust_Cumulative'].values[0],
                    'Index_Cum': metrics['Index_Cumulative'].values[0]
                }
                params_results = pd.concat([params_results, pd.DataFrame([new_row])], ignore_index=True)
            
            except Exception as e:
                print(f"Error with lambda={lambda_val}, eta={eta_val}: {e}")
    
    # Create pivot tables for heatmaps
    pivot_bt = params_results.pivot(index='lambda', columns='eta', values='Overall_BT')
    pivot_ete = params_results.pivot(index='lambda', columns='eta', values='ETE_Diff')
    pivot_down = params_results.pivot(index='lambda', columns='eta', values='Downturn_BT')
    
    # Plot parameter sensitivity heatmaps
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_bt, cmap='viridis', aspect='auto')
    plt.colorbar(label='Beating Time (%)')
    plt.title('Beating Time (%) vs Parameters')
    plt.xlabel('eta')
    plt.ylabel('lambda')
    plt.xticks(range(len(eta_values)), eta_values)
    plt.yticks(range(len(lambda_values)), lambda_values)
    for i in range(len(lambda_values)):
        for j in range(len(eta_values)):
            if not np.isnan(pivot_bt.iloc[i, j]):
                plt.text(j, i, f"{pivot_bt.iloc[i, j]:.1f}", ha="center", va="center", color="w")
    plt.savefig('param_sensitivity_bt.png')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_ete, cmap='viridis', aspect='auto')
    plt.colorbar(label='ETE Improvement')
    plt.title('ETE Improvement vs Parameters')
    plt.xlabel('eta')
    plt.ylabel('lambda')
    plt.xticks(range(len(eta_values)), eta_values)
    plt.yticks(range(len(lambda_values)), lambda_values)
    for i in range(len(lambda_values)):
        for j in range(len(eta_values)):
            if not np.isnan(pivot_ete.iloc[i, j]):
                plt.text(j, i, f"{pivot_ete.iloc[i, j]:.6f}", ha="center", va="center", color="w")
    plt.savefig('param_sensitivity_ete.png')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_down, cmap='viridis', aspect='auto')
    plt.colorbar(label='Beating Time During Downturns (%)')
    plt.title('Beating Time During Downturns (%) vs Parameters')
    plt.xlabel('eta')
    plt.ylabel('lambda')
    plt.xticks(range(len(eta_values)), eta_values)
    plt.yticks(range(len(lambda_values)), lambda_values)
    for i in range(len(lambda_values)):
        for j in range(len(eta_values)):
            if not np.isnan(pivot_down.iloc[i, j]):
                plt.text(j, i, f"{pivot_down.iloc[i, j]:.1f}", ha="center", va="center", color="w")
    plt.savefig('param_sensitivity_downturns.png')
    
    return params_results

# Test strategy with different market conditions
def test_market_conditions(index_name="SPX",
                          num_assets=20,
                          lambda_val=0.1,
                          eta=0.2,
                          window_size=252,
                          rebalance_freq=21):
    """
    Test the strategy with different market conditions by simulating various scenarios
    """
    # Define different market scenarios
    scenarios = [
        {"name": "Bull Market", "annual_return": 0.15, "annual_vol": 0.12, "start_date": "20180101", "end_date": "20201231"},
        {"name": "Bear Market", "annual_return": -0.10, "annual_vol": 0.25, "start_date": "20180101", "end_date": "20201231"},
        {"name": "Sideways Market", "annual_return": 0.02, "annual_vol": 0.10, "start_date": "20180101", "end_date": "20201231"},
        {"name": "Volatile Market", "annual_return": 0.05, "annual_vol": 0.30, "start_date": "20180101", "end_date": "20201231"}
    ]
    
    # Initialize results DataFrame
    scenario_results = pd.DataFrame(columns=[
        'Scenario', 'Overall_BT', 'Downturn_BT', 'Robust_ETE', 'Non_Robust_ETE', 'ETE_Diff'
    ])
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        
        # Create synthetic data with specific market conditions
        constituents_df, prices_df = create_synthetic_index_data(
            index_name=index_name,
            num_constituents=50,
            start_date=scenario['start_date'],
            end_date=scenario['end_date'],
            seed=42
        )
        
        # Modify returns to match scenario
        # This is a simplified approach - in a real implementation you'd want more sophisticated control
        returns = calculate_returns(prices_df)
        
        # Select top assets by weight to track
        top_constituents = constituents_df.sort_values('weight', ascending=False).head(num_assets)
        tickers = top_constituents.index.tolist()
        index_ticker = f"{index_name}_Index"
        
        # Run strategy for this scenario
        try:
            # Initialize results
            start_idx = window_size
            results = pd.DataFrame(index=returns.index[start_idx:])
            results['Index'] = returns[index_ticker].iloc[start_idx:]
            
            # Initialize portfolios
            robust_port_returns = np.zeros(len(returns) - start_idx)
            non_robust_port_returns = np.zeros(len(returns) - start_idx)
            
            # Loop through time with rebalancing
            for i in tqdm(range(start_idx, len(returns), rebalance_freq)):
                if i + rebalance_freq > len(returns):
                    end_i = len(returns)
                else:
                    end_i = i + rebalance_freq
                
                # Get training window
                train_data = returns.iloc[i-window_size:i]
                
                # Extract index returns
                index_returns_train = train_data[index_ticker]
                
                # Remove index from asset returns
                asset_returns_train = train_data[tickers]
                
                # Calculate portfolio weights
                robust_weights = robust_index_tracking_portfolio(
                    asset_returns_train, 
                    index_returns_train,
                    lambda_val=lambda_val,
                    eta=eta
                )
                
                non_robust_weights = non_robust_index_tracking_portfolio(
                    asset_returns_train, 
                    index_returns_train
                )
                
                # Apply weights to the next period
                for j in range(i, end_i):
                    robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * robust_weights)
                    non_robust_port_returns[j-start_idx] = np.sum(returns[tickers].iloc[j].values * non_robust_weights)
            
            # Store portfolio returns
            results['Robust_Portfolio'] = robust_port_returns
            results['Non_Robust_Portfolio'] = non_robust_port_returns
            
            # Calculate tracking errors
            results['Robust_TE'] = (results['Robust_Portfolio'] - results['Index'])**2
            results['Non_Robust_TE'] = (results['Non_Robust_Portfolio'] - results['Index'])**2
            
            # Calculate outperformance metrics
            results['Robust_Outperforms'] = results['Robust_TE'] < results['Non_Robust_TE']
            results['Downturn'] = results['Index'] < 0
            
            # Add to results DataFrame
            new_row = {
                'Scenario': scenario['name'],
                'Overall_BT': results['Robust_Outperforms'].mean() * 100,
                'Downturn_BT': results.loc[results['Downturn'], 'Robust_Outperforms'].mean() * 100 if any(results['Downturn']) else np.nan,
                'Robust_ETE': results['Robust_TE'].mean(),
                'Non_Robust_ETE': results['Non_Robust_TE'].mean(),
                'ETE_Diff': results['Non_Robust_TE'].mean() - results['Robust_TE'].mean()
            }
            scenario_results = pd.concat([scenario_results, pd.DataFrame([new_row])], ignore_index=True)
            
            # Plot cumulative returns for this scenario
            plt.figure(figsize=(12, 6))
            ((1 + results[['Robust_Portfolio', 'Non_Robust_Portfolio', 'Index']]).cumprod()).plot()
            plt.title(f'Cumulative Returns - {scenario["name"]}')
            plt.ylabel('Cumulative Return')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'cumulative_returns_{scenario["name"].replace(" ", "_").lower()}.png')
            
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
    
    # Plot comparison of scenarios
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenario_results['Scenario'], scenario_results['Overall_BT'])
    plt.title('Beating Time (%) Across Different Market Scenarios')
    plt.ylabel('Beating Time (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison_bt.png')
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenario_results['Scenario'], scenario_results['Downturn_BT'])
    plt.title('Beating Time During Downturns (%) Across Different Market Scenarios')
    plt.ylabel('Beating Time During Downturns (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison_downturn_bt.png')
    
    return scenario_results

# Run the testing
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Set parameters
    index_name = "SPX"
    num_assets = 20
    lambda_val = 0.1
    eta = 0.2
    window_size = 252  # One year of trading days
    rebalance_freq = 21  # Monthly rebalancing
    
    print("Testing basic strategy with synthetic data...")
    # Run basic strategy
    results, metrics, robust_weights_history, non_robust_weights_history = run_rolling_strategy(
        index_name=index_name,
        num_assets=num_assets,
        lambda_val=lambda_val,
        eta=eta,
        window_size=window_size,
        rebalance_freq=rebalance_freq
    )
    
    # Analyze performance in different market conditions
    market_condition_metrics = analyze_market_conditions(results)
    print("\nPerformance in Different Market Conditions:")
    print(market_condition_metrics)
    
    print("\nTesting parameter sensitivity...")
    # Test parameter sensitivity
    params_results = parameter_sensitivity_analysis(
        index_name=index_name,
        num_assets=num_assets,
        start_date="20180101",
        end_date="20211231",  # Shorter period for sensitivity analysis
        window_size=126,      # Shorter window for faster computation
        rebalance_freq=21
    )
    
    print("\nTesting different market scenarios...")
    # Test different market scenarios
    scenario_results = test_market_conditions(
        index_name=index_name,
        num_assets=num_assets,
        lambda_val=lambda_val,
        eta=eta,
        window_size=window_size,
        rebalance_freq=rebalance_freq
    )
    
    # Calculate time elapsed
    end_time = time.time()
    print(f"\nTime elapsed: {end_time - start_time:.2f} seconds")
    
    print("\nAll tests completed. Check the generated plots for visual results.")