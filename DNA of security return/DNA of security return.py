import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

def generate_simulated_data(n_assets=30, n_days=526, n_factors=30, 
                           start_date='2010-01-01', seed=42):
    """
    Generate simulated stock data based on independent factors
    
    Parameters:
    -----------
    n_assets: int
        Number of assets to simulate
    n_days: int
        Number of trading days
    n_factors: int
        Number of independent factors
    start_date: str
        Starting date for the simulation
    seed: int
        Random seed
        
    Returns:
    --------
    prices: DataFrame
        DataFrame containing simulated price data
    returns: DataFrame
        DataFrame containing return data
    true_factors: DataFrame
        DataFrame containing the true independent factors
    mixing_matrix: ndarray
        The true mixing matrix used in simulation
    """
    np.random.seed(seed)
    
    # Create date index (excluding weekends)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # Generate independent source signals (factors)
    true_factors = np.zeros((n_factors, n_days-1))
    
    # Make the first factor a "market factor" with stronger mean and lower volatility
    true_factors[0] = np.random.normal(0.0005, 0.010, n_days-1)
    
    # Generate other factors with different characteristics
    for i in range(1, n_factors):
        # Random parameters for each factor
        mu = np.random.uniform(-0.0003, 0.0003)
        sigma = np.random.uniform(0.008, 0.015)
        
        # Non-Gaussian distribution (mixture of different distributions)
        if i % 3 == 0:
            # t-distribution (heavy tails)
            df = np.random.randint(3, 10)
            true_factors[i] = stats.t.rvs(df=df, loc=mu, scale=sigma, size=n_days-1)
        elif i % 3 == 1:
            # Normal distribution
            true_factors[i] = np.random.normal(mu, sigma, n_days-1)
        else:
            # Asymmetric distribution (skewed)
            true_factors[i] = np.random.gamma(shape=3, scale=sigma/3, size=n_days-1) - 0.01 + mu
    
    # Generate random mixing matrix
    mixing_matrix = np.random.uniform(-1, 1, size=(n_assets, n_factors))
    
    # Make first factor (market factor) influential for all stocks
    mixing_matrix[:, 0] = np.random.uniform(0.5, 1.5, n_assets)
    
    # Make some factors more important for specific groups of stocks
    # (simulating sector effects)
    sector_size = n_assets // 5
    for i in range(1, 6):
        start_idx = (i-1) * sector_size
        end_idx = i * sector_size if i < 5 else n_assets
        factor_idx = i % (n_factors-1) + 1
        mixing_matrix[start_idx:end_idx, factor_idx] *= 2
    
    # Generate returns as a mixture of independent factors
    asset_returns = np.dot(mixing_matrix, true_factors)
    
    # Initialize prices starting at random values
    initial_prices = np.random.uniform(20, 200, n_assets)
    prices = np.zeros((n_assets, n_days))
    prices[:, 0] = initial_prices
    
    # Generate price series from returns
    for t in range(1, n_days):
        prices[:, t] = prices[:, t-1] * (1 + asset_returns[:, t-1])
    
    # Convert to DataFrames
    prices_df = pd.DataFrame(prices.T, index=dates, 
                           columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    returns_df = pd.DataFrame(asset_returns.T, index=dates[1:], 
                            columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    factors_df = pd.DataFrame(true_factors.T, index=dates[1:], 
                            columns=[f'Factor_{i+1}' for i in range(n_factors)])
    
    return prices_df, returns_df, factors_df, mixing_matrix

def calculate_RHD(x, y):
    """
    Calculate Relative Hamming Distance between two time series
    
    Parameters:
    -----------
    x, y: array-like
        Time series to compare
    
    Returns:
    --------
    rhd: float
        Relative Hamming Distance
    """
    t = len(x) - 1
    rhd = 0
    
    for i in range(t):
        sign_x = np.sign(x[i+1] - x[i])
        sign_y = np.sign(y[i+1] - y[i])
        
        if sign_x == sign_y:
            d = 0
        elif sign_x == 0 or sign_y == 0:
            d = 1
        else:
            d = 4
        
        rhd += d
    
    return rhd / t

def TnA_algorithm(X, F, k, W_inv):
    """
    Testing-and-Acceptance algorithm for ordering independent components
    
    Parameters:
    -----------
    X: array-like
        Original time series (one stock)
    F: array-like
        Estimated independent components
    k: int
        Index of the stock
    W_inv: array-like
        Inverted mixing matrix (estimated)
    
    Returns:
    --------
    L: list
        Ordered list of factor indices for stock k
    """
    T = X.shape[0]
    N = F.shape[1]
    
    # Initialize ordering list
    L = []
    
    # Get all component indices
    Z = list(range(N))
    
    while Z:
        min_rhd = float('inf')
        best_i = None
        
        for i in Z:
            # Calculate reconstruction without component i
            v_i = np.zeros(T)
            for p in Z:
                if p != i:
                    v_i += W_inv[k, p] * F[:, p]
            
            # Calculate RHD
            rhd_i = calculate_RHD(X[:, k], v_i)
            
            if rhd_i < min_rhd:
                min_rhd = rhd_i
                best_i = i
        
        # Add the best component to the ordering list
        L.append(best_i)
        
        # Remove from remaining components
        Z.remove(best_i)
    
    # Inverse order (from most important to least important)
    L.reverse()
    
    return L

def calculate_R2_bands(returns, factors, mixing_matrix, factor_ordering):
    """
    Calculate R² bands for each stock
    
    Parameters:
    -----------
    returns: array-like
        Original returns
    factors: array-like
        Estimated independent factors
    mixing_matrix: array-like
        Estimated mixing matrix
    factor_ordering: list of lists
        TnA factor ordering for each stock
    
    Returns:
    --------
    R2_bands: dict
        Dictionary containing R² bands for each stock
    """
    n_assets = returns.shape[1]
    n_factors = factors.shape[1]
    R2_bands = {}
    
    R2_thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]
    
    for k in range(n_assets):
        R2_bands[k] = {}
        
        # Original R² with all factors
        reconstructed = np.dot(factors, mixing_matrix[k, :])
        original_r2 = 1 - np.sum((returns[:, k] - reconstructed) ** 2) / np.sum((returns[:, k] - np.mean(returns[:, k])) ** 2)
        
        # Sort factors by importance (least to most)
        ordering = factor_ordering[k]
        
        current_factors = list(range(n_factors))
        current_r2 = original_r2
        
        for threshold in R2_thresholds:
            R2_bands[k][threshold] = []
            
            # Remove factors one by one from the least important
            for factor_idx in ordering:
                if factor_idx not in current_factors:
                    continue
                    
                # Try removing this factor
                test_factors = current_factors.copy()
                test_factors.remove(factor_idx)
                
                # Calculate R² without this factor
                if test_factors:
                    selected_cols = np.array(test_factors)
                    reconstructed = np.dot(factors[:, selected_cols], mixing_matrix[k, selected_cols])
                    test_r2 = 1 - np.sum((returns[:, k] - reconstructed) ** 2) / np.sum((returns[:, k] - np.mean(returns[:, k])) ** 2)
                else:
                    test_r2 = 0
                
                # If R² is still above threshold, remove this factor
                if test_r2 >= threshold:
                    R2_bands[k][threshold].append(factor_idx)
                    current_factors = test_factors
                    current_r2 = test_r2
                else:
                    break
    
    return R2_bands

def identify_signal_factors(R2_bands, factor_ordering):
    """
    Identify signal factors for each stock based on R² bands
    
    Parameters:
    -----------
    R2_bands: dict
        Dictionary containing R² bands
    factor_ordering: list of lists
        TnA factor ordering for each stock
    
    Returns:
    --------
    signal_factors: dict
        Dictionary containing signal factors for each stock
    """
    n_assets = len(R2_bands)
    signal_factors = {}
    
    for k in range(n_assets):
        # Factors in the 80% R² band are noise factors
        noise_factors = set()
        for threshold in [0.99, 0.95, 0.90, 0.85, 0.80]:
            noise_factors.update(R2_bands[k][threshold])
        
        # All other factors are signal factors
        signal_factors[k] = [f for f in factor_ordering[k] if f not in noise_factors]
        
        # Reverse to have most important first
        signal_factors[k].reverse()
        
        # Ensure at least a few signal factors for each stock
        if len(signal_factors[k]) < 3:
            # Get the top 3 most important factors
            signal_factors[k] = factor_ordering[k][-3:]
            signal_factors[k].reverse()
    
    return signal_factors

def dna_trading_strategy(returns, factors, mixing_matrix, signal_factors, window_size=104, prediction_horizon=1):
    """
    Implement a trading strategy based on DNA of security returns
    
    Parameters:
    -----------
    returns: DataFrame
        Original return data
    factors: DataFrame
        Estimated independent factors
    mixing_matrix: ndarray
        Estimated mixing matrix
    signal_factors: dict
        Dictionary containing signal factors for each stock
    window_size: int
        Size of the rolling window for model estimation
    prediction_horizon: int
        Prediction horizon in days
    
    Returns:
    --------
    results: DataFrame
        Trading results
    """
    n_assets = returns.shape[1]
    asset_names = returns.columns
    T = len(returns)
    
    # Initialize results
    results = pd.DataFrame(index=returns.index[window_size:], 
                          columns=[f"{name}_prediction" for name in asset_names] + 
                                  [f"{name}_actual" for name in asset_names] + 
                                  [f"{name}_position" for name in asset_names] + 
                                  ["portfolio_return"])
    results = results.astype(float)  # Ensure numeric data type
    
    # For each time step
    for t in tqdm(range(window_size, T-prediction_horizon)):
        portfolio_return = 0
        positions_count = 0
        
        # For each asset
        for k in range(n_assets):
            asset_name = asset_names[k]
            
            # Get signal factors for this asset
            asset_signal_factors = signal_factors[k]
            
            # Get window of original returns for this asset
            window_returns = returns.iloc[t-window_size:t, k].values
            
            # Create noise-free returns using only signal factors
            noise_free_returns = np.zeros(window_size)
            for factor_idx in asset_signal_factors:
                factor_values = factors.iloc[t-window_size:t, factor_idx].values
                noise_free_returns += mixing_matrix[k, factor_idx] * factor_values
            
            # Add back the mean return to make the series more realistic
            noise_free_returns += np.mean(window_returns)
            
            # Fit ARIMA model to noise-free returns
            try:
                model = ARIMA(noise_free_returns, order=(1,0,1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=prediction_horizon)
                prediction = forecast[prediction_horizon-1]
            except:
                # Fallback if ARIMA fails
                prediction = np.mean(noise_free_returns[-5:])
            
            # Store prediction
            results.loc[returns.index[t], f"{asset_name}_prediction"] = prediction
            
            # Store actual return (ensure we don't go out of bounds)
            if t + prediction_horizon < T:
                actual_return = returns.iloc[t+prediction_horizon, k]
                results.loc[returns.index[t], f"{asset_name}_actual"] = actual_return
            else:
                actual_return = np.nan
                results.loc[returns.index[t], f"{asset_name}_actual"] = np.nan
            
            # Trading decision (1: buy, 0: do not buy)
            position = 1 if prediction > 0 else 0
            results.loc[returns.index[t], f"{asset_name}_position"] = position
            
            # Update portfolio return only if we have a valid actual return
            if not np.isnan(actual_return):
                portfolio_return += position * actual_return
                positions_count += position
        
        # Average portfolio return across selected assets
        if positions_count > 0:
            portfolio_return /= positions_count
        
        results.loc[returns.index[t], "portfolio_return"] = portfolio_return
    
    return results

def evaluate_strategy(results, returns):
    """
    Evaluate the performance of the trading strategy
    
    Parameters:
    -----------
    results: DataFrame
        Trading results
    returns: DataFrame
        Original return data
    
    Returns:
    --------
    performance: dict
        Dictionary containing performance metrics
    """
    # Remove NaN values from portfolio returns
    valid_returns = results["portfolio_return"].dropna()
    
    if len(valid_returns) == 0:
        print("Warning: No valid portfolio returns to evaluate")
        return None
    
    # Calculate portfolio cumulative return
    portfolio_cum_return = (1 + valid_returns).cumprod() - 1
    
    # Calculate benchmark cumulative return (equal weight)
    benchmark_returns = returns.loc[valid_returns.index].mean(axis=1)
    benchmark_cum_return = (1 + benchmark_returns).cumprod() - 1
    
    # Calculate buy-and-hold cumulative returns for each asset
    buy_hold_returns = {}
    for col in returns.columns:
        buy_hold_returns[col] = (1 + returns.loc[valid_returns.index, col]).cumprod() - 1
    
    # Calculate metrics
    total_return = portfolio_cum_return.iloc[-1]
    benchmark_return = benchmark_cum_return.iloc[-1]
    
    annualized_return = (1 + total_return) ** (252 / len(valid_returns)) - 1
    benchmark_ann_return = (1 + benchmark_return) ** (252 / len(valid_returns)) - 1
    
    volatility = valid_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    benchmark_sharpe = benchmark_ann_return / benchmark_vol if benchmark_vol > 0 else 0
    
    # Calculate max drawdown
    portfolio_value = (1 + valid_returns).cumprod()
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Direction accuracy
    direction_accuracy = {}
    for col in returns.columns:
        predictions = results[f"{col}_prediction"].dropna()
        actuals = results[f"{col}_actual"].dropna()
        
        # Make sure we only use common indices
        common_idx = predictions.index.intersection(actuals.index)
        if len(common_idx) > 0:
            predictions = predictions.loc[common_idx]
            actuals = actuals.loc[common_idx]
            
            correct = ((predictions > 0) & (actuals > 0)) | ((predictions <= 0) & (actuals <= 0))
            direction_accuracy[col] = correct.mean()
        else:
            direction_accuracy[col] = np.nan
    
    # Filter out NaN values for average
    valid_accuracies = [acc for acc in direction_accuracy.values() if not np.isnan(acc)]
    avg_direction_accuracy = np.mean(valid_accuracies) if valid_accuracies else np.nan
    
    # Results
    performance = {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "annualized_return": annualized_return,
        "benchmark_ann_return": benchmark_ann_return,
        "volatility": volatility,
        "benchmark_vol": benchmark_vol,
        "sharpe_ratio": sharpe_ratio,
        "benchmark_sharpe": benchmark_sharpe,
        "max_drawdown": max_drawdown,
        "direction_accuracy": direction_accuracy,
        "avg_direction_accuracy": avg_direction_accuracy,
        "portfolio_cum_return": portfolio_cum_return,
        "benchmark_cum_return": benchmark_cum_return,
        "buy_hold_returns": buy_hold_returns
    }
    
    return performance

def plot_results(performance):
    """
    Plot the results of the strategy
    
    Parameters:
    -----------
    performance: dict
        Dictionary containing performance metrics
    """
    if performance is None:
        print("No performance data to plot")
        return
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(performance["portfolio_cum_return"], label='DNA Trading Strategy', linewidth=2)
    plt.plot(performance["benchmark_cum_return"], label='Equal Weight Benchmark', linewidth=2)
    
    # Plot best and worst buy-and-hold
    best_asset = max(performance["buy_hold_returns"].items(), key=lambda x: x[1].iloc[-1])[0]
    worst_asset = min(performance["buy_hold_returns"].items(), key=lambda x: x[1].iloc[-1])[0]
    
    plt.plot(performance["buy_hold_returns"][best_asset], label=f'Best Asset ({best_asset})', linestyle='--')
    plt.plot(performance["buy_hold_returns"][worst_asset], label=f'Worst Asset ({worst_asset})', linestyle='--')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png')
    plt.close()
    
    # Plot direction accuracy
    plt.figure(figsize=(12, 6))
    assets = list(performance["direction_accuracy"].keys())
    accuracies = [performance["direction_accuracy"][asset] for asset in assets]
    
    plt.bar(assets, accuracies)
    plt.axhline(y=0.5, color='r', linestyle='-', label='Random')
    
    if not np.isnan(performance["avg_direction_accuracy"]):
        plt.axhline(y=performance["avg_direction_accuracy"], color='g', linestyle='--', 
                  label=f'Average: {performance["avg_direction_accuracy"]:.2f}')
    
    plt.title('Direction Prediction Accuracy by Asset')
    plt.xlabel('Asset')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('direction_accuracy.png')
    plt.close()
    
    # Performance summary table
    metrics = {
        "Metric": ["Total Return", "Annualized Return", "Volatility", "Sharpe Ratio", "Max Drawdown", "Avg Direction Accuracy"],
        "DNA Strategy": [
            f"{performance['total_return']:.2%}",
            f"{performance['annualized_return']:.2%}",
            f"{performance['volatility']:.2%}",
            f"{performance['sharpe_ratio']:.2f}",
            f"{performance['max_drawdown']:.2%}",
            f"{performance['avg_direction_accuracy']:.2%}" if not np.isnan(performance['avg_direction_accuracy']) else "N/A"
        ],
        "Benchmark": [
            f"{performance['benchmark_return']:.2%}",
            f"{performance['benchmark_ann_return']:.2%}",
            f"{performance['benchmark_vol']:.2%}",
            f"{performance['benchmark_sharpe']:.2f}",
            "N/A",
            "N/A"
        ]
    }
    
    print("\nPerformance Summary:")
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))

# Main execution
if __name__ == "__main__":
    print("Generating simulated stock data based on independent factors...")
    prices, returns, true_factors, true_mixing = generate_simulated_data(
        n_assets=30, n_days=526, n_factors=30)
    
    # Simulate a full dataset with price and return data similar to the paper
    returns_np = returns.values
    factors_np = true_factors.values
    
    print("Performing Independent Component Analysis (ICA)...")
    # FastICA for estimating independent components
    ica = FastICA(n_components=30, random_state=42)
    S = ica.fit_transform(returns_np)  # Estimated source signals
    A = ica.mixing_  # Estimated mixing matrix
    W = ica.components_  # Estimated unmixing matrix
    
    # In practice, we would use the estimated mixing matrix
    # For this simulation, we'll use the true mixing matrix to better illustrate the concept
    W_inv_estimated = true_mixing  # For simplicity, using true mixing matrix
    
    print("Ordering independent factors using TnA algorithm...")
    # Apply TnA algorithm to get factor ordering for each stock
    factor_ordering = []
    for k in range(returns_np.shape[1]):
        ordering = TnA_algorithm(returns_np, factors_np, k, W_inv_estimated)
        factor_ordering.append(ordering)
    
    print("Calculating R² bands to identify signal factors...")
    # Calculate R² bands
    R2_bands = calculate_R2_bands(returns_np, factors_np, true_mixing, factor_ordering)
    
    # Identify signal factors
    signal_factors = identify_signal_factors(R2_bands, factor_ordering)
    
    print("Implementing DNA trading strategy...")
    # Implement trading strategy
    window_size = 104  # About 5 months of trading days
    results = dna_trading_strategy(returns, true_factors, true_mixing, signal_factors, window_size=window_size)
    
    print("Evaluating strategy performance...")
    # Evaluate strategy
    performance = evaluate_strategy(results, returns)
    
    # Plot results
    plot_results(performance)
    
    # Summary of signal factors
    print("\nNumber of signal factors per asset:")
    for k in range(len(signal_factors)):
        print(f"Asset_{k+1}: {len(signal_factors[k])} signal factors - {signal_factors[k][:5]}...")
    
    # Count how many times each factor appears as a signal factor
    factor_counts = {}
    for k in range(len(signal_factors)):
        for f in signal_factors[k]:
            if f in factor_counts:
                factor_counts[f] += 1
            else:
                factor_counts[f] = 1
    
    print("\nMost common signal factors across assets:")
    for f, count in sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Factor_{f+1}: appears in {count} assets")
    
    # Calculate average number of signal factors per asset
    avg_signal_factors = np.mean([len(signal_factors[k]) for k in range(len(signal_factors))])
    print(f"\nAverage number of signal factors per asset: {avg_signal_factors:.2f}")
    
    # Identify systematic vs idiosyncratic factors
    systematic_threshold = len(signal_factors) * 0.7  # If a factor is a signal for >70% of assets
    systematic_factors = [f for f, count in factor_counts.items() if count >= systematic_threshold]
    
    print("\nSystematic factors (common across most assets):")
    for f in systematic_factors:
        print(f"Factor_{f+1}: appears in {factor_counts[f]} assets")