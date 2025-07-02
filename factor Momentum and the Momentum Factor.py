import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_stocks = 1000          # Number of stocks
n_factors = 10           # Number of factors
n_months = 240           # Number of months (20 years)
factor_persistence = 0.2  # Autocorrelation of factor returns
idiosyncratic_vol = 0.08  # Monthly idiosyncratic volatility
factor_vol = 0.04         # Monthly factor volatility

# Create date range
start_date = datetime(2003, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(n_months)]
date_index = pd.DatetimeIndex(dates)

# Function to generate autocorrelated factor returns
def generate_factor_returns(n_factors, n_months, persistence, vol):
    """
    Generate factor returns with autocorrelation
    
    Parameters:
    - n_factors: Number of factors
    - n_months: Number of months
    - persistence: Autocorrelation parameter
    - vol: Monthly volatility of factor returns
    
    Returns:
    - DataFrame of factor returns
    """
    # Initialize factor returns matrix
    factor_returns = np.zeros((n_months, n_factors))
    
    # Generate factor returns with autocorrelation
    for f in range(n_factors):
        # Randomly assign a mean factor return (risk premium)
        factor_mean = np.random.normal(0.002, 0.002)
        
        # Generate the first month
        factor_returns[0, f] = factor_mean + np.random.normal(0, vol)
        
        # Generate subsequent months with autocorrelation
        for t in range(1, n_months):
            factor_returns[t, f] = factor_mean + persistence * (factor_returns[t-1, f] - factor_mean) + np.random.normal(0, vol)
    
    # Convert to DataFrame
    factor_df = pd.DataFrame(factor_returns, index=date_index, 
                            columns=[f'Factor_{i+1}' for i in range(n_factors)])
    
    return factor_df

# Generate factor returns
factor_returns = generate_factor_returns(n_factors, n_months, factor_persistence, factor_vol)

# Generate stock betas (factor loadings)
betas = np.random.normal(0, 1, size=(n_stocks, n_factors))

# Function to generate stock returns based on factors and betas
def generate_stock_returns(factor_returns, betas, idiosyncratic_vol):
    """
    Generate stock returns based on factors and betas
    
    Parameters:
    - factor_returns: DataFrame of factor returns
    - betas: Array of factor loadings (n_stocks × n_factors)
    - idiosyncratic_vol: Monthly idiosyncratic volatility
    
    Returns:
    - DataFrame of stock returns
    """
    n_months = len(factor_returns)
    n_stocks = betas.shape[0]
    
    # Initialize stock returns matrix
    stock_returns = np.zeros((n_months, n_stocks))
    
    # Generate stock returns
    for t in range(n_months):
        # Systematic component based on factors
        systematic_returns = np.dot(betas, factor_returns.iloc[t].values)
        
        # Add idiosyncratic component
        idiosyncratic_returns = np.random.normal(0, idiosyncratic_vol, n_stocks)
        
        # Total returns
        stock_returns[t, :] = systematic_returns + idiosyncratic_returns
    
    # Convert to DataFrame
    stock_df = pd.DataFrame(stock_returns, index=factor_returns.index, 
                           columns=[f'Stock_{i+1}' for i in range(n_stocks)])
    
    return stock_df

# Generate stock returns
stock_returns = generate_stock_returns(factor_returns, betas, idiosyncratic_vol)

# Function to implement factor momentum strategy
def factor_momentum_strategy(factor_returns, lookback=12, holding_period=1):
    """
    Implement factor momentum strategy
    
    Parameters:
    - factor_returns: DataFrame of factor returns
    - lookback: Lookback period in months
    - holding_period: Holding period in months
    
    Returns:
    - DataFrame with strategy returns
    """
    n_months = len(factor_returns)
    n_factors = factor_returns.shape[1]
    
    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=factor_returns.index[lookback:], dtype=float)
    positions = pd.DataFrame(index=factor_returns.index[lookback:], 
                            columns=factor_returns.columns, dtype=float)
    
    # For each month
    for t in range(lookback, n_months):
        # Calculate past returns
        past_returns = factor_returns.iloc[t-lookback:t].mean()
        
        # Take positions based on sign of past returns
        factor_positions = np.sign(past_returns)
        positions.iloc[t-lookback] = factor_positions
        
        # Calculate portfolio return for the holding period
        if t + holding_period <= n_months:
            future_returns = factor_returns.iloc[t:t+holding_period].mean()
            portfolio_returns.iloc[t-lookback] = (factor_positions * future_returns).mean()
    
    return portfolio_returns, positions

# Function to implement individual stock momentum strategy
def stock_momentum_strategy(stock_returns, lookback=12, holding_period=1, percentile=10):
    """
    Implement individual stock momentum strategy
    
    Parameters:
    - stock_returns: DataFrame of stock returns
    - lookback: Lookback period in months
    - holding_period: Holding period in months
    - percentile: Percentile for selecting winners and losers
    
    Returns:
    - Series with strategy returns
    """
    n_months = len(stock_returns)
    n_stocks = stock_returns.shape[1]
    
    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=stock_returns.index[lookback:], dtype=float)
    
    # For each month
    for t in range(lookback, n_months):
        # Calculate past returns
        past_returns = stock_returns.iloc[t-lookback:t].mean()
        
        # Select winners and losers
        winner_threshold = np.percentile(past_returns, 100 - percentile)
        loser_threshold = np.percentile(past_returns, percentile)
        
        winners = past_returns >= winner_threshold
        losers = past_returns <= loser_threshold
        
        # Calculate portfolio return for the holding period
        if t + holding_period <= n_months:
            future_returns = stock_returns.iloc[t:t+holding_period].mean()
            
            # Long winners, short losers
            portfolio_returns.iloc[t-lookback] = future_returns[winners].mean() - future_returns[losers].mean()
    
    return portfolio_returns

# Function to extract principal components (PCs) from factors
def extract_pc_factors(factor_returns, n_components=None):
    """
    Extract principal components from factor returns
    
    Parameters:
    - factor_returns: DataFrame of factor returns
    - n_components: Number of components to extract (default is all)
    
    Returns:
    - DataFrame of PC factor returns
    """
    # Standardize factor returns
    standardized_returns = (factor_returns - factor_returns.mean()) / factor_returns.std()
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(standardized_returns)
    
    # Convert to DataFrame
    pc_df = pd.DataFrame(pc_scores, index=factor_returns.index,
                         columns=[f'PC_{i+1}' for i in range(pc_scores.shape[1])])
    
    return pc_df, pca

# Extract PC factors
pc_factors, pca = extract_pc_factors(factor_returns)

# Print explained variance by each PC
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each PC:")
for i, var in enumerate(explained_variance):
    print(f"PC {i+1}: {var:.4f} ({var*100:.2f}%)")

# Implement factor momentum strategy on original factors
factor_mom_returns, factor_positions = factor_momentum_strategy(factor_returns)

# Implement factor momentum strategy on high-eigenvalue PC factors (top 5)
high_pc_factors = pc_factors.iloc[:, :5]  # Top 5 PCs
high_pc_mom_returns, high_pc_positions = factor_momentum_strategy(high_pc_factors)

# Implement individual stock momentum strategy
stock_mom_returns = stock_momentum_strategy(stock_returns)

# Make sure all return series have the same index
common_index = factor_mom_returns.index.intersection(high_pc_mom_returns.index).intersection(stock_mom_returns.index)
factor_mom_returns = factor_mom_returns[common_index]
high_pc_mom_returns = high_pc_mom_returns[common_index]
stock_mom_returns = stock_mom_returns[common_index]

# Calculate cumulative returns
factor_mom_cumulative = (1 + factor_mom_returns).cumprod()
high_pc_mom_cumulative = (1 + high_pc_mom_returns).cumprod()
stock_mom_cumulative = (1 + stock_mom_returns).cumprod()

# Calculate performance metrics
def calculate_performance(returns):
    """
    Calculate performance metrics for a strategy
    
    Parameters:
    - returns: Series of strategy returns
    
    Returns:
    - Dictionary of performance metrics
    """
    performance = {}
    performance['Mean Return'] = returns.mean() * 100  # Monthly return in %
    performance['Annualized Return'] = returns.mean() * 12 * 100  # Annualized return in %
    performance['Volatility'] = returns.std() * np.sqrt(12) * 100  # Annualized volatility in %
    performance['Sharpe Ratio'] = (returns.mean() * 12) / (returns.std() * np.sqrt(12))  # Annualized Sharpe
    
    # Calculate max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1) * 100
    performance['Max Drawdown'] = drawdown.min()  # Max drawdown in %
    
    performance['Win Rate'] = (returns > 0).mean() * 100  # Win rate in %
    
    return performance

factor_mom_performance = calculate_performance(factor_mom_returns)
high_pc_mom_performance = calculate_performance(high_pc_mom_returns)
stock_mom_performance = calculate_performance(stock_mom_returns)

# Print performance metrics
print("\nFactor Momentum Performance:")
for key, value in factor_mom_performance.items():
    print(f"{key}: {value:.2f}")

print("\nHigh-Eigenvalue PC Factor Momentum Performance:")
for key, value in high_pc_mom_performance.items():
    print(f"{key}: {value:.2f}")

print("\nStock Momentum Performance:")
for key, value in stock_mom_performance.items():
    print(f"{key}: {value:.2f}")

# Fixed regression function using sklearn
def run_regression(y, X):
    """
    Run regression of y on X using sklearn
    
    Parameters:
    - y: Dependent variable (Series)
    - X: Independent variable (Series)
    
    Returns:
    - Dictionary with regression results
    """
    # Convert X to DataFrame if it's a Series
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate statistics
    n = len(y)
    k = X.shape[1] if len(X.shape) > 1 else 1
    residuals = y - y_pred
    
    # R-squared
    r_squared = model.score(X, y)
    
    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # Standard error of regression
    se_regression = np.sqrt(np.sum(residuals**2) / (n - k - 1))
    
    # Standard errors of coefficients
    X_with_const = np.column_stack([np.ones(n), X]) if len(X.shape) == 1 else np.column_stack([np.ones(n), X])
    cov_matrix = se_regression**2 * np.linalg.inv(X_with_const.T @ X_with_const)
    se_coeffs = np.sqrt(np.diag(cov_matrix))
    
    # t-statistics
    alpha = model.intercept_
    beta = model.coef_[0] if len(model.coef_.shape) == 1 else model.coef_
    t_stat_alpha = alpha / se_coeffs[0]
    t_stat_beta = beta / se_coeffs[1] if len(se_coeffs) > 1 else beta / se_coeffs[0]
    
    # Prepare results
    results = {
        'alpha': alpha,
        'beta': beta,
        'se_alpha': se_coeffs[0],
        'se_beta': se_coeffs[1] if len(se_coeffs) > 1 else se_coeffs[0],
        't_stat_alpha': t_stat_alpha,
        't_stat_beta': t_stat_beta,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared
    }
    
    return results

# Run regression to see if factor momentum explains stock momentum
try:
    factor_mom_results = run_regression(stock_mom_returns, factor_mom_returns)
    high_pc_mom_results = run_regression(stock_mom_returns, high_pc_mom_returns)

    # Print regression results
    print("\nRegression of Stock Momentum on Factor Momentum:")
    print(f"Alpha: {factor_mom_results['alpha']:.4f} (t-stat: {factor_mom_results['t_stat_alpha']:.2f})")
    print(f"Beta: {factor_mom_results['beta']:.4f} (t-stat: {factor_mom_results['t_stat_beta']:.2f})")
    print(f"R-squared: {factor_mom_results['r_squared']:.4f}")

    print("\nRegression of Stock Momentum on High-Eigenvalue PC Factor Momentum:")
    print(f"Alpha: {high_pc_mom_results['alpha']:.4f} (t-stat: {high_pc_mom_results['t_stat_alpha']:.2f})")
    print(f"Beta: {high_pc_mom_results['beta']:.4f} (t-stat: {high_pc_mom_results['t_stat_beta']:.2f})")
    print(f"R-squared: {high_pc_mom_results['r_squared']:.4f}")
except Exception as e:
    print(f"Error in regression: {e}")

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(factor_mom_cumulative, label='Factor Momentum')
plt.plot(high_pc_mom_cumulative, label='High-Eigenvalue PC Factor Momentum')
plt.plot(stock_mom_cumulative, label='Stock Momentum')
plt.title('Cumulative Returns of Momentum Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot monthly returns - scatter plot with regression line
plt.figure(figsize=(12, 6))
plt.scatter(factor_mom_returns, stock_mom_returns, alpha=0.6)
plt.xlabel('Factor Momentum Returns')
plt.ylabel('Stock Momentum Returns')
plt.title('Monthly Returns: Stock Momentum vs. Factor Momentum')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add regression line using numpy polyfit
try:
    slope, intercept = np.polyfit(factor_mom_returns, stock_mom_returns, 1)
    x = np.linspace(factor_mom_returns.min(), factor_mom_returns.max(), 100)
    plt.plot(x, intercept + slope * x, 'r-', alpha=0.7, 
             label=f'y = {intercept:.4f} + {slope:.4f}x, R² = {factor_mom_results["r_squared"]:.4f}')
    plt.legend()
except Exception as e:
    print(f"Error plotting regression line: {e}")

plt.tight_layout()
plt.show()

# Create heatmap of factor positions over time
plt.figure(figsize=(12, 8))
sns.heatmap(factor_positions.T, cmap='RdBu_r', center=0, 
            xticklabels=factor_positions.index.strftime('%Y-%m')[::12],
            yticklabels=factor_positions.columns)
plt.title('Factor Momentum Positions Over Time')
plt.xlabel('Date')
plt.ylabel('Factor')
plt.tight_layout()
plt.show()

# Plot 12-month autocorrelation of factors
autocorr = pd.Series(index=factor_returns.columns)
for col in factor_returns.columns:
    # Calculate autocorrelation at 12-month lag
    returns = factor_returns[col].values
    if len(returns) > 12:
        returns_lagged = np.roll(returns, 12)[12:]  # Shift by 12 months and drop first 12 values
        returns_current = returns[12:]  # Drop first 12 values
        autocorr[col] = np.corrcoef(returns_current, returns_lagged)[0, 1]

plt.figure(figsize=(10, 6))
autocorr.plot(kind='bar')
plt.title('12-Month Autocorrelation of Factor Returns')
plt.ylabel('Autocorrelation')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate correlations between different momentum strategies
correlation_matrix = pd.DataFrame({
    'Factor Mom': factor_mom_returns,
    'High PC Mom': high_pc_mom_returns,
    'Stock Mom': stock_mom_returns
}).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Momentum Strategies')
plt.tight_layout()
plt.show()