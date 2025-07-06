import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
import yfinance as yf
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulated data parameters
n_assets = 10  # Number of assets
n_days = 750    # Number of trading days
trading_period = 5  # 5-day trading period as mentioned in the paper

# Simulate price data with correlations
def simulate_correlated_prices(n_assets, n_days, correlation_strength=0.3):
    # Generate a random correlation matrix
    corr_matrix = np.random.rand(n_assets, n_assets)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
    np.fill_diagonal(corr_matrix, 1)  # Diagonal elements are 1
    
    # Apply correlation strength
    corr_matrix = (corr_matrix - np.eye(n_assets)) * correlation_strength + np.eye(n_assets)
    
    # Compute Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, adjust it
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        L = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated daily returns
    daily_returns = np.random.normal(0.0005, 0.01, size=(n_days, n_assets))
    
    # Apply correlation structure
    correlated_returns = daily_returns @ L.T
    
    # Generate prices from returns
    prices = np.cumprod(1 + correlated_returns, axis=0) * 100
    
    return prices

# Generate simulated price data
prices = simulate_correlated_prices(n_assets, n_days)

# Create asset names
asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

# Create a DataFrame for prices
price_df = pd.DataFrame(prices, columns=asset_names)

# Calculate returns
returns_df = price_df.pct_change().dropna()

# Visualize the simulated prices
plt.figure(figsize=(12, 6))
for i in range(n_assets):
    plt.plot(price_df.iloc[:, i], label=asset_names[i])
plt.title('Simulated Asset Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# BL Model implementation
class BlackLittermanModel:
    def __init__(self, returns, risk_aversion=1.0, tau=1.0):
        self.returns = returns
        self.risk_aversion = risk_aversion
        self.tau = tau  # Uncertainty of prior
        self.n_assets = returns.shape[1]
        
    def calculate_covariance(self, window=50):
        """Calculate the historical covariance matrix."""
        return self.returns.iloc[-window:].cov() * 252  # Annualized
    
    def calculate_prior_returns(self, cov_matrix):
        """Calculate prior expected returns using inverse optimization."""
        ones = np.ones(self.n_assets)
        prior_returns = (self.risk_aversion * cov_matrix @ ones) / self.n_assets
        return prior_returns
    
    def incorporate_views(self, views, cov_matrix, confidences):
        """Incorporate subjective views into the prior expected returns."""
        # Views matrix P (in this simplified case, it's an identity matrix for absolute views)
        P = np.eye(self.n_assets)
        
        # Views covariance matrix Omega
        omega = np.diag([(cov_matrix.iloc[i, i] / conf) for i, conf in enumerate(confidences)])
        
        # Convert views to array
        Q = np.array(views)
        
        # Prior expected returns
        prior_returns = self.calculate_prior_returns(cov_matrix)
        
        # Calculate posterior expected returns
        term1 = np.linalg.inv(self.tau * cov_matrix)
        term2 = P.T @ np.linalg.inv(omega) @ P
        term3 = term1 @ prior_returns
        term4 = P.T @ np.linalg.inv(omega) @ Q
        
        posterior_returns = np.linalg.inv(term1 + term2) @ (term3 + term4)
        posterior_cov = cov_matrix + np.linalg.inv(term1 + term2)
        
        return posterior_returns, posterior_cov
    
    def optimal_portfolio(self, posterior_returns, posterior_cov):
        """Calculate optimal portfolio weights."""
        weights = (1 / self.risk_aversion) * np.linalg.inv(posterior_cov) @ posterior_returns
        return weights

# Simplified version of the Transformer network for the DRL agent
def create_view_model(input_shape, n_assets):
    """Create a simplified model to generate views for the BL model."""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    
    # Output layer for views (expected returns)
    views_output = Dense(n_assets, activation='tanh')(x)
    
    # Output layer for risk aversion
    risk_aversion = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=[views_output, risk_aversion])
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Prepare data for the model
def prepare_state_tensor(prices, window=50, trading_period=5):
    """Prepare state tensor from price data."""
    # Calculate returns
    returns = np.log(prices[1:] / prices[:-1])
    
    # Create state tensors
    n_periods = len(returns) // trading_period
    state_tensors = []
    
    for i in range(window, n_periods * trading_period - trading_period + 1, trading_period):
        # Get historical returns for the state
        historical_returns = returns[i-window:i].reshape(1, window, n_assets)
        state_tensors.append(historical_returns)
    
    return np.array(state_tensors)

# Backtest function
def backtest_bl_drl_strategy(price_data, view_model, trading_period=5, initial_capital=1000000, 
                            transaction_cost=0.0005, risk_free_rate=0, lending_rate=0.03):
    """Backtest the BL-DRL strategy."""
    n_assets = price_data.shape[1]
    n_days = price_data.shape[0]
    n_periods = n_days // trading_period
    
    # Initialize portfolio
    portfolio_value = [initial_capital]
    cash = initial_capital
    holdings = np.zeros(n_assets)
    weights_history = []
    
    # Track performance
    daily_returns = []
    
    # BL model
    returns = np.log(price_data.values[1:] / price_data.values[:-1])
    returns_df = pd.DataFrame(returns, columns=price_data.columns)
    bl_model = BlackLittermanModel(returns_df)
    
    # Prepare initial state
    window = 50
    
    for period in range(1, n_periods - 1):
        if period * trading_period < window:
            continue
            
        # Current prices
        current_prices = price_data.iloc[period * trading_period].values
        
        # Prepare state
        state = returns[max(0, period * trading_period - window):period * trading_period].reshape(1, min(window, period * trading_period), n_assets)
        
        # Get model predictions (views and risk aversion)
        views_pred, risk_aversion_pred = view_model.predict(state)
        
        # Scale views to realistic return expectations (-5% to +5% monthly)
        views = views_pred[0] * 0.05
        
        # Scale risk aversion (1 to 5)
        risk_aversion = 1 + risk_aversion_pred[0][0] * 4
        
        # Update BL model parameters
        bl_model.risk_aversion = float(risk_aversion)
        
        # Calculate covariance matrix
        cov_matrix = bl_model.calculate_covariance(window=min(window, period * trading_period))
        
        # Confidence in views (higher for less volatile assets)
        confidences = 1 / (1 + np.diag(cov_matrix) * 10)
        
        # Incorporate views
        posterior_returns, posterior_cov = bl_model.incorporate_views(views, cov_matrix, confidences)
        
        # Calculate optimal weights
        weights = bl_model.optimal_portfolio(posterior_returns, posterior_cov)
        
        # Normalize weights to sum to 1 (for long-only portfolio, for simplicity)
        weights = np.clip(weights, -1, 1)  # Allow short selling but limit leverage
        
        # Store weights
        weights_history.append(weights)
        
        # Rebalance portfolio
        portfolio_value_before = cash + np.sum(holdings * current_prices)
        
        # Calculate target positions
        target_value = weights * portfolio_value_before
        target_holdings = target_value / current_prices
        
        # Calculate trades
        trades = target_holdings - holdings
        
        # Apply transaction costs
        transaction_costs = np.sum(np.abs(trades) * current_prices) * transaction_cost
        cash -= transaction_costs
        
        # Update holdings and cash
        cash -= np.sum(trades * current_prices)
        holdings = target_holdings
        
        # Track portfolio value for the next trading period
        for day in range(1, trading_period + 1):
            if period * trading_period + day >= n_days:
                break
                
            next_prices = price_data.iloc[period * trading_period + day].values
            portfolio_value_after = cash + np.sum(holdings * next_prices)
            portfolio_value.append(portfolio_value_after)
            
            # Calculate daily return
            daily_return = portfolio_value_after / portfolio_value[-2] - 1
            daily_returns.append(daily_return)
    
    # Calculate performance metrics
    cumulative_return = portfolio_value[-1] / initial_capital - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
    
    return {
        'portfolio_value': portfolio_value,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'weights_history': weights_history,
        'daily_returns': daily_returns
    }

# Create and train the model
input_shape = (50, n_assets)  # 50 days of historical returns for 10 assets
view_model = create_view_model(input_shape, n_assets)

# Use initial trained weights rather than full training for this demonstration
# Normally we would train the model using the DRL approach described in the paper

# Prepare state tensor for the model
state_tensors = prepare_state_tensor(price_df.values, window=50, trading_period=trading_period)

# Generate random views for demonstration
sample_views = np.random.normal(0, 0.01, (len(state_tensors), n_assets))
sample_risk_aversion = np.random.uniform(0.5, 1.5, (len(state_tensors), 1))

# Train the model for just a few epochs for demonstration
view_model.fit(state_tensors, [sample_views, sample_risk_aversion], epochs=5, batch_size=16, verbose=1)

# Backtest the strategy
results = backtest_bl_drl_strategy(price_df, view_model, trading_period=trading_period)

# Plot results
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(results['portfolio_value'])
plt.title('Portfolio Value Over Time')
plt.xlabel('Days')
plt.ylabel('Value ($)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.cumsum(results['daily_returns']))
plt.title('Cumulative Returns')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print performance metrics
print(f"Cumulative Return: {results['cumulative_return']:.4f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")

# Plot asset weights over time
plt.figure(figsize=(14, 7))
weights_history = np.array(results['weights_history'])
for i in range(n_assets):
    plt.plot(weights_history[:, i], label=f'Asset {i+1}')
plt.title('Portfolio Weights Over Time')
plt.xlabel('Trading Periods')
plt.ylabel('Weight')
plt.legend()
plt.grid(True)
plt.show()

# Compare against a simple equal-weight strategy
def backtest_equal_weight(price_data, trading_period=5, initial_capital=1000000, transaction_cost=0.0005):
    """Backtest an equal-weight strategy."""
    n_assets = price_data.shape[1]
    n_days = price_data.shape[0]
    n_periods = n_days // trading_period
    
    # Initialize portfolio
    portfolio_value = [initial_capital]
    cash = initial_capital
    holdings = np.zeros(n_assets)
    
    # Track performance
    daily_returns = []
    
    for period in range(1, n_periods - 1):
        # Current prices
        current_prices = price_data.iloc[period * trading_period].values
        
        # Equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Rebalance portfolio
        portfolio_value_before = cash + np.sum(holdings * current_prices)
        
        # Calculate target positions
        target_value = weights * portfolio_value_before
        target_holdings = target_value / current_prices
        
        # Calculate trades
        trades = target_holdings - holdings
        
        # Apply transaction costs
        transaction_costs = np.sum(np.abs(trades) * current_prices) * transaction_cost
        cash -= transaction_costs
        
        # Update holdings and cash
        cash -= np.sum(trades * current_prices)
        holdings = target_holdings
        
        # Track portfolio value for the next trading period
        for day in range(1, trading_period + 1):
            if period * trading_period + day >= n_days:
                break
                
            next_prices = price_data.iloc[period * trading_period + day].values
            portfolio_value_after = cash + np.sum(holdings * next_prices)
            portfolio_value.append(portfolio_value_after)
            
            # Calculate daily return
            daily_return = portfolio_value_after / portfolio_value[-2] - 1
            daily_returns.append(daily_return)
    
    # Calculate performance metrics
    cumulative_return = portfolio_value[-1] / initial_capital - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
    
    return {
        'portfolio_value': portfolio_value,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'daily_returns': daily_returns
    }

# Backtest equal weight strategy
equal_weight_results = backtest_equal_weight(price_df, trading_period=trading_period)

# Compare strategies
plt.figure(figsize=(14, 7))
plt.plot(results['portfolio_value'], label='BL-DRL Strategy')
plt.plot(equal_weight_results['portfolio_value'], label='Equal-Weight Strategy')
plt.title('Strategy Comparison')
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# Print comparison metrics
print("\nStrategy Comparison:")
print(f"BL-DRL Strategy - Cumulative Return: {results['cumulative_return']:.4f}, Sharpe Ratio: {results['sharpe_ratio']:.4f}")
print(f"Equal-Weight Strategy - Cumulative Return: {equal_weight_results['cumulative_return']:.4f}, Sharpe Ratio: {equal_weight_results['sharpe_ratio']:.4f}")

