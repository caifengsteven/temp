import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import datetime as dt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to simulate financial market data
def simulate_financial_data(n_assets=50, n_factors=5, n_characteristics=20, n_days=1000):
    """
    Simulate financial data based on a multi-factor model with time-varying loadings
    
    Args:
        n_assets: Number of assets
        n_factors: Number of latent factors
        n_characteristics: Number of asset characteristics
        n_days: Number of trading days
        
    Returns:
        returns: Asset returns
        factors: Latent factors
        characteristics: Asset characteristics
    """
    # Simulate latent factors (with time dependence)
    factors = np.zeros((n_days, n_factors))
    factors[0] = np.random.normal(0, 1, n_factors)
    
    # Add autoregressive structure to factors
    for t in range(1, n_days):
        factors[t] = 0.8 * factors[t-1] + 0.2 * np.random.normal(0, 1, n_factors)
    
    # Create time-varying characteristics for each asset
    characteristics = np.zeros((n_days, n_assets, n_characteristics))
    
    # Initialize characteristics with random values
    characteristics[0] = np.random.normal(0, 1, (n_assets, n_characteristics))
    
    # Add time-varying component to characteristics
    for t in range(1, n_days):
        characteristics[t] = 0.9 * characteristics[t-1] + 0.1 * np.random.normal(0, 1, (n_assets, n_characteristics))
    
    # Create factor loadings based on characteristics
    beta = np.random.normal(0, 1, (n_characteristics, n_factors))
    
    # Initialize returns array
    returns = np.zeros((n_days, n_assets))
    
    # Generate returns using factor model with non-linear components
    for t in range(n_days):
        # Linear factor component
        linear_component = np.dot(characteristics[t], beta) @ factors[t].T
        
        # Add non-linear component (interactions between factors)
        nonlinear_component = np.zeros(n_assets)
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                interaction = factors[t, i] * factors[t, j]
                nonlinear_component += interaction * np.random.normal(0, 0.2, n_assets)
        
        # Add asset-specific volatility that varies by size
        size_factor = np.exp(-characteristics[t, :, 0])  # Using first characteristic as proxy for size
        idiosyncratic_risk = np.random.normal(0, 0.02, n_assets) * size_factor
        
        # Combine components with noise
        returns[t] = linear_component + 0.3 * nonlinear_component + idiosyncratic_risk
    
    # Create time index
    start_date = dt.datetime(2018, 1, 1)
    dates = [start_date + dt.timedelta(days=i) for i in range(n_days)]
    
    # Convert to pandas DataFrames
    returns_df = pd.DataFrame(returns, index=dates)
    factors_df = pd.DataFrame(factors, index=dates, columns=[f'Factor_{i+1}' for i in range(n_factors)])
    
    # Reshape characteristics for easier analysis
    chars_reshaped = []
    for t in range(n_days):
        chars_t = pd.DataFrame(characteristics[t], 
                             columns=[f'Char_{i+1}' for i in range(n_characteristics)])
        chars_t['Date'] = dates[t]
        chars_t['Asset'] = range(n_assets)
        chars_reshaped.append(chars_t)
    
    characteristics_df = pd.concat(chars_reshaped)
    
    return returns_df, factors_df, characteristics_df

# Generate simulated data
returns, factors, characteristics = simulate_financial_data(n_assets=50, n_factors=5, 
                                                          n_characteristics=20, n_days=1000)

# Display basic statistics of the data
print("Returns summary statistics:")
print(returns.describe())

print("\nFactors summary statistics:")
print(factors.describe())

# Prepare the dataset for machine learning
# We'll predict next-day returns based on current characteristics and past returns

def prepare_ml_dataset(returns, characteristics, lookback=5):
    """
    Prepare dataset for machine learning by combining returns and characteristics
    
    Args:
        returns: DataFrame of asset returns
        characteristics: DataFrame of asset characteristics
        lookback: Number of days of historical returns to include
        
    Returns:
        X: Features for ML models
        y: Target variable (next day returns)
    """
    # Get unique dates and assets
    dates = returns.index.unique()
    assets = range(returns.shape[1])
    
    # Create empty lists to store features and targets
    X_data = []
    y_data = []
    
    # Loop through dates (excluding the first 'lookback' days and the last day)
    for t in range(lookback, len(dates)-1):
        current_date = dates[t]
        next_date = dates[t+1]
        
        # Get characteristics for current date
        current_chars = characteristics[characteristics['Date'] == current_date]
        
        # Loop through each asset
        for asset in assets:
            # Get historical returns for this asset
            hist_returns = returns.iloc[t-lookback:t, asset].values
            
            # Get characteristics for this asset
            asset_chars = current_chars[current_chars['Asset'] == asset].drop(['Date', 'Asset'], axis=1).values.flatten()
            
            # Combine features
            features = np.concatenate([hist_returns, asset_chars])
            X_data.append(features)
            
            # Target is the next day's return
            y_data.append(returns.iloc[t+1, asset])
    
    return np.array(X_data), np.array(y_data)

# Prepare dataset with a 5-day lookback period
X, y = prepare_ml_dataset(returns, characteristics, lookback=5)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train and evaluate different models
# 1. Linear Regression (Traditional approach)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

# 2. Random Forest (Tree-based ML approach)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

# 3. Gradient Boosting (Advanced ML approach)
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)

# 4. Neural Network (Deep Learning approach)
# Reshape data for LSTM
lookback = 5
n_features = X_train_scaled.shape[1] - lookback  # Excluding historical returns

# Create separate arrays for historical returns and other features
X_train_hist = X_train_scaled[:, :lookback]
X_train_other = X_train_scaled[:, lookback:]
X_test_hist = X_test_scaled[:, :lookback]
X_test_other = X_test_scaled[:, lookback:]

# Reshape historical returns for LSTM
X_train_hist_reshaped = X_train_hist.reshape(X_train_hist.shape[0], lookback, 1)
X_test_hist_reshaped = X_test_hist.reshape(X_test_hist.shape[0], lookback, 1)

# Build a hybrid model with LSTM for historical returns and Dense for other features
# Input 1: Historical returns
input_hist = tf.keras.layers.Input(shape=(lookback, 1))
lstm_out = tf.keras.layers.LSTM(16, return_sequences=False)(input_hist)
lstm_out = tf.keras.layers.Dropout(0.2)(lstm_out)

# Input 2: Other features
input_other = tf.keras.layers.Input(shape=(n_features,))
dense_out = tf.keras.layers.Dense(32, activation='relu')(input_other)
dense_out = tf.keras.layers.Dropout(0.2)(dense_out)

# Combine both inputs
combined = tf.keras.layers.Concatenate()([lstm_out, dense_out])
combined = tf.keras.layers.Dense(16, activation='relu')(combined)
output = tf.keras.layers.Dense(1)(combined)

# Create and compile model
nn_model = tf.keras.models.Model(inputs=[input_hist, input_other], outputs=output)
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
nn_history = nn_model.fit(
    [X_train_hist_reshaped, X_train_other], 
    y_train, 
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Evaluate model
nn_pred = nn_model.predict([X_test_hist_reshaped, X_test_other]).flatten()
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_mae = mean_absolute_error(y_test, nn_pred)

# Print model performance
print("\nModel Performance (Test Set):")
print(f"Linear Regression - RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}")
print(f"Random Forest - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
print(f"Gradient Boosting - RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}")
print(f"Neural Network - RMSE: {nn_rmse:.4f}, MAE: {nn_mae:.4f}")

# Plot training history for neural network
plt.figure(figsize=(10, 6))
plt.plot(nn_history.history['loss'], label='Training Loss')
plt.plot(nn_history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Portfolio Construction and Backtesting
# Let's implement a simple long-short portfolio strategy based on predicted returns

def backtest_strategy(model, X_test_scaled, y_test, test_dates, n_assets=50, top_pct=0.2):
    """
    Backtest a simple long-short portfolio strategy based on model predictions
    
    Args:
        model: Trained ML model
        X_test_scaled: Scaled features for test set
        y_test: Actual returns for test set
        test_dates: Dates corresponding to test set
        n_assets: Number of assets
        top_pct: Percentage of assets to go long/short
        
    Returns:
        portfolio_returns: Daily portfolio returns
        cumulative_returns: Cumulative portfolio returns
    """
    # For neural network, we need to handle the input differently
    if isinstance(model, tf.keras.models.Model):
        # Reshape data for LSTM
        X_test_hist = X_test_scaled[:, :lookback]
        X_test_other = X_test_scaled[:, lookback:]
        X_test_hist_reshaped = X_test_hist.reshape(X_test_hist.shape[0], lookback, 1)
        predictions = model.predict([X_test_hist_reshaped, X_test_other]).flatten()
    else:
        # For other models
        predictions = model.predict(X_test_scaled)
    
    # Determine the number of periods in the test set
    n_test_days = len(y_test) // n_assets
    
    # Initialize portfolio returns
    portfolio_returns = []
    
    # Loop through each day in the test set
    for day in range(n_test_days):
        # Get predictions for all assets on this day
        day_predictions = predictions[day * n_assets:(day + 1) * n_assets]
        
        # Get actual returns for all assets on this day
        day_returns = y_test[day * n_assets:(day + 1) * n_assets]
        
        # Rank assets based on predicted returns
        ranked_indices = np.argsort(day_predictions)
        
        # Determine the number of assets to include in each side of the portfolio
        n_include = int(n_assets * top_pct)
        
        # Select top and bottom assets
        long_indices = ranked_indices[-n_include:]
        short_indices = ranked_indices[:n_include]
        
        # Calculate portfolio return (equal-weighted long-short)
        long_return = np.mean(day_returns[long_indices])
        short_return = np.mean(day_returns[short_indices])
        portfolio_return = long_return - short_return
        
        portfolio_returns.append(portfolio_return)
    
    # Convert to numpy array
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    return portfolio_returns, cumulative_returns

# Create a dictionary to store backtesting results
backtest_results = {}

# Backtest Linear Regression
lr_daily_returns, lr_cumulative_returns = backtest_strategy(lr_model, X_test_scaled, y_test, 
                                                           returns.index[-len(y_test)//50:])
backtest_results['Linear Regression'] = lr_cumulative_returns

# Backtest Random Forest
rf_daily_returns, rf_cumulative_returns = backtest_strategy(rf_model, X_test_scaled, y_test, 
                                                          returns.index[-len(y_test)//50:])
backtest_results['Random Forest'] = rf_cumulative_returns

# Backtest Gradient Boosting
gb_daily_returns, gb_cumulative_returns = backtest_strategy(gb_model, X_test_scaled, y_test, 
                                                          returns.index[-len(y_test)//50:])
backtest_results['Gradient Boosting'] = gb_cumulative_returns

# Backtest Neural Network
nn_daily_returns, nn_cumulative_returns = backtest_strategy(nn_model, X_test_scaled, y_test, 
                                                          returns.index[-len(y_test)//50:])
backtest_results['Neural Network'] = nn_cumulative_returns

# Calculate performance metrics for each strategy
def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a portfolio
    
    Args:
        returns: Array of daily portfolio returns
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod() - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = drawdowns.max()
    
    # Calculate win rate
    win_rate = sum(returns > 0) / len(returns)
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate
    }

# Calculate metrics for each strategy
lr_metrics = calculate_performance_metrics(lr_daily_returns)
rf_metrics = calculate_performance_metrics(rf_daily_returns)
gb_metrics = calculate_performance_metrics(gb_daily_returns)
nn_metrics = calculate_performance_metrics(nn_daily_returns)

# Create a DataFrame to compare strategies
metrics_df = pd.DataFrame({
    'Linear Regression': lr_metrics,
    'Random Forest': rf_metrics,
    'Gradient Boosting': gb_metrics,
    'Neural Network': nn_metrics
})

# Display metrics
print("\nPortfolio Performance Metrics:")
print(metrics_df.T)

# Plot cumulative returns for each strategy
plt.figure(figsize=(12, 8))
for name, returns in backtest_results.items():
    plt.plot(returns, label=name)
plt.title('Cumulative Returns of ML-based Trading Strategies')
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Analysis
# For Random Forest, we can extract feature importance
feature_names = [f'Return_t-{i}' for i in range(lookback, 0, -1)] + [f'Char_{i+1}' for i in range(20)]
rf_importance = rf_model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(rf_importance)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), rf_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# For Gradient Boosting, we can also extract feature importance
gb_importance = gb_model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(gb_importance)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), gb_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance (Gradient Boosting)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Visualize the relationship between predicted and actual returns
plt.figure(figsize=(10, 8))

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Linear Regression
axes[0, 0].scatter(lr_pred, y_test, alpha=0.3)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[0, 0].set_title('Linear Regression')
axes[0, 0].set_xlabel('Predicted Returns')
axes[0, 0].set_ylabel('Actual Returns')
axes[0, 0].grid(True)

# Random Forest
axes[0, 1].scatter(rf_pred, y_test, alpha=0.3)
axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[0, 1].set_title('Random Forest')
axes[0, 1].set_xlabel('Predicted Returns')
axes[0, 1].set_ylabel('Actual Returns')
axes[0, 1].grid(True)

# Gradient Boosting
axes[1, 0].scatter(gb_pred, y_test, alpha=0.3)
axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[1, 0].set_title('Gradient Boosting')
axes[1, 0].set_xlabel('Predicted Returns')
axes[1, 0].set_ylabel('Actual Returns')
axes[1, 0].grid(True)

# Neural Network
axes[1, 1].scatter(nn_pred, y_test, alpha=0.3)
axes[1, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[1, 1].set_title('Neural Network')
axes[1, 1].set_xlabel('Predicted Returns')
axes[1, 1].set_ylabel('Actual Returns')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Analyze portfolio turnover
def calculate_turnover(model, X_test_scaled, n_assets=50, top_pct=0.2):
    """
    Calculate portfolio turnover for a given model
    
    Args:
        model: Trained ML model
        X_test_scaled: Scaled features for test set
        n_assets: Number of assets
        top_pct: Percentage of assets to go long/short
        
    Returns:
        turnover: Average daily turnover
    """
    # For neural network, we need to handle the input differently
    if isinstance(model, tf.keras.models.Model):
        # Reshape data for LSTM
        X_test_hist = X_test_scaled[:, :lookback]
        X_test_other = X_test_scaled[:, lookback:]
        X_test_hist_reshaped = X_test_hist.reshape(X_test_hist.shape[0], lookback, 1)
        predictions = model.predict([X_test_hist_reshaped, X_test_other]).flatten()
    else:
        # For other models
        predictions = model.predict(X_test_scaled)
    
    # Determine the number of periods in the test set
    n_test_days = len(predictions) // n_assets
    
    # Initialize turnover list
    turnover_list = []
    
    # Number of assets in each side of the portfolio
    n_include = int(n_assets * top_pct)
    
    # Keep track of previous day's positions
    prev_long = set()
    prev_short = set()
    
    # Loop through each day in the test set
    for day in range(n_test_days - 1):  # -1 because we compare with next day
        # Get predictions for all assets on this day
        day_predictions = predictions[day * n_assets:(day + 1) * n_assets]
        
        # Rank assets based on predicted returns
        ranked_indices = np.argsort(day_predictions)
        
        # Select top and bottom assets
        long_indices = set(ranked_indices[-n_include:])
        short_indices = set(ranked_indices[:n_include])
        
        if day > 0:  # Skip first day
            # Calculate changes in positions
            long_additions = long_indices - prev_long
            long_removals = prev_long - long_indices
            short_additions = short_indices - prev_short
            short_removals = prev_short - short_indices
            
            # Total number of changes
            total_changes = len(long_additions) + len(long_removals) + len(short_additions) + len(short_removals)
            
            # Total positions (long and short)
            total_positions = 2 * n_include
            
            # Calculate turnover as percentage of portfolio that changed
            turnover = total_changes / total_positions
            turnover_list.append(turnover)
        
        # Update previous positions
        prev_long = long_indices
        prev_short = short_indices
    
    # Calculate average turnover
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    
    return avg_turnover

# Calculate turnover for each model
lr_turnover = calculate_turnover(lr_model, X_test_scaled)
rf_turnover = calculate_turnover(rf_model, X_test_scaled)
gb_turnover = calculate_turnover(gb_model, X_test_scaled)
nn_turnover = calculate_turnover(nn_model, X_test_scaled)

print("\nPortfolio Turnover (Average Daily):")
print(f"Linear Regression: {lr_turnover:.4f}")
print(f"Random Forest: {rf_turnover:.4f}")
print(f"Gradient Boosting: {gb_turnover:.4f}")
print(f"Neural Network: {nn_turnover:.4f}")

# Add turnover to metrics DataFrame
metrics_df.loc['Turnover'] = [lr_turnover, rf_turnover, gb_turnover, nn_turnover]

# Display updated metrics
print("\nUpdated Portfolio Performance Metrics:")
print(metrics_df.T)

# Summary of findings
print("\nSummary of Findings:")
print("1. Model Performance Comparison:")
best_model = metrics_df.loc['Sharpe Ratio'].idxmax()
print(f"   - Best performing model based on Sharpe Ratio: {best_model}")
print(f"   - Best total return: {metrics_df.loc['Total Return'].max():.4f} ({metrics_df.loc['Total Return'].idxmax()})")

print("\n2. Feature Importance:")
top_feature_rf = feature_names[np.argmax(rf_importance)]
print(f"   - Most important feature according to Random Forest: {top_feature_rf}")

top_feature_gb = feature_names[np.argmax(gb_importance)]
print(f"   - Most important feature according to Gradient Boosting: {top_feature_gb}")

print("\n3. Portfolio Characteristics:")
print(f"   - Lowest turnover model: {metrics_df.loc['Turnover'].idxmin()} ({metrics_df.loc['Turnover'].min():.4f})")
print(f"   - Highest Sharpe ratio: {metrics_df.loc['Sharpe Ratio'].max():.4f} ({metrics_df.loc['Sharpe Ratio'].idxmax()})")
print(f"   - Lowest maximum drawdown: {metrics_df.loc['Maximum Drawdown'].min():.4f} ({metrics_df.loc['Maximum Drawdown'].idxmin()})")

print("\n4. Model Complexity vs. Performance:")
if metrics_df.loc['Sharpe Ratio']['Neural Network'] > metrics_df.loc['Sharpe Ratio']['Linear Regression']:
    print("   - More complex models (Neural Network) outperform simpler models (Linear Regression)")
else:
    print("   - Simpler models (Linear Regression) perform comparably to complex models (Neural Network)")