import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate limit order book data
def simulate_lob_data(n_samples=10000, n_levels=5):
    """
    Simulate limit order book data with known patterns
    """
    # Initialize arrays for ask and bid prices and volumes
    ask_prices = np.zeros((n_samples, n_levels))
    ask_volumes = np.zeros((n_samples, n_levels))
    bid_prices = np.zeros((n_samples, n_levels))
    bid_volumes = np.zeros((n_samples, n_levels))
    
    # Base price that will evolve over time
    base_price = 100.0
    
    # Price trends
    trends = np.zeros(n_samples)
    
    # Create some trending and mean-reverting periods
    for i in range(0, n_samples, 1000):
        if random.random() > 0.5:
            # Upward trend
            trends[i:i+1000] = np.linspace(0, 0.1, 1000)
        else:
            # Downward trend
            trends[i:i+1000] = np.linspace(0, -0.1, 1000)
    
    # Volatility
    volatility = np.ones(n_samples) * 0.005
    
    # Generate price series with the trends and volatility
    price_changes = trends + volatility * np.random.randn(n_samples)
    prices = base_price + np.cumsum(price_changes)
    
    # Create the order book
    for t in range(n_samples):
        # Spread is larger when volatility is higher
        spread = 0.02 + 0.5 * volatility[t]
        
        # Best ask and bid prices
        ask_prices[t, 0] = prices[t] + spread / 2
        bid_prices[t, 0] = prices[t] - spread / 2
        
        # Create price levels with widening spreads
        for level in range(1, n_levels):
            ask_prices[t, level] = ask_prices[t, level-1] + 0.01 * (level + 1)
            bid_prices[t, level] = bid_prices[t, level-1] - 0.01 * (level + 1)
        
        # Generate volumes
        for level in range(n_levels):
            ask_volumes[t, level] = 100 * (n_levels - level) / n_levels + 20 * np.random.randn()
            bid_volumes[t, level] = 100 * (n_levels - level) / n_levels + 20 * np.random.randn()
            
            # Ensure volumes are positive
            ask_volumes[t, level] = max(1, ask_volumes[t, level])
            bid_volumes[t, level] = max(1, bid_volumes[t, level])
    
    # Create mid prices
    mid_prices = (ask_prices[:, 0] + bid_prices[:, 0]) / 2
    
    # Calculate future mid-price movements for labeling
    future_returns = np.zeros(n_samples)
    horizon = 10  # Prediction horizon (10 steps ahead)
    for t in range(n_samples - horizon):
        future_returns[t] = (mid_prices[t + horizon] - mid_prices[t]) / mid_prices[t]
    
    # Create labels (up, down, stationary)
    threshold = 0.0002  # Threshold for considering a move significant
    labels = np.zeros(n_samples, dtype=int)
    labels[future_returns > threshold] = 1  # Up
    labels[future_returns < -threshold] = -1  # Down
    # 0 remains stationary
    
    # Create DataFrames for easier handling
    data = {
        'mid_price': mid_prices,
        'label': labels
    }
    
    for level in range(n_levels):
        data[f'ask_price_{level+1}'] = ask_prices[:, level]
        data[f'ask_volume_{level+1}'] = ask_volumes[:, level]
        data[f'bid_price_{level+1}'] = bid_prices[:, level]
        data[f'bid_volume_{level+1}'] = bid_volumes[:, level]
    
    df = pd.DataFrame(data)
    return df

# Generate the simulated limit order book data
lob_data = simulate_lob_data(n_samples=20000, n_levels=5)

# Display the first few rows
print("Simulated LOB Data:")
print(lob_data.head())

# Check the distribution of labels
label_counts = lob_data['label'].value_counts()
print("\nLabel distribution:")
print(label_counts)
print(f"Percentage of Up labels: {label_counts.get(1, 0) / len(lob_data) * 100:.2f}%")
print(f"Percentage of Stationary labels: {label_counts.get(0, 0) / len(lob_data) * 100:.2f}%")
print(f"Percentage of Down labels: {label_counts.get(-1, 0) / len(lob_data) * 100:.2f}%")

# Plot the mid price and labels using numpy arrays
plt.figure(figsize=(12, 6))

# Get the first 1000 data points
first_1000_mid_prices = lob_data['mid_price'].values[:1000]
first_1000_labels = lob_data['label'].values[:1000]

# Plot the mid prices
plt.plot(first_1000_mid_prices, label='Mid Price')

# Find indices for up and down movements using numpy
up_indices = np.where(first_1000_labels == 1)[0]
down_indices = np.where(first_1000_labels == -1)[0]

# Plot scatter points
plt.scatter(up_indices, first_1000_mid_prices[up_indices], color='green', marker='^', label='Up')
plt.scatter(down_indices, first_1000_mid_prices[down_indices], color='red', marker='v', label='Down')

plt.title('Simulated Mid Price with Labels (First 1000 points)')
plt.legend()
plt.show()

# Extract basic features directly
def extract_features(df):
    """Extract combined features set"""
    features = {}
    
    # Basic spread and mid-price features
    for level in range(1, 6):
        features[f'spread_{level}'] = df[f'ask_price_{level}'] - df[f'bid_price_{level}']
        features[f'mid_{level}'] = (df[f'ask_price_{level}'] + df[f'bid_price_{level}']) / 2
        features[f'imbalance_{level}'] = (df[f'bid_volume_{level}'] - df[f'ask_volume_{level}']) / (df[f'bid_volume_{level}'] + df[f'ask_volume_{level}'] + 1e-10)
    
    # Moving averages of mid price
    for window in [5, 10, 20]:
        features[f'sma_{window}'] = df['mid_price'].rolling(window=window).mean().fillna(df['mid_price'])
        features[f'ema_{window}'] = df['mid_price'].ewm(span=window, adjust=False).mean().fillna(df['mid_price'])
    
    # Price changes
    features['price_change'] = df['mid_price'].diff().fillna(0)
    features['price_change_pct'] = df['mid_price'].pct_change().fillna(0)
    
    # Volume features
    features['total_bid_volume'] = df[[f'bid_volume_{i}' for i in range(1, 6)]].sum(axis=1)
    features['total_ask_volume'] = df[[f'ask_volume_{i}' for i in range(1, 6)]].sum(axis=1)
    features['volume_ratio'] = features['total_bid_volume'] / (features['total_ask_volume'] + 1e-10)
    
    # Combine into DataFrame
    return pd.DataFrame(features)

# Extract features
all_features = extract_features(lob_data)

# Check feature dimensions
print(f"\nTotal features: {all_features.shape[1]}")

# Split the data into training, validation, and testing sets
train_size = int(0.6 * len(lob_data))
val_size = int(0.2 * len(lob_data))

X_train = all_features.iloc[:train_size].values
y_train = lob_data['label'].iloc[:train_size].values

X_val = all_features.iloc[train_size:train_size+val_size].values
y_val = lob_data['label'].iloc[train_size:train_size+val_size].values

X_test = all_features.iloc[train_size+val_size:].values
y_test = lob_data['label'].iloc[train_size+val_size:].values

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Testing set: {X_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Simple feature selection: select best features based on correlation with target
def select_features(X, y, k=10):
    """Select k best features based on correlation with target"""
    correlations = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        correlations[i] = abs(np.corrcoef(X[:, i], y)[0, 1])
    
    # Handle NaN values
    correlations = np.nan_to_num(correlations)
    
    # Select top k features
    selected_indices = np.argsort(correlations)[-k:]
    
    return selected_indices

# Select best features
k = 10  # Number of features to select
selected_indices = select_features(X_train_scaled, y_train, k=k)

X_train_selected = X_train_scaled[:, selected_indices]
X_val_selected = X_val_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

print(f"\nSelected {k} best features based on correlation")

# Train LDA model
print("\nTraining LDA model...")
lda = LDA()
lda.fit(X_train_selected, y_train)

# Make predictions
y_val_pred = lda.predict(X_val_selected)
y_test_pred = lda.predict(X_test_selected)

# Calculate metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='macro')

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\nValidation results:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"F1 score: {val_f1:.4f}")

print(f"\nTest results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 score: {test_f1:.4f}")

# Simple trading strategy
def trading_strategy(y_pred, mid_prices, transaction_cost=0.0001):
    """Simple trading strategy based on predicted mid-price movements."""
    n = len(y_pred)
    positions = np.zeros(n)
    pnl = np.zeros(n)
    
    # Set positions based on predictions
    positions[y_pred == 1] = 1    # Long position
    positions[y_pred == -1] = -1  # Short position
    
    # Calculate price returns
    price_returns = np.zeros(n)
    price_returns[1:] = (mid_prices[1:] - mid_prices[:-1]) / mid_prices[:-1]
    
    # Calculate PnL with transaction costs
    for i in range(1, n):
        # PnL is the return multiplied by the previous position
        pnl[i] = positions[i-1] * price_returns[i]
        
        # Subtract transaction costs when position changes
        if positions[i] != positions[i-1]:
            pnl[i] -= transaction_cost
    
    # Calculate cumulative PnL
    cum_pnl = np.cumsum(pnl)
    
    return positions, pnl, cum_pnl

# Apply trading strategy to test predictions
test_mid_prices = lob_data['mid_price'].iloc[train_size+val_size:].values
positions, pnl, cum_pnl = trading_strategy(y_test_pred, test_mid_prices)

# Calculate trading metrics
total_return = cum_pnl[-1]
annualized_return = total_return * (252 / len(pnl))  # Assuming 252 trading days in a year
sharpe_ratio = annualized_return / (np.std(pnl) * np.sqrt(252)) if np.std(pnl) > 0 else 0
win_rate = np.sum(pnl > 0) / len(pnl)

print("\nTrading results:")
print(f"Total return: {total_return:.4f}")
print(f"Annualized return: {annualized_return:.4f}")
print(f"Sharpe ratio: {sharpe_ratio:.4f}")
print(f"Win rate: {win_rate:.4f}")

# Plot trading results
plt.figure(figsize=(12, 8))

# Plot 1: Mid-price and positions
plt.subplot(2, 1, 1)
plt.plot(test_mid_prices, label='Mid Price')

# Find position indices without using boolean indexing
long_indices = np.where(positions == 1)[0]
short_indices = np.where(positions == -1)[0]

plt.scatter(long_indices, test_mid_prices[long_indices], color='green', marker='^', alpha=0.5, label='Long')
plt.scatter(short_indices, test_mid_prices[short_indices], color='red', marker='v', alpha=0.5, label='Short')
plt.title('Mid Price and Trading Positions')
plt.legend()

# Plot 2: Cumulative PnL
plt.subplot(2, 1, 2)
plt.plot(cum_pnl, color='blue', label='Cumulative PnL')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Cumulative PnL')
plt.legend()

plt.tight_layout()
plt.show()

# Print top features
feature_names = all_features.columns
print("\nTop features selected:")
for i, idx in enumerate(selected_indices):
    print(f"{i+1}. {feature_names[idx]}")