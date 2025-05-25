import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_stocks = 500  # Number of stocks
n_days = 1000   # Number of days
jump_threshold = 3  # Number of std deviations to classify as jump

# Market parameters
market_mean_return = 0.0005  # Daily market return (≈12% annual)
market_vol = 0.01            # Market volatility

# Firm-specific parameters
firm_mean_return = 0.0005    # Mean firm-specific return
firm_vol = 0.02              # Firm-specific volatility
idio_vol_mean = 0.02         # Mean idiosyncratic volatility
idio_vol_std = 0.01          # Standard deviation of idiosyncratic volatility

# Jump parameters
jump_intensity = 0.018       # Average probability of jump on any day (1.8% as in paper)
jump_mean = 0.025            # Mean jump size (2.5% as in paper)
jump_std = 0.08              # Standard deviation of jump size
jump_pos_prob = 0.62         # Probability of positive jump (vs negative)

# Simulation parameters
decay_factor = 0.94          # Decay factor for EWMA volatility estimation (RiskMetrics standard)

#--------------------------------------------------------------
# Simulate market returns
#--------------------------------------------------------------
market_returns = np.random.normal(market_mean_return, market_vol, n_days)

#--------------------------------------------------------------
# Generate stock data with jumps
#--------------------------------------------------------------

# Function to simulate a stock price path with jumps
def simulate_stock_with_jumps(n_days, market_beta, market_returns, 
                              idio_vol, jump_intensity, jump_mean, jump_std):
    """
    Simulate a stock price path with jumps
    
    Parameters:
    -----------
    n_days: int
        Number of days to simulate
    market_beta: float
        Stock's beta to market
    market_returns: array
        Market returns
    idio_vol: float
        Idiosyncratic volatility
    jump_intensity: float
        Probability of jump per day
    jump_mean: float
        Mean jump size
    jump_std: float
        Standard deviation of jump size
        
    Returns:
    --------
    tuple of arrays:
        returns, idiosyncratic_returns, jumps, jump_sizes
    """
    # Generate idiosyncratic returns (without jumps)
    idio_returns_no_jumps = np.random.normal(0, idio_vol, n_days)
    
    # Generate jumps
    jump_occurs = np.random.binomial(1, jump_intensity, n_days).astype(bool)
    jump_signs = np.random.choice([-1, 1], size=n_days, p=[1-jump_pos_prob, jump_pos_prob])
    jump_sizes = np.zeros(n_days)
    jump_sizes[jump_occurs] = jump_signs[jump_occurs] * np.abs(np.random.normal(jump_mean, jump_std, np.sum(jump_occurs)))
    
    # Compose total idiosyncratic returns
    idio_returns = idio_returns_no_jumps + jump_sizes
    
    # Compose total returns (market + idiosyncratic)
    returns = market_beta * market_returns + idio_returns
    
    return returns, idio_returns, jump_occurs, jump_sizes

# Simulate all stocks
stock_data = []
for i in range(n_stocks):
    # Randomize parameters
    beta = np.random.uniform(0.8, 1.2)
    idio_vol = np.random.normal(idio_vol_mean, idio_vol_std)
    idio_vol = max(0.005, idio_vol)  # Ensure positive vol
    
    # Simulate stock
    returns, idio_returns, jump_occurs, jump_sizes = simulate_stock_with_jumps(
        n_days, beta, market_returns, idio_vol, jump_intensity, jump_mean, jump_std
    )
    
    # Store results
    stock_data.append({
        'stock_id': i,
        'beta': beta,
        'idio_vol': idio_vol,
        'returns': returns,
        'idio_returns': idio_returns,
        'jump_occurs': jump_occurs,
        'jump_sizes': jump_sizes
    })

#--------------------------------------------------------------
# Process data and identify jumps (as in the paper)
#--------------------------------------------------------------

# Implement EWMA volatility estimation and jump detection
for i, stock in enumerate(stock_data):
    idio_returns = stock['idio_returns']
    
    # Calculate EWMA idiosyncratic volatility
    ewma_var = np.zeros(n_days)
    ewma_var[0] = idio_returns[0]**2
    for t in range(1, n_days):
        ewma_var[t] = decay_factor * ewma_var[t-1] + (1 - decay_factor) * idio_returns[t-1]**2
    
    # Ensure minimum variance to avoid division by zero
    ewma_var = np.maximum(ewma_var, 1e-8)
    ewma_vol = np.sqrt(ewma_var)
    
    # Identify jumps
    jump_threshold_adjusted = np.abs(idio_returns) > (jump_threshold * ewma_vol)
    
    # Store results
    stock_data[i]['ewma_vol'] = ewma_vol
    stock_data[i]['detected_jumps'] = jump_threshold_adjusted

#--------------------------------------------------------------
# Create option-based features for jump prediction
#--------------------------------------------------------------

# In the paper, they use option-implied volatility and tail measures
# Since we don't have option data in our simulation, we'll create proxies
# These proxies will be noisy forecasts of true jump probabilities

for i, stock in enumerate(stock_data):
    n_days = len(stock['returns'])
    
    # Create a proxy for implied volatility (IV)
    # IV is typically future realized vol + risk premium + noise
    # Here we'll use future realized vol + noise as a proxy
    future_vol = np.zeros(n_days)
    
    # For each day, calculate realized vol over next 20 days
    lookforward = 20
    for t in range(n_days - lookforward):
        future_vol[t] = np.std(stock['idio_returns'][t:t+lookforward])
    
    # Add some noise and risk premium
    risk_premium = 0.005
    noise_level = 0.003
    implied_vol = future_vol + risk_premium + np.random.normal(0, noise_level, n_days)
    
    # For the last 20 days, use the last valid value
    implied_vol[n_days - lookforward:] = implied_vol[n_days - lookforward - 1]
    
    # Ensure minimum volatility
    implied_vol = np.maximum(implied_vol, 0.001)
    
    # Create a proxy for right-tail and left-tail jump measures
    # Higher when real jump probability is higher, plus noise
    base_jump_prob = np.zeros(n_days)
    for t in range(n_days - lookforward):
        # Check if there's a jump in the next days
        future_jumps = stock['jump_occurs'][t:t+lookforward]
        base_jump_prob[t] = 0.2 if np.any(future_jumps) else 0.05
    
    # Fill the last days
    base_jump_prob[n_days - lookforward:] = base_jump_prob[n_days - lookforward - 1]
    
    # Add noise
    left_tail = base_jump_prob + np.random.normal(0, 0.05, n_days)
    right_tail = base_jump_prob + np.random.normal(0, 0.05, n_days)
    
    # Ensure values are positive
    left_tail = np.maximum(0.01, left_tail)
    right_tail = np.maximum(0.01, right_tail)
    
    # Calculate volatility ratio (IV to realized vol)
    # Use a safer way to calculate the ratio to avoid log(0) or division by zero
    vol_ratio = np.log(np.maximum(implied_vol / np.maximum(stock['ewma_vol'], 1e-8), 1e-8))
    
    # Store features
    stock_data[i]['implied_vol'] = implied_vol
    stock_data[i]['left_tail'] = left_tail
    stock_data[i]['right_tail'] = right_tail
    stock_data[i]['vol_ratio'] = vol_ratio
    stock_data[i]['avg_tail'] = (left_tail + right_tail) / 2

#--------------------------------------------------------------
# Implement the jump prediction model
#--------------------------------------------------------------

# Convert stock data into a DataFrame for easier handling
all_data = []
for i, stock in enumerate(stock_data):
    for t in range(n_days):
        all_data.append({
            'stock_id': stock['stock_id'],
            'day': t,
            'return': stock['returns'][t],
            'idio_return': stock['idio_returns'][t],
            'ewma_vol': stock['ewma_vol'][t],
            'true_jump': stock['jump_occurs'][t],
            'detected_jump': stock['detected_jumps'][t],
            'implied_vol': stock['implied_vol'][t],
            'left_tail': stock['left_tail'][t],
            'right_tail': stock['right_tail'][t],
            'vol_ratio': stock['vol_ratio'][t],
            'avg_tail': stock['avg_tail'][t]
        })

df = pd.DataFrame(all_data)

# Check for any NaN or infinite values
print(f"NaN values in data: {df.isna().sum().sum()}")
print(f"Infinite values in vol_ratio: {np.isinf(df['vol_ratio']).sum()}")

# Replace any remaining NaN or inf values with reasonable values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Split data for training and testing
train_cutoff = int(n_days * 0.6)
df_train = df[df['day'] < train_cutoff]
df_test = df[df['day'] >= train_cutoff]

# Train logistic regression model for jump prediction
# Using the same predictors as in the paper: average tail and volatility ratio
X_train = df_train[['avg_tail', 'vol_ratio']].values
y_train = df_train['detected_jump'].values

jump_model = LogisticRegression(random_state=42)
jump_model.fit(X_train, y_train)

# Make predictions on test set
X_test = df_test[['avg_tail', 'vol_ratio']].values
df_test['jump_prob_predicted'] = jump_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nJump Prediction Model Performance:")
auc = roc_auc_score(df_test['detected_jump'], df_test['jump_prob_predicted'])
print(f"AUC: {auc:.4f}")

# Create quintiles based on predicted jump probabilities
df_test['jump_prob_quintile'] = pd.qcut(df_test['jump_prob_predicted'], 5, labels=False)

#--------------------------------------------------------------
# Backtest the strategy
#--------------------------------------------------------------

# For each day in the test set, form portfolios based on jump probability quintiles
# and calculate returns for the next day

daily_returns = {}
daily_actual_jump_rates = {}

test_days = sorted(df_test['day'].unique())
for day in test_days[:-1]:  # Skip the last day as we need next day returns
    # Get data for the current day
    day_data = df_test[df_test['day'] == day]
    
    # For each quintile, calculate average return for the next day
    for quintile in range(5):
        # Select stocks in this quintile
        quintile_stocks = day_data[day_data['jump_prob_quintile'] == quintile]
        stock_ids = quintile_stocks['stock_id'].values
        
        # Get next day's returns and jump occurrences for these stocks
        next_day = day + 1
        next_day_data = df_test[(df_test['day'] == next_day) & (df_test['stock_id'].isin(stock_ids))]
        
        # Calculate average return and jump rate
        avg_return = next_day_data['return'].mean()
        jump_rate = next_day_data['detected_jump'].mean()
        
        # Store results
        if quintile not in daily_returns:
            daily_returns[quintile] = []
            daily_actual_jump_rates[quintile] = []
            
        daily_returns[quintile].append(avg_return)
        daily_actual_jump_rates[quintile].append(jump_rate)

# Convert to DataFrame for analysis
returns_df = pd.DataFrame(daily_returns)
jump_rates_df = pd.DataFrame(daily_actual_jump_rates)

# Calculate statistics
avg_returns = returns_df.mean()
avg_jump_rates = jump_rates_df.mean()
cumulative_returns = (returns_df + 1).cumprod()

# Calculate long-short portfolio returns (high minus low jump probability)
long_short_returns = returns_df[4] - returns_df[0]
avg_long_short = long_short_returns.mean()
std_long_short = long_short_returns.std()
annual_sharpe = avg_long_short / std_long_short * np.sqrt(252)
cumulative_long_short = (long_short_returns + 1).cumprod()

# Calculate alpha by regressing against market returns
market_returns_test = market_returns[train_cutoff+1:train_cutoff+1+len(long_short_returns)]

X = sm.add_constant(market_returns_test)
model = sm.OLS(long_short_returns, X).fit()
alpha = model.params[0]
annual_alpha = alpha * 252

#--------------------------------------------------------------
# Plot results
#--------------------------------------------------------------

plt.figure(figsize=(12, 18))

# Plot 1: Average returns by jump probability quintile
plt.subplot(3, 1, 1)
plt.bar(avg_returns.index, avg_returns * 252 * 100)  # Annualized percentage
plt.title('Annualized Returns by Jump Probability Quintile')
plt.xlabel('Jump Probability Quintile (0=Low, 4=High)')
plt.ylabel('Annualized Return (%)')

for i, v in enumerate(avg_returns * 252 * 100):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

# Plot 2: Average jump rates by jump probability quintile
plt.subplot(3, 1, 2)
plt.bar(avg_jump_rates.index, avg_jump_rates * 100)
plt.title('Actual Jump Rates by Jump Probability Quintile')
plt.xlabel('Jump Probability Quintile (0=Low, 4=High)')
plt.ylabel('Jump Rate (%)')

for i, v in enumerate(avg_jump_rates * 100):
    plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')

# Plot 3: Cumulative returns of high jump probability minus low jump probability
plt.subplot(3, 1, 3)
plt.plot(cumulative_long_short, linewidth=2)
plt.title(f'Cumulative Returns of High-Low Jump Probability Portfolio\nAnnualized Alpha: {annual_alpha*100:.2f}%, Sharpe: {annual_sharpe:.2f}')
plt.xlabel('Trading Day')
plt.ylabel('Cumulative Return')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results
print("\n==== Strategy Performance ====")
print(f"Annualized Returns by Quintile (Low to High):")
for i, ret in enumerate(avg_returns):
    print(f"  Quintile {i}: {ret*252*100:.2f}%")
    
print(f"\nHigh-Low Jump Probability Portfolio:")
print(f"  Daily Return: {avg_long_short*100:.4f}%")
print(f"  Annualized Return: {avg_long_short*252*100:.2f}%")
print(f"  Annualized Alpha: {annual_alpha*100:.2f}%")
print(f"  Annualized Sharpe Ratio: {annual_sharpe:.2f}")
print(f"  Daily Volatility: {std_long_short*100:.4f}%")

print("\nJump Rates by Quintile (Low to High):")
for i, rate in enumerate(avg_jump_rates):
    print(f"  Quintile {i}: {rate*100:.2f}%")