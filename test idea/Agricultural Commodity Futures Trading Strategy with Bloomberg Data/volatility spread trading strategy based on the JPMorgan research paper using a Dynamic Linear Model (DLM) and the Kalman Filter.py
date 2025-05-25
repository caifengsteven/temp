import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
num_days = 1500
start_date = datetime(2015, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]
dates_df = pd.DataFrame(index=dates)

# Function to simulate price and volatility data
def simulate_volatility_data():
    # Base parameters for the simulation
    # Mean levels - V2X generally higher than VIX
    vix_mean = 17.0
    v2x_mean = 21.0
    
    # Volatility of volatility
    vix_vol = 0.08
    v2x_vol = 0.09
    
    # Mean reversion parameters
    vix_reversion = 0.05
    v2x_reversion = 0.04
    
    # Create correlation between VIX and V2X
    correlation = 0.8
    
    # Simulate paths for VIX and V2X with correlation and mean reversion
    vix = np.zeros(num_days)
    v2x = np.zeros(num_days)
    
    # Initial values
    vix[0] = vix_mean
    v2x[0] = v2x_mean
    
    # Correlated random shocks
    random_shocks = np.random.multivariate_normal(
        mean=[0, 0], 
        cov=[[1, correlation], [correlation, 1]],
        size=num_days
    )
    
    # Simulate paths with mean reversion
    for i in range(1, num_days):
        vix[i] = vix[i-1] + vix_reversion * (vix_mean - vix[i-1]) + vix_vol * vix[i-1] * random_shocks[i, 0]
        v2x[i] = v2x[i-1] + v2x_reversion * (v2x_mean - v2x[i-1]) + v2x_vol * v2x[i-1] * random_shocks[i, 1]
    
    # Ensure volatility doesn't go negative
    vix = np.maximum(vix, 5.0)
    v2x = np.maximum(v2x, 5.0)
    
    # Add occasional volatility spikes
    num_spikes = 5
    spike_indices = np.random.choice(range(100, num_days-100), num_spikes, replace=False)
    
    for idx in spike_indices:
        # Determine spike magnitude (random between 40% and 120% increase)
        vix_spike_magnitude = np.random.uniform(0.4, 1.2)
        v2x_spike_magnitude = np.random.uniform(0.4, 1.2)
        
        # Create spikes
        vix[idx:idx+20] = vix[idx:idx+20] * (1 + vix_spike_magnitude * np.exp(-0.2 * np.arange(20)))
        v2x[idx:idx+20] = v2x[idx:idx+20] * (1 + v2x_spike_magnitude * np.exp(-0.2 * np.arange(20)))
    
    # Simulate futures prices - create a term structure
    # For VIX futures
    vix_1m = np.zeros(num_days)
    vix_2m = np.zeros(num_days)
    
    # For V2X futures
    v2x_1m = np.zeros(num_days)
    v2x_2m = np.zeros(num_days)
    
    # Generate futures prices with term structure effects
    for i in range(num_days):
        # VIX futures
        term_premium_vix = max(0.5, 0.1 * vix[i]) * np.random.normal(1, 0.1)  # Higher premium during higher volatility
        vix_1m[i] = vix[i]
        vix_2m[i] = vix_1m[i] + term_premium_vix
        
        # During spikes, futures curve can become inverted
        if i in spike_indices or i in [idx+j for idx in spike_indices for j in range(1, 10)]:
            if np.random.random() < 0.7:  # 70% chance of inversion during spikes
                vix_2m[i] = vix_1m[i] - abs(np.random.normal(0.3, 0.1) * vix_1m[i])
        
        # V2X futures
        term_premium_v2x = max(0.5, 0.12 * v2x[i]) * np.random.normal(1, 0.1)  # Slightly higher premium for V2X
        v2x_1m[i] = v2x[i]
        v2x_2m[i] = v2x_1m[i] + term_premium_v2x
        
        # During spikes, futures curve can become inverted for V2X too
        if i in spike_indices or i in [idx+j for idx in spike_indices for j in range(1, 10)]:
            if np.random.random() < 0.7:  # 70% chance of inversion during spikes
                v2x_2m[i] = v2x_1m[i] - abs(np.random.normal(0.3, 0.1) * v2x_1m[i])
    
    return vix, v2x, vix_1m, vix_2m, v2x_1m, v2x_2m, spike_indices

# Simulate data
vix, v2x, vix_1m, vix_2m, v2x_1m, v2x_2m, spike_indices = simulate_volatility_data()

# Create a DataFrame with all the data
volatility_data = pd.DataFrame({
    'date': dates,
    'VIX': vix,
    'V2X': v2x,
    'VIX_1M': vix_1m,
    'VIX_2M': vix_2m,
    'V2X_1M': v2x_1m,
    'V2X_2M': v2x_2m
})
volatility_data.set_index('date', inplace=True)

# Calculate rolling percentage for 1M futures (simulating daily rolling)
# Create rolling percentage that repeats every 30 days
roll_cycle = np.linspace(0, 1, 30).tolist()
num_cycles = num_days // 30 + (1 if num_days % 30 > 0 else 0)
roll_pct = (roll_cycle * num_cycles)[:num_days]  # Ensure exact length

volatility_data['VIX_roll_pct'] = roll_pct
volatility_data['V2X_roll_pct'] = roll_pct  # Using same rolling schedule

# Calculate the constant 1M volatility futures
volatility_data['VIX_1M_constant'] = (1 - volatility_data['VIX_roll_pct']) * volatility_data['VIX_1M'] + volatility_data['VIX_roll_pct'] * volatility_data['VIX_2M']
volatility_data['V2X_1M_constant'] = (1 - volatility_data['V2X_roll_pct']) * volatility_data['V2X_1M'] + volatility_data['V2X_roll_pct'] * volatility_data['V2X_2M']

# Calculate volatility spread and carry spread
volatility_data['volatility_spread'] = volatility_data['VIX_1M_constant'] - volatility_data['V2X_1M_constant']
volatility_data['VIX_carry'] = volatility_data['VIX_2M'] - volatility_data['VIX_1M']
volatility_data['V2X_carry'] = volatility_data['V2X_2M'] - volatility_data['V2X_1M']
volatility_data['carry_spread'] = volatility_data['VIX_carry'] - volatility_data['V2X_carry']

# Calculate daily changes
volatility_data['volatility_spread_change'] = volatility_data['volatility_spread'].diff()
volatility_data['carry_spread_change'] = volatility_data['carry_spread'].diff()

# Drop NA values
volatility_data = volatility_data.dropna()

# Plot the simulated data
plt.figure(figsize=(15, 10))

# Plot VIX and V2X (constant 1M)
plt.subplot(2, 2, 1)
plt.plot(volatility_data.index, volatility_data['VIX_1M_constant'], label='VIX 1M', color='blue')
plt.plot(volatility_data.index, volatility_data['V2X_1M_constant'], label='V2X 1M', color='red')
plt.title('Simulated VIX and V2X (1M Constant)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot volatility spread
plt.subplot(2, 2, 2)
plt.plot(volatility_data.index, volatility_data['volatility_spread'], label='Volatility Spread (VIX - V2X)', color='green')
plt.title('Volatility Spread (VIX - V2X)')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Plot carry spreads
plt.subplot(2, 2, 3)
plt.plot(volatility_data.index, volatility_data['VIX_carry'], label='VIX Carry', color='blue')
plt.plot(volatility_data.index, volatility_data['V2X_carry'], label='V2X Carry', color='red')
plt.title('Carry (2M - 1M)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot carry spread
plt.subplot(2, 2, 4)
plt.plot(volatility_data.index, volatility_data['carry_spread'], label='Carry Spread (VIX - V2X)', color='purple')
plt.title('Carry Spread (VIX - V2X)')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check for mean reversion in volatility spread
def mean_reversion_test(series, series_name):
    result = adfuller(series.dropna())
    print(f'ADF Statistic for {series_name}: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] < 0.05:
        print(f"{series_name} appears to be mean-reverting (stationary) at 5% significance level")
    else:
        print(f"{series_name} does not appear to be mean-reverting at 5% significance level")
    print()

mean_reversion_test(volatility_data['volatility_spread'], 'Volatility Spread')

# Analyze relationship between volatility spread and carry spread
plt.figure(figsize=(12, 5))

# Group by volatility spread bins and calculate mean of carry spread
bins = np.linspace(volatility_data['volatility_spread'].min(), 
                   volatility_data['volatility_spread'].max(), 20)
binned_data = pd.cut(volatility_data['volatility_spread'], bins)
grouped_data = volatility_data.groupby(binned_data)['carry_spread'].mean()

plt.scatter(volatility_data['volatility_spread'], volatility_data['carry_spread'], alpha=0.3, color='blue')
plt.plot([(b.left + b.right)/2 for b in grouped_data.index], grouped_data.values, 'ro-', linewidth=2)
plt.title('Relationship between Volatility Spread and Carry Spread')
plt.xlabel('Volatility Spread (VIX - V2X)')
plt.ylabel('Carry Spread (VIX - V2X)')
plt.grid(True, alpha=0.3)
plt.show()

# Analyze relationship between changes in volatility spread and changes in carry spread
plt.figure(figsize=(12, 5))

# Group by volatility spread change bins and calculate mean of carry spread change
bins = np.linspace(volatility_data['volatility_spread_change'].min(), 
                   volatility_data['volatility_spread_change'].max(), 20)
binned_data = pd.cut(volatility_data['volatility_spread_change'], bins)
grouped_data = volatility_data.groupby(binned_data)['carry_spread_change'].mean()

plt.scatter(volatility_data['volatility_spread_change'], volatility_data['carry_spread_change'], alpha=0.3, color='blue')
plt.plot([(b.left + b.right)/2 for b in grouped_data.index], grouped_data.values, 'ro-', linewidth=2)
plt.title('Relationship between Changes in Volatility Spread and Changes in Carry Spread')
plt.xlabel('Change in Volatility Spread')
plt.ylabel('Change in Carry Spread')
plt.grid(True, alpha=0.3)
plt.show()

# Implement the Kalman Filter for the Dynamic Linear Model
class DynamicLinearModel:
    def __init__(self, lookback_window=252):
        self.lookback_window = lookback_window
        # Parameters for the state space model
        self.G = None  # State transition matrix
        self.F = None  # Observation matrix
        self.W = None  # State noise covariance
        self.V = None  # Observation noise covariance
        self.mean_0 = None  # Initial state mean
        self.cov_0 = None  # Initial state covariance
        
        # State estimates
        self.filtered_state_mean = []
        self.filtered_state_cov = []
        self.predicted_state_mean = []
        self.predicted_state_cov = []
        
        # Prediction results
        self.predicted_obs_mean = []
        self.predicted_obs_cov = []
        
    def initialize_parameters(self, vol_spread, carry_spread):
        # State vector: (volatility_spread_t, long_term_mean_t, volatility_spread_t-1, carry_spread_t)
        n_states = 4
        n_obs = 2  # volatility_spread and carry_spread
        
        # Estimate initial parameters
        self.lambda_est = 0.05  # Rate of mean reversion
        self.gamma_est = -0.3   # Relationship between vol spread and carry spread
        
        # State transition matrix G
        self.G = np.array([
            [1 - self.lambda_est, self.lambda_est, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [self.gamma_est, 0, -self.gamma_est, 1]
        ])
        
        # Observation matrix F
        self.F = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        # State noise covariance W (diagonal for simplicity)
        self.W = np.diag([0.1, 0.0001, 0, 0.1])  # Low variance for long-term mean
        
        # Observation noise covariance V (diagonal for simplicity)
        self.V = np.diag([0.05, 0.05])
        
        # Initial state
        self.mean_0 = np.array([vol_spread.iloc[0], vol_spread.iloc[0], 0, carry_spread.iloc[0]])
        self.cov_0 = np.eye(n_states) * 0.1
        
        # Initialize Kalman filter arrays
        self.filtered_state_mean = [self.mean_0]
        self.filtered_state_cov = [self.cov_0]
        self.predicted_state_mean = []
        self.predicted_state_cov = []
        self.predicted_obs_mean = []
        self.predicted_obs_cov = []
    
    def kalman_filter_step(self, observation):
        """Single step of Kalman Filter"""
        # Last filtered state
        m_prev = self.filtered_state_mean[-1]
        C_prev = self.filtered_state_cov[-1]
        
        # Predict state
        a_t = self.G @ m_prev
        R_t = self.G @ C_prev @ self.G.T + self.W
        
        # Predict observation
        f_t = self.F @ a_t
        Q_t = self.F @ R_t @ self.F.T + self.V
        
        # Update state (filter)
        K_t = R_t @ self.F.T @ np.linalg.inv(Q_t)  # Kalman gain
        m_t = a_t + K_t @ (observation - f_t)
        C_t = R_t - K_t @ Q_t @ K_t.T
        
        # Store predictions
        self.predicted_state_mean.append(a_t)
        self.predicted_state_cov.append(R_t)
        self.predicted_obs_mean.append(f_t)
        self.predicted_obs_cov.append(Q_t)
        
        # Store filtered state
        self.filtered_state_mean.append(m_t)
        self.filtered_state_cov.append(C_t)
        
        return f_t, Q_t
    
    def fit(self, vol_spread, carry_spread):
        """Fit the model with historical data"""
        self.initialize_parameters(vol_spread, carry_spread)
        
        # Run Kalman filter
        for i in range(1, len(vol_spread)):
            observation = np.array([vol_spread.iloc[i], carry_spread.iloc[i]])
            self.kalman_filter_step(observation)
    
    def predict_next(self):
        """Predict the next observation"""
        if not self.predicted_obs_mean:
            raise ValueError("Model has not been fitted yet")
            
        return self.predicted_obs_mean[-1], self.predicted_obs_cov[-1]

# Function to forecast PnL using the DLM model
def forecast_pnl(model, vol_spread, carry_spread, current_idx, days_ahead=1):
    """
    Forecast the PnL for a volatility spread strategy using the DLM model.
    Returns the expected PnL and its uncertainty.
    """
    pred_mean, pred_cov = model.predict_next()
    
    # Expected change in volatility spread
    exp_vol_spread_change = pred_mean[0] - vol_spread.iloc[current_idx]
    
    # Expected carry spread
    exp_carry_spread = pred_mean[1]
    
    # Expected PnL components
    pnl_vol_change = exp_vol_spread_change
    pnl_carry = exp_carry_spread
    
    # Total expected PnL
    exp_pnl = pnl_vol_change + pnl_carry
    
    # Uncertainty of PnL
    pnl_uncertainty = np.sqrt(pred_cov[0, 0] + pred_cov[1, 1])
    
    return exp_pnl, pnl_uncertainty

# Implement the volatility spread trading strategy
def volatility_spread_strategy(vol_data, window_size=252, max_weight=0.5, tcost_vix=0.02, tcost_v2x=0.03):
    """
    Implementation of the volatility spread strategy using the DLM model.
    
    Parameters:
    - vol_data: DataFrame with volatility and carry data
    - window_size: Lookback window for model estimation
    - max_weight: Maximum position size for each leg
    - tcost_vix: Transaction cost for VIX futures (in volatility points)
    - tcost_v2x: Transaction cost for V2X futures (in volatility points)
    
    Returns:
    - DataFrame with strategy positions, PnL components, and cumulative returns
    """
    # Create a strategy DataFrame
    strategy_df = pd.DataFrame(index=vol_data.index)
    strategy_df['volatility_spread'] = vol_data['volatility_spread']
    strategy_df['carry_spread'] = vol_data['carry_spread']
    
    # Initialize position weights
    strategy_df['vix_position'] = 0.0
    strategy_df['v2x_position'] = 0.0
    
    # Initialize signals
    strategy_df['signal_strength'] = 0.0
    
    # Initialize PnL components
    strategy_df['pnl_vol_change'] = 0.0
    strategy_df['pnl_carry'] = 0.0
    strategy_df['pnl_total_before_cost'] = 0.0
    strategy_df['transaction_cost'] = 0.0
    strategy_df['pnl_total_after_cost'] = 0.0
    
    # Only start trading after the initial window
    start_idx = window_size
    
    # Create DLM model
    dlm_model = DynamicLinearModel(lookback_window=window_size)
    
    # Run strategy
    for i in range(start_idx, len(vol_data)):
        # Fit model using lookback window
        lookback_vol_spread = vol_data['volatility_spread'].iloc[i-window_size:i]
        lookback_carry_spread = vol_data['carry_spread'].iloc[i-window_size:i]
        
        dlm_model.fit(lookback_vol_spread, lookback_carry_spread)
        
        # Get trading signal
        exp_pnl, pnl_uncertainty = forecast_pnl(dlm_model, vol_data['volatility_spread'], vol_data['carry_spread'], i-1)
        signal = exp_pnl / pnl_uncertainty if pnl_uncertainty > 0 else 0
        
        strategy_df.loc[vol_data.index[i], 'signal_strength'] = signal
        
        # Determine positions based on signal
        if signal > 0:  # Long VIX, short V2X
            vix_pos = min(signal, max_weight)
            v2x_pos = max(-signal, -max_weight)
        else:  # Short VIX, long V2X
            vix_pos = max(signal, -max_weight)
            v2x_pos = min(-signal, max_weight)
        
        # Position size constraints: change by at most 0.2 per day
        prev_vix_pos = strategy_df['vix_position'].iloc[i-1]
        prev_v2x_pos = strategy_df['v2x_position'].iloc[i-1]
        
        delta_vix = vix_pos - prev_vix_pos
        delta_v2x = v2x_pos - prev_v2x_pos
        
        if abs(delta_vix) > 0.2:
            delta_vix = 0.2 * np.sign(delta_vix)
        if abs(delta_v2x) > 0.2:
            delta_v2x = 0.2 * np.sign(delta_v2x)
        
        vix_pos = prev_vix_pos + delta_vix
        v2x_pos = prev_v2x_pos + delta_v2x
        
        # Minimum position size constraint (0.05)
        if abs(vix_pos) < 0.05:
            vix_pos = 0.05 * np.sign(vix_pos) if vix_pos != 0 else 0.05
        if abs(v2x_pos) < 0.05:
            v2x_pos = 0.05 * np.sign(v2x_pos) if v2x_pos != 0 else 0.05
        
        strategy_df.loc[vol_data.index[i], 'vix_position'] = vix_pos
        strategy_df.loc[vol_data.index[i], 'v2x_position'] = v2x_pos
        
        # Calculate PnL components
        if i > start_idx:
            vol_change = vol_data['volatility_spread'].iloc[i] - vol_data['volatility_spread'].iloc[i-1]
            carry = vol_data['carry_spread'].iloc[i-1]
            roll_pct = vol_data['VIX_roll_pct'].iloc[i-1]  # Assuming same for both
            
            # PnL due to change in volatility spread
            pnl_vol_change = vix_pos * (vol_data['VIX_1M_constant'].iloc[i] - vol_data['VIX_1M_constant'].iloc[i-1]) + \
                            v2x_pos * (vol_data['V2X_1M_constant'].iloc[i] - vol_data['V2X_1M_constant'].iloc[i-1])
            
            # PnL due to carry
            pnl_carry = vix_pos * (-roll_pct * vol_data['VIX_carry'].iloc[i-1]) + \
                        v2x_pos * (-roll_pct * vol_data['V2X_carry'].iloc[i-1])
            
            # Total PnL before transaction costs
            pnl_total_before_cost = pnl_vol_change + pnl_carry
            
            # Transaction costs (changes in position + daily rolling)
            tcost = abs(delta_vix) * tcost_vix + abs(delta_v2x) * tcost_v2x + \
                    roll_pct * (abs(vix_pos) * tcost_vix + abs(v2x_pos) * tcost_v2x)
            
            # Total PnL after transaction costs
            pnl_total_after_cost = pnl_total_before_cost - tcost
            
            # Store PnL components
            strategy_df.loc[vol_data.index[i], 'pnl_vol_change'] = pnl_vol_change
            strategy_df.loc[vol_data.index[i], 'pnl_carry'] = pnl_carry
            strategy_df.loc[vol_data.index[i], 'pnl_total_before_cost'] = pnl_total_before_cost
            strategy_df.loc[vol_data.index[i], 'transaction_cost'] = tcost
            strategy_df.loc[vol_data.index[i], 'pnl_total_after_cost'] = pnl_total_after_cost
    
    # Calculate cumulative returns
    strategy_df['cum_pnl_vol_change'] = strategy_df['pnl_vol_change'].cumsum()
    strategy_df['cum_pnl_carry'] = strategy_df['pnl_carry'].cumsum()
    strategy_df['cum_pnl_before_cost'] = strategy_df['pnl_total_before_cost'].cumsum()
    strategy_df['cum_pnl_after_cost'] = strategy_df['pnl_total_after_cost'].cumsum()
    
    return strategy_df

# Run the strategy
strategy_results = volatility_spread_strategy(volatility_data, window_size=252)

# Calculate strategy performance metrics
def calculate_performance_metrics(returns, period="daily"):
    """Calculate performance metrics for a strategy"""
    if period == "daily":
        ann_factor = 252
    elif period == "weekly":
        ann_factor = 52
    else:
        ann_factor = 12  # monthly
    
    # Daily returns
    daily_returns = returns.diff()
    
    # Annualized return
    ann_return = daily_returns.mean() * ann_factor
    
    # Annualized volatility
    ann_vol = daily_returns.std() * np.sqrt(ann_factor)
    
    # Information ratio
    ir = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = returns.copy()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Hit ratio
    hit_ratio = (daily_returns > 0).mean()
    
    # Sortino ratio (using downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(ann_factor)
    sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar ratio
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Information Ratio": ir,
        "Maximum Drawdown": max_drawdown,
        "Hit Ratio": hit_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio
    }

# Calculate performance metrics
performance_metrics = calculate_performance_metrics(strategy_results['cum_pnl_after_cost'])
performance_metrics_no_cost = calculate_performance_metrics(strategy_results['cum_pnl_before_cost'])

# Display performance metrics
metrics_df = pd.DataFrame({
    'With Transaction Costs': performance_metrics,
    'Without Transaction Costs': performance_metrics_no_cost
})
print("\nStrategy Performance Metrics:")
print(metrics_df)

# Plot strategy results
plt.figure(figsize=(15, 12))

# Plot cumulative PnL components
plt.subplot(3, 1, 1)
plt.plot(strategy_results.index, strategy_results['cum_pnl_vol_change'], 'g-', label='PnL from Volatility Change')
plt.plot(strategy_results.index, strategy_results['cum_pnl_carry'], 'orange', label='PnL from Carry')
plt.plot(strategy_results.index, strategy_results['cum_pnl_after_cost'], 'k-', label='Total PnL (after costs)')
plt.title('Cumulative PnL Components')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot positions
plt.subplot(3, 1, 2)
plt.plot(strategy_results.index, strategy_results['vix_position'], 'b-', label='VIX Position')
plt.plot(strategy_results.index, strategy_results['v2x_position'], 'r-', label='V2X Position')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Strategy Positions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot signal strength
plt.subplot(3, 1, 3)
plt.plot(strategy_results.index, strategy_results['signal_strength'], 'purple', label='Signal Strength')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Signal Strength')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze relationship between strategy returns and market conditions
# Create simulated market returns (S&P 500)
np.random.seed(42)
market_returns = pd.Series(np.random.normal(0.0005, 0.01, len(strategy_results)), index=strategy_results.index)
market_returns.iloc[spike_indices] *= 1.5  # Larger market moves during volatility spikes
market_cumulative = (1 + market_returns).cumprod()

# Calculate strategy returns
strategy_returns = strategy_results['pnl_total_after_cost']

# Calculate correlations
correlation = strategy_returns.corr(market_returns)

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(market_returns, strategy_returns, alpha=0.5)
plt.title(f'Strategy Returns vs Market Returns (Correlation: {correlation:.3f})')
plt.xlabel('Market Returns')
plt.ylabel('Strategy Returns')
plt.grid(True, alpha=0.3)
plt.show()

# Analyze how the strategy performs during different market conditions
percentiles = np.percentile(market_returns, [10, 20, 30, 40, 50, 60, 70, 80, 90])
market_bins = pd.cut(market_returns, [-np.inf] + list(percentiles) + [np.inf], labels=range(10))
avg_returns_by_bin = strategy_returns.groupby(market_bins).mean()

plt.figure(figsize=(12, 6))
avg_returns_by_bin.plot(kind='bar')
plt.title('Average Strategy Returns by Market Return Decile')
plt.xlabel('Market Return Decile (0=Worst, 9=Best)')
plt.ylabel('Average Strategy Return')
plt.grid(True, alpha=0.3)
plt.show()

# Create a more detailed analysis of strategy performance during market stress periods
# Define stress periods as days with the largest 5% of negative market returns
stress_threshold = np.percentile(market_returns, 5)
stress_days = market_returns < stress_threshold

stress_performance = pd.DataFrame({
    'Market Returns': market_returns[stress_days],
    'Strategy Returns': strategy_returns[stress_days]
})

print("\nStrategy Performance During Market Stress Periods:")
print(f"Average Market Return: {stress_performance['Market Returns'].mean():.4f}")
print(f"Average Strategy Return: {stress_performance['Strategy Returns'].mean():.4f}")
print(f"Strategy Hit Rate During Stress: {(stress_performance['Strategy Returns'] > 0).mean():.2f}")

# Show the distribution of strategy returns during normal vs. stress periods
plt.figure(figsize=(12, 6))
# Use distplot instead of histplot for older seaborn versions
sns.distplot(strategy_returns[~stress_days], kde=True, color='blue', hist=True, label='Normal Periods')
sns.distplot(strategy_returns[stress_days], kde=True, color='red', hist=True, label='Stress Periods')
plt.title('Distribution of Strategy Returns: Normal vs. Stress Periods')
plt.xlabel('Strategy Returns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Optional: Implement VIX/VNKY strategy
# This is similar to the VIX/V2X strategy but with focus on hedging against VNKY spikes
# We could implement this as described in the paper by always shorting VIX and longing VNKY
# with position sizes determined by the DLM model predictions

# To simulate VNKY data and implement the strategy:
def simulate_vnky_data():
    """Simulate VNKY data similar to V2X but with higher volatility and steeper term structure"""
    # Base parameters
    vnky_mean = 22.0
    vnky_vol = 0.10
    vnky_reversion = 0.04
    
    # Correlation with VIX
    correlation_with_vix = 0.7
    
    # Simulate path
    vnky = np.zeros(num_days)
    vnky[0] = vnky_mean
    
    # Correlated random shocks with VIX
    for i in range(1, num_days):
        # Create correlated shock
        shock = correlation_with_vix * (vix[i] - vix[i-1])/vix[i-1] + \
                (1-correlation_with_vix) * np.random.normal(0, vnky_vol)
                
        # Apply mean reversion
        vnky[i] = vnky[i-1] + vnky_reversion * (vnky_mean - vnky[i-1]) + vnky[i-1] * shock
    
    # Ensure volatility doesn't go negative
    vnky = np.maximum(vnky, 5.0)
    
    # Create spikes (more severe than VIX/V2X)
    for idx in spike_indices:
        vnky_spike_magnitude = np.random.uniform(0.5, 1.5)  # Larger spikes for VNKY
        vnky[idx:idx+20] = vnky[idx:idx+20] * (1 + vnky_spike_magnitude * np.exp(-0.2 * np.arange(20)))
    
    # Simulate futures
    vnky_1m = vnky.copy()
    vnky_2m = np.zeros(num_days)
    
    # Generate futures prices with steeper term structure
    for i in range(num_days):
        term_premium_vnky = max(0.8, 0.15 * vnky[i]) * np.random.normal(1, 0.1)
        vnky_2m[i] = vnky_1m[i] + term_premium_vnky
        
        # During spikes, futures curve can become inverted
        if i in spike_indices or i in [idx+j for idx in spike_indices for j in range(1, 10)]:
            if np.random.random() < 0.7:
                vnky_2m[i] = vnky_1m[i] - abs(np.random.normal(0.3, 0.1) * vnky_1m[i])
    
    return vnky, vnky_1m, vnky_2m

# The VIX/VNKY strategy implementation would be similar to the VIX/V2X strategy
# but with position adjustments to always maintain a short VIX / long VNKY bias