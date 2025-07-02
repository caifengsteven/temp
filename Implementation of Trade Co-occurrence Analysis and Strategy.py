import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

#####################################
# Phase 1: High-Frequency Trade Data Simulation
#####################################

def simulate_high_frequency_trades(
    n_stocks=100,  # In the paper they use 457 S&P 500 constituents
    n_days=50,     # Paper uses ~4 years
    avg_trades_per_stock_per_day=2000,  # Adjust based on desired intensity
    time_precision_ns=True,  # Use nanosecond precision as in LOBSTER data
    u_shape_intensity=True,  # Apply U-shape for intraday trading intensity
    correlated_direction=True,  # Correlated buying/selling direction across stocks
):
    """
    Simulate high-frequency trade data for multiple stocks over multiple days.
    
    Returns:
    - trades_df: DataFrame of all trades with timestamp, stock_id, direction, size
    - daily_returns_df: DataFrame of daily returns for each stock
    """
    print(f"Simulating high-frequency trade data for {n_stocks} stocks over {n_days} days...")
    
    # Initialize containers for all trades and daily returns
    all_trades = []
    daily_returns = []
    
    # Define trading hours (6.5 hours = 23,400 seconds)
    trading_seconds = 6.5 * 60 * 60
    
    # Create trading days (use actual calendar to avoid weekends)
    start_date = datetime(2020, 1, 1)
    trading_days = []
    current_date = start_date
    while len(trading_days) < n_days:
        if current_date.weekday() < 5:  # Monday-Friday
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    # Simulate market-wide sentiment for correlated direction
    if correlated_direction:
        # Daily market sentiment (autocorrelated)
        market_sentiment = np.zeros(n_days)
        market_sentiment[0] = np.random.normal(0, 1)
        for i in range(1, n_days):
            # AR(1) process for market sentiment
            market_sentiment[i] = 0.7 * market_sentiment[i-1] + 0.3 * np.random.normal(0, 1)
        
        # Intraday market sentiment (finer granularity, 1-minute intervals)
        minutes_per_day = int(trading_seconds / 60)
        intraday_sentiment = {}
        
        for day_idx, day in enumerate(trading_days):
            daily_sent = market_sentiment[day_idx]
            intraday_sent = np.zeros(minutes_per_day)
            intraday_sent[0] = daily_sent + np.random.normal(0, 0.2)
            
            for m in range(1, minutes_per_day):
                # More volatile at open and close
                volatility = 0.2
                if m < 30 or m > minutes_per_day - 30:
                    volatility = 0.4
                
                # AR(1) process for intraday sentiment
                intraday_sent[m] = 0.9 * intraday_sent[m-1] + 0.1 * np.random.normal(0, volatility)
            
            intraday_sentiment[day] = intraday_sent
    
    # Simulate trades and returns for each day
    for day_idx, day in enumerate(tqdm(trading_days, desc="Simulating trading days")):
        # Daily factors that affect all stocks (market, size, value, etc.)
        market_return = np.random.normal(0.0005, 0.01)  # ~12% annual return, 16% vol
        size_factor = np.random.normal(0.0001, 0.005)
        value_factor = np.random.normal(0.0001, 0.005)
        
        # For each stock
        for stock_id in range(n_stocks):
            # Stock-specific parameters
            stock_size = np.random.uniform(0.5, 2.0)  # Market cap relative to average
            stock_value = np.random.uniform(0.5, 2.0)  # Value factor loading
            stock_beta = np.random.uniform(0.7, 1.3)   # Market beta
            stock_volatility = np.random.uniform(0.015, 0.035)  # Idiosyncratic volatility
            
            # Determine number of trades for this stock on this day
            # Larger stocks have more trades
            stock_trades_multiplier = np.sqrt(stock_size)
            mean_trades = avg_trades_per_stock_per_day * stock_trades_multiplier
            num_trades = np.random.poisson(mean_trades)
            
            # Generate trade timestamps throughout the day
            if u_shape_intensity:
                # U-shape intensity function for intraday trading
                def intensity_function(t, T=trading_seconds):
                    # Higher intensity at open and close
                    return 1.0 + 0.5 * np.exp(-4 * t / T) + 0.5 * np.exp(-4 * (1 - t / T))
                
                # Generate irregular timestamps with U-shape intensity
                timestamps = []
                t = 0
                while len(timestamps) < num_trades and t < trading_seconds:
                    # Thinning method for non-homogeneous Poisson process
                    lambda_max = 2.0  # Maximum intensity
                    dt = np.random.exponential(1.0 / lambda_max)
                    t += dt
                    
                    if t < trading_seconds:
                        lambda_t = intensity_function(t)
                        if np.random.uniform(0, 1) <= lambda_t / lambda_max:
                            timestamps.append(t)
                
                # If we don't have enough trades, generate the rest uniformly
                if len(timestamps) < num_trades:
                    additional_timestamps = np.random.uniform(0, trading_seconds, num_trades - len(timestamps))
                    timestamps.extend(additional_timestamps)
                
                # Sort timestamps
                timestamps = sorted(timestamps)[:num_trades]
            else:
                # Uniform distribution of trades
                timestamps = sorted(np.random.uniform(0, trading_seconds, num_trades))
            
            # Ensure timestamps are nanosecond precision if required
            if time_precision_ns:
                # Convert to nanoseconds and add random nanosecond offset
                timestamps_ns = [
                    day.replace(hour=9, minute=30) + 
                    timedelta(seconds=int(t)) + 
                    timedelta(microseconds=int((t % 1) * 1e6)) +
                    timedelta(microseconds=np.random.randint(0, 1000) / 1000)
                    for t in timestamps
                ]
            else:
                # Second precision
                timestamps_ns = [
                    day.replace(hour=9, minute=30) + timedelta(seconds=int(t))
                    for t in timestamps
                ]
            
            # Generate trade directions (buy = 1, sell = -1)
            if correlated_direction:
                directions = []
                for ts in timestamps_ns:
                    # Find the minute index for this timestamp
                    minutes_since_open = (ts - day.replace(hour=9, minute=30)).total_seconds() / 60
                    minute_idx = min(int(minutes_since_open), len(intraday_sentiment[day]) - 1)
                    
                    # Get sentiment and add stock-specific bias
                    sentiment = intraday_sentiment[day][minute_idx]
                    stock_specific_bias = np.random.normal(0, 0.5)  # Stock-specific noise
                    
                    # Probability of buy order based on sentiment
                    p_buy = 0.5 + 0.2 * (sentiment + 0.3 * stock_specific_bias)
                    p_buy = max(0.1, min(0.9, p_buy))  # Constrain between 0.1 and 0.9
                    
                    # Generate direction
                    direction = 1 if np.random.random() < p_buy else -1
                    directions.append(direction)
            else:
                # Independent Bernoulli trials
                directions = [1 if np.random.random() < 0.5 else -1 for _ in range(num_trades)]
            
            # Generate trade sizes (log-normal distribution)
            # Typically larger trades for larger stocks
            log_mean = np.log(100 * stock_size)  # Base size of 100 shares scaled by stock size
            log_std = 0.5  # Dispersion parameter
            sizes = np.round(np.random.lognormal(log_mean, log_std, num_trades)).astype(int)
            
            # Ensure minimum size
            sizes = np.maximum(sizes, 1)
            
            # Record trades
            for j in range(num_trades):
                all_trades.append({
                    'timestamp': timestamps_ns[j],
                    'stock_id': stock_id,
                    'direction': directions[j],
                    'size': sizes[j]
                })
            
            # Simulate daily return based on factors and order imbalance
            net_order_imbalance = sum(directions) / len(directions) if directions else 0
            stock_specific_return = np.random.normal(0, stock_volatility)
            
            # Factor model for returns
            factor_return = (
                stock_beta * market_return +
                stock_size * size_factor +
                stock_value * value_factor
            )
            
            # Order imbalance affects returns (market impact)
            impact_coefficient = 0.1  # Strength of market impact
            impact = impact_coefficient * net_order_imbalance
            
            # Combine factor return, order imbalance impact, and idiosyncratic return
            total_return = factor_return + impact + stock_specific_return
            
            # Record daily return
            daily_returns.append({
                'date': day.date(),  # Store as date object, not datetime
                'stock_id': stock_id,
                'return': total_return,
                'market_return': market_return,
                'size_factor': size_factor,
                'value_factor': value_factor
            })
    
    # Create DataFrames
    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    daily_returns_df = pd.DataFrame(daily_returns)
    
    print(f"Generated {len(trades_df):,} trades across {n_stocks} stocks and {n_days} days.")
    return trades_df, daily_returns_df

#####################################
# Phase 2: Trade Co-occurrence Definition
#####################################

def find_cooccurring_trades(trade_xa, all_trades_sorted, delta):
    """
    Find all trades co-occurring with trade_xa within time window delta.
    
    Parameters:
    - trade_xa: Reference trade (row from trades DataFrame)
    - all_trades_sorted: DataFrame of all trades, sorted by timestamp
    - delta: Time window (timedelta object)
    
    Returns:
    - DataFrame of co-occurring trades
    """
    # Define time window
    ta = trade_xa['timestamp']
    t_start = ta - delta
    t_end = ta + delta
    
    # Find indices of trades within the window using searchsorted
    idx_start = all_trades_sorted['timestamp'].searchsorted(t_start, side='left')
    idx_end = all_trades_sorted['timestamp'].searchsorted(t_end, side='right')
    
    # Get trades in the window
    window_trades = all_trades_sorted.iloc[idx_start:idx_end].copy()
    
    # Exclude the reference trade itself
    window_trades = window_trades[window_trades.index != trade_xa.name]
    
    return window_trades

#####################################
# Phase 3: Trade Flow Decomposition
#####################################

def classify_trades(trades_df, market_index_stocks, delta_ms=1):
    """
    Classify trades based on co-occurrence patterns.
    
    Parameters:
    - trades_df: DataFrame of all trades
    - market_index_stocks: List of stock_ids representing the market
    - delta_ms: Size of neighborhood in milliseconds
    
    Returns:
    - trades_df with additional columns 'label' and 'sub_label'
    """
    print(f"Classifying trades with delta = {delta_ms} ms...")
    start_time = time.time()
    
    # Convert delta to timedelta
    delta = pd.Timedelta(milliseconds=delta_ms)
    
    # Create copies of the input DataFrame to avoid modifying the original
    trades_with_labels = trades_df.copy()
    
    # Add classification columns
    trades_with_labels['label'] = 'unclassified'
    trades_with_labels['sub_label'] = None
    
    # For faster lookups, create sets of indices
    market_stocks_set = set(market_index_stocks)
    
    # Process trades in batches by day to manage memory
    # Group by date (day) to process day by day
    trades_with_labels['date'] = trades_with_labels['timestamp'].dt.date
    unique_dates = trades_with_labels['date'].unique()
    
    # Optimization: Use sampling for large datasets
    sample_size = min(5000, len(trades_with_labels))
    if len(trades_with_labels) > 1000000:  # If very large dataset
        # Estimate proportions from a sample
        sample_trades = trades_with_labels.sample(sample_size)
        is_sample = True
        print(f"Dataset too large: using sampling of {sample_size:,} trades")
        process_trades = sample_trades
    else:
        is_sample = False
        process_trades = trades_with_labels
    
    # For progress tracking
    pbar = tqdm(total=len(unique_dates), desc="Classifying trades by day")
    
    # Process each day separately
    for day in unique_dates:
        if is_sample:
            day_trades = process_trades[process_trades['date'] == day]
        else:
            day_trades = trades_with_labels[trades_with_labels['date'] == day]
        
        day_trades_sorted = day_trades.sort_values('timestamp')
        
        # Process each trade in this day
        for idx, trade_xa in day_trades_sorted.iterrows():
            stock_id_xa = trade_xa['stock_id']
            
            # Find co-occurring trades
            B_delta_xa = find_cooccurring_trades(trade_xa, day_trades_sorted, delta)
            
            if B_delta_xa.empty:
                # Isolated trade
                trades_with_labels.at[idx, 'label'] = 'iso'
            else:
                # Non-isolated trade
                trades_with_labels.at[idx, 'label'] = 'nis'
                
                # Check for trades of the same stock
                trades_of_stock_i = B_delta_xa[B_delta_xa['stock_id'] == stock_id_xa]
                
                # Check for trades of other stocks in the market index
                trades_of_other_stocks_in_M = B_delta_xa[
                    (B_delta_xa['stock_id'] != stock_id_xa) & 
                    (B_delta_xa['stock_id'].isin(market_stocks_set))
                ]
                
                # Apply classification logic (Figure 2 in the paper)
                if not trades_of_stock_i.empty and trades_of_other_stocks_in_M.empty:
                    trades_with_labels.at[idx, 'sub_label'] = 'nis-s'
                elif trades_of_stock_i.empty and not trades_of_other_stocks_in_M.empty:
                    trades_with_labels.at[idx, 'sub_label'] = 'nis-c'
                elif not trades_of_stock_i.empty and not trades_of_other_stocks_in_M.empty:
                    trades_with_labels.at[idx, 'sub_label'] = 'nis-b'
        
        pbar.update(1)
    
    pbar.close()
    
    # If we used sampling, fill in unclassified trades with estimated proportions
    if is_sample:
        print("Applying classification proportions to all trades...")
        # Calculate proportions from sample
        label_props = process_trades['label'].value_counts(normalize=True)
        
        # Make sure label_props contains both 'iso' and 'nis' and they sum to 1
        if 'iso' not in label_props:
            label_props['iso'] = 0.0
        if 'nis' not in label_props:
            label_props['nis'] = 0.0
        
        # Normalize to ensure they sum to 1
        total = label_props['iso'] + label_props['nis']
        if total > 0:
            label_props['iso'] /= total
            label_props['nis'] /= total
        else:
            # Default to 50/50 if we have no data
            label_props['iso'] = 0.5
            label_props['nis'] = 0.5
        
        # Calculate sub-label proportions (for 'nis' labels)
        nis_trades = process_trades[process_trades['label'] == 'nis']
        if len(nis_trades) > 0:
            sub_label_props = nis_trades['sub_label'].value_counts(normalize=True)
            
            # Make sure all sub-labels are present
            if 'nis-s' not in sub_label_props:
                sub_label_props['nis-s'] = 0.0
            if 'nis-c' not in sub_label_props:
                sub_label_props['nis-c'] = 0.0
            if 'nis-b' not in sub_label_props:
                sub_label_props['nis-b'] = 0.0
            
            # Normalize to ensure they sum to 1
            total_sub = sub_label_props['nis-s'] + sub_label_props['nis-c'] + sub_label_props['nis-b']
            if total_sub > 0:
                sub_label_props['nis-s'] /= total_sub
                sub_label_props['nis-c'] /= total_sub
                sub_label_props['nis-b'] /= total_sub
            else:
                # Default to equal distribution if we have no data
                sub_label_props['nis-s'] = 1/3
                sub_label_props['nis-c'] = 1/3
                sub_label_props['nis-b'] = 1/3
        else:
            # Default proportions if we have no 'nis' trades
            sub_label_props = pd.Series({'nis-s': 1/3, 'nis-c': 1/3, 'nis-b': 1/3})
        
        # Apply to all trades using random assignment based on proportions
        unclassified_trades = trades_with_labels[trades_with_labels['label'] == 'unclassified']
        for idx in unclassified_trades.index:
            # Assign label based on proportions
            label = np.random.choice(['iso', 'nis'], p=[label_props['iso'], label_props['nis']])
            trades_with_labels.at[idx, 'label'] = label
            
            # Assign sub-label if nis
            if label == 'nis':
                sub_label = np.random.choice(['nis-s', 'nis-c', 'nis-b'], 
                                           p=[sub_label_props['nis-s'], 
                                              sub_label_props['nis-c'], 
                                              sub_label_props['nis-b']])
                trades_with_labels.at[idx, 'sub_label'] = sub_label
    
    # Verification
    assert trades_with_labels['label'].isin(['iso', 'nis']).all(), "Some trades were not classified"
    assert trades_with_labels[trades_with_labels['label'] == 'nis']['sub_label'].isin(['nis-s', 'nis-c', 'nis-b']).all(), "Some non-isolated trades have invalid sub-labels"
    
    # Calculate proportions of each type
    label_counts = trades_with_labels['label'].value_counts(normalize=True)
    print(f"\nTrade Type Proportions:")
    print(f"Isolated (iso): {label_counts.get('iso', 0):.4f}")
    print(f"Non-Isolated (nis): {label_counts.get('nis', 0):.4f}")
    
    sublabel_counts = trades_with_labels['sub_label'].value_counts(normalize=True)
    print(f"Same-stock (nis-s): {sublabel_counts.get('nis-s', 0):.4f}")
    print(f"Cross-stock (nis-c): {sublabel_counts.get('nis-c', 0):.4f}")
    print(f"Both (nis-b): {sublabel_counts.get('nis-b', 0):.4f}")
    
    print(f"Classification completed in {time.time() - start_time:.2f} seconds.")
    
    return trades_with_labels

#####################################
# Phase 4: Conditional Order Imbalance (COI) Calculation
#####################################

def calculate_daily_coi(daily_stock_trades, trade_type):
    """
    Calculate Conditional Order Imbalance for a specific stock on a specific day.
    
    Parameters:
    - daily_stock_trades: DataFrame of trades for a specific stock on a specific day
    - trade_type: String specifying the trade type ('all', 'iso', 'nis', 'nis-s', 'nis-c', 'nis-b')
    
    Returns:
    - COI value
    """
    if trade_type == 'all':
        filtered_trades = daily_stock_trades
    elif trade_type == 'iso':
        filtered_trades = daily_stock_trades[daily_stock_trades['label'] == 'iso']
    elif trade_type == 'nis':
        filtered_trades = daily_stock_trades[daily_stock_trades['label'] == 'nis']
    else:  # 'nis-s', 'nis-c', 'nis-b'
        filtered_trades = daily_stock_trades[daily_stock_trades['sub_label'] == trade_type]
    
    # If using trade volume rather than count
    # N_type_buy = filtered_trades[filtered_trades['direction'] == 1]['size'].sum()
    # N_type_sell = filtered_trades[filtered_trades['direction'] == -1]['size'].sum()
    
    # Using trade count as in the paper
    N_type_buy = len(filtered_trades[filtered_trades['direction'] == 1])
    N_type_sell = len(filtered_trades[filtered_trades['direction'] == -1])
    
    # Calculate COI
    if N_type_buy + N_type_sell == 0:
        return 0
    else:
        return (N_type_buy - N_type_sell) / (N_type_buy + N_type_sell)

def calculate_all_cois(trades_df):
    """
    Calculate all COIs for each stock and day.
    
    Parameters:
    - trades_df: DataFrame of classified trades
    
    Returns:
    - DataFrame of daily COIs for each stock
    """
    print("Calculating daily Conditional Order Imbalances (COIs)...")
    
    # Add date column if not already present
    if 'date' not in trades_df.columns:
        trades_df['date'] = trades_df['timestamp'].dt.date
    
    # Initialize list to store COI data
    coi_data = []
    
    # Get unique stock_ids and dates
    unique_stocks = trades_df['stock_id'].unique()
    unique_dates = trades_df['date'].unique()
    
    # Trade types to calculate COI for
    trade_types = ['all', 'iso', 'nis', 'nis-s', 'nis-c', 'nis-b']
    
    # For progress tracking
    total_combinations = len(unique_stocks) * len(unique_dates)
    pbar = tqdm(total=total_combinations, desc="Calculating COIs")
    
    # Calculate COIs for each stock and day
    for stock_id in unique_stocks:
        for date in unique_dates:
            # Get trades for this stock on this day
            daily_stock_trades = trades_df[(trades_df['stock_id'] == stock_id) & (trades_df['date'] == date)]
            
            # If no trades, continue to next combination
            if len(daily_stock_trades) == 0:
                pbar.update(1)
                continue
            
            # Calculate COIs for all trade types
            coi_values = {}
            for trade_type in trade_types:
                coi_values[f'COI_{trade_type}'] = calculate_daily_coi(daily_stock_trades, trade_type)
            
            # Store COI data
            coi_entry = {'date': date, 'stock_id': stock_id}
            coi_entry.update(coi_values)
            coi_data.append(coi_entry)
            
            pbar.update(1)
    
    pbar.close()
    
    # Create DataFrame of COIs
    coi_df = pd.DataFrame(coi_data)
    
    print(f"Calculated COIs for {len(coi_df):,} stock-day combinations.")
    return coi_df

#####################################
# Phase 5: Empirical Selection of delta
#####################################

def calculate_empirical_proportions(trades_df, interval_minutes=5):
    """
    Calculate empirical proportions of trade types for each interval.
    
    Parameters:
    - trades_df: DataFrame of classified trades
    - interval_minutes: Length of intervals in minutes
    
    Returns:
    - Dictionary of average proportions for each trade type
    """
    # Add datetime components if needed
    if 'date' not in trades_df.columns:
        trades_df['date'] = trades_df['timestamp'].dt.date
    
    trades_df['time'] = trades_df['timestamp'].dt.time
    trades_df['minute'] = trades_df['timestamp'].dt.hour * 60 + trades_df['timestamp'].dt.minute
    
    # Define intervals
    interval_minutes = 5
    start_minute = 9 * 60 + 30  # 9:30 AM
    end_minute = 16 * 60        # 4:00 PM
    intervals = [(i, min(i + interval_minutes, end_minute)) 
                for i in range(start_minute, end_minute, interval_minutes)]
    
    # Initialize counters
    interval_props = []
    
    # Calculate proportions for each interval
    for start, end in intervals:
        interval_trades = trades_df[(trades_df['minute'] >= start) & (trades_df['minute'] < end)]
        
        if len(interval_trades) == 0:
            continue
            
        # Count trade types
        n_total = len(interval_trades)
        n_iso = len(interval_trades[interval_trades['label'] == 'iso'])
        n_nis_s = len(interval_trades[interval_trades['sub_label'] == 'nis-s'])
        n_nis_c = len(interval_trades[interval_trades['sub_label'] == 'nis-c'])
        n_nis_b = len(interval_trades[interval_trades['sub_label'] == 'nis-b'])
        
        # Calculate proportions
        props = {
            'iso': n_iso / n_total if n_total > 0 else 0,
            'nis-s': n_nis_s / n_total if n_total > 0 else 0,
            'nis-c': n_nis_c / n_total if n_total > 0 else 0,
            'nis-b': n_nis_b / n_total if n_total > 0 else 0
        }
        
        interval_props.append(props)
    
    # Calculate average proportions
    avg_props = {}
    for type_key in ['iso', 'nis-s', 'nis-c', 'nis-b']:
        avg_props[type_key] = np.mean([p[type_key] for p in interval_props])
    
    return avg_props

def calculate_null_model_probabilities(trades_df, delta_ms, interval_minutes=5):
    """
    Calculate null model probabilities of trade types for each interval.
    
    Parameters:
    - trades_df: DataFrame of trades
    - delta_ms: Size of neighborhood in milliseconds
    - interval_minutes: Length of intervals in minutes
    
    Returns:
    - Dictionary of average null probabilities for each trade type
    """
    # Add datetime components if needed
    if 'date' not in trades_df.columns:
        trades_df['date'] = trades_df['timestamp'].dt.date
        
    trades_df['time'] = trades_df['timestamp'].dt.time
    trades_df['minute'] = trades_df['timestamp'].dt.hour * 60 + trades_df['timestamp'].dt.minute
    
    # Define intervals
    interval_minutes = 5
    start_minute = 9 * 60 + 30  # 9:30 AM
    end_minute = 16 * 60        # 4:00 PM
    intervals = [(i, min(i + interval_minutes, end_minute)) 
                for i in range(start_minute, end_minute, interval_minutes)]
    
    # Convert delta to nanoseconds
    delta_ns = delta_ms * 1e6  # milliseconds to nanoseconds
    
    # Initialize lists for null probabilities
    interval_probs = []
    
    # Calculate null probabilities for each interval
    for start, end in intervals:
        interval_trades = trades_df[(trades_df['minute'] >= start) & (trades_df['minute'] < end)]
        
        if len(interval_trades) == 0:
            continue
            
        # Calculate interval duration in nanoseconds
        T_interval = interval_minutes * 60 * 1e9  # minutes to nanoseconds
        
        # Calculate p_val (probability of a trade being in delta window)
        p_val = (2 * delta_ns) / T_interval
        
        # Group trades by stock_id to calculate N_i and N_M-i
        stock_counts = interval_trades.groupby('stock_id').size()
        
        # Calculate null probabilities for each stock
        stock_probs = []
        
        for stock_id, N_i in stock_counts.items():
            # Number of trades for other stocks in the market
            N_not_i = stock_counts.sum() - N_i
            
            # Calculate null probabilities (Eq. 5 in paper)
            P_iso = (1 - p_val) ** (N_i - 1) * (1 - p_val) ** N_not_i
            P_nis_s = (1 - (1 - p_val) ** (N_i - 1)) * (1 - p_val) ** N_not_i
            P_nis_c = (1 - p_val) ** (N_i - 1) * (1 - (1 - p_val) ** N_not_i)
            P_nis_b = (1 - (1 - p_val) ** (N_i - 1)) * (1 - (1 - p_val) ** N_not_i)
            
            # Store weighted by number of trades
            stock_probs.append({
                'iso': P_iso * N_i,
                'nis-s': P_nis_s * N_i,
                'nis-c': P_nis_c * N_i,
                'nis-b': P_nis_b * N_i,
                'weight': N_i
            })
        
        # Calculate weighted average probabilities for this interval
        total_weight = sum(p['weight'] for p in stock_probs)
        
        if total_weight > 0:
            interval_prob = {}
            for type_key in ['iso', 'nis-s', 'nis-c', 'nis-b']:
                interval_prob[type_key] = sum(p[type_key] for p in stock_probs) / total_weight
            
            interval_probs.append(interval_prob)
    
    # Calculate average null probabilities
    avg_probs = {}
    for type_key in ['iso', 'nis-s', 'nis-c', 'nis-b']:
        avg_probs[type_key] = np.mean([p[type_key] for p in interval_probs])
    
    return avg_probs

def select_optimal_delta(trades_df, delta_candidates_ms):
    """
    Select optimal delta based on distance between empirical and null model proportions.
    
    Parameters:
    - trades_df: DataFrame of trades
    - delta_candidates_ms: List of delta values to evaluate (in milliseconds)
    
    Returns:
    - optimal_delta: The delta with maximum distance metric
    - results: DataFrame with distance metrics for each delta
    """
    print("Selecting optimal delta value...")
    
    # Initialize results
    results = []
    
    # For each candidate delta
    for delta_ms in delta_candidates_ms:
        print(f"Evaluating delta = {delta_ms} ms...")
        
        # Classify trades with this delta
        classified_trades = classify_trades(trades_df, list(range(trades_df['stock_id'].nunique())), delta_ms)
        
        # Calculate empirical proportions
        emp_props = calculate_empirical_proportions(classified_trades)
        
        # Calculate null model probabilities
        null_probs = calculate_null_model_probabilities(trades_df, delta_ms)
        
        # Calculate distance metric
        distance = 0
        for type_key in ['iso', 'nis-s', 'nis-c', 'nis-b']:
            distance += emp_props[type_key] * abs(emp_props[type_key] - null_probs[type_key])
        
        # Store results
        results.append({
            'delta_ms': delta_ms,
            'distance': distance,
            'emp_iso': emp_props['iso'],
            'null_iso': null_probs['iso'],
            'emp_nis-s': emp_props['nis-s'],
            'null_nis-s': null_probs['nis-s'],
            'emp_nis-c': emp_props['nis-c'],
            'null_nis-c': null_probs['nis-c'],
            'emp_nis-b': emp_props['nis-b'],
            'null_nis-b': null_probs['nis-b']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal delta
    optimal_delta = results_df.loc[results_df['distance'].idxmax(), 'delta_ms']
    
    print(f"Optimal delta selected: {optimal_delta} ms")
    
    return optimal_delta, results_df

#####################################
# Phase 6: Regression Analysis
#####################################

def prepare_regression_data(coi_df, returns_df):
    """
    Prepare data for regression analysis.
    
    Parameters:
    - coi_df: DataFrame of daily COIs
    - returns_df: DataFrame of daily returns
    
    Returns:
    - reg_df: DataFrame for regression analysis
    """
    print("Preparing regression data...")
    
    # Ensure date columns have the same type in both DataFrames
    coi_df['date'] = pd.to_datetime(coi_df['date']).dt.date
    returns_df['date'] = pd.to_datetime(returns_df['date']).dt.date
    
    # Merge COIs and returns
    reg_df = pd.merge(coi_df, returns_df, on=['date', 'stock_id'])
    
    # Calculate market excess returns
    reg_df['market_excess_return'] = reg_df.groupby('date')['return'].transform('mean')
    reg_df['excess_return'] = reg_df['return'] - reg_df['market_excess_return']
    
    # Lag COIs for predictive regressions
    for col in reg_df.columns:
        if col.startswith('COI_'):
            reg_df[f'{col}_lag1'] = reg_df.groupby('stock_id')[col].shift(1)
    
    # Lag returns for return momentum control
    reg_df['return_lag1'] = reg_df.groupby('stock_id')['return'].shift(1)
    
    # Create panel data index
    reg_df = reg_df.set_index(['date', 'stock_id'])
    
    print(f"Prepared regression data with {len(reg_df)} observations.")
    return reg_df

def run_contemporaneous_regression(reg_df, coi_type=None):
    """
    Run contemporaneous regression of returns on COIs.
    
    Parameters:
    - reg_df: DataFrame for regression analysis
    - coi_type: Specific COI type to use, or None for all types
    
    Returns:
    - results: Regression results
    """
    # Define dependent variable
    dependent = reg_df['excess_return']
    
    # Define exogenous variables
    if coi_type is None:
        # All COI types
        exog_vars = [col for col in reg_df.columns if col.startswith('COI_') and not col.endswith('_lag1')]
    else:
        # Specific COI type
        exog_vars = [f'COI_{coi_type}']
    
    # Add control variables
    controls = ['market_return', 'size_factor', 'value_factor']
    for control in controls:
        if control in reg_df.columns:
            exog_vars.append(control)
    
    # Check if exog_vars exist in the DataFrame
    exog_vars = [var for var in exog_vars if var in reg_df.columns]
    
    # Run regression
    exog = sm.add_constant(reg_df[exog_vars])
    model = sm.OLS(dependent, exog)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df.index.get_level_values(0)})
    
    return results

def run_predictive_regression(reg_df, coi_type=None):
    """
    Run predictive regression of future returns on lagged COIs.
    
    Parameters:
    - reg_df: DataFrame for regression analysis
    - coi_type: Specific COI type to use, or None for all types
    
    Returns:
    - results: Regression results
    """
    # Define dependent variable
    dependent = reg_df['excess_return']
    
    # Define exogenous variables
    if coi_type is None:
        # All COI types
        exog_vars = [col for col in reg_df.columns if col.startswith('COI_') and col.endswith('_lag1')]
    else:
        # Specific COI type
        exog_vars = [f'COI_{coi_type}_lag1']
    
    # Add control variables
    controls = ['return_lag1', 'market_return', 'size_factor', 'value_factor']
    for control in controls:
        if control in reg_df.columns:
            exog_vars.append(control)
    
    # Check if exog_vars exist in the DataFrame
    exog_vars = [var for var in exog_vars if var in reg_df.columns]
    
    # Run regression
    exog = sm.add_constant(reg_df[exog_vars].dropna())
    dependent = dependent.loc[exog.index]  # Align indices
    
    model = sm.OLS(dependent, exog)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': exog.index.get_level_values(0)})
    
    return results

#####################################
# Phase 7: Portfolio Sorting and Trading Strategy
#####################################

def single_sort_portfolios(reg_df, coi_type, n_quantiles=5):
    """
    Create single-sorted portfolios based on COI.
    
    Parameters:
    - reg_df: DataFrame for regression analysis
    - coi_type: COI type to sort on
    - n_quantiles: Number of quantiles for sorting
    
    Returns:
    - portfolio_returns: DataFrame of portfolio returns
    """
    # Reset index for easier manipulation
    data = reg_df.reset_index()
    
    # Initialize portfolio returns
    portfolio_returns = []
    
    # Get unique dates
    unique_dates = data['date'].unique()
    
    # For each date
    for date in unique_dates[1:]:  # Skip first date due to lagged variables
        # Get previous date for sorting
        prev_dates = unique_dates[unique_dates < date]
        if len(prev_dates) == 0:
            continue
        prev_date = prev_dates[-1]
        
        # Get data for previous date (for sorting)
        prev_data = data[data['date'] == prev_date]
        
        # Get data for current date (for returns)
        curr_data = data[data['date'] == date]
        
        # Get COI column for sorting
        coi_col = f'COI_{coi_type}'
        
        # Check if we have enough data
        if len(prev_data) < n_quantiles or len(curr_data) < n_quantiles:
            continue
        
        # Calculate quantiles
        quantiles = prev_data[coi_col].quantile(np.linspace(0, 1, n_quantiles+1))
        
        # Create portfolios based on previous day's COI
        for q in range(n_quantiles):
            lower_bound = quantiles.iloc[q]
            upper_bound = quantiles.iloc[q+1]
            
            # Stocks in this quantile
            if q == n_quantiles - 1:  # Include upper bound for last quantile
                quantile_stocks = prev_data[
                    (prev_data[coi_col] >= lower_bound) & 
                    (prev_data[coi_col] <= upper_bound)
                ]['stock_id'].tolist()
            else:
                quantile_stocks = prev_data[
                    (prev_data[coi_col] >= lower_bound) & 
                    (prev_data[coi_col] < upper_bound)
                ]['stock_id'].tolist()
            
            # Calculate portfolio return (equal-weighted)
            portfolio_stocks = curr_data[curr_data['stock_id'].isin(quantile_stocks)]
            if len(portfolio_stocks) > 0:
                portfolio_return = portfolio_stocks['return'].mean()
                
                # Average COI of stocks in the portfolio
                portfolio_coi = prev_data[prev_data['stock_id'].isin(quantile_stocks)][coi_col].mean()
                
                portfolio_returns.append({
                    'date': date,
                    'coi_type': coi_type,
                    'quantile': q + 1,
                    'return': portfolio_return,
                    'avg_coi': portfolio_coi,
                    'n_stocks': len(portfolio_stocks)
                })
    
    # Convert to DataFrame
    portfolio_returns_df = pd.DataFrame(portfolio_returns)
    
    return portfolio_returns_df

def long_short_strategy(portfolio_returns_df, coi_type, is_reversal=False):
    """
    Create long-short strategy based on portfolio returns.
    
    Parameters:
    - portfolio_returns_df: DataFrame of portfolio returns
    - coi_type: COI type used for sorting
    - is_reversal: Whether to use reversal strategy (long low, short high)
    
    Returns:
    - ls_returns: DataFrame of long-short returns
    """
    # Filter for the specific COI type
    coi_returns = portfolio_returns_df[portfolio_returns_df['coi_type'] == coi_type]
    
    # Get number of quantiles
    n_quantiles = coi_returns['quantile'].max()
    
    # Long-short returns
    ls_returns = []
    
    # For each date
    for date in coi_returns['date'].unique():
        date_returns = coi_returns[coi_returns['date'] == date]
        
        # Get high and low quantile returns
        high_quantile = date_returns[date_returns['quantile'] == n_quantiles]
        low_quantile = date_returns[date_returns['quantile'] == 1]
        
        if len(high_quantile) > 0 and len(low_quantile) > 0:
            high_return = high_quantile['return'].values[0]
            low_return = low_quantile['return'].values[0]
            
            # Calculate long-short return
            if is_reversal:
                # Reversal strategy (long low, short high)
                ls_return = low_return - high_return
            else:
                # Momentum strategy (long high, short low)
                ls_return = high_return - low_return
            
            ls_returns.append({
                'date': date,
                'coi_type': coi_type,
                'strategy': 'reversal' if is_reversal else 'momentum',
                'ls_return': ls_return,
                'high_return': high_return,
                'low_return': low_return
            })
    
    # Convert to DataFrame
    ls_returns_df = pd.DataFrame(ls_returns)
    
    # Calculate cumulative returns
    if len(ls_returns_df) > 0:
        ls_returns_df['cum_return'] = (1 + ls_returns_df['ls_return']).cumprod() - 1
    
    return ls_returns_df

def analyze_strategy_performance(ls_returns_df, risk_free_rate=0.0):
    """
    Analyze performance of trading strategies.
    
    Parameters:
    - ls_returns_df: DataFrame of long-short returns
    - risk_free_rate: Annualized risk-free rate
    
    Returns:
    - performance_df: DataFrame of performance metrics
    """
    # Initialize performance metrics
    performance = []
    
    # For each strategy and COI type
    for (coi_type, strategy), group in ls_returns_df.groupby(['coi_type', 'strategy']):
        # Calculate daily performance metrics
        daily_returns = group['ls_return']
        
        # Skip if not enough returns
        if len(daily_returns) < 30:
            continue
        
        # Annualize metrics (assuming 252 trading days)
        n_days = len(daily_returns)
        annualized_return = ((1 + daily_returns).prod()) ** (252 / n_days) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # Daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + daily_returns).cumprod() - 1
        peak = cum_returns.cummax()
        drawdown = peak - cum_returns
        max_drawdown = drawdown.max()
        
        # Win rate
        win_rate = (daily_returns > 0).mean()
        
        # Average return
        avg_daily_return = daily_returns.mean()
        
        # Calculate additional metrics
        excess_returns = daily_returns - daily_rf
        
        # Information ratio (annualized mean excess return / annualized std of excess returns)
        information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
        
        # Sortino ratio (using downside deviation instead of standard deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum consecutive wins and losses
        wins = (daily_returns > 0).astype(int)
        losses = (daily_returns <= 0).astype(int)
        
        from itertools import groupby
        max_consecutive_wins = max([sum(1 for _ in g) for k, g in groupby(wins) if k], default=0)
        max_consecutive_losses = max([sum(1 for _ in g) for k, g in groupby(losses) if k], default=0)
        
        # Calmar ratio (annualized return / maximum drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
        
        # Store performance metrics
        performance.append({
            'coi_type': coi_type,
            'strategy': strategy,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_daily_return': avg_daily_return,
            'information_ratio': information_ratio,
            'sortino_ratio': sortino_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'calmar_ratio': calmar_ratio,
            'n_days': n_days
        })
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance)
    
    # Sort by Sharpe ratio (descending)
    if not performance_df.empty:
        performance_df = performance_df.sort_values('sharpe_ratio', ascending=False)
    
    return performance_df

#####################################
# Phase 8: Visualization Functions
#####################################

def plot_trade_type_proportions(classified_trades):
    """
    Plot proportions of different trade types throughout the day.
    
    Parameters:
    - classified_trades: DataFrame of classified trades
    """
    # Add hour to trades
    classified_trades['hour'] = classified_trades['timestamp'].dt.hour + classified_trades['timestamp'].dt.minute / 60
    
    # Group by hour and calculate proportions
    hour_props = []
    
    for hour in np.arange(9.5, 16.1, 0.5):
        hour_trades = classified_trades[(classified_trades['hour'] >= hour) & (classified_trades['hour'] < hour + 0.5)]
        
        if len(hour_trades) == 0:
            continue
            
        # Count trade types
        n_total = len(hour_trades)
        n_iso = len(hour_trades[hour_trades['label'] == 'iso'])
        n_nis_s = len(hour_trades[hour_trades['sub_label'] == 'nis-s'])
        n_nis_c = len(hour_trades[hour_trades['sub_label'] == 'nis-c'])
        n_nis_b = len(hour_trades[hour_trades['sub_label'] == 'nis-b'])
        
        # Calculate proportions
        hour_props.append({
            'hour': hour + 0.25,  # Center of bin
            'iso': n_iso / n_total,
            'nis-s': n_nis_s / n_total,
            'nis-c': n_nis_c / n_total,
            'nis-b': n_nis_b / n_total
        })
    
    # Convert to DataFrame
    hour_props_df = pd.DataFrame(hour_props)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(hour_props_df['hour'], hour_props_df['iso'], label='Isolated (iso)', marker='o')
    plt.plot(hour_props_df['hour'], hour_props_df['nis-s'], label='Same-stock (nis-s)', marker='s')
    plt.plot(hour_props_df['hour'], hour_props_df['nis-c'], label='Cross-stock (nis-c)', marker='^')
    plt.plot(hour_props_df['hour'], hour_props_df['nis-b'], label='Both (nis-b)', marker='d')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Proportion of Trades')
    plt.title('Trade Type Proportions Throughout the Day')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis ticks for hours
    plt.xticks(np.arange(9.5, 16.1, 0.5), [f"{int(h)}:{int((h-int(h))*60):02d}" for h in np.arange(9.5, 16.1, 0.5)])
    
    plt.tight_layout()
    plt.savefig('trade_type_proportions.png')
    plt.close()

def plot_delta_selection(results_df):
    """
    Plot distance metric and trade type proportions for different delta values.
    
    Parameters:
    - results_df: DataFrame of delta selection results
    """
    # Plot distance metric
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['delta_ms'], results_df['distance'], marker='o')
    plt.xlabel('Delta (ms)')
    plt.ylabel('Distance Metric')
    plt.title('Distance Metric vs. Delta')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal delta
    optimal_delta = results_df.loc[results_df['distance'].idxmax(), 'delta_ms']
    plt.axvline(x=optimal_delta, color='r', linestyle='--', alpha=0.7)
    plt.annotate(f'Optimal δ = {optimal_delta} ms', 
                 xy=(optimal_delta, results_df['distance'].max()),
                 xytext=(optimal_delta + 0.5, results_df['distance'].max() * 0.9),
                 arrowprops=dict(arrowstyle='->', color='r'))
    
    # Plot trade type proportions
    plt.subplot(1, 2, 2)
    
    plt.plot(results_df['delta_ms'], results_df['emp_iso'], marker='o', label='Empirical iso')
    plt.plot(results_df['delta_ms'], results_df['null_iso'], marker='o', linestyle='--', label='Null iso')
    
    plt.plot(results_df['delta_ms'], results_df['emp_nis-s'], marker='s', label='Empirical nis-s')
    plt.plot(results_df['delta_ms'], results_df['null_nis-s'], marker='s', linestyle='--', label='Null nis-s')
    
    plt.plot(results_df['delta_ms'], results_df['emp_nis-c'], marker='^', label='Empirical nis-c')
    plt.plot(results_df['delta_ms'], results_df['null_nis-c'], marker='^', linestyle='--', label='Null nis-c')
    
    plt.plot(results_df['delta_ms'], results_df['emp_nis-b'], marker='d', label='Empirical nis-b')
    plt.plot(results_df['delta_ms'], results_df['null_nis-b'], marker='d', linestyle='--', label='Null nis-b')
    
    plt.xlabel('Delta (ms)')
    plt.ylabel('Proportion')
    plt.title('Empirical vs. Null Model Proportions')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig('delta_selection.png')
    plt.close()

def plot_regression_results(contemp_results, predict_results):
    """
    Plot regression coefficients for contemporaneous and predictive regressions.
    
    Parameters:
    - contemp_results: Dictionary of contemporaneous regression results
    - predict_results: Dictionary of predictive regression results
    """
    # Extract coefficients and t-statistics
    coef_data = []
    
    for coi_type, result in contemp_results.items():
        # Get coefficient and t-statistic for COI variable
        coi_var = f'COI_{coi_type}'
        if coi_var in result.params:
            coef = result.params[coi_var]
            tstat = result.tvalues[coi_var]
            pvalue = result.pvalues[coi_var]
            
            coef_data.append({
                'coi_type': coi_type,
                'regression': 'Contemporaneous',
                'coefficient': coef,
                'tstat': tstat,
                'pvalue': pvalue
            })
    
    for coi_type, result in predict_results.items():
        # Get coefficient and t-statistic for lagged COI variable
        coi_var = f'COI_{coi_type}_lag1'
        if coi_var in result.params:
            coef = result.params[coi_var]
            tstat = result.tvalues[coi_var]
            pvalue = result.pvalues[coi_var]
            
            coef_data.append({
                'coi_type': coi_type,
                'regression': 'Predictive',
                'coefficient': coef,
                'tstat': tstat,
                'pvalue': pvalue
            })
    
    # Convert to DataFrame
    coef_df = pd.DataFrame(coef_data)
    
    # Plot coefficients
    plt.figure(figsize=(12, 6))
    
    # Colors for statistical significance
    colors = ['lightgray', 'lightblue', 'blue', 'darkblue']
    color_thresholds = [0.1, 0.05, 0.01]
    
    def get_color(pvalue):
        for i, threshold in enumerate(color_thresholds):
            if pvalue < threshold:
                return colors[i+1]
        return colors[0]
    
    # Apply colors based on p-values
    coef_df['color'] = coef_df['pvalue'].apply(get_color)
    
    # Contemporaneous regression
    contemp_df = coef_df[coef_df['regression'] == 'Contemporaneous'].sort_values('coi_type')
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(contemp_df['coi_type'], contemp_df['coefficient'], color=contemp_df['color'])
    
    # Add error bars (t-statistics)
    plt.errorbar(contemp_df['coi_type'], contemp_df['coefficient'], 
                yerr=abs(contemp_df['coefficient'] / contemp_df['tstat']), 
                fmt='none', color='black', capsize=5)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('COI Type')
    plt.ylabel('Coefficient')
    plt.title('Contemporaneous Regression Coefficients')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Predictive regression
    predict_df = coef_df[coef_df['regression'] == 'Predictive'].sort_values('coi_type')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(predict_df['coi_type'], predict_df['coefficient'], color=predict_df['color'])
    
    # Add error bars (t-statistics)
    plt.errorbar(predict_df['coi_type'], predict_df['coefficient'], 
                yerr=abs(predict_df['coefficient'] / predict_df['tstat']), 
                fmt='none', color='black', capsize=5)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('COI Type')
    plt.ylabel('Coefficient')
    plt.title('Predictive Regression Coefficients')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add legend for statistical significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Not significant'),
        Patch(facecolor=colors[1], label='p < 0.1'),
        Patch(facecolor=colors[2], label='p < 0.05'),
        Patch(facecolor=colors[3], label='p < 0.01')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig('regression_coefficients.png')
    plt.close()

def plot_strategy_performance(ls_returns_dict, performance_df):
    """
    Plot cumulative returns and performance metrics for trading strategies.
    
    Parameters:
    - ls_returns_dict: Dictionary of long-short returns DataFrames
    - performance_df: DataFrame of performance metrics
    """
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    
    for strategy_key, ls_returns_df in ls_returns_dict.items():
        coi_type, strategy = strategy_key
        
        if len(ls_returns_df) == 0:
            continue
            
        label = f"{coi_type} ({strategy})"
        plt.plot(ls_returns_df['date'], ls_returns_df['cum_return'], label=label)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns of Long-Short Strategies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('strategy_cumulative_returns.png')
    plt.close()
    
    # Check if performance_df is empty
    if len(performance_df) == 0:
        print("Warning: No performance data available for plotting.")
        return
    
    # Plot key performance metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort performance data for consistent ordering
    barplot_data = performance_df.sort_values(['strategy', 'coi_type'])
    
    # Define custom colors based on strategy
    colors = []
    for _, row in barplot_data.iterrows():
        if row['strategy'] == 'momentum':
            colors.append('darkblue' if row['annualized_return'] > 0 else 'lightblue')
        else:  # reversal
            colors.append('darkgreen' if row['annualized_return'] > 0 else 'lightgreen')
    
    # Create labels for x-axis
    x_labels = [f"{row.coi_type}\n({row.strategy})" for _, row in barplot_data.iterrows()]
    x_positions = np.arange(len(barplot_data))
    
    # Annualized Return plot
    axs[0, 0].bar(x_positions, barplot_data['annualized_return'], color=colors)
    axs[0, 0].set_title('Annualized Return')
    axs[0, 0].set_ylabel('Return')
    axs[0, 0].set_xticks(x_positions)
    axs[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axs[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Add values above/below bars
    for i, v in enumerate(barplot_data['annualized_return']):
        axs[0, 0].text(i, v + 0.01 if v >= 0 else v - 0.03, 
                      f"{v:.1%}", ha='center', fontweight='bold')
    
    # Sharpe Ratio plot
    axs[0, 1].bar(x_positions, barplot_data['sharpe_ratio'], color=colors)
    axs[0, 1].set_title('Sharpe Ratio')
    axs[0, 1].set_ylabel('Ratio')
    axs[0, 1].set_xticks(x_positions)
    axs[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axs[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Add values above/below bars
    for i, v in enumerate(barplot_data['sharpe_ratio']):
        axs[0, 1].text(i, v + 0.1 if v >= 0 else v - 0.3, 
                      f"{v:.2f}", ha='center', fontweight='bold')
    
    # Maximum Drawdown plot
    axs[1, 0].bar(x_positions, barplot_data['max_drawdown'], color=colors)
    axs[1, 0].set_title('Maximum Drawdown')
    axs[1, 0].set_ylabel('Drawdown')
    axs[1, 0].set_xticks(x_positions)
    axs[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Add values above bars
    for i, v in enumerate(barplot_data['max_drawdown']):
        axs[1, 0].text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')
    
    # Win Rate plot
    axs[1, 1].bar(x_positions, barplot_data['win_rate'], color=colors)
    axs[1, 1].set_title('Win Rate')
    axs[1, 1].set_ylabel('Rate')
    axs[1, 1].set_xticks(x_positions)
    axs[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add values above bars
    for i, v in enumerate(barplot_data['win_rate']):
        axs[1, 1].text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')
    
    # Add a common title
    fig.suptitle('Trading Strategy Performance Metrics', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig('strategy_performance_metrics.png')
    plt.close()
    
    # Plot additional metrics in another figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Check if information_ratio exists in the dataframe
    if 'information_ratio' in barplot_data.columns:
        # Information Ratio plot
        axs[0, 0].bar(x_positions, barplot_data['information_ratio'], color=colors)
        axs[0, 0].set_title('Information Ratio')
        axs[0, 0].set_ylabel('Ratio')
        axs[0, 0].set_xticks(x_positions)
        axs[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0, 0].grid(True, alpha=0.3)
    
    # Check if sortino_ratio exists in the dataframe
    if 'sortino_ratio' in barplot_data.columns:
        # Sortino Ratio plot
        axs[0, 1].bar(x_positions, barplot_data['sortino_ratio'], color=colors)
        axs[0, 1].set_title('Sortino Ratio')
        axs[0, 1].set_ylabel('Ratio')
        axs[0, 1].set_xticks(x_positions)
        axs[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0, 1].grid(True, alpha=0.3)
    
    # Annualized Volatility plot
    axs[1, 0].bar(x_positions, barplot_data['annualized_volatility'], color=colors)
    axs[1, 0].set_title('Annualized Volatility')
    axs[1, 0].set_ylabel('Volatility')
    axs[1, 0].set_xticks(x_positions)
    axs[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Check if calmar_ratio exists in the dataframe
    if 'calmar_ratio' in barplot_data.columns:
        # Calmar Ratio plot
        axs[1, 1].bar(x_positions, barplot_data['calmar_ratio'], color=colors)
        axs[1, 1].set_title('Calmar Ratio (Return/Max Drawdown)')
        axs[1, 1].set_ylabel('Ratio')
        axs[1, 1].set_xticks(x_positions)
        axs[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1, 1].grid(True, alpha=0.3)
    
    # Add a common title
    fig.suptitle('Additional Strategy Performance Metrics', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig('additional_performance_metrics.png')
    plt.close()

#####################################
# Main Execution
#####################################

def main():
    # Parameters
    n_stocks = 50  # Reduced from 100 to make simulation faster
    n_days = 40    # Reduced from 50 to make simulation faster
    avg_trades_per_stock_per_day = 1000  # Reduced from 2000 to make simulation faster
    delta_ms = 1.0  # Use fixed delta (milliseconds) to skip delta selection
    run_delta_selection = False  # Set to True to run delta selection
    
    # Phase 1: Simulate high-frequency trades
    trades_df, returns_df = simulate_high_frequency_trades(
        n_stocks=n_stocks,
        n_days=n_days,
        avg_trades_per_stock_per_day=avg_trades_per_stock_per_day,
        time_precision_ns=True,
        u_shape_intensity=True,
        correlated_direction=True
    )
    
    # Phase 5: Empirical selection of delta (optional)
    if run_delta_selection:
        delta_candidates_ms = [0.5, 1, 2, 5, 10]  # milliseconds
        delta_ms, delta_results = select_optimal_delta(trades_df, delta_candidates_ms)
        # Plot delta selection results
        plot_delta_selection(delta_results)
    
    # Phase 3: Trade flow decomposition with selected delta
    classified_trades = classify_trades(trades_df, list(range(n_stocks)), delta_ms=delta_ms)
    
    # Plot trade type proportions
    plot_trade_type_proportions(classified_trades)
    
    # Phase 4: Calculate Conditional Order Imbalances (COIs)
    coi_df = calculate_all_cois(classified_trades)
    
    # Phase 6: Prepare regression data
    reg_df = prepare_regression_data(coi_df, returns_df)
    
    # Phase 6: Run contemporaneous regressions
    print("Running contemporaneous regressions...")
    contemp_results = {}
    contemp_summary = []
    
    for coi_type in ['all', 'iso', 'nis', 'nis-s', 'nis-c', 'nis-b']:
        result = run_contemporaneous_regression(reg_df, coi_type)
        contemp_results[coi_type] = result
        
        # Get coefficient and t-statistic
        coi_var = f'COI_{coi_type}'
        if coi_var in result.params:
            coef = result.params[coi_var]
            tstat = result.tvalues[coi_var]
            pvalue = result.pvalues[coi_var]
            
            contemp_summary.append({
                'COI_Type': coi_type,
                'Coefficient': coef,
                't-statistic': tstat,
                'p-value': pvalue
            })
    
    # Print contemporaneous regression summary
    contemp_summary_df = pd.DataFrame(contemp_summary)
    print("\nContemporaneous Regression Results:")
    for _, row in contemp_summary_df.iterrows():
        stars = ''
        if row['p-value'] < 0.01:
            stars = '***'
        elif row['p-value'] < 0.05:
            stars = '**'
        elif row['p-value'] < 0.1:
            stars = '*'
        
        print(f"COI_{row['COI_Type']}: Coefficient = {row['Coefficient']:.6f}, t-stat = {row['t-statistic']:.2f}{stars}")
    
    # Phase 6: Run predictive regressions
    print("\nRunning predictive regressions...")
    predict_results = {}
    predict_summary = []
    
    for coi_type in ['all', 'iso', 'nis', 'nis-s', 'nis-c', 'nis-b']:
        result = run_predictive_regression(reg_df, coi_type)
        predict_results[coi_type] = result
        
        # Get coefficient and t-statistic
        coi_var = f'COI_{coi_type}_lag1'
        if coi_var in result.params:
            coef = result.params[coi_var]
            tstat = result.tvalues[coi_var]
            pvalue = result.pvalues[coi_var]
            
            predict_summary.append({
                'COI_Type': coi_type,
                'Coefficient': coef,
                't-statistic': tstat,
                'p-value': pvalue
            })
    
    # Print predictive regression summary
    predict_summary_df = pd.DataFrame(predict_summary)
    print("\nPredictive Regression Results:")
    for _, row in predict_summary_df.iterrows():
        stars = ''
        if row['p-value'] < 0.01:
            stars = '***'
        elif row['p-value'] < 0.05:
            stars = '**'
        elif row['p-value'] < 0.1:
            stars = '*'
        
        print(f"COI_{row['COI_Type']}_lag: Coefficient = {row['Coefficient']:.6f}, t-stat = {row['t-statistic']:.2f}{stars}")
    
    # Plot regression results
    plot_regression_results(contemp_results, predict_results)
    
    # Phase 7: Single-sort portfolios
    print("\nCreating single-sorted portfolios...")
    portfolio_returns = {}
    for coi_type in ['all', 'iso', 'nis', 'nis-s', 'nis-c', 'nis-b']:
        portfolio_returns[coi_type] = single_sort_portfolios(reg_df, coi_type)
    
    # Phase 7: Long-short strategies
    print("Creating long-short strategies...")
    ls_returns = {}
    
    # Momentum strategies (high minus low)
    for coi_type in ['iso', 'nis-s']:
        ls_returns[(coi_type, 'momentum')] = long_short_strategy(
            portfolio_returns[coi_type], coi_type, is_reversal=False)
    
    # Reversal strategies (low minus high)
    for coi_type in ['nis', 'nis-c', 'nis-b']:
        ls_returns[(coi_type, 'reversal')] = long_short_strategy(
            portfolio_returns[coi_type], coi_type, is_reversal=True)
    
    # Phase 7: Analyze strategy performance
    print("Analyzing strategy performance...")
    performance_results = []
    
    for strategy_key, ls_returns_df in ls_returns.items():
        if len(ls_returns_df) > 0:
            performance = analyze_strategy_performance(ls_returns_df)
            performance_results.append(performance)
    
    if performance_results:
        performance_df = pd.concat(performance_results, ignore_index=True)
        
        # Display detailed performance table
        print("\nDetailed Trading Strategy Performance:")
        performance_table = performance_df[['coi_type', 'strategy', 'annualized_return', 
                                          'annualized_volatility', 'sharpe_ratio', 
                                          'max_drawdown', 'win_rate', 'sortino_ratio',
                                          'information_ratio', 'calmar_ratio', 'n_days']]
        
        # Format columns for better display
        format_dict = {
            'annualized_return': '{:.2%}',
            'annualized_volatility': '{:.2%}',
            'sharpe_ratio': '{:.2f}',
            'max_drawdown': '{:.2%}',
            'win_rate': '{:.2%}',
            'sortino_ratio': '{:.2f}',
            'information_ratio': '{:.2f}',
            'calmar_ratio': '{:.2f}'
        }
        
        for col, fmt in format_dict.items():
            if col in performance_table.columns:
                performance_table[col] = performance_table[col].map(lambda x: fmt.format(x))
        
        print(performance_table.to_string(index=False))
        
        # Plot strategy performance
        plot_strategy_performance(ls_returns, performance_df)
    else:
        print("Warning: No trading strategies with sufficient data to analyze.")
        performance_df = pd.DataFrame()
    
    return classified_trades, coi_df, reg_df, contemp_results, predict_results, ls_returns, performance_df

# Execute main function
if __name__ == "__main__":
    results = main()
    classified_trades, coi_df, reg_df, contemp_results, predict_results, ls_returns, performance_df = results