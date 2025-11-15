"""
Kalman Filter Pairs Trading Strategy for EWA-EWC
EWA: iShares MSCI Australia ETF
EWC: iShares MSCI Canada ETF
"""

import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from datetime import datetime

def connect_to_mysql():
    """Connect to MySQL database"""
    print("Connecting to MySQL server...")
    try:
        conn = pymysql.connect(
            host='192.168.50.230',
            port=3306,
            user='root',
            password='352471Cf!1',
            database='us_stock_sip_min_aggs'
        )
        print("Successfully connected to MySQL!")
        return conn
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        raise

def get_table_names(start_date='2015-01-01'):
    """Generate list of table names from start_date to current date"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.now()
    
    tables = []
    current = start
    while current <= end:
        table_name = current.strftime('%Y%m')
        tables.append(table_name)
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return tables

def fetch_stock_data(conn, symbol, start_date='2015-01-01'):
    """Fetch stock data from multiple monthly tables"""
    print(f"Fetching {symbol} data from {start_date}...")
    
    tables = get_table_names(start_date)
    all_data = []
    
    for table in tables:
        try:
            query = f"""
                SELECT window_start, close 
                FROM `{table}` 
                WHERE ticker = %s
                ORDER BY window_start
            """
            df = pd.read_sql(query, conn, params=(symbol,))
            if len(df) > 0:
                all_data.append(df)
                print(f"  Retrieved {len(df)} records from table {table}")
        except Exception as e:
            # Table might not exist or no data for this symbol
            pass
    
    if not all_data:
        raise ValueError(f"No data found for {symbol}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total retrieved: {len(combined_df)} records for {symbol}")
    
    return combined_df

def prepare_data(ewa_df, ewc_df):
    """Prepare and align data for both stocks"""
    print("Preparing and aligning data...")

    # Convert window_start (nanoseconds) to datetime
    ewa_df['timestamp'] = pd.to_datetime(ewa_df['window_start'], unit='ns')
    ewc_df['timestamp'] = pd.to_datetime(ewc_df['window_start'], unit='ns')

    # Convert close to float
    ewa_df['close'] = ewa_df['close'].astype(float)
    ewc_df['close'] = ewc_df['close'].astype(float)

    # Merge on timestamp
    merged = pd.merge(ewa_df[['timestamp', 'close']],
                      ewc_df[['timestamp', 'close']],
                      on='timestamp',
                      suffixes=('_ewa', '_ewc'))

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    print(f"Aligned data: {len(merged)} records")

    return merged

def resample_to_5min(data):
    """Resample 1-minute data to 5-minute intervals"""
    print("Resampling to 5-minute intervals...")
    
    # Set timestamp as index
    data = data.set_index('timestamp')
    
    # Resample to 5-minute intervals, taking the last close price
    resampled = data.resample('5T').agg({
        'close_ewa': 'last',
        'close_ewc': 'last'
    }).dropna()
    
    # Reset index to get timestamp back as column
    resampled = resampled.reset_index()
    
    print(f"Resampled data: {len(resampled)} records")
    
    return resampled

def kalman_filter_regression(x, y):
    """
    Apply Kalman Filter to estimate dynamic hedge ratio
    x: independent variable (EWA prices)
    y: dependent variable (EWC prices)
    Returns: state_means containing [beta, alpha] for each time step
    """
    print("Applying Kalman Filter...")
    
    # Observation matrix: y = beta * x + alpha
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    
    # Initialize Kalman Filter
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=0.01 * np.eye(2)
    )
    
    # Run filter
    state_means, _ = kf.filter(y)

    return state_means

def generate_trading_signals(data, state_means, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading signals based on Kalman filter estimates
    """
    print("Generating trading signals...")
    print(f"  Entry threshold: ±{entry_threshold}, Exit threshold: ±{exit_threshold}")

    # Extract hedge ratio (beta) and intercept (alpha)
    beta = state_means[:, 0]
    alpha = state_means[:, 1]

    # Calculate spread
    spread = data['close_ewc'].values - beta * data['close_ewa'].values - alpha

    # Calculate z-score of spread with longer window for stability
    window = 500  # Increased from 100 for more stable statistics
    spread_mean = pd.Series(spread).rolling(window=window, min_periods=100).mean().values
    spread_std = pd.Series(spread).rolling(window=window, min_periods=100).std().values
    z_score = (spread - spread_mean) / spread_std

    # Generate signals
    signals = pd.DataFrame(index=data.index)
    signals['timestamp'] = data['timestamp']
    signals['ewa_price'] = data['close_ewa']
    signals['ewc_price'] = data['close_ewc']
    signals['beta'] = beta
    signals['spread'] = spread
    signals['z_score'] = z_score

    # Trading logic using vectorized operations with state machine
    # Long spread when z-score < -entry_threshold (spread is low)
    # Short spread when z-score > entry_threshold (spread is high)
    # Exit when |z-score| < exit_threshold

    print("  Computing positions (optimized)...")
    position = np.zeros(len(signals))
    current_pos = 0

    # Use numpy arrays for faster iteration
    z_score_arr = z_score

    for i in range(len(position)):
        if i % 100000 == 0:
            print(f"    Processed {i}/{len(position)} records...")

        if current_pos == 0:  # No position
            if not np.isnan(z_score_arr[i]):
                if z_score_arr[i] < -entry_threshold:
                    current_pos = 1  # Long spread (long EWC, short EWA)
                elif z_score_arr[i] > entry_threshold:
                    current_pos = -1  # Short spread (short EWC, long EWA)
        elif current_pos == 1:  # Long spread
            if not np.isnan(z_score_arr[i]) and z_score_arr[i] > -exit_threshold:
                current_pos = 0
        elif current_pos == -1:  # Short spread
            if not np.isnan(z_score_arr[i]) and z_score_arr[i] < exit_threshold:
                current_pos = 0

        position[i] = current_pos

    signals['position'] = position
    print("  Position computation complete!")

    return signals

def calculate_returns(signals, transaction_cost=0.0001):
    """Calculate strategy returns with proper portfolio calculation"""
    print("Calculating returns...")
    print(f"  Transaction cost: {transaction_cost*100:.2f}% per trade")

    # Calculate individual asset returns
    signals['ewa_return'] = signals['ewa_price'].pct_change()
    signals['ewc_return'] = signals['ewc_price'].pct_change()

    # Portfolio return based on position
    # Position = 1: Long EWC, Short EWA (beta units)
    # Position = -1: Short EWC, Long EWA (beta units)
    # Position = 0: No position

    signals['portfolio_return'] = 0.0

    # When position = 1 (long spread): profit when EWC outperforms EWA
    # When position = -1 (short spread): profit when EWA outperforms EWC
    for i in range(1, len(signals)):
        pos = signals['position'].iloc[i-1]  # Use previous position
        if pos != 0:
            # Portfolio return = EWC return - beta * EWA return (for long spread)
            # For short spread, reverse the sign
            beta = signals['beta'].iloc[i]
            ewc_ret = signals['ewc_return'].iloc[i]
            ewa_ret = signals['ewa_return'].iloc[i]

            if not np.isnan(ewc_ret) and not np.isnan(ewa_ret):
                signals.loc[signals.index[i], 'portfolio_return'] = pos * (ewc_ret - beta * ewa_ret)

    # Apply transaction costs when position changes
    position_change = signals['position'].diff().abs()
    signals['transaction_cost'] = position_change * transaction_cost

    # Net strategy return
    signals['strategy_return'] = signals['portfolio_return'] - signals['transaction_cost']

    # Cumulative returns
    signals['cumulative_return'] = (1 + signals['strategy_return'].fillna(0)).cumprod()

    return signals

def print_performance_metrics(signals):
    """Print performance metrics"""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)

    strategy_returns = signals['strategy_return'].dropna()

    # Total return
    total_return = (signals['cumulative_return'].iloc[-1] - 1) * 100
    print(f"Total Return: {total_return:.2f}%")

    # Annualized return (assuming 252 trading days, 390 minutes per day)
    n_periods = len(strategy_returns)
    n_years = n_periods / (252 * 390)
    annualized_return = ((signals['cumulative_return'].iloc[-1]) ** (1/n_years) - 1) * 100
    print(f"Annualized Return: {annualized_return:.2f}%")

    # Sharpe ratio (annualized)
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 390) if std_return > 0 else 0
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Maximum drawdown
    cumulative = signals['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (strategy_returns != 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")

    # Number of trades
    position_changes = (signals['position'].diff() != 0).sum()
    print(f"Number of Position Changes: {position_changes}")
    print("="*60)

def plot_results(signals, entry_threshold=2.0, exit_threshold=0.5, filename='kalman_ewa_ewc_results.png'):
    """Plot trading results"""
    print("Generating plots...")

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Sample data for plotting (plot every 100th point to avoid overcrowding)
    sample_idx = range(0, len(signals), 100)

    # Plot 1: Prices
    ax1 = axes[0]
    ax1.plot(signals.iloc[sample_idx]['timestamp'],
             signals.iloc[sample_idx]['ewa_price'],
             label='EWA (Australia)', alpha=0.7)
    ax1.plot(signals.iloc[sample_idx]['timestamp'],
             signals.iloc[sample_idx]['ewc_price'],
             label='EWC (Canada)', alpha=0.7)
    ax1.set_title('EWA and EWC Prices')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hedge Ratio (Beta)
    ax2 = axes[1]
    ax2.plot(signals.iloc[sample_idx]['timestamp'],
             signals.iloc[sample_idx]['beta'],
             label='Hedge Ratio (Beta)', color='green')
    ax2.set_title('Kalman Filter Hedge Ratio')
    ax2.set_ylabel('Beta')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Z-Score and Positions
    ax3 = axes[2]
    ax3.plot(signals.iloc[sample_idx]['timestamp'],
             signals.iloc[sample_idx]['z_score'],
             label='Z-Score', color='blue', alpha=0.7)
    ax3.axhline(y=entry_threshold, color='r', linestyle='--', alpha=0.5, label='Entry Threshold')
    ax3.axhline(y=-entry_threshold, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=exit_threshold, color='g', linestyle='--', alpha=0.5, label='Exit Threshold')
    ax3.axhline(y=-exit_threshold, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Spread Z-Score')
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative Returns
    ax4 = axes[3]
    ax4.plot(signals.iloc[sample_idx]['timestamp'],
             signals.iloc[sample_idx]['cumulative_return'],
             label='Strategy Returns', color='purple', linewidth=2)
    ax4.set_title('Cumulative Returns')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")
    plt.close()

def main(use_5min=False, entry_threshold=2.5, exit_threshold=0.5, transaction_cost=0.0001):
    """
    Main execution function

    Parameters:
    - use_5min: If True, resample to 5-minute data; if False, use 1-minute data
    - entry_threshold: Z-score threshold for entering positions (default: 2.5)
    - exit_threshold: Z-score threshold for exiting positions (default: 0.5)
    - transaction_cost: Transaction cost per trade as fraction (default: 0.0001 = 0.01%)
    """
    try:
        print("\n" + "="*60)
        print("KALMAN FILTER PAIRS TRADING ANALYSIS: EWA-EWC")
        print("="*60)
        print(f"Time interval: {'5-minute' if use_5min else '1-minute'}")
        print(f"Entry threshold: ±{entry_threshold}")
        print(f"Exit threshold: ±{exit_threshold}")
        print(f"Transaction cost: {transaction_cost*100:.2f}%")
        print("="*60 + "\n")

        # Connect to database
        conn = connect_to_mysql()

        # Fetch data
        ewa_df = fetch_stock_data(conn, 'EWA', start_date='2015-01-01')
        ewc_df = fetch_stock_data(conn, 'EWC', start_date='2015-01-01')

        # Close connection
        conn.close()
        print("Database connection closed.")

        # Prepare data
        data = prepare_data(ewa_df, ewc_df)

        # Resample to 5-minute if requested
        if use_5min:
            data = resample_to_5min(data)

        # Apply Kalman Filter
        state_means = kalman_filter_regression(
            data['close_ewa'].values,
            data['close_ewc'].values
        )

        # Generate trading signals
        signals = generate_trading_signals(data, state_means,
                                          entry_threshold=entry_threshold,
                                          exit_threshold=exit_threshold)

        # Calculate returns
        signals = calculate_returns(signals, transaction_cost=transaction_cost)

        # Print performance metrics
        print_performance_metrics(signals)

        # Save results to CSV
        output_file = f'ewa_ewc_signals_{"5min" if use_5min else "1min"}.csv'
        signals.to_csv(output_file, index=False)
        print(f"Trading signals saved to '{output_file}'")

        # Plot results
        plot_file = f'kalman_ewa_ewc_results_{"5min" if use_5min else "1min"}.png'
        plot_results(signals, entry_threshold, exit_threshold, plot_file)

        print("\nAnalysis complete!")

        return signals

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run with 1-minute data and adjusted parameters
    print("\n### EWA-EWC PAIRS TRADING: 1-minute data ###")
    signals_1min = main(use_5min=False, entry_threshold=2.5, exit_threshold=0.5, transaction_cost=0.0001)

