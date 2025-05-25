import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pdblp
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create directories for saving results
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Function to connect to Bloomberg
def connect_to_bloomberg():
    """
    Establishes connection to Bloomberg
    """
    try:
        print("Connecting to Bloomberg...")
        con = pdblp.BCon(debug=True, port=8194, timeout=10000)  # Increase timeout to 10 seconds
        con.start()
        print("Successfully connected to Bloomberg")
        return con
    except Exception as e:
        print(f"Failed to connect to Bloomberg: {e}")
        print("Will use sample data for demonstration")
        return None

# Function to get historical BFIX and WMR data
def get_fixing_data(con, currency_pairs, start_date, end_date):
    """
    Retrieves historical BFIX and WMR fixing rates from Bloomberg

    Parameters:
    con: Bloomberg connection
    currency_pairs: List of currency pairs to retrieve data for
    start_date: Start date for data retrieval (YYYY-MM-DD)
    end_date: End date for data retrieval (YYYY-MM-DD)

    Returns:
    Dictionary with BFIX and WMR data for each currency pair
    """
    if con is None:
        return load_sample_data(currency_pairs, start_date, end_date)

    data = {}

    for pair in currency_pairs:
        print(f"Retrieving fixing data for {pair}...")

        # Get BFIX rates (3:30pm and 4pm London time)
        bfix_tickers = [
            f"{pair} Curncy",  # Regular currency pair
            f"{pair} BGN Curncy"    # 3:30pm BFIX
        ]

        print(f"Requesting data for tickers: {bfix_tickers}")
        try:
            bfix_data = con.bdh(
                bfix_tickers,
                ["PX_LAST"],
                start_date,
                end_date
            )
            print(f"Successfully retrieved BFIX data for {pair}")
        except Exception as e:
            print(f"Error retrieving BFIX data for {pair}: {e}")
            return load_sample_data(currency_pairs, start_date, end_date)

        # Get WMR rates (4pm London time)
        wmr_tickers = [f"{pair} Curncy"]  # Use the same ticker as BFIX for now
        print(f"Using WMR ticker: {wmr_tickers}")

        print(f"Requesting data for tickers: {wmr_tickers}")
        try:
            wmr_data = con.bdh(
                wmr_tickers,
                ["PX_LAST"],
                start_date,
                end_date
            )
            print(f"Successfully retrieved WMR data for {pair}")
        except Exception as e:
            print(f"Error retrieving WMR data for {pair}: {e}")
            return load_sample_data(currency_pairs, start_date, end_date)

        # Combine data
        bfix_data.columns = bfix_data.columns.droplevel(1)  # Drop PX_LAST level
        wmr_data.columns = wmr_data.columns.droplevel(1)    # Drop PX_LAST level

        combined = pd.merge(
            bfix_data,
            wmr_data,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Rename columns to match our expected format
        # First column is EURUSD Curncy (regular rate), second is BGN (3:30pm), third is WMR (4pm)
        combined.columns = ['BFIX_16', 'BFIX_1530', 'WMR_16']
        data[pair] = combined

    return data

# Function to load sample data if Bloomberg is not available
def load_sample_data(currency_pairs, start_date, end_date):
    """
    Loads or generates sample data for demonstration purposes

    Parameters:
    currency_pairs: List of currency pairs
    start_date: Start date (YYYY-MM-DD)
    end_date: End date (YYYY-MM-DD)

    Returns:
    Dictionary with BFIX and WMR data for each currency pair
    """
    print("Loading sample data...")
    data = {}

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Parameters for synthetic data
    pair_params = {
        'EURUSD': {'mean': 1.15, 'std': 0.05, 'trend_factor': 0.0002},
        'GBPUSD': {'mean': 1.30, 'std': 0.08, 'trend_factor': 0.0003},
        'USDJPY': {'mean': 110.0, 'std': 5.0, 'trend_factor': 0.0002},
        'EURJPY': {'mean': 126.0, 'std': 6.0, 'trend_factor': 0.0003},
        'GBPJPY': {'mean': 142.0, 'std': 7.0, 'trend_factor': 0.0003},
        'EURGBP': {'mean': 0.88, 'std': 0.03, 'trend_factor': 0.0001}
    }

    # Set seed for reproducibility
    np.random.seed(42)

    for pair in currency_pairs:
        # Get parameters for this pair
        params = pair_params.get(pair, {'mean': 1.0, 'std': 0.05, 'trend_factor': 0.0002})

        # Generate base price series with random walk
        base_prices = np.random.normal(0, params['std'] / 20, len(dates)).cumsum()
        base_prices = base_prices + params['mean']

        # Add trends to make the data more realistic
        trends = np.sin(np.linspace(0, 20, len(dates))) * params['trend_factor'] * params['mean']
        base_prices = base_prices + trends

        # Create DataFrame with dates
        df = pd.DataFrame(index=dates)

        # Generate BFIX rates with slight variations
        df['BFIX_1530'] = base_prices

        # Add some trend-following behavior between 3:30pm and 4pm
        df['BFIX_16'] = df['BFIX_1530'] + np.random.normal(0, params['std'] / 100, len(dates))

        # Generate WMR rates with trend-following behavior (after Feb 15, 2015)
        wmr_trend_factor = np.where(dates > pd.Timestamp('2015-02-15'), 0.7, 0.3)
        trend_signal = np.sign(df['BFIX_1530'].rolling(10).mean() - df['BFIX_1530'].shift(10))
        df['WMR_16'] = df['BFIX_16'] + wmr_trend_factor * trend_signal * params['std'] / 100

        # Add noise
        df['BFIX_1530'] = df['BFIX_1530'] + np.random.normal(0, params['std'] / 200, len(dates))
        df['BFIX_16'] = df['BFIX_16'] + np.random.normal(0, params['std'] / 200, len(dates))
        df['WMR_16'] = df['WMR_16'] + np.random.normal(0, params['std'] / 200, len(dates))

        data[pair] = df

    return data

# Function to calculate trend signals
def calculate_trend_signals(data, lookback_days=10, trigger_level=0.0001):
    """
    Calculates trend signals based on the methodology in the paper

    Parameters:
    data: Dictionary with fixing data for each currency pair
    lookback_days: Number of days to look back for calculating trends
    trigger_level: Minimum difference to consider a trend significant

    Returns:
    Dictionary with trend signals for each currency pair
    """
    signals = {}

    for pair, df in data.items():
        print(f"Calculating trend signals for {pair}...")

        # Make a copy of the DataFrame
        signals_df = df.copy()

        # Calculate log price and 10-day moving average
        signals_df['log_price'] = np.log(signals_df['BFIX_1530'])
        signals_df['log_ma'] = signals_df['log_price'].rolling(window=lookback_days).mean()

        # Calculate trend as difference between current price and moving average
        signals_df['trend_diff'] = signals_df['log_price'] - signals_df['log_ma']

        # Apply trigger level to determine trend direction
        signals_df['trend_signal'] = 0
        signals_df.loc[signals_df['trend_diff'] >= trigger_level, 'trend_signal'] = 1
        signals_df.loc[signals_df['trend_diff'] <= -trigger_level, 'trend_signal'] = -1

        signals[pair] = signals_df

    return signals

# Function to simulate the trading strategy
def simulate_strategy(signals, start_date=None, end_date=None, only_uk_days=True):
    """
    Simulates the trading strategy described in the paper

    Parameters:
    signals: Dictionary with trend signals for each currency pair
    start_date: Start date for simulation (YYYY-MM-DD)
    end_date: End date for simulation (YYYY-MM-DD)
    only_uk_days: Whether to filter for UK business days

    Returns:
    Dictionary with simulation results for each currency pair
    """
    results = {}

    for pair, df in signals.items():
        print(f"Simulating strategy for {pair}...")

        # Make a copy of the DataFrame
        sim_df = df.copy()

        # Filter by date if specified
        if start_date:
            sim_df = sim_df[sim_df.index >= pd.Timestamp(start_date)]
        if end_date:
            sim_df = sim_df[sim_df.index <= pd.Timestamp(end_date)]

        # Filter for UK business days if specified
        if only_uk_days:
            # UK holidays could be added here for more precise filtering
            sim_df = sim_df[sim_df.index.dayofweek < 5]  # Weekdays only

        # Calculate strategy returns
        # Buy at BFIX, sell at WMR if trend is positive
        # Sell at BFIX, buy at WMR if trend is negative
        sim_df['strategy_return_bp'] = sim_df['trend_signal'] * (sim_df['WMR_16'] - sim_df['BFIX_16']) / sim_df['BFIX_16'] * 10000

        # Calculate cumulative returns
        sim_df['cumulative_return_bp'] = sim_df['strategy_return_bp'].cumsum()

        # Calculate statistics
        stats = {
            'total_trades': len(sim_df),
            'positive_trades': sum(sim_df['strategy_return_bp'] > 0),
            'negative_trades': sum(sim_df['strategy_return_bp'] < 0),
            'win_rate': sum(sim_df['strategy_return_bp'] > 0) / len(sim_df) if len(sim_df) > 0 else 0,
            'avg_return_bp': sim_df['strategy_return_bp'].mean(),
            'std_return_bp': sim_df['strategy_return_bp'].std(),
            'max_return_bp': sim_df['strategy_return_bp'].max(),
            'min_return_bp': sim_df['strategy_return_bp'].min(),
            'total_return_bp': sim_df['strategy_return_bp'].sum(),
            'sharpe_ratio': sim_df['strategy_return_bp'].mean() / sim_df['strategy_return_bp'].std() if sim_df['strategy_return_bp'].std() > 0 else 0,
            'data': sim_df
        }

        results[pair] = stats

    return results

# Function to analyze pre and post WMR methodology change
def analyze_methodology_change(results, change_date='2015-02-16'):
    """
    Analyzes strategy performance before and after WMR methodology change

    Parameters:
    results: Dictionary with simulation results for each currency pair
    change_date: Date of WMR methodology change (YYYY-MM-DD)

    Returns:
    Dictionary with analysis results for each currency pair
    """
    analysis = {}

    for pair, result in results.items():
        df = result['data']

        # Split data into before and after methodology change
        before_change = df[df.index < pd.Timestamp(change_date)]
        after_change = df[df.index >= pd.Timestamp(change_date)]

        # Calculate statistics for both periods
        before_stats = {
            'total_trades': len(before_change),
            'avg_return_bp': before_change['strategy_return_bp'].mean() if len(before_change) > 0 else 0,
            'std_return_bp': before_change['strategy_return_bp'].std() if len(before_change) > 0 else 0,
            'total_return_bp': before_change['strategy_return_bp'].sum() if len(before_change) > 0 else 0,
            'data': before_change
        }

        after_stats = {
            'total_trades': len(after_change),
            'avg_return_bp': after_change['strategy_return_bp'].mean() if len(after_change) > 0 else 0,
            'std_return_bp': after_change['strategy_return_bp'].std() if len(after_change) > 0 else 0,
            'total_return_bp': after_change['strategy_return_bp'].sum() if len(after_change) > 0 else 0,
            'data': after_change
        }

        analysis[pair] = {
            'before_change': before_stats,
            'after_change': after_stats
        }

    return analysis

# Function to plot cumulative returns
def plot_cumulative_returns(results, title="Cumulative Returns from BFIX-WMR Strategy"):
    """
    Plots cumulative returns for all currency pairs

    Parameters:
    results: Dictionary with simulation results for each currency pair
    title: Title for the plot
    """
    plt.figure(figsize=(14, 8))

    for pair, result in results.items():
        df = result['data']
        plt.plot(df.index, df['cumulative_return_bp'], label=pair)

    plt.axvline(x=pd.Timestamp('2015-02-16'), color='grey', linestyle='--',
                label="WMR Methodology Change (Feb 2015)")

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (basis points)')
    plt.legend()
    plt.grid(True)

    # Format x-axis to show dates better
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig('results/cumulative_returns.png')
    plt.close()

# Function to plot average returns by trend
def plot_average_returns_by_trend(results, title="Average Returns by Trend Signal"):
    """
    Plots average returns for positive and negative trend signals

    Parameters:
    results: Dictionary with simulation results for each currency pair
    title: Title for the plot
    """
    pairs = list(results.keys())
    positive_trends = []
    negative_trends = []

    for pair, result in results.items():
        df = result['data']
        pos_ret = df.loc[df['trend_signal'] == 1, 'strategy_return_bp'].mean()
        neg_ret = df.loc[df['trend_signal'] == -1, 'strategy_return_bp'].mean()

        positive_trends.append(pos_ret)
        negative_trends.append(neg_ret)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Positive Trend': positive_trends,
        'Negative Trend': negative_trends
    }, index=pairs)

    plt.figure(figsize=(12, 8))
    plot_df.plot(kind='bar')

    plt.title(title)
    plt.xlabel('Currency Pair')
    plt.ylabel('Average Return (basis points)')
    plt.grid(True, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/average_returns_by_trend.png')
    plt.close()

# Function to plot intraday price movements
def plot_intraday_movements(signals, after_change_date='2015-02-16'):
    """
    Plots average intraday price movements as function of trend signal

    Parameters:
    signals: Dictionary with trend signals for each currency pair
    after_change_date: Date after which to analyze data (YYYY-MM-DD)
    """
    plt.figure(figsize=(16, 10))

    for i, (pair, df) in enumerate(signals.items()):
        # Filter data after methodology change
        df_after = df[df.index >= pd.Timestamp(after_change_date)].copy()

        # Calculate returns from BFIX 3:30 to BFIX 4:00 and from BFIX 4:00 to WMR 4:00
        df_after['bfix_ret_bp'] = (df_after['BFIX_16'] - df_after['BFIX_1530']) / df_after['BFIX_1530'] * 10000
        df_after['wmr_ret_bp'] = (df_after['WMR_16'] - df_after['BFIX_16']) / df_after['BFIX_16'] * 10000

        # Plot returns by trend signal
        plt.subplot(2, 3, i + 1)

        # Calculate average returns for positive and negative trends
        pos_bfix_ret = df_after.loc[df_after['trend_signal'] == 1, 'bfix_ret_bp'].mean()
        pos_wmr_ret = df_after.loc[df_after['trend_signal'] == 1, 'wmr_ret_bp'].mean()
        neg_bfix_ret = df_after.loc[df_after['trend_signal'] == -1, 'bfix_ret_bp'].mean()
        neg_wmr_ret = df_after.loc[df_after['trend_signal'] == -1, 'wmr_ret_bp'].mean()

        # Create bar chart
        plt.bar([0, 1], [pos_bfix_ret, pos_wmr_ret], width=0.4, alpha=0.6, color='green', label='Positive Trend')
        plt.bar([0.5, 1.5], [neg_bfix_ret, neg_wmr_ret], width=0.4, alpha=0.6, color='red', label='Negative Trend')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks([0.25, 1.25], ['3:30pm to 4:00pm BFIX', '4:00pm BFIX to WMR'])
        plt.title(f"{pair} Intraday Moves by Trend")
        plt.ylabel('Return (basis points)')

        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig('results/intraday_movements.png')
    plt.close()

# Function to test parameter sensitivity
def test_parameter_sensitivity(data, lookback_range=[1, 5, 10, 15, 20, 25],
                              trigger_range=[0.0001, 0.001, 0.002, 0.003, 0.004, 0.005],
                              after_date='2015-02-16'):
    """
    Tests sensitivity of the strategy to different parameters

    Parameters:
    data: Dictionary with fixing data for each currency pair
    lookback_range: List of lookback periods to test
    trigger_range: List of trigger levels to test
    after_date: Date after which to analyze data (YYYY-MM-DD)

    Returns:
    Dictionary with sensitivity analysis results
    """
    sensitivity_results = {}

    # Test lookback sensitivity
    lookback_results = {}
    for lookback in lookback_range:
        signals = calculate_trend_signals(data, lookback_days=lookback)
        results = simulate_strategy(signals)

        # Filter for results after methodology change
        avg_returns = {}
        for pair, result in results.items():
            df = result['data']
            df_after = df[df.index >= pd.Timestamp(after_date)]
            avg_returns[pair] = df_after['strategy_return_bp'].mean() if len(df_after) > 0 else 0

        lookback_results[lookback] = avg_returns

    # Test trigger sensitivity
    trigger_results = {}
    for trigger in trigger_range:
        signals = calculate_trend_signals(data, trigger_level=trigger)
        results = simulate_strategy(signals)

        # Filter for results after methodology change
        avg_returns = {}
        trade_counts = {}
        for pair, result in results.items():
            df = result['data']
            df_after = df[df.index >= pd.Timestamp(after_date)]
            avg_returns[pair] = df_after['strategy_return_bp'].mean() if len(df_after) > 0 else 0
            trade_counts[pair] = len(df_after[df_after['trend_signal'] != 0])

        trigger_results[trigger] = {
            'avg_returns': avg_returns,
            'trade_counts': trade_counts
        }

    sensitivity_results['lookback'] = lookback_results
    sensitivity_results['trigger'] = trigger_results

    return sensitivity_results

# Function to plot parameter sensitivity
def plot_parameter_sensitivity(sensitivity_results, pairs):
    """
    Plots sensitivity of the strategy to different parameters

    Parameters:
    sensitivity_results: Dictionary with sensitivity analysis results
    pairs: List of currency pairs to include in the plot
    """
    # Plot lookback sensitivity
    plt.figure(figsize=(12, 8))

    lookback_results = sensitivity_results['lookback']
    lookback_range = sorted(lookback_results.keys())

    for pair in pairs:
        avg_returns = [lookback_results[lb][pair] for lb in lookback_range]
        plt.plot(lookback_range, avg_returns, marker='o', label=pair)

    plt.title("Sensitivity to Lookback Period")
    plt.xlabel('Lookback Period (Days)')
    plt.ylabel('Average Return (basis points)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/lookback_sensitivity.png')
    plt.close()

    # Plot trigger sensitivity
    plt.figure(figsize=(14, 10))

    trigger_results = sensitivity_results['trigger']
    trigger_range = sorted(trigger_results.keys())

    # Plot average returns
    plt.subplot(2, 1, 1)
    for pair in pairs:
        avg_returns = [trigger_results[t]['avg_returns'][pair] for t in trigger_range]
        plt.plot(trigger_range, avg_returns, marker='o', label=pair)

    plt.title("Sensitivity to Trigger Level (Average Returns)")
    plt.xlabel('Trigger Level')
    plt.ylabel('Average Return (basis points)')
    plt.grid(True)
    plt.legend()

    # Plot trade counts
    plt.subplot(2, 1, 2)
    for pair in pairs:
        trade_counts = [trigger_results[t]['trade_counts'][pair] for t in trigger_range]
        plt.plot(trigger_range, trade_counts, marker='o', label=pair)

    plt.title("Sensitivity to Trigger Level (Trade Counts)")
    plt.xlabel('Trigger Level')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/trigger_sensitivity.png')
    plt.close()

# Function to create summary table
def create_summary_table(results, analysis):
    """
    Creates a summary table with key strategy metrics

    Parameters:
    results: Dictionary with simulation results for each currency pair
    analysis: Dictionary with analysis results for each currency pair

    Returns:
    DataFrame with summary statistics
    """
    summary = []

    for pair in results.keys():
        # Overall statistics
        overall = results[pair]

        # Before and after statistics
        before = analysis[pair]['before_change']
        after = analysis[pair]['after_change']

        # Create summary row
        row = {
            'Currency Pair': pair,
            'Total Trades': overall['total_trades'],
            'Win Rate': overall['win_rate'],
            'Avg Return (bp)': overall['avg_return_bp'],
            'Std Dev (bp)': overall['std_return_bp'],
            'Sharpe Ratio': overall['sharpe_ratio'],
            'Total Return (bp)': overall['total_return_bp'],
            'Pre-Change Avg (bp)': before['avg_return_bp'],
            'Post-Change Avg (bp)': after['avg_return_bp'],
            'Improvement Factor': after['avg_return_bp'] / before['avg_return_bp'] if before['avg_return_bp'] != 0 else float('inf')
        }

        summary.append(row)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)

    # Set Currency Pair as index
    summary_df.set_index('Currency Pair', inplace=True)

    return summary_df

# Main function
def main():
    # Define currency pairs to analyze
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'GBPJPY', 'EURGBP']

    # Define date range - using a shorter date range
    start_date = '20230101'  # January 1, 2023 in YYYYMMDD format
    end_date = '20230430'    # April 30, 2023 in YYYYMMDD format

    print(f"Using date range: {start_date} to {end_date}")

    # Connect to Bloomberg
    con = connect_to_bloomberg()

    # Get fixing data
    fixing_data = get_fixing_data(con, currency_pairs, start_date, end_date)

    # Calculate trend signals
    trend_signals = calculate_trend_signals(fixing_data)

    # Simulate the strategy
    strategy_results = simulate_strategy(trend_signals)

    # Analyze pre and post WMR methodology change
    change_analysis = analyze_methodology_change(strategy_results)

    # Create summary table
    summary_table = create_summary_table(strategy_results, change_analysis)
    print("\nStrategy Summary:")
    print(summary_table)

    # Save summary table
    summary_table.to_csv('results/strategy_summary.csv')

    # Plot results
    plot_cumulative_returns(strategy_results)
    plot_average_returns_by_trend(strategy_results)
    plot_intraday_movements(trend_signals)

    # Test parameter sensitivity
    sensitivity_results = test_parameter_sensitivity(fixing_data)

    # Plot parameter sensitivity
    plot_parameter_sensitivity(sensitivity_results, currency_pairs[:3])  # Use only main 3 pairs for clarity

    # Reporting implementation information for production use
    print("\nImplementation Guidelines for Production:")
    print("1. Monitor BFIX 3:30pm London rate daily")
    print("2. Calculate trend signal using 10-day lookback window")
    print("3. For trend-following flow with WMR benchmark, shift to BFIX execution")
    print("4. For counter-trend flow with BFIX benchmark, shift to WMR execution")
    print("5. Expect higher benefits in highly traded pairs (EURUSD, GBPUSD, USDJPY)")

    print("\nAnalysis complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()