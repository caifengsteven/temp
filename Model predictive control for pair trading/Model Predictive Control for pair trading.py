import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Import the BloombergDataFetcher from your module
from bloomberg_data_fetcher import BloombergDataFetcher

class MPCPairsTrading:
    """
    MPC pairs trading implementation with Bloomberg data support
    """
    
    def __init__(self):
        """Initialize the model"""
        # Pairs data
        self.pairs = []
        self.spread_matrix = None
        self.prices = {}
        
        # Model parameters
        self.mean_reversion = 0.85  # Mean reversion strength
        self.spread_volatility = 1.0  # Volatility scaling
        
        # Trading parameters
        self.wealth = 1.0
        self.positions = None
        self.wealth_history = []
        self.position_history = []
        
        # MPC parameters
        self.prediction_horizon = 5
        self.transaction_cost = 0.0004
        self.position_limit = 0.2
        
        # Bloomberg fetcher
        self.bbg_fetcher = None
    
    def connect_to_bloomberg(self):
        """Connect to Bloomberg API using the fetcher"""
        try:
            self.bbg_fetcher = BloombergDataFetcher()
            if self.bbg_fetcher.start_session():
                print("Connected to Bloomberg API")
                return True
            else:
                print("Failed to connect to Bloomberg API")
                return False
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            return False
    
    def add_pair(self, ticker1, ticker2, beta=None):
        """Add a pair to trade"""
        self.pairs.append({
            'ticker1': ticker1,
            'ticker2': ticker2,
            'beta': beta
        })
        print(f"Added pair: {ticker1} - {ticker2}")
    
    def load_data_from_bloomberg(self, start_date=None, end_date=None, interval=30):
        """
        Load price data from Bloomberg using the fetcher
        
        Parameters:
        -----------
        start_date : datetime, optional
            Start date for data retrieval
        end_date : datetime, optional
            End date for data retrieval
        interval : int
            Bar interval in minutes (default: 30)
        """
        if self.bbg_fetcher is None:
            if not self.connect_to_bloomberg():
                return False
        
        # Create output directory if needed
        output_dir = "bloomberg_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # If dates not provided, use default range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            # Use two years of data (adjust for Bloomberg's 140-day limit for intraday)
            if interval < 1440:  # Intraday
                start_date = end_date - timedelta(days=140)
            else:  # Daily
                start_date = end_date - timedelta(days=365*2)
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Create instruments.txt file for the fetcher
        tickers = []
        for pair in self.pairs:
            tickers.append(pair['ticker1'])
            tickers.append(pair['ticker2'])
        
        # Remove duplicates
        tickers = list(set(tickers))
        
        # Write tickers to file
        with open("instruments.txt", "w") as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")
        
        # Get data for each ticker
        all_data_success = True
        
        for ticker in tickers:
            try:
                # Get intraday or daily data based on interval
                if interval < 1440:  # Intraday
                    data = self.bbg_fetcher.get_intraday_bars(
                        ticker,
                        event_type="TRADE",
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date
                    )
                    # Resample to daily for the pairs trading strategy
                    if not data.empty:
                        data.set_index('time', inplace=True)
                        daily_data = data['close'].resample('D').last().dropna()
                        self.prices[ticker] = daily_data
                else:  # Daily data
                    # For daily data we would need to modify the fetcher
                    # For now, we'll use intraday data resampled to daily
                    data = self.bbg_fetcher.get_intraday_bars(
                        ticker,
                        event_type="TRADE",
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not data.empty:
                        data.set_index('time', inplace=True)
                        daily_data = data['close'].resample('D').last().dropna()
                        self.prices[ticker] = daily_data
                
                # Save data to file
                if not data.empty:
                    self.bbg_fetcher.save_data_to_csv(data, ticker, output_dir)
                else:
                    print(f"No data retrieved for {ticker}")
                    all_data_success = False
            
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                all_data_success = False
        
        # Calculate spreads
        if all_data_success and self.prices:
            print(f"Successfully retrieved data for {len(self.prices)} tickers")
            self.calculate_spreads()
            return True
        else:
            print("Failed to retrieve data for all tickers")
            return False
    
    def calculate_spreads(self):
        """Calculate spreads from price data"""
        if not self.prices:
            print("No price data available")
            return False
        
        # Get common dates across all price series
        common_dates = None
        
        for ticker, prices in self.prices.items():
            if common_dates is None:
                common_dates = set(prices.index)
            else:
                common_dates = common_dates.intersection(set(prices.index))
        
        if not common_dates:
            print("No common dates found across all price series")
            return False
        
        # Sort dates
        common_dates = sorted(list(common_dates))
        
        # Combine all prices into a DataFrame with common dates
        price_df = pd.DataFrame(index=common_dates)
        
        for ticker, prices in self.prices.items():
            price_df[ticker] = prices.reindex(common_dates)
        
        # Calculate spreads for each pair
        spread_data = {}
        
        for pair in self.pairs:
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            
            # Make sure both tickers are in the data
            if ticker1 not in price_df.columns or ticker2 not in price_df.columns:
                print(f"Warning: Missing price data for {ticker1} or {ticker2}")
                continue
            
            # Estimate beta if not provided
            if pair['beta'] is None:
                # Calculate returns for regression
                returns1 = price_df[ticker1].pct_change().dropna()
                returns2 = price_df[ticker2].pct_change().dropna()
                
                # Get common dates
                common_returns_dates = returns1.index.intersection(returns2.index)
                
                if len(common_returns_dates) < 30:  # Need enough data for regression
                    print(f"Not enough data for regression for {ticker1}-{ticker2}")
                    # Use default beta of 1.0
                    pair['beta'] = 1.0
                else:
                    # Simple OLS regression on returns
                    X = returns2.loc[common_returns_dates].values.reshape(-1, 1)
                    y = returns1.loc[common_returns_dates].values
                    
                    # Add constant
                    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
                    
                    # OLS formula: (X'X)^(-1)X'y
                    try:
                        beta_vector = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
                        pair['beta'] = beta_vector[1]  # Beta is the slope
                    except np.linalg.LinAlgError:
                        # Fallback if matrix is singular
                        pair['beta'] = 1.0
                    
                print(f"Estimated beta for {ticker1}-{ticker2}: {pair['beta']:.4f}")
            
            # Calculate spread
            spread = price_df[ticker1] - pair['beta'] * price_df[ticker2]
            spread_data[f"{ticker1}-{ticker2}"] = spread
        
        # Create spread matrix
        self.spread_matrix = pd.DataFrame(spread_data)
        
        # Calculate spread volatility for position sizing
        self.spread_volatility = np.mean([spread.std() for spread in self.spread_matrix.values.T])
        
        print(f"Calculated spreads for {len(spread_data)} pairs")
        print(f"Data range: {self.spread_matrix.index[0]} to {self.spread_matrix.index[-1]}")
        
        return True
    
    def generate_simulated_data(self, n_days=500, seed=42):
        """Generate simple mean-reverting spread data"""
        np.random.seed(seed)
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate spread data
        n_pairs = len(self.pairs)
        spread_data = np.zeros((len(dates), n_pairs))
        pair_names = []
        
        for i, pair in enumerate(self.pairs):
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            
            # Generate Ornstein-Uhlenbeck process
            mean = 0.0
            mean_rev = self.mean_reversion
            vol = 0.5 + np.random.rand()  # Different volatility for each pair
            
            spread = np.zeros(len(dates))
            spread[0] = mean + np.random.randn()
            
            for t in range(1, len(dates)):
                spread[t] = mean + mean_rev * (spread[t-1] - mean) + vol * np.random.randn()
            
            # Store spread
            spread_data[:, i] = spread
            pair_names.append(f"{ticker1}-{ticker2}")
        
        # Create spread matrix
        self.spread_matrix = pd.DataFrame(spread_data, index=dates, columns=pair_names)
        
        print(f"Generated simulated data for {n_pairs} pairs from {dates[0]} to {dates[-1]}")
        return True
    
    def predict_spreads(self, current_spreads, horizon):
        """Predict future spreads using simple AR(1) model"""
        n_pairs = len(current_spreads)
        predicted_spreads = np.zeros((horizon + 1, n_pairs))
        predicted_spreads[0] = current_spreads
        
        # Simple AR(1) prediction for each spread
        for t in range(1, horizon + 1):
            predicted_spreads[t] = self.mean_reversion * predicted_spreads[t-1]
        
        return predicted_spreads
    
    def calculate_optimal_positions(self, current_spreads, current_positions, current_wealth):
        """
        Calculate optimal positions using simplified MPC approach
        """
        n_pairs = len(current_spreads)
        horizon = self.prediction_horizon
        
        # Predict future spreads
        predicted_spreads = self.predict_spreads(current_spreads, horizon)
        
        # Calculate expected spread changes
        expected_changes = np.diff(predicted_spreads, axis=0)
        
        # Signal strength based on mean reversion
        signal_strength = np.abs(current_spreads) / self.spread_volatility
        
        # Base position sizes on signal strength
        raw_positions = -np.sign(current_spreads) * signal_strength
        
        # Scale positions to respect position limit
        max_pos = np.max(np.abs(raw_positions))
        if max_pos > 0:
            scale_factor = min(1.0, self.position_limit / max_pos)
            positions = raw_positions * scale_factor
        else:
            positions = raw_positions
        
        # Consider transaction costs - reduce position changes for small signals
        position_changes = positions - current_positions
        
        # Only take trades where expected profit exceeds transaction cost
        expected_profit = np.abs(expected_changes[0] * position_changes)
        transaction_cost = self.transaction_cost * np.abs(position_changes)
        
        # Adjust positions where expected profit < transaction cost
        for i in range(n_pairs):
            if expected_profit[i] < transaction_cost[i]:
                positions[i] = current_positions[i]  # Don't change position
        
        return positions
    
    def backtest(self, prediction_horizon=5, transaction_cost=0.0004, position_limit=0.2, initial_wealth=1.0):
        """Run a backtest of the strategy"""
        if self.spread_matrix is None or self.spread_matrix.empty:
            print("No spread data available. Load or generate data first.")
            return None
        
        # Set parameters
        self.prediction_horizon = prediction_horizon
        self.transaction_cost = transaction_cost
        self.position_limit = position_limit
        self.wealth = initial_wealth
        
        # Reset history
        self.wealth_history = [initial_wealth]
        self.position_history = []
        
        # Initialize positions
        n_pairs = len(self.pairs)
        self.positions = np.zeros(n_pairs)
        
        # Get dates for backtest
        dates = self.spread_matrix.index
        
        # Backtest over all but the last day
        backtest_dates = dates[:-1]
        
        print(f"Running backtest from {backtest_dates[0]} to {backtest_dates[-1]} ({len(backtest_dates)} days)...")
        
        daily_returns = []
        position_sizes = []
        
        for i, current_date in enumerate(backtest_dates):
            try:
                # Get current spreads
                current_spreads = self.spread_matrix.loc[current_date].values
                
                # Calculate optimal positions
                new_positions = self.calculate_optimal_positions(
                    current_spreads, self.positions, self.wealth
                )
                
                # Calculate transaction costs
                position_changes = new_positions - self.positions
                transaction_costs = self.transaction_cost * np.sum(np.abs(position_changes))
                
                # Pre-trade wealth
                pre_trade_wealth = self.wealth
                
                # Subtract transaction costs
                self.wealth -= transaction_costs
                
                # Update positions
                self.positions = new_positions
                
                # Get next day's spreads
                next_date = dates[dates.get_loc(current_date) + 1]
                next_spreads = self.spread_matrix.loc[next_date].values
                
                # Calculate spread changes
                spread_changes = next_spreads - current_spreads
                
                # Calculate P&L
                pnl = np.dot(self.positions, spread_changes)
                
                # Update wealth
                self.wealth += pnl
                
                # Enforce minimum wealth
                self.wealth = max(0.01, self.wealth)
                
                # Calculate daily return
                daily_return = (self.wealth / pre_trade_wealth) - 1
                daily_returns.append(daily_return)
                
                # Calculate position size
                position_size = np.sum(np.abs(self.positions)) / self.wealth
                position_sizes.append(position_size)
                
                # Save history
                self.position_history.append(self.positions.copy())
                self.wealth_history.append(self.wealth)
                
                # Print progress
                if (i + 1) % 100 == 0 or i == len(backtest_dates) - 1:
                    print(f"Processed {i+1}/{len(backtest_dates)} days. " +
                          f"Wealth: {self.wealth:.4f}, " +
                          f"Daily return: {daily_return:.2%}")
            
            except Exception as e:
                print(f"Error on date {current_date}: {e}")
                # Skip this date
                continue
        
        # Calculate performance metrics
        wealth_series = pd.Series(self.wealth_history, index=[backtest_dates[0]] + list(dates[1:len(backtest_dates)+1]))
        returns = pd.Series(daily_returns, index=dates[1:len(backtest_dates)+1])
        positions = pd.Series(position_sizes, index=dates[1:len(backtest_dates)+1])
        
        # Annualization factor
        annual_factor = 252
        
        # Calculate metrics
        total_return = (self.wealth - initial_wealth) / initial_wealth
        annual_return = (1 + total_return) ** (annual_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        running_max = wealth_series.cummax()
        drawdown = (wealth_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Compile results
        results = {
            'wealth_series': wealth_series,
            'returns': returns,
            'positions': positions,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        print("\nBacktest Results:")
        print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"Annualized Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
        print(f"Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        print(f"Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
        
        return results
    
    def plot_results(self, results):
        """Plot backtest results"""
        if results is None:
            print("No results to plot.")
            return
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot wealth curve
        axes[0].plot(results['wealth_series'], 'b-')
        axes[0].set_title('Portfolio Wealth')
        axes[0].set_ylabel('Wealth')
        axes[0].grid(True)
        
        # Plot daily returns
        axes[1].plot(results['returns'], 'g-')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Daily Returns')
        axes[1].set_ylabel('Return')
        axes[1].grid(True)
        
        # Plot position sizes
        if 'positions' in results:
            axes[2].plot(results['positions'], 'b-')
            axes[2].set_title('Total Position Size (Multiple of Capital)')
            axes[2].set_ylabel('Position Size')
            axes[2].grid(True)
        
        # Plot drawdowns
        wealth_series = results['wealth_series']
        running_max = wealth_series.cummax()
        drawdown = (wealth_series - running_max) / running_max
        
        axes[3].fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
        axes[3].set_title('Drawdowns')
        axes[3].set_ylabel('Drawdown')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot individual pair spreads and positions
        n_pairs = len(self.pairs)
        
        # Calculate number of rows and columns for subplots
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), squeeze=False)
        
        # Get position history
        positions = np.array(self.position_history)
        
        for i, pair in enumerate(self.pairs):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            pair_name = f"{ticker1}-{ticker2}"
            
            # Plot positions
            dates = results['wealth_series'].index[1:len(positions)+1]
            ax.plot(dates, positions[:, i], 'b-', label='Position')
            
            # Plot normalized spread
            if pair_name in self.spread_matrix:
                spread = self.spread_matrix[pair_name].loc[dates[0]:dates[-1]]
                
                # Normalize spread for visual comparison
                spread_norm = (spread - spread.mean()) / spread.std()
                
                # Scale to match position range for better visualization
                pos_range = np.max(np.abs(positions[:, i])) if len(positions) > 0 else 1
                spread_norm = spread_norm * pos_range * 0.5
                
                # Plot on secondary y-axis
                ax2 = ax.twinx()
                ax2.plot(spread.index, spread_norm, 'r-', alpha=0.5, label='Spread')
                ax2.set_ylabel('Normalized Spread', color='r')
            
            ax.set_title(f"{ticker1}-{ticker2}")
            ax.set_ylabel('Position')
            ax.grid(True)
        
        # Hide any unused subplots
        for i in range(n_pairs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()

def run_bloomberg_example():
    """Run MPC pairs trading example with Bloomberg data"""
    # Create model
    mpc = MPCPairsTrading()
    
    # Add pairs to trade
    # Using common pairs across different sectors
    pairs = [
        # US Tech pairs
        ("AAPL US Equity", "MSFT US Equity"),
        ("GOOGL US Equity", "META US Equity"),
        
        # US Financial pairs
        ("JPM US Equity", "BAC US Equity"),
        ("GS US Equity", "MS US Equity"),
        
        # US Healthcare pairs
        ("PFE US Equity", "JNJ US Equity"),
        ("AMGN US Equity", "ABBV US Equity"),
        
        # US Consumer pairs
        ("KO US Equity", "PEP US Equity"),
        ("WMT US Equity", "TGT US Equity"),
    ]
    
    for ticker1, ticker2 in pairs:
        mpc.add_pair(ticker1, ticker2)
    
    # Load data from Bloomberg
    # Default is last 140 days for intraday data
    success = mpc.load_data_from_bloomberg()
    
    if not success:
        print("Could not load Bloomberg data. Using simulated data instead.")
        mpc.generate_simulated_data(n_days=500)
    
    # 1. Compare transaction costs
    print("\n1. Testing different transaction cost levels with horizon=5:")
    
    # Run backtests with different transaction costs
    results_0bps = mpc.backtest(
        transaction_cost=0.0000,  # 0 bps
        prediction_horizon=5,
        position_limit=0.2
    )
    
    results_10bps = mpc.backtest(
        transaction_cost=0.0010,  # 10 bps
        prediction_horizon=5,
        position_limit=0.2
    )
    
    results_40bps = mpc.backtest(
        transaction_cost=0.0040,  # 40 bps
        prediction_horizon=5,
        position_limit=0.2
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(results_0bps['wealth_series'], 'b-', label='0 bps')
    plt.plot(results_10bps['wealth_series'], 'g-', label='10 bps')
    plt.plot(results_40bps['wealth_series'], 'r-', label='40 bps')
    plt.title('Portfolio Wealth with Different Transaction Cost Levels')
    plt.xlabel('Date')
    plt.ylabel('Wealth')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 2. Compare prediction horizons
    print("\n2. Testing different prediction horizons with transaction_cost=0.0020:")
    
    # Run backtests with different horizons
    results_horizon_1 = mpc.backtest(
        prediction_horizon=1,  # Myopic
        transaction_cost=0.0020,  # 20 bps
        position_limit=0.2
    )
    
    results_horizon_5 = mpc.backtest(
        prediction_horizon=5,  # Medium horizon
        transaction_cost=0.0020,  # 20 bps
        position_limit=0.2
    )
    
    results_horizon_10 = mpc.backtest(
        prediction_horizon=10,  # Long horizon
        transaction_cost=0.0020,  # 20 bps
        position_limit=0.2
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(results_horizon_1['wealth_series'], 'b-', label='Horizon = 1 (Myopic)')
    plt.plot(results_horizon_5['wealth_series'], 'g-', label='Horizon = 5')
    plt.plot(results_horizon_10['wealth_series'], 'r-', label='Horizon = 10')
    plt.title('Portfolio Wealth with Different Prediction Horizons')
    plt.xlabel('Date')
    plt.ylabel('Wealth')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print metrics comparison
    print("\nPerformance Metrics:")
    print("\nTransaction Cost Levels (τ = 5):")
    print(f"  0 bps:  Sharpe = {results_0bps['sharpe_ratio']:.4f}, Return = {results_0bps['annual_return']*100:.2f}%, Volatility = {results_0bps['volatility']*100:.2f}%")
    print(f"  10 bps: Sharpe = {results_10bps['sharpe_ratio']:.4f}, Return = {results_10bps['annual_return']*100:.2f}%, Volatility = {results_10bps['volatility']*100:.2f}%")
    print(f"  40 bps: Sharpe = {results_40bps['sharpe_ratio']:.4f}, Return = {results_40bps['annual_return']*100:.2f}%, Volatility = {results_40bps['volatility']*100:.2f}%")
    
    print("\nPrediction Horizons (20 bps):")
    print(f"  τ = 1:  Sharpe = {results_horizon_1['sharpe_ratio']:.4f}, Return = {results_horizon_1['annual_return']*100:.2f}%, Volatility = {results_horizon_1['volatility']*100:.2f}%")
    print(f"  τ = 5:  Sharpe = {results_horizon_5['sharpe_ratio']:.4f}, Return = {results_horizon_5['annual_return']*100:.2f}%, Volatility = {results_horizon_5['volatility']*100:.2f}%")
    print(f"  τ = 10: Sharpe = {results_horizon_10['sharpe_ratio']:.4f}, Return = {results_horizon_10['annual_return']*100:.2f}%, Volatility = {results_horizon_10['volatility']*100:.2f}%")
    
    # Plot detailed results for best configuration
    print("\nDetailed results for best configuration:")
    mpc.plot_results(results_horizon_5)
    
    # Clean up Bloomberg connection
    if mpc.bbg_fetcher:
        mpc.bbg_fetcher.stop_session()

if __name__ == "__main__":
    run_bloomberg_example()