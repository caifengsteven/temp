import pdblp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import random

# Create directory for results
os.makedirs('results', exist_ok=True)

# Connect to Bloomberg
print("Connecting to Bloomberg...")
con = pdblp.BCon(timeout=5000)
con.start()

class IntrinsicEntropyStrategy:
    def __init__(self, tickers, start_date, end_date, reference_price='previous', use_simulated_data=True):
        """
        Initialize the Intrinsic Entropy Strategy
        
        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date in format 'YYYYMMDD'
            end_date: End date in format 'YYYYMMDD'
            reference_price: Type of reference price ('opening', 'previous', or 'vwap')
            use_simulated_data: Whether to use simulated data instead of trying to fetch intraday data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.reference_price = reference_price
        self.use_simulated_data = use_simulated_data
        self.trade_data = {}
        self.intraday_data = {}
        self.entropy_data = {}
        self.returns = {}
        
    def fetch_trade_data(self, con):
        """Fetch trade data from Bloomberg or generate completely synthetic data"""
        print(f"Fetching trade data for {len(self.tickers)} tickers...")
        
        # Parse start and end dates
        start_date_dt = pd.to_datetime(self.start_date, format='%Y%m%d')
        end_date_dt = pd.to_datetime(self.end_date, format='%Y%m%d')
        date_range = pd.date_range(start=start_date_dt, end=end_date_dt)
        
        # Create business days mask - exclude weekends
        is_business_day = [d.weekday() < 5 for d in date_range]
        business_days = date_range[is_business_day]
        
        for ticker in self.tickers:
            print(f"Getting data for {ticker}...")
            
            # Generate fully synthetic daily data for this ticker
            synthetic_daily_data = self.generate_synthetic_daily_data(ticker, business_days)
            self.trade_data[ticker] = synthetic_daily_data
            
            # Initialize intraday data with an empty dataframe
            self.intraday_data[ticker] = pd.DataFrame()
            
            # Generate synthetic intraday data for each business day
            for date in business_days:
                date_str = date.strftime('%Y-%m-%d')
                print(f"Generating intraday data for {ticker} on {date_str}...")
                
                # Get daily data for this date from our synthetic daily data
                daily_row = synthetic_daily_data.loc[date_str]
                
                # Generate synthetic intraday trades for this day
                synthetic_intraday = self.generate_synthetic_intraday(ticker, date, daily_row)
                self.intraday_data[ticker] = pd.concat([self.intraday_data[ticker], synthetic_intraday])
    
    def generate_synthetic_daily_data(self, ticker, business_days):
        """Generate synthetic daily data for a ticker"""
        print(f"Generating synthetic daily data for {ticker}")
        
        # Initialize with a starting price around 100
        base_prices = {
            'AGG US Equity': 107.5,
            'DBC US Equity': 15.2,
            'VIX Index': 18.5,
            'VTI US Equity': 240.0,
            'SPY US Equity': 450.0,
            'QQQ US Equity': 380.0,
            'IWM US Equity': 195.0,
            'GLD US Equity': 185.0
        }
        
        # Use the ticker's base price or default to 100
        starting_price = base_prices.get(ticker, 100.0) + np.random.normal(0, base_prices.get(ticker, 100.0) * 0.05)
        
        # Generate daily price changes with persistence
        returns = np.random.normal(0.0001, 0.01, len(business_days))
        for i in range(1, len(returns)):
            # Add some autocorrelation
            returns[i] = 0.6 * returns[i-1] + 0.4 * returns[i]
        
        # Convert returns to prices
        prices = starting_price * np.cumprod(1 + returns)
        
        # Generate other OHLCV data
        data = []
        for i, date in enumerate(business_days):
            price = prices[i]
            
            # Create high and low with some randomness
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            
            # Make sure high >= price >= low
            high = max(high, price)
            low = min(low, price)
            
            # Generate open price between yesterday's close and today's close
            if i == 0:
                open_price = price * (1 + np.random.normal(0, 0.005))
            else:
                prev_close = prices[i-1]
                open_price = prev_close + (price - prev_close) * np.random.random()
            
            # Make sure open is within high-low range
            open_price = max(min(open_price, high), low)
            
            # Generate volume based on ticker type
            if 'Equity' in ticker:
                # Higher volume for equities
                volume = np.random.randint(500000, 5000000)
            else:
                # Lower volume for indices
                volume = np.random.randint(100000, 1000000)
            
            data.append({
                'PX_LAST': price,
                'PX_OPEN': open_price,
                'PX_HIGH': high,
                'PX_LOW': low,
                'PX_VOLUME': volume
            })
        
        # Create DataFrame with dates as index
        df = pd.DataFrame(data, index=[d.strftime('%Y-%m-%d') for d in business_days])
        
        return df
    
    def generate_synthetic_intraday(self, ticker, date, daily_row):
        """Generate synthetic intraday data for a given day"""
        # Extract daily values from the Series
        open_price = daily_row['PX_OPEN']
        close_price = daily_row['PX_LAST']
        high_price = daily_row['PX_HIGH']
        low_price = daily_row['PX_LOW']
        volume = daily_row['PX_VOLUME']
        
        # Create 20 synthetic trades throughout the day
        n_trades = 20
        
        # Create trading times during market hours
        market_open = pd.Timestamp.combine(date.date(), pd.Timestamp('09:30:00').time())
        market_close = pd.Timestamp.combine(date.date(), pd.Timestamp('16:00:00').time())
        seconds_in_trading_day = (market_close - market_open).total_seconds()
        
        # Generate random times within market hours
        random_seconds = sorted(random.sample(range(int(seconds_in_trading_day)), n_trades))
        times = [market_open + timedelta(seconds=s) for s in random_seconds]
        
        # Simulate realistic price path
        # Start at open, end at close, with proper high/low constraints
        prices = []
        
        # Brownian bridge - constrained random walk from open to close
        curr_price = open_price
        for i in range(n_trades):
            # How far along the day are we (0 to 1)
            progress = i / (n_trades - 1) if n_trades > 1 else 1
            
            # Target moves from open to close as day progresses
            target = open_price * (1 - progress) + close_price * progress
            
            # Random walk with drift toward target
            drift = (target - curr_price) * 0.2  # Strength of pull toward target
            volatility = (high_price - low_price) * 0.05  # Random component
            change = drift + np.random.normal(0, volatility)
            
            # Ensure price stays within daily high/low bounds
            curr_price = curr_price + change
            curr_price = max(min(curr_price, high_price), low_price)
            prices.append(curr_price)
        
        # Make sure the last price matches the close
        if n_trades > 0:
            prices[-1] = close_price
        
        # Distribute volume across trades based on time of day
        time_weights = []
        for t in times:
            # Time of day in hours
            hour = t.hour + t.minute/60
            
            # Weight more heavily around market open (9:30), lunch (12:00), and close (16:00)
            open_weight = max(0, 1 - abs(hour - 9.5) * 0.5)
            lunch_weight = max(0, 1 - abs(hour - 12) * 0.5)
            close_weight = max(0, 1 - abs(hour - 16) * 0.5)
            
            # Combined weight
            weight = 0.5 + open_weight + lunch_weight + close_weight
            time_weights.append(weight)
        
        # Normalize weights to sum to 1
        if time_weights:
            time_weights = np.array(time_weights) / np.sum(time_weights)
            
            # Distribute volume with some randomness
            raw_volumes = time_weights * volume
            volumes = np.round(raw_volumes * (1 + np.random.normal(0, 0.2, n_trades)))
            volumes = np.maximum(volumes, 1)  # Ensure minimum of 1 share per trade
            
            # Adjust to match daily volume
            volumes = volumes * (volume / volumes.sum())
            volumes = np.round(volumes).astype(int)
            volumes[volumes < 1] = 1  # Ensure minimum volume
        else:
            volumes = []
        
        # Create DataFrame for this day's synthetic data
        if len(times) > 0 and len(prices) > 0 and len(volumes) > 0:
            synthetic_data = pd.DataFrame({
                'time': times,
                'price': prices,
                'value': volumes,
                'date': date
            })
            print(f"Generated {n_trades} synthetic trades for {ticker} on {date.strftime('%Y-%m-%d')}")
            return synthetic_data
        else:
            print(f"Warning: Could not generate trades for {ticker} on {date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
    
    def calculate_intrinsic_entropy(self):
        """Calculate intrinsic entropy for each ticker and trading day"""
        print("Calculating intrinsic entropy...")
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            intraday_df = self.intraday_data[ticker]
            
            if intraday_df.empty:
                print(f"No intraday data for {ticker}")
                continue
            
            # Group by date
            grouped = intraday_df.groupby('date')
            entropy_results = {}
            
            for date, group in grouped:
                # Sort by time
                group = group.sort_values('time')
                
                # Skip if less than 5 trades
                if len(group) < 5:  # Reduced minimum to 5 trades due to potential data limitations
                    print(f"Skipping {ticker} on {date.strftime('%Y-%m-%d')} - not enough trades ({len(group)})")
                    continue
                
                # Calculate cumulative traded quantity
                group['cumulative_quantity'] = group['value'].cumsum()
                
                # Initialize arrays for entropy and components
                n_trades = len(group)
                entropy_values = np.zeros(n_trades)
                price_weights = np.zeros(n_trades)
                probability_values = np.zeros(n_trades)
                log_probability_values = np.zeros(n_trades)
                product_values = np.zeros(n_trades)
                
                # Set reference prices based on selected method
                if self.reference_price == 'opening':
                    # Use day's opening price
                    daily_data = self.trade_data[ticker]
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str in daily_data.index:
                        ref_price = daily_data.loc[date_str, 'PX_OPEN']
                        ref_prices = np.full(n_trades, ref_price)
                    else:
                        # If opening price not available, use first trade price
                        ref_prices = np.full(n_trades, group.iloc[0]['price'])
                        
                elif self.reference_price == 'previous':
                    # Use previous trade price (Markov chain)
                    ref_prices = np.zeros(n_trades)
                    ref_prices[0] = group.iloc[0]['price']  # First trade uses itself as reference
                    ref_prices[1:] = group['price'].values[:-1]  # Shift prices for rest
                    
                elif self.reference_price == 'vwap':
                    # Use VWAP up to previous trade
                    ref_prices = np.zeros(n_trades)
                    ref_prices[0] = group.iloc[0]['price']  # First trade uses itself as reference
                    
                    # Calculate running VWAP
                    for i in range(1, n_trades):
                        prev_data = group.iloc[:i]
                        vwap = (prev_data['price'] * prev_data['value']).sum() / prev_data['value'].sum()
                        ref_prices[i] = vwap
                
                # Calculate entropy components for each trade
                for i in range(n_trades):
                    price = group.iloc[i]['price']
                    quantity = group.iloc[i]['value']
                    total_quantity = group.iloc[i]['cumulative_quantity']
                    
                    # Calculate probability (q_i / Q)
                    probability = quantity / total_quantity
                    probability_values[i] = probability
                    
                    # Calculate log of probability
                    log_prob = np.log(probability)
                    log_probability_values[i] = log_prob
                    
                    # Calculate price weight (p_i / p_ref - 1)
                    price_weight = price / ref_prices[i] - 1
                    price_weights[i] = price_weight
                    
                    # Calculate product for entropy component
                    product = price_weight * probability * log_prob
                    product_values[i] = product
                    
                    # Cumulative entropy up to this trade
                    entropy_values[i] = -np.sum(product_values[:i+1])
                
                # Store results for this date
                entropy_results[date] = {
                    'trades': group,
                    'entropy': entropy_values,
                    'price_weights': price_weights,
                    'probabilities': probability_values,
                    'log_probabilities': log_probability_values,
                    'products': product_values
                }
                
                print(f"Calculated entropy for {ticker} on {date.strftime('%Y-%m-%d')} with {n_trades} trades")
            
            self.entropy_data[ticker] = entropy_results
    
    def backtest_strategy(self, min_trades=5, cost_rate=0.0002):
        """
        Backtest the entropy-based trading strategy
        
        Args:
            min_trades: Minimum number of trades needed before making decisions
            cost_rate: Transaction cost rate
        """
        print("Backtesting strategies...")
        
        # Results storage
        entropy_strategy_results = {}
        vwap_strategy_results = {}
        
        for ticker in self.tickers:
            if ticker not in self.entropy_data:
                print(f"No entropy data for {ticker}, skipping backtest")
                continue
                
            entropy_results = self.entropy_data[ticker]
            print(f"Backtesting {ticker} with {len(entropy_results)} days of data")
            
            # Process each trading day
            for date, data in entropy_results.items():
                trades = data['trades']
                entropy_values = data['entropy']
                
                # Skip if not enough trades
                if len(trades) < min_trades:
                    continue
                
                # Calculate VWAP for each trade
                trades['running_vwap'] = (trades['price'] * trades['value']).cumsum() / trades['value'].cumsum()
                
                # Entropy-based strategy
                entropy_buy_flag = False
                entropy_sell_flag = False
                entropy_buy_price = 0
                entropy_buy_trade_no = 0
                entropy_sell_price = 0
                entropy_sell_trade_no = 0
                
                # VWAP-only strategy
                vwap_buy_flag = False
                vwap_sell_flag = False
                vwap_buy_price = 0
                vwap_buy_trade_no = 0
                vwap_sell_price = 0
                vwap_sell_trade_no = 0
                
                # Loop through trades starting from min_trades
                for i in range(min_trades, len(trades)):
                    trade = trades.iloc[i]
                    price = trade['price']
                    vwap = trade['running_vwap']
                    entropy = entropy_values[i]
                    
                    # Entropy strategy
                    if not entropy_buy_flag and entropy > 0 and price < vwap:
                        # Buy signal
                        entropy_buy_flag = True
                        entropy_buy_price = price
                        entropy_buy_trade_no = i
                    
                    elif entropy_buy_flag and not entropy_sell_flag and i > entropy_buy_trade_no:
                        # Sell conditions: entropy turned negative or price above VWAP or end of day
                        if entropy < 0 or price > vwap or i >= len(trades) - 1:
                            entropy_sell_flag = True
                            entropy_sell_price = price
                            entropy_sell_trade_no = i
                    
                    # VWAP-only strategy
                    if not vwap_buy_flag and price < vwap:
                        # Buy signal
                        vwap_buy_flag = True
                        vwap_buy_price = price
                        vwap_buy_trade_no = i
                    
                    elif vwap_buy_flag and not vwap_sell_flag and i > vwap_buy_trade_no:
                        # Sell conditions: price above VWAP or end of day
                        if price > vwap or i >= len(trades) - 1:
                            vwap_sell_flag = True
                            vwap_sell_price = price
                            vwap_sell_trade_no = i
                
                # Calculate returns if trades were executed
                entropy_return = 0
                if entropy_buy_flag and entropy_sell_flag:
                    # Apply transaction costs
                    cost = cost_rate * (entropy_buy_price + entropy_sell_price)
                    entropy_return = (entropy_sell_price - entropy_buy_price) / entropy_buy_price - cost
                
                vwap_return = 0
                if vwap_buy_flag and vwap_sell_flag:
                    # Apply transaction costs
                    cost = cost_rate * (vwap_buy_price + vwap_sell_price)
                    vwap_return = (vwap_sell_price - vwap_buy_price) / vwap_buy_price - cost
                
                # Store results
                date_str = date.strftime('%Y-%m-%d')
                
                if ticker not in entropy_strategy_results:
                    entropy_strategy_results[ticker] = {}
                entropy_strategy_results[ticker][date_str] = {
                    'buy_executed': entropy_buy_flag,
                    'sell_executed': entropy_sell_flag,
                    'buy_price': entropy_buy_price,
                    'sell_price': entropy_sell_price,
                    'buy_trade_no': entropy_buy_trade_no,
                    'sell_trade_no': entropy_sell_trade_no,
                    'return': entropy_return * 100  # Convert to percentage
                }
                
                if ticker not in vwap_strategy_results:
                    vwap_strategy_results[ticker] = {}
                vwap_strategy_results[ticker][date_str] = {
                    'buy_executed': vwap_buy_flag,
                    'sell_executed': vwap_sell_flag,
                    'buy_price': vwap_buy_price,
                    'sell_price': vwap_sell_price,
                    'buy_trade_no': vwap_buy_trade_no,
                    'sell_trade_no': vwap_sell_trade_no,
                    'return': vwap_return * 100  # Convert to percentage
                }
        
        # Compile results to compare strategies
        entropy_returns = []
        vwap_returns = []
        
        for ticker in self.tickers:
            if ticker in entropy_strategy_results and ticker in vwap_strategy_results:
                for date in entropy_strategy_results[ticker]:
                    if entropy_strategy_results[ticker][date]['buy_executed'] and entropy_strategy_results[ticker][date]['sell_executed']:
                        entropy_returns.append({
                            'ticker': ticker,
                            'date': date,
                            'return': entropy_strategy_results[ticker][date]['return']
                        })
                    
                    if vwap_strategy_results[ticker][date]['buy_executed'] and vwap_strategy_results[ticker][date]['sell_executed']:
                        vwap_returns.append({
                            'ticker': ticker,
                            'date': date,
                            'return': vwap_strategy_results[ticker][date]['return']
                        })
        
        entropy_returns_df = pd.DataFrame(entropy_returns)
        vwap_returns_df = pd.DataFrame(vwap_returns)
        
        # Calculate overall statistics
        if not entropy_returns_df.empty:
            entropy_total_return = entropy_returns_df['return'].sum()
            entropy_avg_return = entropy_returns_df['return'].mean()
            entropy_num_trades = len(entropy_returns_df)
        else:
            entropy_total_return = 0
            entropy_avg_return = 0
            entropy_num_trades = 0
        
        if not vwap_returns_df.empty:
            vwap_total_return = vwap_returns_df['return'].sum()
            vwap_avg_return = vwap_returns_df['return'].mean()
            vwap_num_trades = len(vwap_returns_df)
        else:
            vwap_total_return = 0
            vwap_avg_return = 0
            vwap_num_trades = 0
        
        # Store results
        self.returns = {
            'entropy_strategy': {
                'returns_df': entropy_returns_df,
                'total_return': entropy_total_return,
                'avg_return': entropy_avg_return,
                'num_trades': entropy_num_trades
            },
            'vwap_strategy': {
                'returns_df': vwap_returns_df,
                'total_return': vwap_total_return,
                'avg_return': vwap_avg_return,
                'num_trades': vwap_num_trades
            }
        }
        
        print("\nStrategy Results Summary:")
        print(f"Entropy Strategy: Total Return = {entropy_total_return:.2f}%, Avg Return = {entropy_avg_return:.2f}%, Trades = {entropy_num_trades}")
        print(f"VWAP Strategy: Total Return = {vwap_total_return:.2f}%, Avg Return = {vwap_avg_return:.2f}%, Trades = {vwap_num_trades}")
        
        # Save detailed results to CSV
        if not entropy_returns_df.empty:
            entropy_returns_df.to_csv('results/entropy_strategy_returns.csv', index=False)
        if not vwap_returns_df.empty:
            vwap_returns_df.to_csv('results/vwap_strategy_returns.csv', index=False)
        
        return self.returns
    
    def plot_entropy_market_map(self, date):
        """
        Plot entropy market map for a specific date
        
        Args:
            date: Date string in format 'YYYY-MM-DD'
        """
        print(f"Generating entropy market map for {date}...")
        
        # Get market cap data
        market_caps = {}
        final_entropy = {}
        
        for ticker in self.tickers:
            if ticker in self.entropy_data and date in [d.strftime('%Y-%m-%d') for d in self.entropy_data[ticker].keys()]:
                # Find the date object that matches the string
                date_obj = next((d for d in self.entropy_data[ticker].keys() if d.strftime('%Y-%m-%d') == date), None)
                
                if date_obj:
                    # Get the final entropy value for this ticker on this date
                    entropy_values = self.entropy_data[ticker][date_obj]['entropy']
                    if len(entropy_values) > 0:
                        final_entropy[ticker] = entropy_values[-1]
                        
                        # Get market cap from our synthetic data
                        try:
                            daily_data = self.trade_data[ticker]
                            date_str = date_obj.strftime('%Y-%m-%d')
                            if date_str in daily_data.index:
                                price = daily_data.loc[date_str, 'PX_LAST']
                                
                                # Use price * a large number as a proxy for market cap
                                # Weight larger for SPY, QQQ and give relative size to others
                                if ticker == 'SPY US Equity':
                                    market_caps[ticker] = price * 10000000
                                elif ticker == 'QQQ US Equity':
                                    market_caps[ticker] = price * 8000000
                                elif ticker == 'VTI US Equity':
                                    market_caps[ticker] = price * 6000000
                                elif ticker == 'IWM US Equity':
                                    market_caps[ticker] = price * 4000000
                                elif ticker == 'AGG US Equity':
                                    market_caps[ticker] = price * 3000000
                                elif ticker == 'GLD US Equity':
                                    market_caps[ticker] = price * 2000000
                                else:
                                    market_caps[ticker] = price * 1000000
                            else:
                                market_caps[ticker] = 1000000
                        except Exception as e:
                            print(f"Error getting market cap for {ticker}: {str(e)}")
                            market_caps[ticker] = 1000000
        
        if not market_caps:
            print(f"No data available for market map on {date}")
            return
        
        # Normalize market caps for rectangle sizes
        total_market_cap = sum(market_caps.values())
        for ticker in market_caps:
            market_caps[ticker] = market_caps[ticker] / total_market_cap
        
        # Create colormap for entropy values
        cmap = LinearSegmentedColormap.from_list('entropy_cmap', 
                                               [(0.7, 0, 0),      # Dark red for very negative
                                                (1, 0.4, 0.4),    # Light red for slightly negative
                                                (0, 0, 0),        # Black for zero
                                                (0.4, 1, 0.4),    # Light green for slightly positive
                                                (0, 0.7, 0)],     # Dark green for very positive
                                               N=100)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort tickers by market cap (descending)
        sorted_tickers = sorted(market_caps.keys(), key=lambda x: market_caps[x], reverse=True)
        
        # Find the min and max entropy values for color scaling
        entropy_values = list(final_entropy.values())
        if not entropy_values:
            print("No entropy values available for the map")
            return
            
        min_entropy = min(entropy_values)
        max_entropy = max(entropy_values)
        max_abs_entropy = max(abs(min_entropy), abs(max_entropy))
        
        # Set the colormap normalization range
        norm_range = max(0.01, max_abs_entropy)  # Avoid division by zero
        
        # Initialize variables for treemap layout
        x, y = 0, 0
        width = 100
        height = 60
        row_width = width
        row_height = 0
        
        # Draw rectangles for each ticker
        for ticker in sorted_tickers:
            rect_width = market_caps[ticker] * width
            
            # Check if we need to start a new row
            if x + rect_width > width:
                # Move to next row
                y += row_height
                x = 0
                row_height = 0
            
            # Determine rectangle height (proportional to sqrt of market cap)
            rect_height = 10 * np.sqrt(market_caps[ticker])
            row_height = max(row_height, rect_height)
            
            # Determine color based on entropy value
            entropy = final_entropy.get(ticker, 0)
            if abs(entropy) < 0.001:  # Very close to zero
                color = 'black'
            else:
                # Normalize entropy value to [-1, 1] range for colormap
                norm_entropy = max(min(entropy / norm_range, 1), -1)
                color = cmap((norm_entropy + 1) / 2)  # Map [-1, 1] to [0, 1] for colormap
            
            # Create rectangle
            rect = patches.Rectangle((x, y), rect_width, rect_height, linewidth=1, 
                                   edgecolor='white', facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            
            # Add ticker label if rectangle is big enough
            if rect_width > 3 and rect_height > 3:
                # Extract just the ticker part without the exchange
                ticker_display = ticker.split()[0]
                plt.text(x + rect_width/2, y + rect_height/2, ticker_display, 
                       ha='center', va='center', color='white',
                       fontsize=8)
            
            # Update x position for next rectangle
            x += rect_width
        
        # Set axis limits and remove axis ticks
        ax.set_xlim(0, width)
        ax.set_ylim(0, y + row_height)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-norm_range, norm_range))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Intrinsic Entropy')
        
        # Set title and save figure
        plt.title(f'Stock Intrinsic Entropy Market Map - {date}')
        plt.tight_layout()
        plt.savefig(f'results/entropy_market_map_{date}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entropy_examples(self, ticker, date):
        """
        Plot entropy evolution with price and volume for a specific ticker and date
        
        Args:
            ticker: Bloomberg ticker symbol
            date: Date string in format 'YYYY-MM-DD'
        """
        print(f"Generating entropy example plot for {ticker} on {date}...")
        
        # Convert string date to datetime object
        date_obj = pd.to_datetime(date)
        
        # Check if data exists
        if ticker not in self.entropy_data or date_obj not in self.entropy_data[ticker]:
            print(f"No data available for {ticker} on {date}")
            return
        
        # Get data
        data = self.entropy_data[ticker][date_obj]
        trades = data['trades']
        entropy_values = data['entropy']
        price_weights = data['price_weights']
        probabilities = data['probabilities']
        
        # Calculate VWAP for each trade
        trades['running_vwap'] = (trades['price'] * trades['value']).cumsum() / trades['value'].cumsum()
        
        # Use trade numbers for x-axis (more reliable than times)
        trades['trade_number'] = range(len(trades))
        trade_numbers = trades['trade_number'].values
        
        # Create figure with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # Plot 1: Price and VWAP
        ax1.plot(trade_numbers, trades['price'], label='Price', color='blue', linewidth=1)
        ax1.plot(trade_numbers, trades['running_vwap'], label='VWAP', color='red', linewidth=1, linestyle='--')
        ax1.set_ylabel('Price')
        ax1.set_title(f'{ticker} - {date}')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Entropy
        ax2.plot(trade_numbers, entropy_values, label='Intrinsic Entropy', color='green', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Entropy')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Entropy Components
        ax3.plot(trade_numbers, price_weights, label='Price Weights', color='purple', linewidth=1)
        ax3.plot(trade_numbers, probabilities, label='Probabilities', color='orange', linewidth=1)
        ax3.set_ylabel('Components')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Trade Volumes
        ax4.bar(trade_numbers, trades['value'], label='Volume', color='gray', alpha=0.7)
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Trade Number')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/{ticker.split()[0]}_{date}_entropy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, con):
        """Run the complete analysis pipeline"""
        try:
            # Step 1: Fetch trade data
            self.fetch_trade_data(con)
            
            # Step 2: Calculate intrinsic entropy
            self.calculate_intrinsic_entropy()
            
            # Step 3: Backtest trading strategies
            returns = self.backtest_strategy()
            
            # Step 4: Generate visualizations for sample dates and tickers
            if self.entropy_data:
                # Find a date with entropy data
                sample_dates = []
                for ticker in self.tickers:
                    if ticker in self.entropy_data:
                        for date_obj in self.entropy_data[ticker]:
                            date_str = date_obj.strftime('%Y-%m-%d')
                            if date_str not in sample_dates:
                                sample_dates.append(date_str)
                
                if sample_dates:
                    # Generate market map for the first date
                    self.plot_entropy_market_map(sample_dates[0])
                    
                    # Generate example plots for a few tickers
                    for ticker in self.tickers[:min(3, len(self.tickers))]:
                        if ticker in self.entropy_data:
                            for date_str in sample_dates[:1]:
                                self.plot_entropy_examples(ticker, date_str)
            else:
                print("No entropy data available for visualization")
            
            return returns
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Main execution
if __name__ == "__main__":
    # Define parameters
    tickers = ['AGG US Equity', 'DBC US Equity', 'VIX Index', 'VTI US Equity', 
             'SPY US Equity', 'QQQ US Equity', 'IWM US Equity', 'GLD US Equity']
    
    # Use specified date range
    start_date = "20250412"
    end_date = "20250426"
    
    print(f"Analysis period: {start_date} to {end_date}")
    
    # Initialize strategy with simulated data option
    strategy = IntrinsicEntropyStrategy(tickers, start_date, end_date, reference_price='previous', use_simulated_data=True)
    
    try:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Run analysis
        results = strategy.run_analysis(con)
        
        if results:
            # Print final summary
            print("\nFinal Results:")
            print("==============")
            print(f"Entropy Strategy:")
            print(f"  Total Return: {results['entropy_strategy']['total_return']:.2f}%")
            print(f"  Average Return per Trade: {results['entropy_strategy']['avg_return']:.2f}%")
            print(f"  Number of Trades: {results['entropy_strategy']['num_trades']}")
            print("")
            print(f"VWAP Strategy:")
            print(f"  Total Return: {results['vwap_strategy']['total_return']:.2f}%")
            print(f"  Average Return per Trade: {results['vwap_strategy']['avg_return']:.2f}%")
            print(f"  Number of Trades: {results['vwap_strategy']['num_trades']}")
            
            # Compare strategies
            if results['entropy_strategy']['num_trades'] > 0 and results['vwap_strategy']['num_trades'] > 0:
                performance_ratio = results['entropy_strategy']['total_return'] / results['vwap_strategy']['total_return'] if results['vwap_strategy']['total_return'] != 0 else float('inf')
                print(f"\nPerformance comparison: Entropy vs VWAP = {performance_ratio:.2f}x")
                
                if performance_ratio > 1:
                    print("The Intrinsic Entropy strategy outperformed the VWAP-only strategy")
                elif performance_ratio < 1:
                    print("The VWAP-only strategy outperformed the Intrinsic Entropy strategy")
                else:
                    print("Both strategies performed equally")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close Bloomberg connection
        con.stop()