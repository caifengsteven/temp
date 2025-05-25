import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimalCausalPathStrategy:
    """
    Implementation of the Optimal Causal Path algorithm for statistical arbitrage
    """
    
    def __init__(self):
        """Initialize the OCP strategy"""
        # Strategy parameters
        self.pairs = []
        self.prices = {}
        self.returns = {}
        self.true_lags = {}  # For storing true lags in simulation
        self.wealth = 1.0
        self.wealth_history = []
        self.positions = {}
        self.trades_history = []
        
        # Parameters - revised for better trading
        self.k = 0.5  # Bollinger band width (reduced for more signals)
        self.d = 5    # Moving window length (very short)
        self.top_s = 10  # Number of top pairs
        self.transaction_cost = 0.00005  # 0.5 basis point (reduced for simulation)
        self.min_formation_size = 20  # Reduced minimum data for formation
        self.min_profit_target = 0.0002  # 2 basis points minimum profit target
        self.min_correlation = 0.5  # Higher threshold for more reliable patterns
        self.min_lag = 1  # Minimum lag to consider
        self.max_loss = -0.001  # Maximum loss tolerance (10 basis points)
        self.profit_taking_threshold = 0.0005  # Take profits at 5 basis points
    
    def add_pair(self, ticker1, ticker2):
        """Add a pair to the universe"""
        self.pairs.append({
            'ticker1': ticker1,
            'ticker2': ticker2
        })
        print(f"Added pair: {ticker1} - {ticker2}")
    
    def generate_perfect_data(self, n_days=10, minutes_per_day=100):
        """
        Generate perfect price data with obvious lead-lag relationships that are easy to profit from
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        minutes_per_day : int
            Number of minutes per trading day
            
        Returns:
        --------
        bool
            True if data was generated successfully
        """
        # Create date range
        end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=n_days*2)  # Double the days to account for weekends
        
        # Generate trading days (excluding weekends)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in all_dates if d.weekday() < 5]  # Monday to Friday
        trading_days = trading_days[-n_days:]  # Take the last n_days trading days
        
        # Create minute-by-minute timestamps for each trading day
        timestamps = []
        for day in trading_days:
            day_open = day.replace(hour=9, minute=30, second=0)
            day_minutes = pd.date_range(
                start=day_open,
                periods=minutes_per_day,
                freq='1min'
            )
            timestamps.extend(day_minutes)
        
        print(f"Creating data for {len(timestamps)} minutes across {len(trading_days)} trading days")
        
        # Get unique tickers from pairs
        tickers = set()
        for pair in self.pairs:
            tickers.add(pair['ticker1'])
            tickers.add(pair['ticker2'])
        tickers = list(tickers)
        
        n_timestamps = len(timestamps)
        
        # First generate all lead tickers with simple patterns
        for ticker in tickers:
            # Create a price series with clear, predictable patterns
            prices = np.ones(n_timestamps) * 100.0
            
            # Add a pattern with clear zigzags for easy trading signals
            for i in range(1, n_timestamps):
                # Create a predictable zigzag pattern
                cycle_length = 20  # Complete cycle in 20 minutes
                position_in_cycle = i % cycle_length
                
                if position_in_cycle < cycle_length / 2:
                    # Upward trend in first half of cycle
                    change = 0.002  # 20 basis points per minute (very large for testing)
                else:
                    # Downward trend in second half
                    change = -0.002
                
                # Add tiny noise
                noise = 0.0001 * np.random.randn()
                
                # Update price
                prices[i] = prices[i-1] * (1 + change + noise)
            
            # Store prices
            self.prices[ticker] = pd.Series(prices, index=timestamps)
            
            # Calculate returns
            self.returns[ticker] = self.prices[ticker].pct_change().fillna(0)
        
        # Now create perfect follow patterns with fixed lags
        lag_values = [5, 10, 15, 20]
        
        for i, pair in enumerate(self.pairs):
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            
            # Determine lag - cycle through lag values
            lag = lag_values[i % len(lag_values)]
            
            # For even indices, ticker1 leads ticker2
            # For odd indices, ticker2 leads ticker1
            if i % 2 == 0:
                lead_ticker = ticker1
                follow_ticker = ticker2
            else:
                lead_ticker = ticker2
                follow_ticker = ticker1
            
            # Get lead prices
            lead_prices = self.prices[lead_ticker].values
            
            # Create follow prices by exactly copying lead prices with lag
            follow_prices = np.zeros(n_timestamps)
            
            # First few values based on lead with small offset
            for t in range(lag):
                follow_prices[t] = lead_prices[0] * (1 + 0.001 * np.random.randn())
            
            # Rest follow the lead with a fixed lag - almost exact copy
            for t in range(lag, n_timestamps):
                # Direct copy with 99.9% accuracy and tiny noise
                lead_value = lead_prices[t-lag]
                noise = 0.00005 * np.random.randn()  # Very small noise
                follow_prices[t] = lead_value * (1 + noise)
            
            # Override the original prices for the follow ticker
            self.prices[follow_ticker] = pd.Series(follow_prices, index=timestamps)
            
            # Recalculate returns
            self.returns[follow_ticker] = self.prices[follow_ticker].pct_change().fillna(0)
            
            # Store the true lag
            pair_id = f"{ticker1}-{ticker2}"
            self.true_lags[pair_id] = (lead_ticker, follow_ticker, lag)
            
            print(f"Created perfect lead-lag: {lead_ticker} leads {follow_ticker} by {lag} minutes")
        
        print(f"Successfully generated perfect data from {timestamps[0]} to {timestamps[-1]}")
        return True
    
    def calc_lagged_correlation(self, x, y, max_lag=30):
        """
        Calculate lagged correlations between two time series
        
        Parameters:
        -----------
        x : array-like
            First time series
        y : array-like
            Second time series
        max_lag : int
            Maximum lag to consider
            
        Returns:
        --------
        best_lag : int
            Lag with highest absolute correlation
        best_corr : float
            Highest correlation value (absolute)
        all_corrs : list
            All correlation values for each lag
        """
        x = np.array(x)
        y = np.array(y)
        
        # Calculate correlation for different lags
        corrs = []
        for lag in range(max_lag + 1):
            if lag >= len(x) or lag >= len(y):
                corrs.append(0)
                continue
                
            # x leads y with lag
            x_lagged = x[:-lag] if lag > 0 else x
            y_lagged = y[lag:] if lag > 0 else y
            
            # Ensure same length
            min_len = min(len(x_lagged), len(y_lagged))
            if min_len < 5:  # Need enough data
                corrs.append(0)
                continue
                
            x_lagged = x_lagged[:min_len]
            y_lagged = y_lagged[:min_len]
            
            # Calculate correlation
            if np.std(x_lagged) > 0 and np.std(y_lagged) > 0:
                corr = np.corrcoef(x_lagged, y_lagged)[0, 1]
                corrs.append(corr)
            else:
                corrs.append(0)
        
        # Find best lag (highest absolute correlation)
        best_lag = np.argmax(np.abs(corrs))
        best_corr = corrs[best_lag]
        
        return best_lag, best_corr, corrs
    
    def formation_period(self, period_start, period_end):
        """
        Implement the formation period logic
        
        Parameters:
        -----------
        period_start : datetime
            Start of the formation period
        period_end : datetime
            End of the formation period
            
        Returns:
        --------
        top_pairs : list
            List of the top s pairs with the most stable lead-lag structure
        """
        print(f"Running formation period from {period_start} to {period_end}")
        
        # Calculate lead-lag structure for all pairs
        pairs_analysis = []
        
        for pair in self.pairs:
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            pair_id = f"{ticker1}-{ticker2}"
            
            # Get returns for both tickers in the formation period
            if ticker1 not in self.returns or ticker2 not in self.returns:
                continue
                
            r1 = self.returns[ticker1].loc[period_start:period_end]
            r2 = self.returns[ticker2].loc[period_start:period_end]
            
            # Ensure we have data for both tickers
            common_index = r1.index.intersection(r2.index)
            if len(common_index) < self.min_formation_size:  # Need enough data
                continue
                
            r1 = r1.loc[common_index]
            r2 = r2.loc[common_index]
            
            try:
                # Check if ticker1 leads ticker2
                lag_1_2, corr_1_2, _ = self.calc_lagged_correlation(r1.values, r2.values)
                
                # Check if ticker2 leads ticker1
                lag_2_1, corr_2_1, _ = self.calc_lagged_correlation(r2.values, r1.values)
                
                # Determine which direction has stronger correlation
                if abs(corr_1_2) > abs(corr_2_1) and lag_1_2 >= self.min_lag:
                    # ticker1 leads ticker2
                    lag = lag_1_2
                    correlation = corr_1_2
                    lead_ticker = ticker1
                    follow_ticker = ticker2
                    fluctuation = 1 / (abs(correlation) + 0.1)  # Lower correlation = higher fluctuation
                elif lag_2_1 >= self.min_lag:
                    # ticker2 leads ticker1
                    lag = lag_2_1
                    correlation = corr_2_1
                    lead_ticker = ticker2
                    follow_ticker = ticker1
                    fluctuation = 1 / (abs(correlation) + 0.1)  # Lower correlation = higher fluctuation
                else:
                    # No significant lag detected
                    continue
                
                # If correlation meets minimum threshold, add to candidates
                if abs(correlation) > self.min_correlation:
                    pairs_analysis.append({
                        'pair_id': pair_id,
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'lead_ticker': lead_ticker,
                        'follow_ticker': follow_ticker,
                        'lag': lag,
                        'fluctuation': fluctuation,
                        'correlation': correlation
                    })
                    
                    # Debug - print detected relationships
                    print(f"  Detected: {lead_ticker} leads {follow_ticker} by {lag} minutes " +
                          f"(corr: {correlation:.4f})")
                          
                    # Check against true lag if this is simulated data
                    if pair_id in self.true_lags:
                        true_lead, true_follow, true_lag = self.true_lags[pair_id]
                        detected_correctly = (true_lead == lead_ticker and true_follow == follow_ticker)
                        print(f"    True lag: {true_lead} leads {true_follow} by {true_lag} min. " +
                              f"Detected correctly: {detected_correctly}")
                    
            except Exception as e:
                print(f"Error analyzing pair {ticker1}-{ticker2}: {e}")
        
        # Select top s pairs with highest absolute correlation
        pairs_analysis.sort(key=lambda x: -abs(x['correlation']))
        top_pairs = pairs_analysis[:self.top_s]
        
        print(f"Selected {len(top_pairs)} top pairs for trading")
        return top_pairs
    
    def trading_period(self, top_pairs, period_start, period_end):
        """
        Implement the trading period logic
        
        Parameters:
        -----------
        top_pairs : list
            List of the top pairs from the formation period
        period_start : datetime
            Start of the trading period
        period_end : datetime
            End of the trading period
            
        Returns:
        --------
        returns : float
            Returns for the trading period
        """
        print(f"Running trading period from {period_start} to {period_end}")
        
        # Initialize positions
        self.positions = {p['pair_id']: 0 for p in top_pairs}
        trades = []
        
        # Process each pair
        for pair in top_pairs:
            pair_id = pair['pair_id']
            lead_ticker = pair['lead_ticker']
            follow_ticker = pair['follow_ticker']
            lag = pair['lag']
            correlation = pair['correlation']
            
            # Get returns for both tickers in the trading period
            if lead_ticker not in self.returns or follow_ticker not in self.returns:
                continue
                
            r_lead = self.returns[lead_ticker].loc[period_start:period_end]
            r_follow = self.returns[follow_ticker].loc[period_start:period_end]
            
            # Get prices for the trading period
            p_lead = self.prices[lead_ticker].loc[period_start:period_end]
            p_follow = self.prices[follow_ticker].loc[period_start:period_end]
            
            # Ensure we have data for both tickers
            common_index = r_lead.index.intersection(r_follow.index)
            if len(common_index) < lag + 5:  # Need enough data considering the lag
                continue
            
            # Convert to lists for easier indexing
            common_index = list(common_index)
            r_lead = r_lead.loc[common_index].values
            r_follow = r_follow.loc[common_index].values
            p_lead = p_lead.loc[common_index].values
            p_follow = p_follow.loc[common_index].values
            
            # For each time step in the trading period
            for i in range(lag, len(common_index)):
                current_time = common_index[i]
                
                # Skip if we already have an open position
                if self.positions[pair_id] != 0:
                    continue
                
                # Get the leading return (lag timesteps ago)
                leading_return = r_lead[i - lag]
                
                # Calculate Bollinger Bands on the leading returns
                window = min(self.d, i)
                history = r_lead[max(0, i-window):i]
                
                mean_level = np.mean(history)
                std_level = np.std(history)
                
                upper_band = mean_level + self.k * std_level
                lower_band = mean_level - self.k * std_level
                
                # Entry signals
                entry_signal = 0
                
                # Check if the leading return exceeds the transaction cost
                if abs(leading_return) > self.transaction_cost:
                    if (leading_return > upper_band and correlation > 0) or \
                       (leading_return < lower_band and correlation < 0):
                        # The following stock is expected to increase
                        entry_signal = 1
                    elif (leading_return < lower_band and correlation > 0) or \
                         (leading_return > upper_band and correlation < 0):
                        # The following stock is expected to decrease
                        entry_signal = -1
                
                # Execute trades
                if entry_signal != 0:
                    entry_time = current_time
                    entry_idx = i
                    
                    # Set position
                    self.positions[pair_id] = entry_signal
                    
                    # Get entry price
                    entry_price = p_follow[i]
                    
                    # Look for exit opportunity in the next time steps
                    exit_time = None
                    exit_price = None
                    exit_idx = None
                    
                    # Define the confidence interval around the lag
                    max_exit_window = min(lag * 5, len(common_index) - i - 1)  # Longer window
                    max_exit_window = max(1, max_exit_window)  # Ensure at least 1 step
                    
                    for j in range(1, max_exit_window + 1):
                        if i + j >= len(common_index):
                            break
                            
                        # Check if returns exceed the threshold
                        current_exit_price = p_follow[i + j]
                        trade_return = (current_exit_price / entry_price - 1) * entry_signal
                        
                        # Exit on profit threshold
                        if trade_return > self.profit_taking_threshold:
                            exit_time = common_index[i + j]
                            exit_price = current_exit_price
                            exit_idx = i + j
                            break
                        
                        # Exit on max loss
                        if trade_return < self.max_loss:
                            exit_time = common_index[i + j]
                            exit_price = current_exit_price
                            exit_idx = i + j
                            break
                    
                    # If no exit found, use the last available price
                    if exit_time is None and i + 1 < len(common_index):
                        exit_time = common_index[-1]
                        exit_price = p_follow[-1]
                        exit_idx = len(common_index) - 1
                    
                    # Calculate P&L if exit was found
                    pnl = None
                    if exit_price is not None:
                        trade_return = (exit_price / entry_price - 1) * entry_signal
                        pnl = trade_return - 2 * self.transaction_cost  # Round-trip cost
                    
                    # Record trade
                    trades.append({
                        'pair_id': pair_id,
                        'lead_ticker': lead_ticker,
                        'follow_ticker': follow_ticker,
                        'entry_time': entry_time,
                        'entry_idx': entry_idx,
                        'entry_signal': entry_signal,
                        'entry_price': entry_price,
                        'exit_time': exit_time,
                        'exit_idx': exit_idx,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'lag': lag,
                        'correlation': correlation
                    })
                    
                    # Reset position if exit was found
                    if exit_time is not None:
                        self.positions[pair_id] = 0
                    
                    print(f"Trade for {pair_id}: Entry={entry_time}, Exit={exit_time}, PnL={pnl:.4f}" if pnl else "Open trade")
        
        # Calculate period returns
        period_return = 0
        
        if trades:
            # Calculate average PnL across all trades
            total_pnl = sum(t['pnl'] for t in trades if t['pnl'] is not None)
            num_trades = sum(1 for t in trades if t['pnl'] is not None)
            
            if num_trades > 0:
                period_return = total_pnl / num_trades
        
        self.trades_history.extend(trades)
        print(f"Trading period return: {period_return:.4f}")
        
        return period_return
    
    def backtest(self, start_date, end_date, formation_days=1, trading_days=1):
        """
        Run a backtest of the OCP strategy
        
        Parameters:
        -----------
        start_date : datetime
            Start date for the backtest
        end_date : datetime
            End date for the backtest
        formation_days : int
            Number of days for the formation period
        trading_days : int
            Number of days for the trading period
            
        Returns:
        --------
        results : dict
            Backtest results
        """
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize wealth and history
        self.wealth = 1.0
        self.wealth_history = [(start_date, self.wealth)]
        self.trades_history = []
        
        # Create date range for the backtest
        current_date = start_date
        
        while current_date < end_date:
            # Define formation period
            formation_start = current_date
            formation_end = min(formation_start + timedelta(days=formation_days), end_date)
            
            # Define trading period
            trading_start = formation_end
            trading_end = min(trading_start + timedelta(days=trading_days), end_date)
            
            # Run formation period
            top_pairs = self.formation_period(formation_start, formation_end)
            
            # Run trading period if we have pairs
            if top_pairs:
                period_return = self.trading_period(top_pairs, trading_start, trading_end)
                
                # Update wealth
                self.wealth *= (1 + period_return)
                self.wealth_history.append((trading_end, self.wealth))
                
                print(f"Wealth after period: {self.wealth:.4f}")
            else:
                # If no pairs found, just add the current date to history without change
                self.wealth_history.append((trading_end, self.wealth))
            
            # Move to next period
            current_date = trading_end
        
        # Calculate performance metrics
        if len(self.wealth_history) > 1:
            total_return = self.wealth_history[-1][1] / self.wealth_history[0][1] - 1
            
            # Convert wealth history to DataFrame for easier analysis
            wealth_df = pd.DataFrame(self.wealth_history, columns=['date', 'wealth'])
            wealth_df.set_index('date', inplace=True)
            
            # Calculate daily returns
            daily_returns = wealth_df['wealth'].pct_change().dropna()
            
            # Calculate metrics
            days_traded = (end_date - start_date).days
            annualized_return = ((1 + total_return) ** (252 / max(days_traded, 1))) - 1
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdowns
            wealth_series = wealth_df['wealth']
            running_max = wealth_series.cummax()
            drawdown = (wealth_series - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            winning_trades = sum(1 for t in self.trades_history if t['pnl'] is not None and t['pnl'] > 0)
            total_closed_trades = sum(1 for t in self.trades_history if t['pnl'] is not None)
            win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
            
            # Calculate average profit and loss
            profits = [t['pnl'] for t in self.trades_history if t['pnl'] is not None and t['pnl'] > 0]
            losses = [t['pnl'] for t in self.trades_history if t['pnl'] is not None and t['pnl'] <= 0]
            
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            # Compile results
            results = {
                'wealth_history': wealth_df,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'trades': self.trades_history
            }
            
            # Print summary
            print("\nBacktest Results:")
            print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
            print(f"Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
            print(f"Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
            print(f"Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
            print(f"Average Profit: {avg_profit:.6f}")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Profit/Loss Ratio: {profit_loss_ratio:.4f}")
            print(f"Number of Trades: {len(self.trades_history)}")
            
            return results
        else:
            print("No trading occurred during the backtest period")
            return None
    
    def analyze_simulation_accuracy(self):
        """Analyze how accurately the OCP algorithm detected the simulated lead-lag relationships"""
        # This is only for simulated data where we know the true relationships
        print("\nAnalyzing OCP Algorithm Accuracy on Simulated Data:")
        
        # We need both true_lags and trade history
        if not self.true_lags or not self.trades_history:
            print("No true lags or trades available to analyze")
            return
        
        # Get unique pairs from trades
        trade_pairs = set(trade['pair_id'] for trade in self.trades_history)
        
        # Count how many times we detected the correct relationship
        total_traded_pairs = 0
        correct_detections = 0
        
        for pair_id in trade_pairs:
            # Check if we have ground truth for this pair
            if pair_id in self.true_lags:
                total_traded_pairs += 1
                
                # Get the true lead-lag relationship
                true_lead, true_follow, true_lag = self.true_lags[pair_id]
                
                # Find the trading details
                for trade in self.trades_history:
                    if trade['pair_id'] == pair_id:
                        # Check if we detected the correct leader
                        detected_lead = trade['lead_ticker']
                        if detected_lead == true_lead:
                            correct_detections += 1
                        break
        
        if total_traded_pairs > 0:
            accuracy = correct_detections / total_traded_pairs
            print(f"OCP Algorithm Accuracy: {accuracy:.4f} ({correct_detections}/{total_traded_pairs})")
        else:
            print("No pairs with known lead-lag relationships were traded")
    
    def plot_results(self, results):
        """
        Plot the backtest results
        
        Parameters:
        -----------
        results : dict
            Backtest results
        """
        if results is None:
            print("No results to plot")
            return
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot wealth curve
        axes[0].plot(results['wealth_history'].index, results['wealth_history']['wealth'], 'b-')
        axes[0].set_title('Portfolio Wealth')
        axes[0].set_ylabel('Wealth')
        axes[0].grid(True)
        
        # Plot daily returns
        daily_returns = results['wealth_history']['wealth'].pct_change().dropna()
        axes[1].plot(daily_returns.index, daily_returns, 'g-')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Daily Returns')
        axes[1].set_ylabel('Return')
        axes[1].grid(True)
        
        # Plot drawdowns
        wealth_series = results['wealth_history']['wealth']
        running_max = wealth_series.cummax()
        drawdown = (wealth_series - running_max) / running_max
        
        axes[2].fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
        axes[2].set_title('Drawdowns')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Create a DataFrame of trades for analysis
        if results['trades']:
            trade_df = pd.DataFrame(results['trades'])
            
            # Filter out trades without exit
            valid_trades = trade_df[trade_df['exit_time'].notna()].copy()
            
            if not valid_trades.empty:
                # Calculate trade durations
                if 'entry_idx' in valid_trades.columns and 'exit_idx' in valid_trades.columns:
                    valid_trades['duration'] = valid_trades['exit_idx'] - valid_trades['entry_idx']
                else:
                    valid_trades['duration'] = (pd.to_datetime(valid_trades['exit_time']) - 
                                            pd.to_datetime(valid_trades['entry_time'])).dt.total_seconds() / 60  # in minutes
                
                # Plot trade P&L distribution
                plt.figure(figsize=(10, 6))
                valid_trades['pnl'].hist(bins=50)
                plt.title('Trade P&L Distribution')
                plt.xlabel('P&L')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                # Plot scatter of PnL vs lag
                plt.figure(figsize=(10, 6))
                plt.scatter(valid_trades['lag'], valid_trades['pnl'])
                plt.title('PnL vs Lag')
                plt.xlabel('Lag (minutes)')
                plt.ylabel('PnL')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                # Plot scatter of PnL vs correlation
                plt.figure(figsize=(10, 6))
                plt.scatter(valid_trades['correlation'], valid_trades['pnl'])
                plt.title('PnL vs Correlation')
                plt.xlabel('Correlation')
                plt.ylabel('PnL')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                # Plot trade duration distribution
                plt.figure(figsize=(10, 6))
                valid_trades['duration'].hist(bins=50)
                plt.title('Trade Duration Distribution (minutes)')
                plt.xlabel('Duration (minutes)')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

def run_ocp_strategy_with_simulated_data():
    """Run the OCP strategy on simulated data"""
    # Create strategy instance
    ocp = OptimalCausalPathStrategy()
    
    # Add pairs to test - fewer pairs for clarity
    pairs = [
        ("AAPL", "MSFT"),    # Apple - Microsoft (tech)
        ("GOOGL", "META"),   # Google - Meta (tech)
        ("JPM", "BAC"),      # JPMorgan - Bank of America (finance)
        ("GS", "MS"),        # Goldman Sachs - Morgan Stanley (finance)
        ("PFE", "JNJ"),      # Pfizer - Johnson & Johnson (healthcare)
    ]
    
    for ticker1, ticker2 in pairs:
        ocp.add_pair(ticker1, ticker2)
    
    # Generate perfect data with large, predictable moves
    ocp.generate_perfect_data(n_days=10, minutes_per_day=100)
    
    # Define backtest period
    end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=10)
    
    # Run backtest - shorter formation period for more frequent rebalancing
    results = ocp.backtest(start_date, end_date, formation_days=1, trading_days=1)
    
    # Analyze OCP algorithm accuracy
    ocp.analyze_simulation_accuracy()
    
    # Plot results
    if results:
        ocp.plot_results(results)

if __name__ == "__main__":
    run_ocp_strategy_with_simulated_data()