import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class AlphaEngine:
    def __init__(self, thresholds=[0.0025, 0.005, 0.01, 0.015]):
        """
        Initialize the Alpha Engine with the directional change thresholds.
        
        Parameters:
        -----------
        thresholds : list
            The list of directional change thresholds (as percentage values)
        """
        self.thresholds = thresholds
        self.coastline_traders = {}
        
        # Initialize coastline traders for each threshold
        for i, threshold in enumerate(thresholds):
            self.coastline_traders[i] = {
                'threshold_up': threshold,
                'threshold_down': threshold,
                'mode': None,  # 'up' or 'down'
                'extreme_price': None,
                'dc_price': None,  # Price at directional change
                'position': 0,  # Current position size
                'trades': [],  # List to store trade details
                'last_event': None,  # Last event type: 'dc' (directional change) or 'os' (overshoot)
                'probability_indicator': 1.0,  # L value
                'entry_prices': [],  # Entry prices for position tracking
                'entry_sizes': []  # Entry sizes for position tracking
            }
            
    def detect_events(self, prices):
        """
        Detect directional change and overshoot events in the price series.
        
        Parameters:
        -----------
        prices : array-like
            Price series
            
        Returns:
        --------
        events : dict
            Dictionary with event information
        """
        events = {}
        
        for i, threshold in enumerate(self.thresholds):
            events[i] = {
                'dc_events': [],  # Directional change events: (index, price, direction)
                'os_events': []   # Overshoot events: (index, price, direction)
            }
            
            trader = self.coastline_traders[i]
            
            # Initialize with first price
            if trader['mode'] is None:
                trader['mode'] = 'up'  # Assume initially in up mode
                trader['extreme_price'] = prices[0]
                trader['dc_price'] = prices[0]
            
            # Process each price
            for t in range(1, len(prices)):
                price = prices[t]
                
                if trader['mode'] == 'up':
                    # In up mode - looking for new high or a directional change down
                    if price > trader['extreme_price']:
                        # New high - update extreme price
                        trader['extreme_price'] = price
                        
                        # If we're in overshoot state, check for additional overshoot events
                        if trader['last_event'] == 'dc' or trader['last_event'] == 'os':
                            # Check if we've moved up by another threshold since the DC
                            threshold_size = trader['threshold_up']
                            if price >= trader['dc_price'] * (1 + threshold_size):
                                events[i]['os_events'].append((t, price, 'up'))
                                trader['last_event'] = 'os'
                                trader['dc_price'] = price  # Reset for next overshoot
                    
                    # Check for directional change down
                    elif price <= trader['extreme_price'] * (1 - trader['threshold_down']):
                        # Directional change detected
                        events[i]['dc_events'].append((t, price, 'down'))
                        trader['mode'] = 'down'
                        trader['extreme_price'] = price
                        trader['dc_price'] = price
                        trader['last_event'] = 'dc'
                
                else:  # mode == 'down'
                    # In down mode - looking for new low or a directional change up
                    if price < trader['extreme_price']:
                        # New low - update extreme price
                        trader['extreme_price'] = price
                        
                        # If we're in overshoot state, check for additional overshoot events
                        if trader['last_event'] == 'dc' or trader['last_event'] == 'os':
                            # Check if we've moved down by another threshold since the DC
                            threshold_size = trader['threshold_down']
                            if price <= trader['dc_price'] * (1 - threshold_size):
                                events[i]['os_events'].append((t, price, 'down'))
                                trader['last_event'] = 'os'
                                trader['dc_price'] = price  # Reset for next overshoot
                    
                    # Check for directional change up
                    elif price >= trader['extreme_price'] * (1 + trader['threshold_up']):
                        # Directional change detected
                        events[i]['dc_events'].append((t, price, 'up'))
                        trader['mode'] = 'up'
                        trader['extreme_price'] = price
                        trader['dc_price'] = price
                        trader['last_event'] = 'dc'
        
        return events
    
    def calculate_probability_indicator(self, prices, window=100):
        """
        Calculate probability indicator L for each threshold.
        This is a simplified implementation as the full information theory approach 
        from the paper would be quite complex.
        
        Parameters:
        -----------
        prices : array-like
            Price series
        window : int
            Window size for volatility calculation
            
        Returns:
        --------
        probability_indicators : dict
            Dictionary with probability indicator values for each threshold
        """
        probability_indicators = {}
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate rolling volatility
        if len(returns) < window:
            rolling_vol = np.std(returns) * np.ones_like(returns)
        else:
            rolling_vol = np.array([np.std(returns[max(0, i-window):i]) if i > 0 else 0 
                                  for i in range(len(returns))])
        
        # Normalize returns by volatility to get surprise values
        normalized_returns = np.zeros_like(returns)
        nonzero_indices = rolling_vol > 0
        normalized_returns[nonzero_indices] = returns[nonzero_indices] / rolling_vol[nonzero_indices]
        
        # Calculate L for each threshold
        for i, threshold in enumerate(self.thresholds):
            # Calculate surprise as the cumulative sum of squared normalized returns
            # over a window sized based on the threshold (larger threshold -> larger window)
            window_size = int(window * threshold / self.thresholds[0])
            window_size = max(10, min(window_size, len(normalized_returns)))
            
            # Calculate rolling sum of squared normalized returns
            if len(normalized_returns) < window_size:
                surprise = np.sum(normalized_returns**2)
                # L is a scalar in this case
                L_value = 1 - min(1, max(0, surprise / (window_size * 3)))
            else:
                # Calculate rolling surprise
                surprise = np.array([np.sum(normalized_returns[max(0, i-window_size):i]**2)
                                   for i in range(1, len(normalized_returns)+1)])
                
                # Convert to probability (L value) using normal CDF approximation
                # Higher surprise -> lower probability
                L = 1 - np.minimum(1, np.maximum(0, surprise / (window_size * 3)))
                L_value = L[-1] if len(L) > 0 else 1.0
            
            # Store the latest L value
            probability_indicators[i] = L_value
        
        return probability_indicators
    
    def update_thresholds(self, trader_id):
        """
        Update thresholds based on inventory size (asymmetric thresholds).
        
        Parameters:
        -----------
        trader_id : int
            Trader identifier
            
        Returns:
        --------
        None (updates trader in place)
        """
        trader = self.coastline_traders[trader_id]
        position = trader['position']
        original_threshold = self.thresholds[trader_id]
        
        if position > 15:  # Long position
            trader['threshold_up'] = original_threshold * 1.5
            trader['threshold_down'] = original_threshold * 0.75
        elif position > 30:
            trader['threshold_up'] = original_threshold * 2.0
            trader['threshold_down'] = original_threshold * 0.5
        elif position < -15:  # Short position
            trader['threshold_up'] = original_threshold * 0.75
            trader['threshold_down'] = original_threshold * 1.5
        elif position < -30:
            trader['threshold_up'] = original_threshold * 0.5
            trader['threshold_down'] = original_threshold * 2.0
        else:
            trader['threshold_up'] = original_threshold
            trader['threshold_down'] = original_threshold
    
    def calculate_position_size(self, trader):
        """
        Calculate position size based on probability indicator.
        
        Parameters:
        -----------
        trader : dict
            Trader information
            
        Returns:
        --------
        position_size : float
            Position size to trade
        """
        L = trader['probability_indicator']
        
        if L < 0.1:
            return 0.1  # Reduce position size to 0.1 units
        elif L < 0.5:
            return 0.5  # Reduce position size to 0.5 units
        else:
            return 1.0  # Normal position size of 1 unit
    
    def execute_trades(self, events, prices, timestamp):
        """
        Execute trades based on detected events.
        
        Parameters:
        -----------
        events : dict
            Dictionary with event information
        prices : array-like
            Price series
        timestamp : datetime
            Current timestamp
            
        Returns:
        --------
        trades : list
            List of executed trades
        """
        trades = []
        current_price = prices[-1]
        
        # Update probability indicators
        probability_indicators = self.calculate_probability_indicator(prices)
        
        for i, trader_events in events.items():
            trader = self.coastline_traders[i]
            trader['probability_indicator'] = probability_indicators[i]
            
            # Update thresholds based on inventory
            self.update_thresholds(i)
            
            # Process directional change events
            for t, price, direction in trader_events['dc_events']:
                if t == len(prices) - 1:  # Only consider the latest event
                    if direction == 'up' and trader['position'] <= 0:
                        # Directional change up while we're short or flat - initiate long or reduce short
                        position_size = self.calculate_position_size(trader)
                        trades.append({
                            'trader_id': i,
                            'timestamp': timestamp,
                            'price': price,
                            'size': position_size,
                            'type': 'dc_entry_long' if trader['position'] == 0 else 'dc_decascade'
                        })
                        trader['position'] += position_size
                        trader['entry_prices'].append(price)
                        trader['entry_sizes'].append(position_size)
                    
                    elif direction == 'down' and trader['position'] >= 0:
                        # Directional change down while we're long or flat - initiate short or reduce long
                        position_size = self.calculate_position_size(trader)
                        trades.append({
                            'trader_id': i,
                            'timestamp': timestamp,
                            'price': price,
                            'size': -position_size,
                            'type': 'dc_entry_short' if trader['position'] == 0 else 'dc_decascade'
                        })
                        trader['position'] -= position_size
                        trader['entry_prices'].append(price)
                        trader['entry_sizes'].append(-position_size)
            
            # Process overshoot events
            for t, price, direction in trader_events['os_events']:
                if t == len(prices) - 1:  # Only consider the latest event
                    if direction == 'up' and trader['position'] < 0:
                        # Overshoot up while we're short - cascade (increase short position)
                        position_size = self.calculate_position_size(trader)
                        
                        # Apply fractional position changes based on inventory
                        if abs(trader['position']) > 30:
                            position_size *= 0.25  # 1/4 unit for very large inventory
                        elif abs(trader['position']) > 15:
                            position_size *= 0.5   # 1/2 unit for large inventory
                        
                        trades.append({
                            'trader_id': i,
                            'timestamp': timestamp,
                            'price': price,
                            'size': -position_size,  # Negative for short
                            'type': 'os_cascade'
                        })
                        trader['position'] -= position_size
                        trader['entry_prices'].append(price)
                        trader['entry_sizes'].append(-position_size)
                    
                    elif direction == 'down' and trader['position'] > 0:
                        # Overshoot down while we're long - cascade (increase long position)
                        position_size = self.calculate_position_size(trader)
                        
                        # Apply fractional position changes based on inventory
                        if abs(trader['position']) > 30:
                            position_size *= 0.25  # 1/4 unit for very large inventory
                        elif abs(trader['position']) > 15:
                            position_size *= 0.5   # 1/2 unit for large inventory
                        
                        trades.append({
                            'trader_id': i,
                            'timestamp': timestamp,
                            'price': price,
                            'size': position_size,  # Positive for long
                            'type': 'os_cascade'
                        })
                        trader['position'] += position_size
                        trader['entry_prices'].append(price)
                        trader['entry_sizes'].append(position_size)
            
            # Store trades for this trader
            trader['trades'].extend([t for t in trades if t['trader_id'] == i])
        
        return trades
    
    def backtest(self, prices, timestamps=None):
        """
        Backtest the Alpha Engine on historical data.
        
        Parameters:
        -----------
        prices : array-like
            Historical price series
        timestamps : array-like
            Timestamps for the price series (optional)
            
        Returns:
        --------
        results : dict
            Backtest results
        """
        if timestamps is None:
            # Generate timestamps if not provided
            start_date = datetime.now() - timedelta(days=len(prices))
            timestamps = [start_date + timedelta(days=i) for i in range(len(prices))]
        
        all_trades = []
        equity_curve = [0]  # Start with 0 P&L
        positions = []
        
        # Process each price update
        for t in range(1, len(prices)):
            # Get price window up to current time
            price_window = prices[:t+1]
            
            # Detect events in the price window
            events = self.detect_events(price_window)
            
            # Execute trades based on events
            trades = self.execute_trades(events, price_window, timestamps[t])
            all_trades.extend(trades)
            
            # Calculate P&L for this step
            step_pnl = 0
            current_positions = {}
            
            for i, trader in self.coastline_traders.items():
                # Calculate unrealized P&L for current positions
                unrealized_pnl = sum([(prices[t] - entry_price) * size 
                                     for entry_price, size in zip(trader['entry_prices'], trader['entry_sizes'])])
                
                step_pnl += unrealized_pnl
                current_positions[i] = trader['position']
            
            # Update equity curve
            equity_curve.append(step_pnl)
            positions.append(current_positions)
        
        # Calculate performance metrics
        returns = np.diff(equity_curve)
        cumulative_returns = np.array(equity_curve)
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / np.maximum(1, peak)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Store results
        results = {
            'trades': all_trades,
            'equity_curve': equity_curve,
            'positions': positions,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        }
        
        return results

def generate_simulated_price_data(initial_price=100, n_steps=1000, mu=0, sigma=0.01, trend=None):
    """
    Generate simulated price data following a geometric Brownian motion.
    
    Parameters:
    -----------
    initial_price : float
        Initial price
    n_steps : int
        Number of steps to simulate
    mu : float
        Drift parameter
    sigma : float
        Volatility parameter
    trend : str
        None for random walk, 'up' for uptrend, 'down' for downtrend
        
    Returns:
    --------
    prices : array
        Simulated price series
    """
    # Set drift based on trend
    if trend == 'up':
        mu = 0.0001  # Small positive drift
    elif trend == 'down':
        mu = -0.0001  # Small negative drift
    
    # Generate random shocks
    random_shocks = np.random.normal(mu, sigma, n_steps)
    
    # Calculate price series
    prices = np.zeros(n_steps)
    prices[0] = initial_price
    
    for i in range(1, n_steps):
        prices[i] = prices[i-1] * np.exp(random_shocks[i])
    
    return prices

def plot_results(prices, results, title='Alpha Engine Backtest Results'):
    """
    Plot backtest results.
    
    Parameters:
    -----------
    prices : array-like
        Price series
    results : dict
        Backtest results
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price series
    axes[0].plot(prices, 'b-', label='Price')
    
    # Plot trades on price chart
    for trade in results['trades']:
        idx = list(prices).index(trade['price']) if trade['price'] in prices else -1
        if idx >= 0:
            if trade['size'] > 0:
                axes[0].scatter(idx, prices[idx], marker='^', color='g', s=100)
            else:
                axes[0].scatter(idx, prices[idx], marker='v', color='r', s=100)
    
    axes[0].set_title(title)
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot equity curve
    axes[1].plot(results['equity_curve'], 'g-', label='Equity Curve')
    axes[1].set_ylabel('P&L')
    axes[1].set_xlabel('Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown
    plt.figure(figsize=(12, 4))
    plt.plot(results['drawdown'] * 100, 'r-', label='Drawdown (%)')
    plt.title('Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print(f"Total Return: {results['total_return']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {results['max_drawdown']*100:.4f}%")
    print(f"Number of Trades: {len(results['trades'])}")

def main():
    # Generate simulated data
    print("Generating simulated data...")
    
    # Random walk
    random_walk_prices = generate_simulated_price_data(n_steps=10000)
    
    # Uptrend
    uptrend_prices = generate_simulated_price_data(n_steps=10000, trend='up')
    
    # Downtrend
    downtrend_prices = generate_simulated_price_data(n_steps=10000, trend='down')
    
    # Simulate market volatility
    volatility_regime_prices = np.concatenate([
        generate_simulated_price_data(n_steps=2000, sigma=0.005),  # Low volatility
        generate_simulated_price_data(initial_price=random_walk_prices[2000], n_steps=2000, sigma=0.02),  # High volatility
        generate_simulated_price_data(initial_price=random_walk_prices[4000], n_steps=2000, sigma=0.005),  # Low volatility
        generate_simulated_price_data(initial_price=random_walk_prices[6000], n_steps=2000, sigma=0.02),  # High volatility
        generate_simulated_price_data(initial_price=random_walk_prices[8000], n_steps=2000, sigma=0.005)   # Low volatility
    ])
    
    # Initialize Alpha Engine
    alpha_engine = AlphaEngine()
    
    # Run backtest on random walk
    print("\nBacktesting on random walk...")
    random_walk_results = alpha_engine.backtest(random_walk_prices)
    plot_results(random_walk_prices, random_walk_results, 'Alpha Engine - Random Walk')
    
    # Reset Alpha Engine for each test
    alpha_engine = AlphaEngine()
    
    # Run backtest on uptrend
    print("\nBacktesting on uptrend...")
    uptrend_results = alpha_engine.backtest(uptrend_prices)
    plot_results(uptrend_prices, uptrend_results, 'Alpha Engine - Uptrend')
    
    # Reset Alpha Engine
    alpha_engine = AlphaEngine()
    
    # Run backtest on downtrend
    print("\nBacktesting on downtrend...")
    downtrend_results = alpha_engine.backtest(downtrend_prices)
    plot_results(downtrend_prices, downtrend_results, 'Alpha Engine - Downtrend')
    
    # Reset Alpha Engine
    alpha_engine = AlphaEngine()
    
    # Run backtest on volatility regime
    print("\nBacktesting on volatility regime changes...")
    volatility_results = alpha_engine.backtest(volatility_regime_prices)
    plot_results(volatility_regime_prices, volatility_results, 'Alpha Engine - Volatility Regime Changes')

if __name__ == "__main__":
    main()