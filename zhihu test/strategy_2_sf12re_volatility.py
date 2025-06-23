"""
Strategy 2: SF12Re Volatility Algorithm
Adaptive Interval + Volatility Timing Strategy

Based on a new volatility calculation method instead of traditional ATR:
- Core indicator calculation using R1 and R2 periods
- Volatility ratio calculation (NTD1/NTD2)
- Adaptive interval construction
- Volatility timing module
- Dynamic trailing stop loss

Key Features:
1. Smart interval adjustment based on market volatility
2. Multi-period analysis (short-term R1 and medium-term R2=2×R1)
3. Strict risk management with dynamic trailing stops
4. Self-adaptive mechanism for different market environments

Note: This version uses simulated stock data with varying volatility regimes
to properly test the adaptive volatility algorithm. The simulated data includes
high volatility periods (25% of time) and low volatility periods (15% of time)
to demonstrate the strategy's adaptive capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SF12ReVolatilityStrategy:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date='2024-12-31', 
                 r1_period=45, x_factor=0.5, trailing_stop_rate=0.8):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.r1_period = r1_period  # Short-term period
        self.r2_period = r1_period * 2  # Medium-term period
        self.x_factor = x_factor  # Interval width coefficient
        self.trailing_stop_rate = trailing_stop_rate  # Trailing stop tightness
        self.data = None
        self.signals = None
        
    def fetch_data(self):
        """Generate simulated stock data with higher volatility patterns"""
        try:
            # Parse date strings
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)

            # Generate date range (business days only)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = dates[dates.weekday < 5]

            n_days = len(dates)

            # Set random seed for reproducible results
            np.random.seed(123)  # Different seed for different volatility patterns

            # Generate stock price data with varying volatility (key for SF12Re strategy)
            initial_price = 150.0
            base_drift = 0.0003  # Base daily drift
            base_volatility = 0.015  # Base daily volatility

            # Create volatility regimes (important for SF12Re testing)
            volatility_regimes = np.ones(n_days) * base_volatility

            # Add high volatility periods (25% of time)
            high_vol_periods = np.random.choice(n_days, size=int(n_days * 0.25), replace=False)
            volatility_regimes[high_vol_periods] *= 2.5  # 2.5x higher volatility

            # Add low volatility periods (15% of time)
            low_vol_periods = np.random.choice(n_days, size=int(n_days * 0.15), replace=False)
            volatility_regimes[low_vol_periods] *= 0.4  # 40% of base volatility

            # Generate returns with time-varying volatility
            returns = np.random.normal(base_drift, volatility_regimes)

            # Add momentum and mean reversion patterns
            for i in range(1, n_days):
                # Momentum effect (trending)
                momentum = 0.1 * returns[i-1] if i > 0 else 0

                # Mean reversion effect
                if i >= 20:
                    recent_returns = returns[i-20:i]
                    cumulative_return = np.sum(recent_returns)
                    mean_reversion = -0.05 * cumulative_return
                else:
                    mean_reversion = 0

                returns[i] += momentum + mean_reversion

            # Calculate cumulative prices
            price_multipliers = np.exp(np.cumsum(returns))
            close_prices = initial_price * price_multipliers

            # Generate OHLV data with realistic intraday patterns
            high_prices = np.zeros(n_days)
            low_prices = np.zeros(n_days)
            open_prices = np.zeros(n_days)

            open_prices[0] = initial_price

            for i in range(n_days):
                if i > 0:
                    # Gap from previous close
                    gap = np.random.normal(0, volatility_regimes[i] * 0.3)
                    open_prices[i] = close_prices[i-1] * (1 + gap)

                # Intraday range based on volatility
                daily_range = volatility_regimes[i] * close_prices[i] * np.random.uniform(0.8, 1.5)

                # High and low around the close price
                high_prices[i] = max(open_prices[i], close_prices[i]) + daily_range * np.random.uniform(0.3, 0.7)
                low_prices[i] = min(open_prices[i], close_prices[i]) - daily_range * np.random.uniform(0.3, 0.7)

                # Ensure OHLC consistency
                high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
                low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])

            # Generate volume data (higher volume during high volatility)
            base_volume = 2000000
            volume_multiplier = 1 + (volatility_regimes / base_volatility - 1) * 0.5
            volume = base_volume * volume_multiplier * np.exp(np.random.normal(0, 0.4, n_days))

            # Create DataFrame
            self.data = pd.DataFrame({
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volume.astype(int)
            }, index=dates)

            print(f"Successfully generated {len(self.data)} days of simulated data for {self.symbol}")
            print(f"Price range: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
            print(f"Volatility patterns: High vol periods: {len(high_vol_periods)}, Low vol periods: {len(low_vol_periods)}")
            return True

        except Exception as e:
            print(f"Error generating simulated data: {e}")
            return False
    
    def locate_extremes(self, data, period, price_type='high'):
        """Locate highest/lowest points within specified period"""
        if price_type == 'high':
            rolling_max = data['high'].rolling(window=period).max()
            return rolling_max
        else:  # low
            rolling_min = data['low'].rolling(window=period).min()
            return rolling_min
    
    def calculate_volatility_indicators(self):
        """Calculate SF12Re volatility indicators"""
        if self.data is None or len(self.data) < self.r2_period:
            print("Insufficient data for calculation")
            return False
            
        df = self.data.copy()
        
        # Locate R1 period extremes
        df['h1'] = self.locate_extremes(df, self.r1_period, 'high')
        df['l1'] = self.locate_extremes(df, self.r1_period, 'low')
        
        # Locate R2 period extremes  
        df['h2'] = self.locate_extremes(df, self.r2_period, 'high')
        df['l2'] = self.locate_extremes(df, self.r2_period, 'low')
        
        # Calculate NTD (Normalized Trading Distance) indicators
        # NTD1: Short-term volatility measure
        df['ntd1'] = (df['h1'] - df['l1']) / df['close']
        
        # NTD2: Medium-term volatility measure
        df['ntd2'] = (df['h2'] - df['l2']) / df['close']
        
        # Calculate average NTD for adaptive interval
        df['avg_ntd'] = df['ntd1'].rolling(window=20).mean()
        
        # Volatility ratio for timing
        df['volatility_ratio'] = df['ntd1'] / (df['ntd2'] + 1e-8)  # Avoid division by zero
        
        # HL reference point (midpoint of R1 extremes)
        df['hl'] = (df['h1'] + df['l1']) / 2
        
        # Condition filter for adaptive interval selection
        # High volatility condition: volatility ratio > threshold
        df['condition_filter'] = df['volatility_ratio'] > df['volatility_ratio'].rolling(window=20).mean()
        
        # Adaptive interval calculation
        df['h_max'] = np.where(df['condition_filter'], 
                              df['hl'] + df['avg_ntd'] * df['close'] * self.x_factor,
                              df['h1'])
        
        df['l_min'] = np.where(df['condition_filter'],
                              df['hl'] - df['avg_ntd'] * df['close'] * self.x_factor,
                              df['l1'])
        
        self.data = df
        return True
    
    def generate_signals(self):
        """Generate buy/sell signals based on SF12Re strategy"""
        if self.data is None:
            return False
            
        df = self.data.copy()
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['position'] = 0  # 0: no position, 1: long, -1: short
        
        position = 0
        entry_price = 0
        stop_loss = 0
        
        for i in range(self.r2_period, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            h_max = df['h_max'].iloc[i]
            l_min = df['l_min'].iloc[i]
            volatility_ratio = df['volatility_ratio'].iloc[i]
            avg_volatility = df['volatility_ratio'].rolling(window=20).mean().iloc[i]
            
            if position == 0:  # No position
                # Long entry: price breaks above upper band with volatility confirmation
                if (current_high > h_max and 
                    volatility_ratio > avg_volatility * 1.2):  # Volatility timing
                    df.loc[df.index[i], 'signal'] = 1
                    position = 1
                    entry_price = current_close
                    stop_loss = l_min  # Initial stop loss
                    
                # Short entry: price breaks below lower band with volatility confirmation
                elif (current_low < l_min and 
                      volatility_ratio > avg_volatility * 1.2):
                    df.loc[df.index[i], 'signal'] = -1
                    position = -1
                    entry_price = current_close
                    stop_loss = h_max  # Initial stop loss
                    
            elif position == 1:  # Long position
                # Update trailing stop loss
                if current_close > entry_price:
                    profit_ratio = (current_close - entry_price) / entry_price
                    new_stop = entry_price + (current_close - entry_price) * self.trailing_stop_rate
                    stop_loss = max(stop_loss, new_stop)
                
                # Exit conditions
                if (current_close < stop_loss or  # Stop loss
                    current_low < l_min):  # Reverse breakout
                    df.loc[df.index[i], 'signal'] = -1
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    
            elif position == -1:  # Short position
                # Update trailing stop loss
                if current_close < entry_price:
                    profit_ratio = (entry_price - current_close) / entry_price
                    new_stop = entry_price - (entry_price - current_close) * self.trailing_stop_rate
                    stop_loss = min(stop_loss, new_stop)
                
                # Exit conditions
                if (current_close > stop_loss or  # Stop loss
                    current_high > h_max):  # Reverse breakout
                    df.loc[df.index[i], 'signal'] = 1
                    position = 0
                    entry_price = 0
                    stop_loss = 0
            
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'stop_loss'] = stop_loss
        
        self.signals = df
        return True
    
    def backtest(self, initial_capital=10000, commission=0.001):
        """Perform backtesting"""
        if self.signals is None:
            print("No signals generated. Run generate_signals() first.")
            return None
            
        df = self.signals.copy()
        capital = initial_capital
        shares = 0
        trades = []
        position_type = 0  # 0: no position, 1: long, -1: short
        
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            price = df['close'].iloc[i]
            
            if signal == 1 and position_type <= 0:  # Buy signal
                if position_type == -1:  # Close short position first
                    proceeds = shares * price * (1 - commission)
                    capital += proceeds
                    profit = proceeds - (shares * trades[-1]['price'] * (1 + commission))
                    trades.append({
                        'date': df.index[i],
                        'action': 'COVER',
                        'price': price,
                        'shares': shares,
                        'capital': capital,
                        'profit': profit
                    })
                
                # Open long position
                shares = int(capital * 0.95 / price)
                cost = shares * price * (1 + commission)
                capital -= cost
                position_type = 1
                trades.append({
                    'date': df.index[i],
                    'action': 'BUY',
                    'price': price,
                    'shares': shares,
                    'capital': capital
                })
                
            elif signal == -1 and position_type >= 0:  # Sell signal
                if position_type == 1:  # Close long position first
                    proceeds = shares * price * (1 - commission)
                    capital += proceeds
                    profit = proceeds - (shares * trades[-1]['price'] * (1 + commission))
                    trades.append({
                        'date': df.index[i],
                        'action': 'SELL',
                        'price': price,
                        'shares': shares,
                        'capital': capital,
                        'profit': profit
                    })
                
                # Open short position (simulated)
                shares = int(capital * 0.95 / price)
                proceeds = shares * price * (1 - commission)
                capital += proceeds
                position_type = -1
                trades.append({
                    'date': df.index[i],
                    'action': 'SHORT',
                    'price': price,
                    'shares': shares,
                    'capital': capital
                })
        
        # Calculate final portfolio value
        final_price = df['close'].iloc[-1]
        if position_type == 1:  # Long position
            final_value = capital + shares * final_price * (1 - commission)
        elif position_type == -1:  # Short position
            final_value = capital - shares * final_price * (1 + commission)
        else:
            final_value = capital
            
        # Calculate performance metrics
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate win rate
        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
        total_trades = len([t for t in trades if 'profit' in t])
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades
        }
        
        return results
    
    def plot_results(self, results=None):
        """Plot price chart with signals and adaptive intervals"""
        if self.signals is None:
            print("No signals to plot")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot price and adaptive intervals
        df = self.signals
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
        ax1.plot(df.index, df['h_max'], label='Upper Band (H_MAX)', alpha=0.7, color='red')
        ax1.plot(df.index, df['l_min'], label='Lower Band (L_MIN)', alpha=0.7, color='green')
        ax1.fill_between(df.index, df['h_max'], df['l_min'], alpha=0.1, color='gray')
        
        # Plot buy/sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.symbol} - SF12Re Volatility Strategy')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot volatility indicators
        ax2.plot(df.index, df['ntd1'], label='NTD1 (Short-term)', alpha=0.8)
        ax2.plot(df.index, df['ntd2'], label='NTD2 (Medium-term)', alpha=0.8)
        ax2.plot(df.index, df['avg_ntd'], label='Average NTD', alpha=0.8)
        ax2.set_ylabel('Volatility Measures')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot volatility ratio and condition filter
        ax3.plot(df.index, df['volatility_ratio'], label='Volatility Ratio', alpha=0.8)
        ax3.plot(df.index, df['volatility_ratio'].rolling(window=20).mean(), 
                label='20-period MA', alpha=0.8)
        ax3.fill_between(df.index, 0, df['condition_filter'], 
                        alpha=0.3, label='High Volatility Periods')
        ax3.set_ylabel('Volatility Ratio')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        if results:
            print(f"\n=== SF12Re Volatility Strategy Results ===")
            print(f"Symbol: {self.symbol}")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"R1 Period: {self.r1_period}, R2 Period: {self.r2_period}")
            print(f"X Factor: {self.x_factor}, Trailing Stop Rate: {self.trailing_stop_rate}")
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.1%}")

def main():
    """Main function to run the strategy"""
    # Test with simulated stock data
    strategy = SF12ReVolatilityStrategy(
        symbol='SIMULATED_VOLATILE_STOCK',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("Generating simulated data with volatility regimes...")
    if not strategy.fetch_data():
        return

    print("Calculating volatility indicators...")
    if not strategy.calculate_volatility_indicators():
        return

    print("Generating signals...")
    if not strategy.generate_signals():
        return

    print("Running backtest...")
    results = strategy.backtest(initial_capital=10000)

    if results:
        print("\n=== SF12Re Volatility Strategy Results ===")
        print(f"Symbol: {strategy.symbol}")
        print(f"Period: {strategy.start_date} to {strategy.end_date}")
        print(f"R1 Period: {strategy.r1_period}, R2 Period: {strategy.r2_period}")
        print(f"X Factor: {strategy.x_factor}, Trailing Stop Rate: {strategy.trailing_stop_rate}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")

        # Show some trade details
        if results['trades']:
            print(f"\nFirst few trades:")
            for i, trade in enumerate(results['trades'][:5]):
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} at ${trade['price']:.2f}")

        print("\nStrategy completed successfully!")
        print("Note: Plotting disabled for faster execution. Use plot_results() to see charts.")
    else:
        print("Backtest failed to produce results.")

if __name__ == "__main__":
    main()
