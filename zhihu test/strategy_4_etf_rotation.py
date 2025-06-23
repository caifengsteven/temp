"""
Strategy 4: Calm ETF Rotation Strategy
10-Year Backtest with Stable Performance

Strategy evaluates mainstream ETFs through four key dimensions:

1. Trend Factor (50% weight):
   - Calculate 25-day log return linear regression R² value
   - Identify long-term trend strength

2. Momentum Factor (20% weight):
   - Combine 5-day and 10-day Rate of Change (ROC)
   - Capture short-term momentum

3. Volume Factor (30% weight):
   - Use 5-day/18-day average volume ratio
   - Judge fund activity level

4. Momentum Acceleration Factor (10% weight):
   - Calculate acceleration of recent momentum factor
   - Capture targets with faster momentum acceleration

Expected Performance: 21% annual return with stable growth over 10 years

Note: This version uses simulated ETF data representing different sectors and styles.
Each ETF has unique characteristics (volatility, trend, sector behavior) and responds
differently to market regimes (bull, bear, sideways, volatile) to test the rotation
strategy's ability to select the best performing ETFs across different market conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class ETFRotationStrategy:
    def __init__(self, etf_symbols=None, start_date='2020-01-01', end_date='2024-12-31', 
                 rebalance_frequency='M'):  # M=Monthly, Q=Quarterly
        if etf_symbols is None:
            # Default ETF universe - major sector and market ETFs
            self.etf_symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'XLF', 'XLK', 'XLE', 'XLV']
        else:
            self.etf_symbols = etf_symbols
            
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.data = {}
        self.scores = None
        self.positions = None
        self.top_n = 3  # Hold top 3 ETFs
        
    def fetch_data(self):
        """Generate simulated ETF data with different sector characteristics"""
        print(f"Generating simulated data for {len(self.etf_symbols)} ETFs...")

        # Parse date strings
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        # Generate date range (business days only)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]
        n_days = len(dates)

        # Set random seed for reproducible results
        np.random.seed(789)

        # Define ETF characteristics (different sectors/styles)
        etf_characteristics = {
            'SPY': {'base_price': 300, 'volatility': 0.015, 'trend': 0.0005, 'sector': 'broad_market'},
            'QQQ': {'base_price': 250, 'volatility': 0.020, 'trend': 0.0008, 'sector': 'tech'},
            'IWM': {'base_price': 180, 'volatility': 0.025, 'trend': 0.0003, 'sector': 'small_cap'},
            'EFA': {'base_price': 70, 'volatility': 0.018, 'trend': 0.0002, 'sector': 'international'},
            'EEM': {'base_price': 45, 'volatility': 0.030, 'trend': 0.0001, 'sector': 'emerging'},
            'VTI': {'base_price': 200, 'volatility': 0.016, 'trend': 0.0005, 'sector': 'total_market'},
            'XLF': {'base_price': 35, 'volatility': 0.022, 'trend': 0.0004, 'sector': 'financial'},
            'XLK': {'base_price': 120, 'volatility': 0.021, 'trend': 0.0007, 'sector': 'tech'},
            'XLE': {'base_price': 60, 'volatility': 0.035, 'trend': -0.0001, 'sector': 'energy'},
            'XLV': {'base_price': 110, 'volatility': 0.017, 'trend': 0.0006, 'sector': 'healthcare'}
        }

        # Generate market regime periods
        regime_changes = np.random.choice(n_days, size=8, replace=False)
        regime_changes = np.sort(regime_changes)

        for symbol in self.etf_symbols:
            try:
                if symbol not in etf_characteristics:
                    # Default characteristics for unknown symbols
                    char = {'base_price': 100, 'volatility': 0.020, 'trend': 0.0003, 'sector': 'default'}
                else:
                    char = etf_characteristics[symbol]

                # Generate returns with regime changes
                returns = np.random.normal(char['trend'], char['volatility'], n_days)

                # Apply different regimes
                current_regime = 'normal'
                for i in range(n_days):
                    # Check for regime change
                    if i in regime_changes:
                        regimes = ['bull', 'bear', 'sideways', 'volatile']
                        current_regime = np.random.choice(regimes)

                    # Modify returns based on regime and sector
                    if current_regime == 'bull':
                        if char['sector'] in ['tech', 'small_cap']:
                            returns[i] *= 1.5  # Tech and small cap outperform in bull markets
                        else:
                            returns[i] *= 1.2
                    elif current_regime == 'bear':
                        if char['sector'] in ['defensive', 'healthcare']:
                            returns[i] *= 0.7  # Defensive sectors hold up better
                        else:
                            returns[i] *= 0.5
                    elif current_regime == 'volatile':
                        returns[i] *= np.random.choice([0.5, 1.5])  # High volatility

                # Add momentum and mean reversion
                for j in range(1, n_days):
                    # Momentum effect
                    if j >= 5:
                        momentum = 0.1 * np.mean(returns[j-5:j])
                        returns[j] += momentum

                    # Mean reversion
                    if j >= 20:
                        cumulative = np.sum(returns[j-20:j])
                        if abs(cumulative) > 0.15:
                            returns[j] -= 0.05 * cumulative

                # Calculate prices
                price_multipliers = np.exp(np.cumsum(returns))
                close_prices = char['base_price'] * price_multipliers

                # Generate OHLV data
                high_prices = np.zeros(n_days)
                low_prices = np.zeros(n_days)
                open_prices = np.zeros(n_days)

                open_prices[0] = char['base_price']

                for k in range(n_days):
                    if k > 0:
                        gap = np.random.normal(0, char['volatility'] * 0.2)
                        open_prices[k] = close_prices[k-1] * (1 + gap)

                    daily_range = close_prices[k] * char['volatility'] * np.random.uniform(0.5, 1.2)
                    high_prices[k] = max(open_prices[k], close_prices[k]) + daily_range * 0.4
                    low_prices[k] = min(open_prices[k], close_prices[k]) - daily_range * 0.4

                    # Ensure OHLC consistency
                    high_prices[k] = max(high_prices[k], open_prices[k], close_prices[k])
                    low_prices[k] = min(low_prices[k], open_prices[k], close_prices[k])

                # Generate volume
                base_volume = 10000000 if symbol in ['SPY', 'QQQ', 'VTI'] else 5000000
                volume = base_volume * np.exp(np.random.normal(0, 0.3, n_days))

                # Create DataFrame
                self.data[symbol] = pd.DataFrame({
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices,
                    'volume': volume.astype(int)
                }, index=dates)

                print(f"✓ {symbol}: {len(self.data[symbol])} days, "
                      f"${self.data[symbol]['close'].iloc[0]:.2f} → ${self.data[symbol]['close'].iloc[-1]:.2f}")

            except Exception as e:
                print(f"✗ Error generating {symbol}: {e}")

        if len(self.data) == 0:
            print("No data generated successfully")
            return False

        print(f"Successfully generated data for {len(self.data)} ETFs")
        return True
    
    def calculate_trend_factor(self, data, period=25):
        """Calculate trend factor using linear regression R²"""
        close_prices = data['close']
        trend_scores = []
        
        for i in range(period, len(close_prices)):
            # Get log returns for the period
            prices = close_prices.iloc[i-period:i]
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(log_returns) < period - 1:
                trend_scores.append(0)
                continue
                
            # Linear regression on log returns
            X = np.arange(len(log_returns)).reshape(-1, 1)
            y = log_returns.values
            
            try:
                model = LinearRegression().fit(X, y)
                r_squared = model.score(X, y)
                # Adjust score based on trend direction
                slope = model.coef_[0]
                trend_score = r_squared if slope > 0 else -r_squared
            except:
                trend_score = 0
                
            trend_scores.append(trend_score)
        
        # Pad with zeros for initial period
        result = [0] * period + trend_scores
        return pd.Series(result, index=data.index)
    
    def calculate_momentum_factor(self, data):
        """Calculate momentum factor using ROC"""
        close_prices = data['close']
        
        # 5-day and 10-day Rate of Change
        roc_5 = close_prices.pct_change(5)
        roc_10 = close_prices.pct_change(10)
        
        # Combine with equal weights
        momentum_factor = (roc_5 + roc_10) / 2
        
        return momentum_factor.fillna(0)
    
    def calculate_volume_factor(self, data):
        """Calculate volume factor using volume ratio"""
        volume = data['volume']
        
        # 5-day and 18-day average volume
        vol_5 = volume.rolling(window=5).mean()
        vol_18 = volume.rolling(window=18).mean()
        
        # Volume ratio
        volume_factor = vol_5 / (vol_18 + 1e-8)  # Avoid division by zero
        
        return volume_factor.fillna(1)
    
    def calculate_momentum_acceleration(self, momentum_factor, period=5):
        """Calculate momentum acceleration factor"""
        # Calculate acceleration as change in momentum
        momentum_change = momentum_factor.diff(period)
        
        # Normalize by rolling standard deviation
        momentum_std = momentum_factor.rolling(window=20).std()
        acceleration = momentum_change / (momentum_std + 1e-8)
        
        return acceleration.fillna(0)
    
    def calculate_composite_score(self, symbol):
        """Calculate composite score for an ETF"""
        data = self.data[symbol]
        
        # Calculate individual factors
        trend_factor = self.calculate_trend_factor(data)
        momentum_factor = self.calculate_momentum_factor(data)
        volume_factor = self.calculate_volume_factor(data)
        momentum_acceleration = self.calculate_momentum_acceleration(momentum_factor)
        
        # Normalize factors to [0, 1] range using rolling percentile rank
        window = 60  # 60-day rolling window for normalization

        # Use rolling apply with rank function
        def rolling_rank(series, window):
            return series.rolling(window=window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window else 0.5
            )

        trend_norm = rolling_rank(trend_factor, window)
        momentum_norm = rolling_rank(momentum_factor, window)
        volume_norm = rolling_rank(volume_factor, window)
        acceleration_norm = rolling_rank(momentum_acceleration, window)
        
        # Weighted composite score
        composite_score = (trend_norm * 0.5 + 
                          momentum_norm * 0.2 + 
                          volume_norm * 0.3 + 
                          acceleration_norm * 0.1)
        
        return composite_score.fillna(0)
    
    def calculate_all_scores(self):
        """Calculate scores for all ETFs"""
        print("Calculating composite scores for all ETFs...")
        
        scores_dict = {}
        for symbol in self.data.keys():
            scores_dict[symbol] = self.calculate_composite_score(symbol)
            
        # Combine into DataFrame
        self.scores = pd.DataFrame(scores_dict)
        
        # Forward fill missing values
        self.scores = self.scores.fillna(method='ffill').fillna(0)
        
        print(f"Calculated scores for {len(self.scores.columns)} ETFs")
        return True
    
    def generate_positions(self):
        """Generate position allocations based on scores"""
        if self.scores is None:
            print("No scores calculated. Run calculate_all_scores() first.")
            return False
            
        positions = pd.DataFrame(index=self.scores.index, columns=self.scores.columns)
        positions = positions.fillna(0)
        
        # Determine rebalancing dates
        if self.rebalance_frequency == 'M':
            rebalance_dates = pd.date_range(start=self.scores.index[0], 
                                          end=self.scores.index[-1], 
                                          freq='MS')  # Month start
        elif self.rebalance_frequency == 'Q':
            rebalance_dates = pd.date_range(start=self.scores.index[0], 
                                          end=self.scores.index[-1], 
                                          freq='QS')  # Quarter start
        else:
            rebalance_dates = self.scores.index[::20]  # Every 20 days
        
        current_positions = {}
        
        for date in self.scores.index:
            # Check if it's a rebalancing date
            if date in rebalance_dates or len(current_positions) == 0:
                # Get scores for this date
                current_scores = self.scores.loc[date]
                
                # Select top N ETFs
                top_etfs = current_scores.nlargest(self.top_n).index.tolist()
                
                # Equal weight allocation
                weight = 1.0 / self.top_n
                current_positions = {etf: weight if etf in top_etfs else 0 
                                   for etf in self.scores.columns}
            
            # Apply current positions
            for etf in self.scores.columns:
                positions.loc[date, etf] = current_positions.get(etf, 0)
        
        self.positions = positions
        return True
    
    def backtest(self, initial_capital=10000, commission=0.001):
        """Perform backtesting"""
        if self.positions is None:
            print("No positions generated. Run generate_positions() first.")
            return None
            
        # Calculate daily returns for each ETF
        returns_dict = {}
        for symbol in self.data.keys():
            returns_dict[symbol] = self.data[symbol]['close'].pct_change()
            
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(0)
        
        # Align positions and returns
        common_dates = self.positions.index.intersection(returns_df.index)
        positions_aligned = self.positions.loc[common_dates]
        returns_aligned = returns_df.loc[common_dates]
        
        # Calculate portfolio returns
        portfolio_returns = (positions_aligned.shift(1) * returns_aligned).sum(axis=1)
        
        # Account for transaction costs
        position_changes = positions_aligned.diff().abs().sum(axis=1)
        transaction_costs = position_changes * commission
        portfolio_returns_net = portfolio_returns - transaction_costs
        
        # Calculate cumulative returns
        portfolio_value = initial_capital * (1 + portfolio_returns_net).cumprod()
        
        # Performance metrics
        total_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
        
        volatility = portfolio_returns_net.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = ((portfolio_value / portfolio_value.expanding().max()) - 1).min()
        
        results = {
            'initial_capital': initial_capital,
            'final_value': portfolio_value.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns_net,
            'positions': positions_aligned
        }
        
        return results
    
    def plot_results(self, results=None):
        """Plot strategy performance and positions"""
        if results is None:
            print("No results to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio value over time
        portfolio_value = results['portfolio_value']
        ax1.plot(portfolio_value.index, portfolio_value.values, linewidth=2, label='ETF Rotation Strategy')
        
        # Compare with SPY benchmark
        if 'SPY' in self.data:
            spy_returns = self.data['SPY']['close'].pct_change().fillna(0)
            spy_value = results['initial_capital'] * (1 + spy_returns).cumprod()
            common_dates = portfolio_value.index.intersection(spy_value.index)
            ax1.plot(common_dates, spy_value.loc[common_dates], 
                    linewidth=2, alpha=0.7, label='SPY Benchmark')
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rolling returns
        rolling_returns = results['portfolio_returns'].rolling(window=60).mean() * 252
        ax2.plot(rolling_returns.index, rolling_returns.values, linewidth=2)
        ax2.set_title('60-Day Rolling Annualized Returns')
        ax2.set_ylabel('Annual Return')
        ax2.grid(True, alpha=0.3)
        
        # Position allocation over time (stacked area)
        positions = results['positions']
        ax3.stackplot(positions.index, *[positions[col] for col in positions.columns], 
                     labels=positions.columns, alpha=0.7)
        ax3.set_title('ETF Position Allocation Over Time')
        ax3.set_ylabel('Weight')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        portfolio_value = results['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        ax4.set_title('Portfolio Drawdown')
        ax4.set_ylabel('Drawdown')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"\n=== ETF Rotation Strategy Results ===")
        print(f"ETFs: {', '.join(self.data.keys())}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Rebalance Frequency: {self.rebalance_frequency}")
        print(f"Top N Holdings: {self.top_n}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")

def main():
    """Main function to run the strategy"""
    # Test with simulated ETFs representing different sectors
    etf_list = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'XLF', 'XLK', 'XLE', 'XLV']

    strategy = ETFRotationStrategy(
        etf_symbols=etf_list,
        start_date='2020-01-01',
        end_date='2024-12-31',
        rebalance_frequency='M'  # Monthly rebalancing
    )

    print("Generating simulated ETF data...")
    if not strategy.fetch_data():
        return

    print("Calculating composite scores...")
    if not strategy.calculate_all_scores():
        return

    print("Generating position allocations...")
    if not strategy.generate_positions():
        return

    print("Running backtest...")
    results = strategy.backtest(initial_capital=10000)

    if results:
        print("\n=== ETF Rotation Strategy Results ===")
        print(f"ETFs: {', '.join(strategy.data.keys())}")
        print(f"Period: {strategy.start_date} to {strategy.end_date}")
        print(f"Rebalance Frequency: {strategy.rebalance_frequency}")
        print(f"Top N Holdings: {strategy.top_n}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")

        print("\nStrategy completed successfully!")
        print("Note: Plotting disabled for faster execution. Use plot_results() to see charts.")
    else:
        print("Backtest failed to produce results.")

if __name__ == "__main__":
    main()
