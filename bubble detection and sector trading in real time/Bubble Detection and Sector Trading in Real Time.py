import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
from scipy import stats

class BubbleDetector:
    """Simplified bubble detector using price acceleration and threshold crossing"""
    
    def __init__(self, window_size=20, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
    
    def detect_bubbles(self, series):
        """
        Detect bubbles using price acceleration and threshold crossing
        
        Parameters:
        -----------
        series : pd.Series
            Time series of price-earnings ratios
            
        Returns:
        --------
        pd.Series
            Binary bubble indicator (1 for bubble, 0 for no bubble)
        """
        if isinstance(series, pd.DataFrame):
            if len(series.columns) > 0:
                # Take the first column if it's a DataFrame
                series = series.iloc[:, 0]
            else:
                return pd.Series(0, index=series.index)
                
        # Calculate returns and accelerations
        returns = series.pct_change()
        accelerations = returns.rolling(window=self.window_size).mean() / returns.rolling(window=self.window_size).std()
        
        # Identify bubbles when acceleration exceeds threshold
        bubble_indicator = (accelerations > self.threshold).astype(int)
        
        # Apply smoothing: require bubble signal to persist for at least 2 periods
        smoothed_indicator = bubble_indicator.rolling(window=3, min_periods=1).mean()
        smoothed_indicator = (smoothed_indicator >= 0.67).astype(int)  # 2 out of 3 periods showing bubble
        
        return smoothed_indicator


class SectorTradingStrategy:
    """Trading strategy for sectors based on bubble indicators"""
    
    def __init__(self, transaction_cost=0.005):
        self.transaction_cost = transaction_cost
        self.market_data = None
        self.sector_data = {}
        self.bubble_indicators = {}
        self.signals = {}
        self.positions = None
        self.returns = None
    
    def generate_synthetic_data(self, start_date, end_date, n_sectors=11):
        """
        Generate synthetic data for market index and sectors
        
        Parameters:
        -----------
        start_date : str
            Start date (YYYYMMDD)
        end_date : str
            End date (YYYYMMDD)
        n_sectors : int
            Number of sectors to generate
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate weekly dates
        dates = pd.date_range(start=start, end=end, freq='W-WED')
        
        # Create sector names
        sector_names = [
            'Oil & Gas', 'Basic Materials', 'Industrials', 'Consumer Goods',
            'Health Care', 'Consumer Services', 'Telecom', 'Utilities',
            'Technology', 'Financials', 'REITs'
        ][:n_sectors]
        
        # Generate market data
        print("Generating market data...")
        np.random.seed(42)
        
        market_prices = pd.Series(index=dates, name='Price')
        market_div_yields = pd.Series(index=dates, name='DivYield')
        
        # Initial price and parameters
        price = 100.0
        annual_drift = 0.08  # 8% annual return
        annual_vol = 0.15    # 15% annual volatility
        weekly_drift = annual_drift / 52
        weekly_vol = annual_vol / np.sqrt(52)
        
        # Generate price series
        for i, date in enumerate(dates):
            if i == 0:
                market_prices[date] = price
            else:
                ret = np.random.normal(weekly_drift, weekly_vol)
                price *= (1 + ret)
                market_prices[date] = price
            
            # Dividend yield (around 2%)
            market_div_yields[date] = np.random.normal(2.0, 0.3)
        
        # Store market data
        self.market_data = pd.DataFrame({
            'Price': market_prices,
            'DivYield': market_div_yields
        })
        
        # Generate sector data
        print("Generating sector data...")
        self.sector_data = {}
        
        for i, sector in enumerate(sector_names):
            np.random.seed(42 + i)  # Different seed for each sector
            
            # Sector parameters
            beta = np.random.uniform(0.7, 1.3)
            alpha = np.random.normal(0, 0.001) * 52 / 52  # Weekly alpha
            idiosyncratic_vol = np.random.uniform(0.1, 0.2) / np.sqrt(52)
            
            # Generate prices
            prices = pd.Series(index=dates, name='Price')
            div_yields = pd.Series(index=dates, name='DivYield')
            pe_ratios = pd.Series(index=dates, name='PE')
            
            # Initial values
            prices[dates[0]] = 100.0
            pe_ratios[dates[0]] = np.random.uniform(12, 25)
            
            # Generate series
            for j, date in enumerate(dates):
                if j == 0:
                    continue
                    
                # Price calculation based on CAPM
                market_ret = (market_prices[date] / market_prices[dates[j-1]]) - 1
                sector_ret = alpha + beta * market_ret + np.random.normal(0, idiosyncratic_vol)
                prices[date] = prices[dates[j-1]] * (1 + sector_ret)
                
                # Dividend yield
                div_yields[date] = market_div_yields[date] + np.random.normal(0, 0.3)
                div_yields[date] = max(0.1, div_yields[date])  # Minimum 0.1%
                
                # P/E ratio - random walk with drift
                pe_change = np.random.normal(0, 0.02)
                pe_ratios[date] = pe_ratios[dates[j-1]] * (1 + pe_change)
                pe_ratios[date] = max(5, min(50, pe_ratios[date]))  # Keep in reasonable range
            
            # Create artificial bubbles
            n_bubbles = np.random.randint(1, 4)
            
            for _ in range(n_bubbles):
                # Choose bubble period
                bubble_start = np.random.randint(len(dates) // 4, 3 * len(dates) // 4)
                bubble_duration = np.random.randint(4, 13)  # 1-3 months
                bubble_end = min(bubble_start + bubble_duration, len(dates) - 1)
                
                # Bubble growth rate
                bubble_growth = np.random.uniform(0.03, 0.10)  # 3-10% weekly growth
                
                # Apply bubble to PE ratio
                for j in range(bubble_start, bubble_end):
                    date = dates[j]
                    next_date = dates[j+1] if j+1 < len(dates) else dates[j]
                    pe_ratios[next_date] = pe_ratios[date] * (1 + bubble_growth)
                
                # Bubble burst
                if bubble_end < len(dates) - 1:
                    burst_date = dates[bubble_end]
                    next_date = dates[bubble_end + 1]
                    pe_ratios[next_date] = pe_ratios[burst_date] * (1 - np.random.uniform(0.1, 0.3))
            
            # Store sector data
            self.sector_data[sector] = pd.DataFrame({
                'Price': prices,
                'DivYield': div_yields,
                'PE': pe_ratios
            })
            
            print(f"  Generated data for {sector}")
        
        print(f"Generated data from {start_date} to {end_date} for {n_sectors} sectors")
    
    def detect_bubbles(self):
        """Detect bubbles in each sector's PE ratio"""
        if not self.sector_data:
            print("No sector data available")
            return None
        
        print("Detecting bubbles...")
        detector = BubbleDetector(window_size=8, threshold=1.5)
        
        for sector, data in self.sector_data.items():
            print(f"  Processing {sector}...")
            pe_series = data['PE']
            
            # Detect bubbles
            bubble_indicator = detector.detect_bubbles(pe_series)
            self.bubble_indicators[sector] = bubble_indicator
            
            print(f"    Detected {bubble_indicator.sum()} bubble periods")
        
        return self.bubble_indicators
    
    def generate_signals(self):
        """Generate trading signals for each sector"""
        if not self.bubble_indicators or not self.sector_data:
            print("Bubble indicators or sector data not available")
            return None
        
        print("Generating signals...")
        
        for sector, data in self.sector_data.items():
            print(f"  Generating signals for {sector}...")
            
            # Calculate returns
            returns = data['Price'].pct_change().fillna(0)
            
            # Get bubble indicator
            bubble_indicator = self.bubble_indicators[sector]
            
            # Create signal DataFrame
            signals = pd.DataFrame(index=data.index)
            
            # PSY Bubble Indicator (renamed to avoid hyphen)
            signals['PSY_BI'] = bubble_indicator
            
            # Directional signal (positive momentum)
            signals['DS'] = (returns > 0).astype(int)
            
            # Modified bubble indicator (bubble + positive momentum)
            signals['MBI'] = (signals['PSY_BI'] & signals['DS']).astype(int)
            
            self.signals[sector] = signals
            
            print(f"    Generated {signals['PSY_BI'].sum()} PSY_BI signals, {signals['MBI'].sum()} MBI signals")
        
        return self.signals
    
    def run_strategy(self, strategy_type='MBI'):
        """
        Run the trading strategy
        
        Parameters:
        -----------
        strategy_type : str
            Strategy type: 'MBI', 'PSY_BI', 'DS', or 'BH' (buy-and-hold)
            
        Returns:
        --------
        pd.DataFrame
            Strategy returns
        """
        # Fix strategy name to avoid hyphen
        if strategy_type == 'PSY-BI':
            strategy_type = 'PSY_BI'
            
        if strategy_type not in ['MBI', 'PSY_BI', 'DS', 'BH']:
            print(f"Invalid strategy type: {strategy_type}")
            return None
        
        if self.market_data is None:
            print("Market data not available")
            return None
        
        if len(self.sector_data) == 0:
            print("Sector data not available")
            return None
        
        if strategy_type != 'BH' and (not self.signals or len(self.signals) == 0):
            print(f"No signals for {strategy_type} strategy")
            return None
        
        print(f"Running {strategy_type} strategy...")
        
        # Get common dates
        dates = self.market_data.index
        
        # Initialize positions and returns
        positions = pd.DataFrame(0, index=dates, columns=['MARKET'] + list(self.sector_data.keys()))
        returns = pd.DataFrame(0, index=dates, columns=['RETURN', 'CUMULATIVE_RETURN'])
        
        # Calculate market and sector returns
        market_returns = self.market_data['Price'].pct_change().fillna(0)
        
        sector_returns = {}
        for sector, data in self.sector_data.items():
            # Calculate price return
            price_return = data['Price'].pct_change().fillna(0)
            
            # Add dividend yield (convert from annual to weekly)
            div_yield = data['DivYield'] / 100 / 52
            
            # Total return = price return + dividend yield
            total_return = price_return + div_yield
            sector_returns[sector] = total_return
        
        # Buy and hold strategy
        if strategy_type == 'BH':
            positions['MARKET'] = 1
            returns['RETURN'] = market_returns
            returns['CUMULATIVE_RETURN'] = (1 + returns['RETURN']).cumprod() - 1
            self.positions = positions
            self.returns = returns
            return returns
        
        # Track transaction costs
        transaction_costs = pd.Series(0, index=dates)
        
        # Run strategy
        for i, date in enumerate(dates):
            if i == 0:
                continue  # Skip first day (no returns)
                
            # Get sector signals
            today_signals = {}
            for sector in self.sector_data.keys():
                if date in self.signals[sector].index:
                    today_signals[sector] = self.signals[sector].loc[date, strategy_type]
                else:
                    today_signals[sector] = 0
            
            # Count positive signals
            positive_sectors = [s for s, signal in today_signals.items() if signal == 1]
            n_positive = len(positive_sectors)
            
            # Update positions
            if n_positive > 0:
                # Equal weight in positive sectors
                for sector in self.sector_data.keys():
                    if sector in positive_sectors:
                        positions.loc[date, sector] = 1 / n_positive
                    else:
                        positions.loc[date, sector] = 0
                positions.loc[date, 'MARKET'] = 0
            else:
                # If no positive signals, invest in market
                positions.loc[date, 'MARKET'] = 1
                for sector in self.sector_data.keys():
                    positions.loc[date, sector] = 0
            
            # Calculate transaction costs
            if i > 0:
                prev_date = dates[i-1]
                position_changes = positions.loc[date] - positions.loc[prev_date]
                transaction_cost = self.transaction_cost * abs(position_changes).sum()
                transaction_costs[date] = transaction_cost
            
            # Calculate returns
            day_return = 0
            
            # Market component
            if positions.loc[date, 'MARKET'] > 0 and date in market_returns.index:
                day_return += positions.loc[date, 'MARKET'] * market_returns[date]
            
            # Sector components
            for sector in self.sector_data.keys():
                if positions.loc[date, sector] > 0 and date in sector_returns[sector].index:
                    day_return += positions.loc[date, sector] * sector_returns[sector][date]
            
            # Subtract transaction costs
            day_return -= transaction_costs[date]
            
            # Store return
            returns.loc[date, 'RETURN'] = day_return
        
        # Calculate cumulative returns
        returns['CUMULATIVE_RETURN'] = (1 + returns['RETURN']).cumprod() - 1
        
        self.positions = positions
        self.returns = returns
        
        return returns
    
    def calculate_performance_metrics(self, risk_free_rate=0.02):
        """
        Calculate performance metrics
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        dict
            Performance metrics
        """
        if self.returns is None or len(self.returns) == 0:
            print("No strategy returns available")
            return None
        
        # Get strategy returns
        returns = self.returns['RETURN'].fillna(0)
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        n_years = len(returns) / 52  # Assuming weekly data
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = returns.std() * np.sqrt(52)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Final metrics dictionary
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Final Wealth': (1 + total_return)
        }
        
        return metrics
    
    def plot_returns(self, strategies=None, save_path=None):
        """
        Plot cumulative returns for multiple strategies
        
        Parameters:
        -----------
        strategies : dict
            Dictionary of strategy names and returns DataFrames
        save_path : str
            Path to save the plot
        """
        if strategies is None:
            if self.returns is None:
                print("No strategy returns to plot")
                return
            strategies = {'Strategy': self.returns}
        
        plt.figure(figsize=(12, 6))
        
        for name, returns in strategies.items():
            plt.plot(returns['CUMULATIVE_RETURN'], label=name)
        
        plt.title('Cumulative Strategy Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


def main():
    # Create output directory
    output_dir = "bubble_strategy_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize strategy
    strategy = SectorTradingStrategy(transaction_cost=0.005)
    
    # Set date range (5 years)
    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=5*365)).strftime("%Y%m%d")
    
    # Generate synthetic data
    strategy.generate_synthetic_data(start_date, end_date, n_sectors=11)
    
    # Detect bubbles
    strategy.detect_bubbles()
    
    # Generate signals
    strategy.generate_signals()
    
    # Run strategies and collect returns
    strategy_returns = {}
    strategy_metrics = {}
    
    # Key fix: handle both 'PSY_BI' and 'PSY-BI' by standardizing to 'PSY_BI'
    for strategy_type in ['MBI', 'PSY-BI', 'DS', 'BH']:
        original_name = strategy_type  # Keep original name for display
        
        print(f"\nRunning {original_name} strategy...")
        
        # Run strategy (inside the function we correct PSY-BI to PSY_BI)
        returns = strategy.run_strategy(strategy_type=strategy_type)
        if returns is not None:
            strategy_returns[original_name] = returns
            
            # Calculate performance metrics
            metrics = strategy.calculate_performance_metrics()
            strategy_metrics[original_name] = metrics
            
            # Print metrics
            print(f"\n{original_name} Strategy Performance:")
            print(f"Final Wealth: ${metrics['Final Wealth']:.2f}")
            print(f"Annualized Return: {metrics['Annualized Return']*100:.2f}%")
            print(f"Annualized Volatility: {metrics['Annualized Volatility']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown: {metrics['Max Drawdown']*100:.2f}%")
    
    # Plot returns
    strategy.plot_returns(strategies=strategy_returns,
                          save_path=os.path.join(output_dir, "strategy_returns.png"))
    
    # Save results
    if strategy_metrics:
        try:
            # Create a DataFrame with strategy metrics
            metrics_df = pd.DataFrame({
                strat: [
                    metrics['Final Wealth'],
                    metrics['Annualized Return'],
                    metrics['Sharpe Ratio']
                ] for strat, metrics in strategy_metrics.items()
            }, index=['Final Wealth', 'Annualized Return', 'Sharpe Ratio'])
            
            # Save to CSV
            metrics_df.to_csv(os.path.join(output_dir, "strategy_metrics.csv"))
            
            # Save returns
            for strat, returns in strategy_returns.items():
                returns.to_csv(os.path.join(output_dir, f"{strat.replace('-', '_')}_returns.csv"))
            
            print(f"\nResults saved to {output_dir}")
            
            # Print strategy comparison
            print("\nStrategy Performance Comparison:")
            print(metrics_df)
            
            # Identify best strategy
            best_strategy = metrics_df.loc['Sharpe Ratio'].idxmax()
            print(f"\nBest strategy by Sharpe ratio: {best_strategy}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()