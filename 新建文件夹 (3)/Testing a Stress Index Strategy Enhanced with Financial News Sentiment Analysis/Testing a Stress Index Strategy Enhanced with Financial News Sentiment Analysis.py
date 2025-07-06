import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import datetime
from dateutil.relativedelta import relativedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FinancialDataSimulator:
    """Class to simulate financial market data including prices, volatility, and news sentiment."""
    
    def __init__(self, start_date='2005-01-01', end_date='2024-01-31', freq='D'):
        """
        Initialize the simulator.
        
        Parameters:
        - start_date: Start date for simulation
        - end_date: End date for simulation
        - freq: Frequency of data ('D' for daily)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        self.n_days = len(self.dates)
        
    def simulate_market_data(self, markets=['SPX', 'NDX', 'EMM'], annual_return=0.08, 
                             annual_vol=0.15, correlation_matrix=None):
        """
        Simulate market price data for multiple indices.
        
        Parameters:
        - markets: List of market names
        - annual_return: Expected annual return
        - annual_vol: Expected annual volatility
        - correlation_matrix: Correlation matrix between markets
        
        Returns:
        - DataFrame with simulated market prices
        """
        n_markets = len(markets)
        
        # Default correlation matrix if not provided
        if correlation_matrix is None:
            # Create a correlation matrix with 0.7 correlation between markets
            correlation_matrix = np.ones((n_markets, n_markets)) * 0.7
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Daily parameters
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Generate correlated random returns
        cholesky = np.linalg.cholesky(correlation_matrix)
        random_returns = np.random.normal(0, 1, size=(self.n_days, n_markets))
        correlated_returns = np.dot(random_returns, cholesky.T)
        
        # Apply drift and volatility
        returns = daily_return + daily_vol * correlated_returns
        
        # Add some autocorrelation to returns (momentum effect)
        for i in range(1, self.n_days):
            returns[i] += 0.05 * returns[i-1]
        
        # Generate prices from returns
        prices = np.zeros((self.n_days, n_markets))
        prices[0] = 100  # Starting price
        for i in range(1, self.n_days):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        # Create DataFrame
        market_data = pd.DataFrame(prices, index=self.dates, columns=markets)
        return market_data
    
    def simulate_stress_index(self, market_data, window=252, crisis_periods=None):
        """
        Simulate a financial stress index based on market volatility and credit spreads.
        
        Parameters:
        - market_data: DataFrame with market prices
        - window: Window for rolling calculations (default: 252 days)
        - crisis_periods: List of tuples with (start_date, end_date) for simulated crisis periods
        
        Returns:
        - Series with stress index values
        """
        # Calculate returns
        returns = market_data.pct_change().dropna()
        
        # Calculate volatility component (average of rolling volatilities)
        volatility = returns.rolling(window=22).std().mean(axis=1) * np.sqrt(252)
        
        # Simulate credit spread component
        credit_spread = pd.Series(np.random.normal(0, 0.01, size=len(returns)), index=returns.index)
        
        # Add some autocorrelation to credit spread
        for i in range(1, len(credit_spread)):
            credit_spread.iloc[i] = 0.8 * credit_spread.iloc[i-1] + 0.2 * credit_spread.iloc[i]
        
        # Ensure credit spread is positive
        credit_spread = credit_spread - credit_spread.min() + 0.005
        
        # Combine components
        stress_raw = 0.7 * volatility.rank(pct=True) + 0.3 * credit_spread.rank(pct=True)
        
        # Add crisis periods
        if crisis_periods:
            for start, end in crisis_periods:
                if start in stress_raw.index and end in stress_raw.index:
                    crisis_mask = (stress_raw.index >= start) & (stress_raw.index <= end)
                    # Amplify stress during crisis periods
                    stress_raw.loc[crisis_mask] = stress_raw.loc[crisis_mask] * 1.5
        
        # Z-score the stress index
        stress_zscore = (stress_raw - stress_raw.rolling(window=window).mean()) / stress_raw.rolling(window=window).std()
        stress_zscore = stress_zscore.fillna(0)
        
        # Apply CDF to get values between 0 and 1
        stress_index = pd.Series(norm.cdf(stress_zscore), index=stress_zscore.index)
        
        return stress_index
    
    def simulate_vix_index(self, market_data, window=252, beta=1.2):
        """
        Simulate a VIX-like index based on market volatility.
        
        Parameters:
        - market_data: DataFrame with market prices
        - window: Window for rolling calculations
        - beta: Amplification factor for volatility
        
        Returns:
        - Series with VIX-like values
        """
        # Calculate returns for the first market (assuming it's S&P 500 like)
        returns = market_data.iloc[:, 0].pct_change().dropna()
        
        # Calculate rolling volatility and annualize
        rolling_vol = returns.rolling(window=22).std() * np.sqrt(252)
        
        # Add some randomness to mimic the forward-looking nature of VIX
        vix_raw = beta * rolling_vol + np.random.normal(0, 0.02, size=len(rolling_vol))
        
        # Ensure VIX is positive and has realistic values
        vix = vix_raw * 100  # Scale to percentage points
        vix = vix.clip(lower=10, upper=80)  # Realistic VIX range
        
        # Fill NAs with first valid value
        vix = vix.fillna(method='bfill')
        
        return vix
    
    def simulate_news_sentiment(self, market_data, positive_bias=0.05, noise_level=0.2):
        """
        Simulate news sentiment scores.
        
        Parameters:
        - market_data: DataFrame with market prices
        - positive_bias: Bias towards positive news (slight upward bias in markets)
        - noise_level: Amount of random noise in sentiment
        
        Returns:
        - DataFrame with simulated sentiment scores
        """
        # Calculate returns
        returns = market_data.pct_change().dropna()
        
        # Generate base sentiment that's correlated with returns but with a lag
        # (news often follows price moves with some lag)
        lagged_returns = returns.shift(-5)  # News partially anticipates future returns
        
        # Create noise component
        noise = pd.DataFrame(
            np.random.normal(positive_bias, noise_level, size=returns.shape),
            index=returns.index,
            columns=returns.columns
        )
        
        # Combine signals: 30% lagged returns (news reaction to recent market moves)
        # 30% future returns (news anticipating future moves)
        # 40% noise (random news)
        sentiment_raw = 0.3 * returns + 0.3 * lagged_returns + 0.4 * noise
        
        # Fill NAs
        sentiment_raw = sentiment_raw.fillna(0)
        
        # Process sentiment as described in the paper
        # 1. Convert to binary (-1, 0, 1) based on sign
        sentiment_sign = np.sign(sentiment_raw)
        
        # Replace zeros with random values (-1 or 1)
        random_signs = np.random.choice([-1, 1], size=sentiment_sign.shape)
        sentiment_sign = np.where(sentiment_sign == 0, random_signs, sentiment_sign)
        
        # 2. Calculate 10-day average
        sentiment_10d_avg = pd.DataFrame(sentiment_sign).rolling(window=10).mean()
        
        # 3. Z-score the 10-day average
        scaler = StandardScaler()
        sentiment_zscore = pd.DataFrame(
            scaler.fit_transform(sentiment_10d_avg.fillna(0)),
            index=sentiment_10d_avg.index,
            columns=sentiment_10d_avg.columns
        )
        
        # 4. Calculate mean of z-scored signal over preceding 10 days
        sentiment_signal = sentiment_zscore.rolling(window=10).mean()
        
        # 5. Convert to binary signal (positive vs negative)
        sentiment_binary = (sentiment_signal > 0).astype(int)
        
        # Create a single aggregate sentiment signal
        aggregate_sentiment = sentiment_binary.mean(axis=1)
        
        return aggregate_sentiment
    
    def simulate_full_dataset(self, markets=['SPX', 'NDX', 'EMM', 'NIKKEI', 'EUROSTOXX', 'EM'], 
                              crisis_periods=None):
        """
        Generate a complete simulated dataset including market prices, stress index, VIX, and news sentiment.
        
        Parameters:
        - markets: List of market names
        - crisis_periods: List of tuples with (start_date, end_date) for simulated crisis periods
        
        Returns:
        - Dictionary containing all simulated data
        """
        # If no crisis periods are provided, define some realistic ones
        if crisis_periods is None:
            crisis_periods = [
                ('2008-09-01', '2009-03-31'),  # Global Financial Crisis
                ('2011-08-01', '2011-09-30'),  # European Debt Crisis
                ('2015-08-01', '2015-10-31'),  # China Slowdown
                ('2018-12-01', '2018-12-31'),  # Fed Rate Hike Concerns
                ('2020-02-20', '2020-04-30'),  # COVID-19 Crash
                ('2022-01-01', '2022-06-30')   # Inflation Concerns
            ]
        
        # Simulate market prices
        market_data = self.simulate_market_data(markets=markets)
        
        # Simulate stress index
        stress_index = self.simulate_stress_index(market_data, crisis_periods=crisis_periods)
        
        # Simulate VIX index
        vix_index = self.simulate_vix_index(market_data)
        
        # Simulate news sentiment
        news_sentiment = self.simulate_news_sentiment(market_data)
        
        return {
            'market_data': market_data,
            'stress_index': stress_index,
            'vix_index': vix_index,
            'news_sentiment': news_sentiment,
            'dates': self.dates[1:],  # Excluding first day due to return calculation
        }

class StressIndexStrategy:
    """Class to implement and test the stress index strategy with news sentiment."""
    
    def __init__(self, data, transaction_cost=0.0002):
        """
        Initialize the strategy.
        
        Parameters:
        - data: Dictionary with market data, stress index, VIX, and news sentiment
        - transaction_cost: Transaction cost in basis points (default: 2bps)
        """
        self.data = data
        self.transaction_cost = transaction_cost
        
        # Extract key data components
        self.market_data = data['market_data']
        self.stress_index = data['stress_index']
        self.vix_index = data['vix_index']
        self.news_sentiment = data['news_sentiment']
        
        # Align all data to the same index
        self.align_data()
        
    def align_data(self):
        """Align all data to the same index."""
        # Get common index
        common_idx = self.market_data.index.intersection(
            self.stress_index.index.intersection(
                self.vix_index.index.intersection(
                    self.news_sentiment.index
                )
            )
        )
        
        # Reindex all data
        self.market_data = self.market_data.loc[common_idx]
        self.stress_index = self.stress_index.loc[common_idx]
        self.vix_index = self.vix_index.loc[common_idx]
        self.news_sentiment = self.news_sentiment.loc[common_idx]
        
        # Calculate market returns
        self.market_returns = self.market_data.pct_change().fillna(0)
    
    def compute_vix_signal(self, percentile=80):
        """
        Compute the VIX-based signal.
        
        Parameters:
        - percentile: Percentile threshold for high VIX (default: 80)
        
        Returns:
        - Series with VIX signal (1 for risk-on, 0 for risk-off)
        """
        # Calculate threshold
        threshold = self.vix_index.quantile(percentile/100)
        
        # Generate signal (1 for low VIX, 0 for high VIX)
        vix_signal = (self.vix_index <= threshold).astype(int)
        
        return vix_signal
    
    def compute_si_signal(self):
        """
        Compute the stress index signal.
        
        Returns:
        - Series with stress index signal (1 for risk-on, 0 for risk-off)
        """
        # Use the stress index directly as a risk-off signal (higher = more stress)
        # Invert it to get a risk-on signal (1 - stress_index)
        si_signal = 1 - self.stress_index
        
        return si_signal
    
    def compute_news_signal(self):
        """
        Compute the news sentiment signal.
        
        Returns:
        - Series with news sentiment signal (1 for positive, 0 for negative)
        """
        # The news sentiment is already binary (1 for positive, 0 for negative)
        return self.news_sentiment
    
    def compute_si_news_signal(self):
        """
        Compute the combined stress index and news signal.
        
        Returns:
        - Series with combined signal (product of stress index and news signals)
        """
        si_signal = self.compute_si_signal()
        news_signal = self.compute_news_signal()
        
        # Multiply the two signals as described in the paper
        combined_signal = si_signal * news_signal
        
        return combined_signal
    
    def compute_dynamic_signal(self, lookback=250, performance_window=20):
        """
        Compute the dynamic selection between SI and SI+News signals.
        
        Parameters:
        - lookback: Period for calculating Sharpe ratio (default: 250 days)
        - performance_window: Window for selecting strategy (default: 20 days)
        
        Returns:
        - Series with dynamic signal
        """
        # Compute the two component signals
        si_signal = self.compute_si_signal()
        si_news_signal = self.compute_si_news_signal()
        
        # Initialize dynamic signal with SI signal
        dynamic_signal = si_signal.copy()
        
        # Calculate strategy returns
        si_returns = si_signal.shift(1) * self.market_returns.iloc[:, 0]
        si_news_returns = si_news_signal.shift(1) * self.market_returns.iloc[:, 0]
        
        # Calculate rolling Sharpe ratios
        si_sharpe = si_returns.rolling(window=lookback).mean() / si_returns.rolling(window=lookback).std()
        si_news_sharpe = si_news_returns.rolling(window=lookback).mean() / si_news_returns.rolling(window=lookback).std()
        
        # Compare Sharpe ratios for strategy selection
        for i in range(lookback + performance_window, len(dynamic_signal)):
            start_idx = i - performance_window
            end_idx = i
            
            # Check which strategy had better Sharpe ratio over the last month
            si_avg_sharpe = si_sharpe.iloc[start_idx:end_idx].mean()
            si_news_avg_sharpe = si_news_sharpe.iloc[start_idx:end_idx].mean()
            
            # Select strategy based on higher Sharpe ratio
            if si_news_avg_sharpe > si_avg_sharpe:
                dynamic_signal.iloc[i] = si_news_signal.iloc[i]
            else:
                dynamic_signal.iloc[i] = si_signal.iloc[i]
        
        return dynamic_signal
    
    def backtest_strategy(self, signal, market_idx=0, initial_capital=1.0):
        """
        Backtest a strategy based on the provided signal.
        
        Parameters:
        - signal: Series with strategy signal (1 for long, 0 for cash)
        - market_idx: Index of the market to trade (default: 0 for first market)
        - initial_capital: Initial capital (default: 1.0)
        
        Returns:
        - DataFrame with strategy performance
        """
        # Get market returns for the selected market
        market_returns = self.market_returns.iloc[:, market_idx]
        
        # Initialize strategy performance
        strategy_returns = pd.Series(0, index=signal.index)
        
        # Calculate positions (lagged signal)
        positions = signal.shift(1).fillna(0)
        
        # Calculate transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        
        # Calculate strategy returns (with transaction costs)
        strategy_returns = positions * market_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate drawdowns
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        
        # Calculate portfolio value
        portfolio_value = initial_capital * cumulative_returns
        
        # Turnover calculation
        turnover = position_changes.sum()
        
        # Performance metrics
        total_return = portfolio_value.iloc[-1] / initial_capital - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = drawdown.min()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Create performance summary
        performance = {
            'portfolio_value': portfolio_value,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown,
            'positions': positions,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'turnover': turnover
        }
        
        return performance
    
    def run_all_strategies(self, market_idx=0):
        """
        Run all strategies for a specific market.
        
        Parameters:
        - market_idx: Index of the market to trade (default: 0 for first market)
        
        Returns:
        - Dictionary with performance results for all strategies
        """
        # Compute signals
        vix_signal = self.compute_vix_signal()
        si_signal = self.compute_si_signal()
        news_signal = self.compute_news_signal()
        si_news_signal = self.compute_si_news_signal()
        dynamic_signal = self.compute_dynamic_signal()
        
        # Create a long-only signal for benchmark
        long_only_signal = pd.Series(1, index=self.market_data.index)
        
        # Run backtests
        results = {
            'Long Only': self.backtest_strategy(long_only_signal, market_idx),
            'VIX': self.backtest_strategy(vix_signal, market_idx),
            'SI': self.backtest_strategy(si_signal, market_idx),
            'News': self.backtest_strategy(news_signal, market_idx),
            'SI+News': self.backtest_strategy(si_news_signal, market_idx),
            'Dynamic SI+News': self.backtest_strategy(dynamic_signal, market_idx)
        }
        
        return results
    
    def run_all_markets(self):
        """
        Run all strategies for all markets.
        
        Returns:
        - Dictionary with performance results for all strategies and markets
        """
        all_results = {}
        
        # Run strategies for each market
        for i, market in enumerate(self.market_data.columns):
            all_results[market] = self.run_all_strategies(i)
        
        # Run strategies for an equally weighted basket of all markets
        # Create an equally weighted basket of returns
        basket_returns = self.market_returns.mean(axis=1)
        
        # Save original market returns
        original_returns = self.market_returns
        
        # Replace with basket returns
        self.market_returns = pd.DataFrame(basket_returns, columns=['Basket'])
        
        # Run strategies on the basket
        all_results['Basket'] = self.run_all_strategies(0)
        
        # Restore original market returns
        self.market_returns = original_returns
        
        return all_results

    def normalize_volatility(self, strategy_perf, target_vol=0.075):
        """
        Rescale a strategy to match a target volatility level.
        
        Parameters:
        - strategy_perf: Strategy performance dictionary
        - target_vol: Target annualized volatility (default: 7.5%)
        
        Returns:
        - Dictionary with rescaled strategy performance
        """
        # Get current volatility
        current_vol = strategy_perf['annual_volatility']
        
        # Calculate scaling factor
        scale = target_vol / current_vol if current_vol > 0 else 1.0
        
        # Create a copy of the performance dictionary
        rescaled_perf = strategy_perf.copy()
        
        # Rescale relevant metrics
        rescaled_perf['returns'] = strategy_perf['returns'] * scale
        rescaled_perf['cumulative_returns'] = (1 + rescaled_perf['returns']).cumprod()
        rescaled_perf['portfolio_value'] = rescaled_perf['cumulative_returns']
        
        # Recalculate drawdowns
        peak = rescaled_perf['cumulative_returns'].cummax()
        rescaled_perf['drawdown'] = (rescaled_perf['cumulative_returns'] - peak) / peak
        
        # Recalculate performance metrics
        rescaled_perf['total_return'] = rescaled_perf['portfolio_value'].iloc[-1] - 1
        rescaled_perf['annual_return'] = (1 + rescaled_perf['total_return']) ** (252 / len(rescaled_perf['portfolio_value'])) - 1
        rescaled_perf['annual_volatility'] = rescaled_perf['returns'].std() * np.sqrt(252)
        rescaled_perf['sharpe_ratio'] = rescaled_perf['annual_return'] / rescaled_perf['annual_volatility'] if rescaled_perf['annual_volatility'] > 0 else 0
        rescaled_perf['max_drawdown'] = rescaled_perf['drawdown'].min()
        rescaled_perf['calmar_ratio'] = rescaled_perf['annual_return'] / abs(rescaled_perf['max_drawdown']) if rescaled_perf['max_drawdown'] < 0 else float('inf')
        
        return rescaled_perf
    
    def compare_strategies(self, results, market, dynamic_vol=None):
        """
        Compare strategy performance for a specific market.
        
        Parameters:
        - results: Dictionary with strategy results
        - market: Market name
        - dynamic_vol: Volatility of the Dynamic SI+News strategy for rescaling (default: None)
        
        Returns:
        - DataFrame with performance metrics
        """
        strategies = list(results.keys())
        metrics = ['sharpe_ratio', 'calmar_ratio', 'annual_volatility', 'max_drawdown', 'turnover']
        comparison = pd.DataFrame(index=strategies, columns=metrics)
        
        # Fill the comparison table
        for strategy in strategies:
            for metric in metrics:
                comparison.loc[strategy, metric] = results[strategy][metric]
        
        # Rescale the benchmark (Long Only) to match Dynamic SI+News volatility
        if dynamic_vol is not None:
            benchmark_vol = results['Long Only']['annual_volatility']
            scale = dynamic_vol / benchmark_vol
            comparison.loc['Long Only (Scaled)', 'annual_volatility'] = dynamic_vol
            comparison.loc['Long Only (Scaled)', 'sharpe_ratio'] = results['Long Only']['sharpe_ratio']
            comparison.loc['Long Only (Scaled)', 'max_drawdown'] = results['Long Only']['max_drawdown'] * scale
            comparison.loc['Long Only (Scaled)', 'calmar_ratio'] = comparison.loc['Long Only', 'calmar_ratio'] / scale
            comparison.loc['Long Only (Scaled)', 'turnover'] = results['Long Only']['turnover']
        
        return comparison
    
    def plot_strategy_performance(self, results, market, figsize=(15, 10)):
        """
        Plot strategy performance for a specific market.
        
        Parameters:
        - results: Dictionary with strategy results
        - market: Market name
        - figsize: Figure size (default: (15, 10))
        """
        # Get Dynamic SI+News volatility for rescaling
        dynamic_vol = results['Dynamic SI+News']['annual_volatility']
        
        # Rescale the benchmark to match Dynamic SI+News volatility
        benchmark_rescaled = self.normalize_volatility(results['Long Only'], dynamic_vol)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio values
        axes[0].plot(results['Dynamic SI+News']['portfolio_value'], 
                   label='Dynamic SI+News', linewidth=2)
        axes[0].plot(benchmark_rescaled['portfolio_value'], 
                   label='Long Only (rescaled)', linewidth=2, linestyle='--')
        
        # Add title and labels
        axes[0].set_title(f'Dynamic SI+News vs Long Only Strategy for {market}', fontsize=14)
        axes[0].set_ylabel('Portfolio Value', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot allocation (positions)
        axes[1].plot(results['Dynamic SI+News']['positions'], color='green', linewidth=1)
        axes[1].set_ylabel('Allocation', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_strategies(self, results, market, figsize=(15, 10)):
        """
        Plot all strategy performances for a specific market.
        
        Parameters:
        - results: Dictionary with strategy results
        - market: Market name
        - figsize: Figure size (default: (15, 10))
        """
        plt.figure(figsize=figsize)
        
        # Plot portfolio values for all strategies
        for strategy, perf in results.items():
            plt.plot(perf['portfolio_value'], label=strategy, linewidth=2)
        
        # Add title and labels
        plt.title(f'Strategy Comparison for {market}', fontsize=14)
        plt.ylabel('Portfolio Value', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_performance_summary(self, all_results):
        """
        Generate a comprehensive performance summary for all markets and strategies.
        
        Parameters:
        - all_results: Dictionary with results for all markets and strategies
        
        Returns:
        - Dictionary with performance summaries
        """
        summary = {}
        
        for market, results in all_results.items():
            # Get Dynamic SI+News volatility for rescaling
            dynamic_vol = results['Dynamic SI+News']['annual_volatility']
            
            # Generate comparison table
            summary[market] = self.compare_strategies(results, market, dynamic_vol)
        
        return summary

def main():
    """Main function to run the simulations and strategy tests."""
    print("Simulating financial market data...")
    
    # Initialize simulator
    simulator = FinancialDataSimulator(start_date='2005-01-01', end_date='2024-01-31')
    
    # Generate simulated data
    simulated_data = simulator.simulate_full_dataset()
    
    print("Data simulation complete.")
    print(f"Generated {len(simulated_data['market_data'])} days of data for {len(simulated_data['market_data'].columns)} markets.")
    
    # Initialize strategy
    print("\nInitializing and testing strategies...")
    strategy = StressIndexStrategy(simulated_data)
    
    # Run strategies for all markets
    all_results = strategy.run_all_markets()
    
    # Generate performance summary
    summary = strategy.generate_performance_summary(all_results)
    
    # Print performance summary for each market
    for market, comparison in summary.items():
        print(f"\nPerformance Summary for {market}:")
        print(comparison.round(3))
    
    # Plot performance for S&P 500, NASDAQ, and the basket
    markets_to_plot = ['SPX', 'NDX', 'Basket']
    for market in markets_to_plot:
        if market in all_results:
            print(f"\nPlotting performance for {market}...")
            strategy.plot_strategy_performance(all_results[market], market)
            strategy.plot_all_strategies(all_results[market], market)
    
    print("\nStrategy testing complete.")

if __name__ == "__main__":
    main()