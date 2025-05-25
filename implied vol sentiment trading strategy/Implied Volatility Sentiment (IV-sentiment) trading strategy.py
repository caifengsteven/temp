import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

class IVSentimentStrategy:
    """
    Implementation of the Implied Volatility Sentiment strategy from:
    "Implied volatility sentiment: a tale of two tails" (Félix et al., 2020)
    """
    
    def __init__(self, start_date='20150101', end_date='20230101'):
        """
        Initialize the strategy with date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYYMMDD' format
        end_date : str
            End date in 'YYYYMMDD' format
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy parameters
        self.moneyness_levels = [80, 90, 110, 120]  # 80% and 90% for puts, 110% and 120% for calls
        self.option_maturities = [3, 6, 12]  # 3-month, 6-month and 12-month options
        
        # Data containers
        self.index_iv_data = {}
        self.single_stock_iv_data = {}
        self.market_data = None
        self.cash_data = None
        self.iv_sentiment = {}
        self.positions = pd.DataFrame()
        self.returns = pd.DataFrame()
        
        # Strategy results
        self.performance = {}
        
        print("Using simulated data for IV-sentiment strategy")
            
    def fetch_data(self):
        """
        Generate simulated data for strategy testing
        """
        self._generate_simulated_data()
    
    def _generate_simulated_data(self):
        """
        Generate simulated IV data and market data
        """
        print("Generating simulated data...")
        
        # Convert dates to datetime objects
        start_date = dt.datetime.strptime(self.start_date, '%Y%m%d')
        end_date = dt.datetime.strptime(self.end_date, '%Y%m%d')
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate market data with realistic properties
        np.random.seed(42)
        
        # Create market regimes (bull, bear, volatile)
        n_days = len(date_range)
        regime_changes = np.random.choice(range(50, n_days-50), size=8, replace=False)
        regime_changes.sort()
        
        regimes = np.ones(n_days)
        for i, change_point in enumerate(regime_changes):
            if i % 3 == 0:  # Bull market
                regimes[change_point:] = 1
            elif i % 3 == 1:  # Bear market
                regimes[change_point:] = 2
            else:  # Volatile market
                regimes[change_point:] = 3
        
        # Market returns (S&P 500)
        daily_returns = np.zeros(n_days)
        daily_vols = np.zeros(n_days)
        
        for i in range(n_days):
            if regimes[i] == 1:  # Bull market
                daily_returns[i] = np.random.normal(0.0008, 0.008)  # ~20% annual return
                daily_vols[i] = np.random.uniform(0.10, 0.15)  # Low volatility
            elif regimes[i] == 2:  # Bear market
                daily_returns[i] = np.random.normal(-0.0008, 0.015)  # ~-20% annual return
                daily_vols[i] = np.random.uniform(0.20, 0.35)  # High volatility
            else:  # Volatile market
                daily_returns[i] = np.random.normal(0.0002, 0.020)  # ~5% annual return, high vol
                daily_vols[i] = np.random.uniform(0.25, 0.40)  # Very high volatility
        
        # Add some momentum and mean-reversion effects
        for i in range(5, n_days):
            # Momentum component (trailing 5-day return influence)
            momentum = np.sum(daily_returns[i-5:i]) * 0.1
            # Mean-reversion component (if extreme moves)
            mean_reversion = -np.sign(np.sum(daily_returns[i-3:i])) * max(0, (abs(np.sum(daily_returns[i-3:i])) - 0.05)) * 0.15
            daily_returns[i] += momentum + mean_reversion
        
        # Calculate price levels
        levels = 100 * np.cumprod(1 + daily_returns)
        
        # Create market data
        self.market_data = pd.DataFrame({
            'PX_LAST': levels,
            'VOLATILITY_90D': daily_vols
        }, index=date_range)
        
        # Risk-free rate (3-month Treasury bill)
        self.cash_data = pd.DataFrame({
            'PX_LAST': np.random.uniform(0.005, 0.025, len(date_range))  # Between 0.5% and 2.5%
        }, index=date_range)
        
        # Generate IV data with realistic skew
        for maturity in self.option_maturities:
            # Dynamic IV skew that depends on market regimes
            implied_vol_base = pd.Series(daily_vols * 1.1, index=date_range)  # Base IV level slightly higher than realized vol
            
            # Calculate moving averages for trend signals
            market_50d_ma = self.market_data['PX_LAST'].rolling(window=50).mean()
            market_200d_ma = self.market_data['PX_LAST'].rolling(window=200).mean()
            trend_signal = (market_50d_ma > market_200d_ma).astype(int).replace(0, -1)
            
            # Volatility regime (1=low, 2=medium, 3=high)
            vol_regime = pd.Series(regimes, index=date_range)
            
            # Index options with dynamic skew
            self.index_iv_data[maturity] = {}
            
            # Add more skew for higher IV periods (vol of vol)
            for moneyness in self.moneyness_levels + [100]:
                field_name = f'IVOL_CALL_{maturity}M_{moneyness}' if moneyness >= 100 else f'IVOL_PUT_{maturity}M_{moneyness}'
                
                if moneyness == 100:  # ATM
                    self.index_iv_data[maturity][moneyness] = pd.DataFrame({
                        field_name: implied_vol_base
                    }, index=date_range)
                elif moneyness < 100:  # OTM puts
                    # More skew for OTM puts (80% strike has higher IV than 90% strike)
                    skew_factor = (100 - moneyness) / 10  # 2 for 80%, 1 for 90%
                    
                    # Base skew level depends on volatility regime
                    base_skew = vol_regime.apply(lambda x: 0.08 if x == 1 else (0.12 if x == 2 else 0.18))
                    
                    # Enhanced skew when market is in downtrend
                    trend_adjustment = 0.05 * (1 - trend_signal)
                    
                    iv_premium = implied_vol_base * (base_skew * skew_factor + trend_adjustment)
                    
                    self.index_iv_data[maturity][moneyness] = pd.DataFrame({
                        field_name: implied_vol_base + iv_premium
                    }, index=date_range)
                else:  # OTM calls
                    # Less skew for OTM calls
                    skew_factor = (moneyness - 100) / 20  # 0.5 for 110%, 1 for 120%
                    
                    # Base call skew depends on volatility regime (less skew in bull markets)
                    base_call_skew = 0.03 * vol_regime
                    
                    # Lower call skew when market is in uptrend
                    trend_adjustment = 0.02 * trend_signal
                    
                    iv_premium = implied_vol_base * (base_call_skew * skew_factor + trend_adjustment)
                    
                    self.index_iv_data[maturity][moneyness] = pd.DataFrame({
                        field_name: implied_vol_base + iv_premium
                    }, index=date_range)
            
            # Single stock options with different skew patterns
            self.single_stock_iv_data[maturity] = {}
            
            # Generate 30 stocks with different IV levels and skews
            for moneyness in self.moneyness_levels + [100]:
                self.single_stock_iv_data[maturity][moneyness] = pd.DataFrame(index=date_range)
                
                for i in range(30):
                    # Individual stock IV is more varied than index IV
                    stock_iv_base = implied_vol_base * np.random.uniform(0.8, 1.5)
                    
                    # Each stock has some unique characteristics
                    stock_beta = np.random.uniform(0.7, 1.3)  # Market sensitivity
                    stock_trend = trend_signal * (1 + np.random.uniform(-0.5, 0.5))  # Individual trend
                    
                    if moneyness == 100:  # ATM
                        self.single_stock_iv_data[maturity][moneyness][f'Stock_{i}'] = stock_iv_base
                    elif moneyness < 100:  # OTM puts
                        # Less skew than index for single stock OTM puts
                        skew_factor = (100 - moneyness) / 20
                        
                        # Base skew level
                        base_skew = vol_regime.apply(lambda x: 0.05 if x == 1 else (0.07 if x == 2 else 0.10))
                        
                        # Adjust based on stock characteristics
                        trend_adjustment = 0.02 * stock_trend * stock_beta
                        
                        iv_premium = stock_iv_base * (base_skew * skew_factor + trend_adjustment)
                        
                        self.single_stock_iv_data[maturity][moneyness][f'Stock_{i}'] = stock_iv_base + iv_premium
                    else:  # OTM calls
                        # More skew for single stock OTM calls (bullish sentiment for single stocks)
                        skew_factor = (moneyness - 100) / 10
                        
                        # Base skew level increases during bull markets for single stocks
                        bull_market_effect = 1.5 * (trend_signal > 0).astype(int)
                        
                        base_skew = 0.05 + 0.02 * bull_market_effect
                        
                        # Enhanced skew based on stock characteristics
                        trend_adjustment = 0.03 * stock_trend * (2 - stock_beta)  # More pronounced for low beta stocks
                        
                        iv_premium = stock_iv_base * (base_skew * skew_factor + trend_adjustment)
                        
                        self.single_stock_iv_data[maturity][moneyness][f'Stock_{i}'] = stock_iv_base + iv_premium
        
        print(f"Generated simulated data from {date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}")
    
    def calculate_iv_sentiment(self):
        """
        Calculate IV sentiment measures for different maturities and moneyness levels
        
        Formula: IV-sentiment = OTM index put IV - OTM single stock call IV
        """
        print("Calculating IV sentiment measures...")
        
        for maturity in self.option_maturities:
            self.iv_sentiment[maturity] = {}
            
            # Calculate IV-sentiment 90-110
            self._calculate_specific_iv_sentiment(maturity, 90, 110, '90-110')
            
            # Calculate IV-sentiment 80-120
            self._calculate_specific_iv_sentiment(maturity, 80, 120, '80-120')
    
    def _calculate_specific_iv_sentiment(self, maturity, put_moneyness, call_moneyness, label):
        """
        Calculate a specific IV sentiment measure
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        put_moneyness : int
            Moneyness level for index puts (80 or 90)
        call_moneyness : int
            Moneyness level for single stock calls (110 or 120)
        label : str
            Label for the sentiment measure (e.g., '90-110')
        """
        try:
            # Get index put IV data
            index_put_field = f"IVOL_PUT_{maturity}M_{put_moneyness}"
            index_put_iv = self.index_iv_data[maturity][put_moneyness].copy()
            
            # Get single stock call IV data
            single_stock_call_iv = self.single_stock_iv_data[maturity][call_moneyness].copy()
            
            # Calculate weighted average single stock call IV
            if not single_stock_call_iv.empty:
                # In real implementation, we would weight by market cap
                # For simplicity, use equal weighting
                avg_single_stock_call_iv = single_stock_call_iv.mean(axis=1)
                
                # Calculate IV sentiment
                iv_sentiment = index_put_iv.iloc[:, 0] - avg_single_stock_call_iv
                
                self.iv_sentiment[maturity][label] = pd.DataFrame({
                    'IV_Sentiment': iv_sentiment
                }, index=iv_sentiment.index)
                
                # Calculate Z-score for trading signals
                iv_sentiment_zscore = pd.DataFrame({
                    'Z_Score': self._calculate_rolling_zscore(iv_sentiment)
                }, index=iv_sentiment.index)
                
                self.iv_sentiment[maturity][f"{label}_zscore"] = iv_sentiment_zscore
                
        except Exception as e:
            print(f"Error calculating IV sentiment {label} for {maturity}M: {e}")
    
    def _calculate_rolling_zscore(self, series, window=252):
        """
        Calculate rolling Z-score for a time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to calculate Z-score for
        window : int
            Rolling window length in days (default: 252, approximately 1 year)
            
        Returns:
        --------
        pd.Series
            Rolling Z-score
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Create Z-score (handle the initial window)
        zscore = pd.Series(index=series.index)
        zscore.iloc[window:] = (series.iloc[window:] - rolling_mean.iloc[window:]) / rolling_std.iloc[window:]
        
        return zscore
    
    def generate_trading_signals(self, maturity=3, iv_type='90-110', threshold=2.0):
        """
        Generate trading signals based on IV sentiment Z-scores
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        iv_type : str
            Type of IV sentiment measure ('90-110' or '80-120')
        threshold : float
            Z-score threshold for trading signals (default: 2.0)
            
        Returns:
        --------
        pd.DataFrame
            Trading signals (1 for long, -1 for short, 0 for cash)
        """
        print(f"Generating trading signals for {maturity}M {iv_type} IV sentiment...")
        
        try:
            # Get Z-scores
            zscore_key = f"{iv_type}_zscore"
            if zscore_key not in self.iv_sentiment[maturity]:
                print(f"Z-score data not found for {maturity}M {iv_type}")
                return pd.DataFrame()
            
            zscores = self.iv_sentiment[maturity][zscore_key]['Z_Score']
            
            # Initialize signal series
            signals = pd.Series(0, index=zscores.index)
            
            # Previous signal for state tracking
            prev_signal = 0
            
            # Iterate through Z-scores
            for i, (date, zscore) in enumerate(zscores.dropna().items()):
                if prev_signal == 0:  # Currently in cash
                    if zscore > threshold:  # Extreme bearishness (high IV-sentiment)
                        signals.loc[date] = 1  # Go long (contrarian)
                        prev_signal = 1
                    elif zscore < -threshold:  # Extreme bullishness (low IV-sentiment)
                        signals.loc[date] = -1  # Go short (contrarian)
                        prev_signal = -1
                elif prev_signal == 1:  # Currently long
                    if zscore <= 0:  # IV-sentiment reverts to mean
                        signals.loc[date] = 0  # Go to cash
                        prev_signal = 0
                elif prev_signal == -1:  # Currently short
                    if zscore >= 0:  # IV-sentiment reverts to mean
                        signals.loc[date] = 0  # Go to cash
                        prev_signal = 0
            
            # Create position dataframe
            positions = pd.DataFrame({
                'Signal': signals,
                'Position': signals * 0.05  # 5% allocation when signal is active
            }, index=signals.index)
            
            return positions
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, maturity=3, iv_type='90-110', threshold=2.0, transaction_cost=0.0005):
        """
        Backtest the IV sentiment strategy
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        iv_type : str
            Type of IV sentiment measure ('90-110' or '80-120')
        threshold : float
            Z-score threshold for trading signals (default: 2.0)
        transaction_cost : float
            Transaction cost per trade as a percentage (default: 0.05%)
            
        Returns:
        --------
        dict
            Strategy performance metrics
        """
        print(f"Backtesting {maturity}M {iv_type} IV sentiment strategy...")
        
        # Generate trading signals
        positions = self.generate_trading_signals(maturity, iv_type, threshold)
        if positions.empty:
            print("No trading signals generated. Aborting backtest.")
            return {}
        
        # Get market data
        market_returns = self.market_data['PX_LAST'].pct_change().dropna()
        
        # Align data
        common_dates = positions.index.intersection(market_returns.index)
        positions = positions.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        # Calculate strategy returns
        strategy_returns = positions['Position'].shift(1) * market_returns
        
        # Add transaction costs
        position_changes = positions['Position'].diff().abs() * transaction_cost
        strategy_returns = strategy_returns - position_changes
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1]
        annualized_return = ((1 + total_return) ** (252 / len(strategy_returns))) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Calculate win rate
        winning_days = strategy_returns[strategy_returns > 0].count()
        total_active_days = strategy_returns[strategy_returns != 0].count()
        win_rate = winning_days / total_active_days if total_active_days > 0 else 0
        
        # Calculate skewness and kurtosis
        skewness = strategy_returns.skew()
        kurtosis = strategy_returns.kurtosis()
        
        # Store results
        results = {
            'positions': positions,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # Store in class variable
        self.performance[f"{maturity}M_{iv_type}"] = results
        
        return results
    
    def _calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown from a series of returns
        
        Parameters:
        -----------
        returns : pd.Series
            Cumulative returns series
            
        Returns:
        --------
        float
            Maximum drawdown as a positive percentage
        """
        # Calculate running maximum
        running_max = returns.cummax()
        
        # Calculate drawdown
        drawdown = (running_max - returns) / (1 + running_max)
        
        # Get maximum drawdown
        max_drawdown = drawdown.max()
        
        return max_drawdown
    
    def run_multiple_backtests(self, thresholds=[1.5, 2.0, 2.5], all_combinations=True):
        """
        Run multiple backtests with different parameters
        
        Parameters:
        -----------
        thresholds : list
            List of Z-score thresholds to test
        all_combinations : bool
            Whether to test all combinations of maturities and IV types
            
        Returns:
        --------
        pd.DataFrame
            Summary of backtest results
        """
        print("Running multiple backtests...")
        
        results = []
        
        # Define test configurations
        maturities = self.option_maturities if all_combinations else [3]
        iv_types = ['90-110', '80-120'] if all_combinations else ['90-110']
        
        # Run backtests for all configurations
        for maturity in maturities:
            for iv_type in iv_types:
                for threshold in thresholds:
                    # Skip if data not available
                    if maturity not in self.iv_sentiment or iv_type not in self.iv_sentiment[maturity]:
                        continue
                    
                    # Run backtest
                    backtest_result = self.backtest_strategy(maturity, iv_type, threshold)
                    
                    if backtest_result:
                        results.append({
                            'Maturity': f"{maturity}M",
                            'IV Type': iv_type,
                            'Threshold': threshold,
                            'Ann. Return': backtest_result['annualized_return'],
                            'Volatility': backtest_result['volatility'],
                            'Sharpe Ratio': backtest_result['sharpe_ratio'],
                            'Max Drawdown': backtest_result['max_drawdown'],
                            'Win Rate': backtest_result['win_rate'],
                            'Skewness': backtest_result['skewness'],
                            'Kurtosis': backtest_result['kurtosis']
                        })
        
        # Create summary dataframe
        if results:
            summary = pd.DataFrame(results)
            return summary.sort_values('Sharpe Ratio', ascending=False)
        else:
            return pd.DataFrame()
    
    def plot_strategy_performance(self, maturity=3, iv_type='90-110', threshold=2.0):
        """
        Plot strategy performance charts
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        iv_type : str
            Type of IV sentiment measure ('90-110' or '80-120')
        threshold : float
            Z-score threshold for trading signals
            
        Returns:
        --------
        None
        """
        key = f"{maturity}M_{iv_type}"
        if key not in self.performance:
            print(f"Performance data not found for {key}. Run backtest first.")
            return
        
        perf = self.performance[key]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: IV Sentiment with Z-scores
        zscore_data = self.iv_sentiment[maturity][f"{iv_type}_zscore"]['Z_Score']
        sentiment_data = self.iv_sentiment[maturity][iv_type]['IV_Sentiment']
        
        ax1.plot(sentiment_data.index, sentiment_data, 'b-', label='IV Sentiment')
        ax1.set_ylabel('IV Sentiment', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(zscore_data.index, zscore_data, 'r-', label='Z-Score')
        ax1_twin.axhline(y=threshold, color='g', linestyle='--', alpha=0.7, label=f'Threshold (+{threshold})')
        ax1_twin.axhline(y=-threshold, color='g', linestyle='--', alpha=0.7, label=f'Threshold (-{threshold})')
        ax1_twin.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1_twin.set_ylabel('Z-Score', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title(f'IV Sentiment ({maturity}M {iv_type}) and Z-Score')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Strategy positions
        positions = perf['positions']['Position']
        ax2.fill_between(positions.index, positions, 0, where=positions > 0, facecolor='green', alpha=0.3, label='Long')
        ax2.fill_between(positions.index, positions, 0, where=positions < 0, facecolor='red', alpha=0.3, label='Short')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Position Size')
        ax2.set_title('Strategy Positions')
        ax2.legend(loc='upper left')
        
        # Plot 3: Cumulative returns comparison
        market_returns = self.market_data['PX_LAST'].pct_change().dropna()
        market_cumulative = (1 + market_returns.loc[perf['cumulative_returns'].index]).cumprod() - 1
        
        ax3.plot(perf['cumulative_returns'].index, perf['cumulative_returns'], 'g-', label='IV Sentiment Strategy')
        ax3.plot(market_cumulative.index, market_cumulative, 'b-', label='S&P 500')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Cumulative Return')
        ax3.set_title('Cumulative Returns Comparison')
        ax3.legend(loc='upper left')
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Add performance summary text
        strategy_text = (
            f"Strategy Performance:\n"
            f"Annual Return: {perf['annualized_return']:.2%}\n"
            f"Volatility: {perf['volatility']:.2%}\n"
            f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {perf['max_drawdown']:.2%}\n"
            f"Skewness: {perf['skewness']:.2f}"
        )
        ax3.annotate(strategy_text, xy=(0.02, 0.05), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def combine_with_other_strategies(self, maturity=3, iv_type='90-110', threshold=2.0, combine_with='momentum'):
        """
        Combine IV sentiment strategy with other strategies (buyhold, momentum, etc.)
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        iv_type : str
            Type of IV sentiment measure ('90-110' or '80-120')
        threshold : float
            Z-score threshold for trading signals
        combine_with : str
            Strategy to combine with ('buyhold', 'momentum', 'time_series_momentum')
            
        Returns:
        --------
        dict
            Combined strategy performance metrics
        """
        key = f"{maturity}M_{iv_type}"
        if key not in self.performance:
            print(f"Performance data not found for {key}. Run backtest first.")
            return {}
        
        iv_strategy_returns = self.performance[key]['returns']
        
        # Get market data
        market_returns = self.market_data['PX_LAST'].pct_change().dropna()
        
        # Align data
        common_dates = iv_strategy_returns.index.intersection(market_returns.index)
        iv_strategy_returns = iv_strategy_returns.loc[common_dates]
        market_returns = market_returns.loc[common_dates]
        
        # Generate other strategy returns
        other_strategy_returns = None
        
        if combine_with == 'buyhold':
            # Simple buy and hold
            other_strategy_returns = market_returns
        elif combine_with == 'momentum':
            # Cross-sectional momentum (simulated)
            # In real implementation, use actual momentum factor returns
            momentum_factor = market_returns.rolling(window=252).mean() * 12  # Momentum strength proxy
            momentum_signal = np.sign(momentum_factor)
            other_strategy_returns = momentum_signal.shift(1) * market_returns
        elif combine_with == 'time_series_momentum':
            # Time series momentum (trend following)
            # Use 12-month return sign as signal
            signal = np.sign(market_returns.rolling(window=252).sum())
            other_strategy_returns = signal.shift(1) * market_returns
        else:
            print(f"Unknown strategy: {combine_with}")
            return {}
        
        # Equal weight combination (50/50)
        combined_returns = 0.5 * iv_strategy_returns + 0.5 * other_strategy_returns
        
        # Calculate performance metrics
        cumulative_returns = (1 + combined_returns).cumprod() - 1
        total_return = cumulative_returns.iloc[-1]
        annualized_return = ((1 + total_return) ** (252 / len(combined_returns))) - 1
        volatility = combined_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Calculate correlation
        correlation = iv_strategy_returns.corr(other_strategy_returns)
        
        # Calculate conditional co-crash probability
        combined_tail_risk = self._calculate_tail_dependency(iv_strategy_returns, other_strategy_returns)
        
        # Calculate combined strategy stats vs individual strategies
        iv_only_sharpe = self.performance[key]['sharpe_ratio']
        other_only_sharpe = self._calculate_sharpe_ratio(other_strategy_returns)
        diversification_benefit = sharpe_ratio - max(iv_only_sharpe, other_only_sharpe)
        
        # Store results
        results = {
            'returns': combined_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'correlation': correlation,
            'tail_dependency': combined_tail_risk,
            'diversification_benefit': diversification_benefit
        }
        
        return results
    
    def _calculate_sharpe_ratio(self, returns, risk_free=0):
        """
        Calculate the Sharpe ratio for a returns series
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
        risk_free : float
            Risk-free rate (annual)
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        daily_rf = risk_free / 252
        excess_returns = returns - daily_rf
        annualized_return = ((1 + returns.mean()) ** 252) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol == 0:
            return 0
            
        sharpe = (annualized_return - risk_free) / annualized_vol
        return sharpe
    
    def _calculate_tail_dependency(self, returns1, returns2, percentile=0.05):
        """
        Calculate the probability of one strategy having a tail event conditional on another
        
        Parameters:
        -----------
        returns1 : pd.Series
            Returns of first strategy
        returns2 : pd.Series
            Returns of second strategy
        percentile : float
            Threshold for tail events (default: 0.05, i.e., 5th percentile)
            
        Returns:
        --------
        float
            Conditional co-crash probability
        """
        # Calculate return thresholds for tail events
        threshold1 = returns1.quantile(percentile)
        threshold2 = returns2.quantile(percentile)
        
        # Count joint and individual tail events
        tail_events1 = returns1 <= threshold1
        tail_events2 = returns2 <= threshold2
        joint_tail_events = (tail_events1 & tail_events2).sum()
        
        # Calculate conditional probabilities
        conditional_prob1 = joint_tail_events / tail_events1.sum() if tail_events1.sum() > 0 else 0
        conditional_prob2 = joint_tail_events / tail_events2.sum() if tail_events2.sum() > 0 else 0
        
        # Average the conditional probabilities
        conditional_coprob = (conditional_prob1 + conditional_prob2) / 2
        
        return conditional_coprob
    
    def plot_combined_strategy(self, maturity=3, iv_type='90-110', threshold=2.0):
        """
        Plot combined strategy performance
        
        Parameters:
        -----------
        maturity : int
            Option maturity in months (3, 6, or 12)
        iv_type : str
            Type of IV sentiment measure ('90-110' or '80-120')
        threshold : float
            Z-score threshold for trading signals
            
        Returns:
        --------
        None
        """
        key = f"{maturity}M_{iv_type}"
        if key not in self.performance:
            print(f"Performance data not found for {key}. Run backtest first.")
            return
        
        # Get combined strategy results
        buyhold_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'buyhold')
        momentum_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'momentum')
        ts_momentum_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'time_series_momentum')
        
        if not buyhold_combo or not momentum_combo or not ts_momentum_combo:
            print("Could not calculate all combination strategies")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Get data
        iv_strategy_returns = self.performance[key]['returns']
        market_returns = self.market_data['PX_LAST'].pct_change().dropna()
        iv_cum_returns = (1 + iv_strategy_returns).cumprod() - 1
        market_cum_returns = (1 + market_returns.loc[iv_cum_returns.index]).cumprod() - 1
        
        # Plot cumulative returns
        ax1.plot(iv_cum_returns.index, iv_cum_returns, 'g-', label='IV Sentiment Only')
        ax1.plot(market_cum_returns.index, market_cum_returns, 'b-', label='Buy & Hold')
        ax1.plot(buyhold_combo['cumulative_returns'].index, buyhold_combo['cumulative_returns'], 'm-', 
                label='IV + Buy & Hold')
        ax1.plot(momentum_combo['cumulative_returns'].index, momentum_combo['cumulative_returns'], 'r-', 
                label='IV + Momentum')
        ax1.plot(ts_momentum_combo['cumulative_returns'].index, ts_momentum_combo['cumulative_returns'], 'c-', 
                label='IV + TS Momentum')
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Cumulative Returns of Combined Strategies')
        ax1.legend(loc='upper left')
        
        # Calculate drawdowns
        iv_drawdowns = self._calculate_drawdown_series(iv_strategy_returns)
        market_drawdowns = self._calculate_drawdown_series(market_returns.loc[iv_drawdowns.index])
        buyhold_drawdowns = self._calculate_drawdown_series(buyhold_combo['returns'])
        momentum_drawdowns = self._calculate_drawdown_series(momentum_combo['returns'])
        ts_momentum_drawdowns = self._calculate_drawdown_series(ts_momentum_combo['returns'])
        
        # Plot drawdowns
        ax2.fill_between(iv_drawdowns.index, iv_drawdowns, 0, alpha=0.3, color='g', label='IV Sentiment Only')
        ax2.fill_between(market_drawdowns.index, market_drawdowns, 0, alpha=0.3, color='b', label='Buy & Hold')
        ax2.fill_between(buyhold_drawdowns.index, buyhold_drawdowns, 0, alpha=0.3, color='m', label='IV + Buy & Hold')
        ax2.fill_between(momentum_drawdowns.index, momentum_drawdowns, 0, alpha=0.3, color='r', label='IV + Momentum')
        ax2.fill_between(ts_momentum_drawdowns.index, ts_momentum_drawdowns, 0, alpha=0.3, color='c', label='IV + TS Momentum')
        
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Strategy Drawdowns')
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Create performance summary table
        performance_data = [
            ['Strategy', 'Ann. Return', 'Volatility', 'Sharpe', 'Max DD', 'Correlation with IV'],
            ['IV Sentiment', f"{self.performance[key]['annualized_return']:.2%}", 
             f"{self.performance[key]['volatility']:.2%}", f"{self.performance[key]['sharpe_ratio']:.2f}", 
             f"{self.performance[key]['max_drawdown']:.2%}", "1.00"],
            ['Buy & Hold', f"{self._calculate_annualized_return(market_returns.loc[iv_strategy_returns.index]):.2%}", 
             f"{market_returns.loc[iv_strategy_returns.index].std() * np.sqrt(252):.2%}", 
             f"{self._calculate_sharpe_ratio(market_returns.loc[iv_strategy_returns.index]):.2f}", 
             f"{self._calculate_max_drawdown((1 + market_returns.loc[iv_strategy_returns.index]).cumprod() - 1):.2%}",
             f"{iv_strategy_returns.corr(market_returns.loc[iv_strategy_returns.index]):.2f}"],
            ['IV + Buy & Hold', f"{buyhold_combo['annualized_return']:.2%}", 
             f"{buyhold_combo['volatility']:.2%}", f"{buyhold_combo['sharpe_ratio']:.2f}", 
             f"{buyhold_combo['max_drawdown']:.2%}", f"{buyhold_combo['correlation']:.2f}"],
            ['IV + Momentum', f"{momentum_combo['annualized_return']:.2%}", 
             f"{momentum_combo['volatility']:.2%}", f"{momentum_combo['sharpe_ratio']:.2f}", 
             f"{momentum_combo['max_drawdown']:.2%}", f"{momentum_combo['correlation']:.2f}"],
            ['IV + TS Momentum', f"{ts_momentum_combo['annualized_return']:.2%}", 
             f"{ts_momentum_combo['volatility']:.2%}", f"{ts_momentum_combo['sharpe_ratio']:.2f}", 
             f"{ts_momentum_combo['max_drawdown']:.2%}", f"{ts_momentum_combo['correlation']:.2f}"]
        ]
        
        # Add table to plot
        table = ax1.table(cellText=performance_data, loc='upper right', cellLoc='center', bbox=[0.5, 0.05, 0.45, 0.42])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_drawdown_series(self, returns):
        """
        Calculate drawdown series from returns
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        Returns:
        --------
        pd.Series
            Drawdown series
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        return drawdown
    
    def _calculate_annualized_return(self, returns):
        """
        Calculate annualized return from a returns series
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        Returns:
        --------
        float
            Annualized return
        """
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        annualized = (1 + total_return) ** (1 / years) - 1
        return annualized
    
    def run(self):
        """
        Run the full pipeline: fetch data, calculate IV sentiment, backtest strategy, plot results
        """
        # Fetch data
        self.fetch_data()
        
        # Calculate IV sentiment measures
        self.calculate_iv_sentiment()
        
        # Run backtests for different configurations
        summary = self.run_multiple_backtests()
        if not summary.empty:
            print("\nBacktest results summary:")
            print(summary.to_string())
        
        # Plot performance of best strategy
        if not summary.empty:
            best_config = summary.iloc[0]
            maturity = int(best_config['Maturity'].replace('M', ''))
            iv_type = best_config['IV Type']
            threshold = best_config['Threshold']
            
            print(f"\nPlotting performance of best strategy: {maturity}M {iv_type} with threshold {threshold}")
            self.plot_strategy_performance(maturity, iv_type, threshold)
            
            # Plot combined strategies
            print("\nPlotting combined strategies performance...")
            self.plot_combined_strategy(maturity, iv_type, threshold)
            
            # Detailed analysis of strategy combinations
            buyhold_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'buyhold')
            momentum_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'momentum')
            ts_momentum_combo = self.combine_with_other_strategies(maturity, iv_type, threshold, 'time_series_momentum')
            
            if buyhold_combo and momentum_combo and ts_momentum_combo:
                combo_results = pd.DataFrame({
                    'IV Sentiment Only': [
                        self.performance[f"{maturity}M_{iv_type}"]['annualized_return'],
                        self.performance[f"{maturity}M_{iv_type}"]['volatility'],
                        self.performance[f"{maturity}M_{iv_type}"]['sharpe_ratio'],
                        self.performance[f"{maturity}M_{iv_type}"]['max_drawdown'],
                        self.performance[f"{maturity}M_{iv_type}"]['skewness'],
                        "N/A"
                    ],
                    'IV + Buy & Hold': [
                        buyhold_combo['annualized_return'],
                        buyhold_combo['volatility'],
                        buyhold_combo['sharpe_ratio'],
                        buyhold_combo['max_drawdown'],
                        'N/A',
                        buyhold_combo['correlation']
                    ],
                    'IV + Momentum': [
                        momentum_combo['annualized_return'],
                        momentum_combo['volatility'],
                        momentum_combo['sharpe_ratio'],
                        momentum_combo['max_drawdown'],
                        'N/A',
                        momentum_combo['correlation']
                    ],
                    'IV + TS Momentum': [
                        ts_momentum_combo['annualized_return'],
                        ts_momentum_combo['volatility'],
                        ts_momentum_combo['sharpe_ratio'],
                        ts_momentum_combo['max_drawdown'],
                        'N/A',
                        ts_momentum_combo['correlation']
                    ]
                }, index=['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Skewness', 'Correlation with IV'])
                
                print("\nCombined strategies performance:")
                print(combo_results.to_string())
                
                # Calculate and print tail dependency metrics
                tail_risk = pd.DataFrame({
                    'Buy & Hold': [buyhold_combo['tail_dependency']],
                    'Momentum': [momentum_combo['tail_dependency']],
                    'TS Momentum': [ts_momentum_combo['tail_dependency']]
                }, index=['Conditional Co-Crash Probability'])
                
                print("\nTail dependency with IV-sentiment strategy:")
                print(tail_risk.to_string())
                
                # Calculate and print diversification benefits
                div_benefits = pd.DataFrame({
                    'Buy & Hold': [buyhold_combo['diversification_benefit']],
                    'Momentum': [momentum_combo['diversification_benefit']],
                    'TS Momentum': [ts_momentum_combo['diversification_benefit']]
                }, index=['Sharpe Ratio Improvement'])
                
                print("\nDiversification benefits (Sharpe ratio improvement):")
                print(div_benefits.to_string())


def main():
    # Initialize and run the strategy
    strategy = IVSentimentStrategy(start_date='20170101', end_date='20221231')
    strategy.run()


if __name__ == "__main__":
    main()