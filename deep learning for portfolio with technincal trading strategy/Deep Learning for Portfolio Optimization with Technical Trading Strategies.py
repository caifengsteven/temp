"""
Deep Learning for Portfolio Optimization with Technical Trading Strategies
Based on 'Learning the dynamics of technical trading strategies' by Murphy & Gebbie (2021)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import time
from tqdm import tqdm
import logging
from scipy.optimize import minimize
from scipy import stats
import warnings
import traceback
import blpapi  # Bloomberg API

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("technical_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Suppress warnings
warnings.filterwarnings('ignore')

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg terminal."""
    
    def __init__(self):
        """Initialize Bloomberg connection."""
        self.session = None
        self.refdata_service = None
        
    def start_session(self):
        """Start Bloomberg session."""
        logger.info("Starting Bloomberg session...")
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)
        
        self.session = blpapi.Session(sessionOptions)
        if not self.session.start():
            raise RuntimeError("Failed to start session.")
        
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata")
        
        self.refdata_service = self.session.getService("//blp/refdata")
        logger.info("Bloomberg session started successfully.")
    
    def stop_session(self):
        """Stop Bloomberg session."""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg session stopped.")
    
    def fetch_historical_data(self, tickers, fields, start_date, end_date, period='DAILY'):
        """
        Fetch historical data from Bloomberg.
        
        Parameters:
        -----------
        tickers : list
            List of Bloomberg tickers
        fields : list
            List of Bloomberg fields (e.g., 'PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'VOLUME')
        start_date : datetime
            Start date
        end_date : datetime
            End date
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY', or intraday like '1', '5', '15', '30', '60')
            
        Returns:
        --------
        DataFrame
            DataFrame with historical data
        """
        # Convert dates to string format for Bloomberg
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        # Set request parameters
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        request.set("startDate", start_date_str)
        request.set("endDate", end_date_str)
        request.set("periodicitySelection", period)
        
        logger.info(f"Sending request for {len(tickers)} securities...")
        
        # Send request
        self.session.sendRequest(request)
        
        # Process response
        data = []
        
        while True:
            event = self.session.nextEvent(500)
            for msg in event:
                if msg.messageType() == "HistoricalDataResponse":
                    security_data = msg.getElement("securityData")
                    security_name = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")
                    
                    for i in range(field_data.numValues()):
                        field_value = field_data.getValue(i)
                        date = field_value.getElementAsDatetime("date").strftime('%Y-%m-%d')
                        
                        row_data = {'date': date, 'security': security_name}
                        
                        for field in fields:
                            if field_value.hasElement(field):
                                row_data[field] = field_value.getElementAsFloat(field)
                            else:
                                row_data[field] = np.nan
                        
                        data.append(row_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No data received from Bloomberg.")
            return None
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create wide format with securities as columns
        wide_data = {}
        for field in fields:
            field_data = df.pivot(index='date', columns='security', values=field)
            wide_data[field] = field_data
            
        # Create multi-index columns DataFrame
        result = pd.concat(wide_data, axis=1)
        
        # Sort by date
        result.sort_index(inplace=True)
        
        logger.info(f"Received data for {len(tickers)} securities from {result.index[0]} to {result.index[-1]}.")
        
        # Save the column structure for debugging
        with open('data/column_structure.txt', 'w') as f:
            f.write(str(result.columns))
        
        return result
    
    def fetch_risk_free_rate(self, start_date, end_date, region='US', period='DAILY'):
        """
        Fetch risk-free rate data.
        
        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
        region : str
            Region for risk-free rate ('US', 'EU', 'ASIA')
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
        --------
        DataFrame
            DataFrame with risk-free rate data
        """
        # Select appropriate risk-free rate based on region
        if region == 'EU':
            ticker = "EUR001M Index"  # EU 1-month T-bill
        elif region == 'US':
            ticker = "US0001M Index"  # US 1-month LIBOR
        elif region == 'ZA':
            ticker = "STEFI Index"    # South Africa STeFI 
        else:
            ticker = "US0001M Index"  # Default to US
            
        fields = ["PX_LAST"]
        
        rf_data = self.fetch_historical_data(
            [ticker], fields, start_date, end_date, period
        )
        
        if rf_data is None:
            logger.warning("No risk-free rate data received. Using zero as default.")
            # Create a DataFrame with zeros for the risk-free rate
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            rf_data = pd.DataFrame(0, index=dates, columns=pd.MultiIndex.from_product([['PX_LAST'], [ticker]]))
        
        # Convert annual rate to daily/weekly/monthly
        if period == 'DAILY':
            trading_days = 252
        elif period == 'WEEKLY':
            trading_days = 52
        else:  # MONTHLY
            trading_days = 12
            
        rf_data = rf_data / 100 / trading_days  # Convert percentage to decimal and annualized to period
        
        return rf_data

class TechnicalIndicators:
    """Class to calculate technical indicators"""
    
    @staticmethod
    def SMA(prices, n):
        """Simple Moving Average"""
        # Ensure n is an integer
        n = int(n)
        return prices.rolling(window=n).mean()
    
    @staticmethod
    def EMA(prices, n):
        """Exponential Moving Average"""
        # Ensure n is an integer
        n = int(n)
        return prices.ewm(span=n, adjust=False).mean()
    
    @staticmethod
    def MACD(prices, n_fast=12, n_slow=26, n_signal=9):
        """Moving Average Convergence Divergence"""
        # Ensure parameters are integers
        n_fast = int(n_fast)
        n_slow = int(n_slow)
        n_signal = int(n_signal)
        
        ema_fast = TechnicalIndicators.EMA(prices, n_fast)
        ema_slow = TechnicalIndicators.EMA(prices, n_slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.EMA(macd_line, n_signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def RSI(prices, n=14):
        """Relative Strength Index"""
        # Ensure n is an integer
        n = int(n)
        
        delta = prices.diff()
        # Handle NaN values
        delta = delta.fillna(0)
        
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Handle zero division
        down = down.replace(0, np.finfo(float).eps)  # Replace zeros with small value
        
        ema_up = up.ewm(com=n-1, adjust=False).mean()
        ema_down = down.ewm(com=n-1, adjust=False).mean()
        
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def Bollinger_Bands(prices, n=20, ndev=2):
        """Bollinger Bands"""
        # Ensure n is an integer
        n = int(n)
        
        mean = prices.rolling(window=n).mean()
        std = prices.rolling(window=n).std()
        upper_band = mean + (std * ndev)
        lower_band = mean - (std * ndev)
        return upper_band, mean, lower_band
    
    @staticmethod
    def Stochastic_Oscillator(close, high, low, n=14, d=3):
        """Stochastic Oscillator"""
        # Ensure parameters are integers
        n = int(n)
        d = int(d)
        
        # Ensure high and low have some values
        if high.equals(close):
            high = close * 1.001  # Add 0.1% to create some high values
        
        if low.equals(close):
            low = close * 0.999  # Subtract 0.1% to create some low values
        
        # Fast %K
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.fillna(np.finfo(float).eps)  # Replace NaNs with small value
        denominator = denominator.replace(0, np.finfo(float).eps)  # Replace zeros with small value
        
        fast_k = 100 * ((close - lowest_low) / denominator)
        
        # Fast %D
        fast_d = fast_k.rolling(window=d).mean()
        
        # Slow %K and %D
        slow_k = fast_d
        slow_d = slow_k.rolling(window=d).mean()
        
        return fast_k, fast_d, slow_k, slow_d
    
    @staticmethod
    def Williams_R(close, high, low, n=14):
        """Williams %R"""
        # Ensure n is an integer
        n = int(n)
        
        # Ensure high and low have some values
        if high.equals(close):
            high = close * 1.001  # Add 0.1% to create some high values
        
        if low.equals(close):
            low = close * 0.999  # Subtract 0.1% to create some low values
        
        highest_high = high.rolling(window=n).max()
        lowest_low = low.rolling(window=n).min()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.fillna(np.finfo(float).eps)  # Replace NaNs with small value
        denominator = denominator.replace(0, np.finfo(float).eps)  # Replace zeros with small value
        
        williams_r = -100 * ((highest_high - close) / denominator)
        return williams_r
    
    @staticmethod
    def momentum(prices, n=14):
        """Momentum"""
        # Ensure n is an integer
        n = int(n)
        return prices.diff(n)
    
    @staticmethod
    def acceleration(prices, n=14):
        """Acceleration"""
        # Ensure n is an integer
        n = int(n)
        mom = TechnicalIndicators.momentum(prices, n)
        return mom.diff()
    
    @staticmethod
    def ichimoku(high, low, close, n1=9, n2=26, n3=52):
        """Ichimoku Kinko Hyo"""
        # Ensure parameters are integers
        n1 = int(n1)
        n2 = int(n2)
        n3 = int(n3)
        
        # Ensure high and low have some values
        if high.equals(close):
            high = close * 1.001  # Add 0.1% to create some high values
        
        if low.equals(close):
            low = close * 0.999  # Subtract 0.1% to create some low values
        
        # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past n1 periods
        tenkan_sen = (high.rolling(window=n1).max() + low.rolling(window=n1).min()) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past n2 periods
        kijun_sen = (high.rolling(window=n2).max() + low.rolling(window=n2).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 plotted n2 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(n2)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past n3 periods plotted n2 periods ahead
        senkou_span_b = ((high.rolling(window=n3).max() + low.rolling(window=n3).min()) / 2).shift(n2)
        
        # Chikou Span (Lagging Span): Close price plotted n2 periods back
        chikou_span = close.shift(-n2)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    @staticmethod
    def PROC(prices, n=12):
        """Price Rate of Change"""
        # Ensure n is an integer
        n = int(n)
        return prices.pct_change(n) * 100

class TechnicalTradingExpert:
    """Class representing a technical trading expert"""
    
    def __init__(self, strategy_type, n1=None, n2=None, ticker_universe=None):
        """
        Initialize a technical trading expert.
        
        Parameters:
        -----------
        strategy_type : str
            Type of technical trading strategy
        n1 : int
            First lookback parameter
        n2 : int
            Second lookback parameter (optional, depends on strategy)
        ticker_universe : list
            List of tickers this expert trades
        """
        self.strategy_type = strategy_type
        self.n1 = n1
        self.n2 = n2
        self.ticker_universe = ticker_universe
        self.signals = None
        self.portfolio_weights = None
        self.wealth = 1.0
        
    def generate_signals(self, data):
        """
        Generate trading signals based on the expert's strategy.
        
        Parameters:
        -----------
        data : DataFrame
            OHLCV data for all assets
            
        Returns:
        --------
        DataFrame
            DataFrame with signals (-1, 0, 1) for each asset
        """
        # Check if strategy requires two parameters and if they are provided
        two_param_strategies = ['SMA_Crossover', 'EMA_Crossover', 'MACD', 'Ichimoku']
        if self.strategy_type in two_param_strategies and (self.n1 is None or self.n2 is None):
            return pd.DataFrame(0, index=data.index, columns=self.ticker_universe)  # Return empty signals
            
        signals = pd.DataFrame(0, index=data.index, columns=self.ticker_universe)
        
        # Identify field names and securities in the data
        if isinstance(data.columns, pd.MultiIndex):
            fields = data.columns.levels[0]
            securities = data.columns.levels[1]
        else:
            fields = ["PX_LAST"]
            securities = data.columns
        
        for ticker in self.ticker_universe:
            # Check if the ticker is in the data
            if ticker not in securities:
                continue
                
            # Get appropriate data for this ticker
            try:
                close = data['PX_LAST'][ticker] if 'PX_LAST' in fields else data['CLOSE'][ticker]
                high = data['PX_HIGH'][ticker] if 'PX_HIGH' in fields else close
                low = data['PX_LOW'][ticker] if 'PX_LOW' in fields else close
                volume = data['VOLUME'][ticker] if 'VOLUME' in fields else None
                
                # Ensure data is numeric
                close = pd.to_numeric(close, errors='coerce')
                high = pd.to_numeric(high, errors='coerce')
                low = pd.to_numeric(low, errors='coerce')
                
                # Fill NaN values with previous values
                close = close.fillna(method='ffill')
                high = high.fillna(method='ffill')
                low = low.fillna(method='ffill')
                
                # Ensure we have valid data
                if close.isnull().all() or high.isnull().all() or low.isnull().all():
                    continue
            except Exception as e:
                continue
            
            try:
                if self.strategy_type == 'SMA_Crossover':
                    # Simple Moving Average Crossover
                    short_ma = TechnicalIndicators.SMA(close, self.n1)
                    long_ma = TechnicalIndicators.SMA(close, self.n2)
                    
                    # Buy signal: Short MA crosses above Long MA
                    buy_signal = (short_ma.shift(1) < long_ma.shift(1)) & (short_ma >= long_ma)
                    
                    # Sell signal: Short MA crosses below Long MA
                    sell_signal = (short_ma.shift(1) > long_ma.shift(1)) & (short_ma <= long_ma)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'EMA_Crossover':
                    # Exponential Moving Average Crossover
                    short_ema = TechnicalIndicators.EMA(close, self.n1)
                    long_ema = TechnicalIndicators.EMA(close, self.n2)
                    
                    # Buy signal: Short EMA crosses above Long EMA
                    buy_signal = (short_ema.shift(1) < long_ema.shift(1)) & (short_ema >= long_ema)
                    
                    # Sell signal: Short EMA crosses below Long EMA
                    sell_signal = (short_ema.shift(1) > long_ema.shift(1)) & (short_ema <= long_ema)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'MACD':
                    # MACD Strategy
                    macd_line, signal_line, _ = TechnicalIndicators.MACD(close, self.n1, self.n2, 9)
                    
                    # Buy signal: MACD Line crosses above Signal Line
                    buy_signal = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
                    
                    # Sell signal: MACD Line crosses below Signal Line
                    sell_signal = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'RSI':
                    # RSI Strategy
                    rsi = TechnicalIndicators.RSI(close, self.n1)
                    
                    # Buy signal: RSI crosses above 30 (oversold)
                    buy_signal = (rsi.shift(1) <= 30) & (rsi > 30)
                    
                    # Sell signal: RSI crosses below 70 (overbought)
                    sell_signal = (rsi.shift(1) >= 70) & (rsi < 70)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'BB':
                    # Bollinger Bands Strategy
                    upper, middle, lower = TechnicalIndicators.Bollinger_Bands(close, self.n1, 2)
                    
                    # Buy signal: Price crosses below lower band
                    buy_signal = (close.shift(1) <= lower.shift(1)) & (close > lower)
                    
                    # Sell signal: Price crosses above upper band
                    sell_signal = (close.shift(1) >= upper.shift(1)) & (close < upper)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'Stochastic':
                    # Fast Stochastic Strategy
                    _, _, slow_k, slow_d = TechnicalIndicators.Stochastic_Oscillator(close, high, low, self.n1, 3)
                    
                    # Buy signal: %K crosses above %D in oversold territory
                    buy_signal = (slow_k.shift(1) <= slow_d.shift(1)) & (slow_k > slow_d) & (slow_k < 20)
                    
                    # Sell signal: %K crosses below %D in overbought territory
                    sell_signal = (slow_k.shift(1) >= slow_d.shift(1)) & (slow_k < slow_d) & (slow_k > 80)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'Momentum':
                    # Momentum Strategy
                    mom = TechnicalIndicators.momentum(close, self.n1)
                    
                    # Buy signal: Momentum crosses above zero
                    buy_signal = (mom.shift(1) <= 0) & (mom > 0)
                    
                    # Sell signal: Momentum crosses below zero
                    sell_signal = (mom.shift(1) >= 0) & (mom < 0)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'Ichimoku':
                    # Ichimoku Strategy
                    tenkan, kijun, senkou_a, senkou_b, chikou = TechnicalIndicators.ichimoku(
                        high, low, close, 9, self.n1, self.n2
                    )
                    
                    # Buy signal: Price crosses above Kijun-sen (Base Line)
                    buy_signal = (close.shift(1) < kijun.shift(1)) & (close >= kijun)
                    
                    # Sell signal: Price crosses below Kijun-sen (Base Line)
                    sell_signal = (close.shift(1) > kijun.shift(1)) & (close <= kijun)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                    
                elif self.strategy_type == 'Williams_R':
                    # Williams %R Strategy
                    williams = TechnicalIndicators.Williams_R(close, high, low, self.n1)
                    
                    # Buy signal: %R crosses above -80 from below
                    buy_signal = (williams.shift(1) <= -80) & (williams > -80)
                    
                    # Sell signal: %R crosses below -20 from above
                    sell_signal = (williams.shift(1) >= -20) & (williams < -20)
                    
                    signals.loc[buy_signal, ticker] = 1
                    signals.loc[sell_signal, ticker] = -1
                
            except Exception as e:
                logger.error(f"Error generating signals for expert {self.strategy_type} on {ticker}: {str(e)}")
                continue
        
        self.signals = signals
        return signals
    
    def signals_to_weights(self, signals, transaction_cost_func=None):
        """
        Transform signals into portfolio weights.
        Uses a simplified approach for weight allocation.
        
        Parameters:
        -----------
        signals : DataFrame
            DataFrame with signals (-1, 0, 1) for each asset
        transaction_cost_func : function
            Function to calculate transaction costs
            
        Returns:
        --------
        DataFrame
            DataFrame with portfolio weights for each asset
        """
        if signals is None:
            raise ValueError("Signals must be generated first")
        
        # Initialize weights with zeros
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        # For each time period
        for t in range(len(signals.index)):
            current_signals = signals.iloc[t]
            
            # Count positive and negative signals
            buy_count = (current_signals == 1).sum()
            sell_count = (current_signals == -1).sum()
            
            if buy_count == 0 and sell_count == 0:
                # No trades, weights remain 0
                continue
            
            # Allocate equal weights to buys and sells, ensuring they sum to 0 (self-financing)
            # and the absolute sum is 1 (unit leverage)
            for ticker in signals.columns:
                if current_signals[ticker] == 1:  # Buy signal
                    weights.iloc[t, weights.columns.get_loc(ticker)] = 0.5 / buy_count if buy_count > 0 else 0
                elif current_signals[ticker] == -1:  # Sell signal
                    weights.iloc[t, weights.columns.get_loc(ticker)] = -0.5 / sell_count if sell_count > 0 else 0
        
        self.portfolio_weights = weights
        return weights
    
    def update_wealth(self, weights, returns, transaction_costs=None):
        """
        Update the expert's wealth based on portfolio weights and asset returns.
        
        Parameters:
        -----------
        weights : DataFrame
            Portfolio weights for each asset
        returns : DataFrame
            Asset returns for each period
        transaction_costs : DataFrame
            Transaction costs for each period (if any)
            
        Returns:
        --------
        wealth : float
            Updated wealth of the expert
        """
        if weights is None or returns is None:
            raise ValueError("Weights and returns must be provided")
        
        # Make sure weights and returns have aligned indices
        common_index = weights.index.intersection(returns.index)
        if len(common_index) == 0:
            return self.wealth  # No common dates, return current wealth
        
        weights = weights.loc[common_index]
        returns = returns.loc[common_index]
        
        # Calculate portfolio returns for each period
        # Ensure common columns
        common_columns = weights.columns.intersection(returns.columns)
        if len(common_columns) == 0:
            return self.wealth  # No common columns, return current wealth
        
        portfolio_returns = (weights[common_columns] * returns[common_columns]).sum(axis=1)
        
        # Subtract transaction costs if provided
        if transaction_costs is not None:
            transaction_costs = transaction_costs.loc[common_index]
            portfolio_returns = portfolio_returns - transaction_costs
        
        # Compound returns to get wealth
        if len(portfolio_returns) > 0:
            wealth_series = (1 + portfolio_returns).cumprod()
            self.wealth = wealth_series.iloc[-1]
        
        return self.wealth

class OnlineLearningPortfolio:
    """Class for the online learning portfolio algorithm"""
    
    def __init__(self, config):
        """
        Initialize the online learning portfolio.
        
        Parameters:
        -----------
        config : dict
            Configuration parameters
        """
        self.config = config
        self.experts = []
        self.expert_weights = None
        self.portfolio_weights = None
        self.wealth = 1.0
        self.profits_and_losses = []
        
    def generate_experts(self):
        """Generate a population of experts based on configuration"""
        experts = []
        
        # Define strategies, lookback parameters, and object clusters
        strategies = self.config.get('strategies', [])
        short_lookbacks = self.config.get('short_lookbacks', [])
        long_lookbacks = self.config.get('long_lookbacks', [])
        object_clusters = self.config.get('object_clusters', {})
        
        logger.info(f"Generating experts with {len(strategies)} strategies, "
                  f"{len(short_lookbacks)} short lookbacks, "
                  f"{len(long_lookbacks)} long lookbacks, "
                  f"{len(object_clusters)} object clusters")
        
        # Strategies that require two parameters
        two_param_strategies = ['SMA_Crossover', 'EMA_Crossover', 'MACD', 'Ichimoku']
        # Strategies that require only one parameter
        one_param_strategies = [s for s in strategies if s not in two_param_strategies]
        
        # Generate experts for each strategy, lookback parameter, and object cluster
        for cluster_name, tickers in object_clusters.items():
            # Generate experts for one-parameter strategies
            for strategy in one_param_strategies:
                for n1 in short_lookbacks:
                    expert = TechnicalTradingExpert(strategy, n1=n1, ticker_universe=tickers)
                    experts.append(expert)
            
            # Generate experts for two-parameter strategies
            for strategy in [s for s in strategies if s in two_param_strategies]:
                for n1 in short_lookbacks:
                    for n2 in long_lookbacks:
                        if n1 < n2:  # Ensure short-term lookback is smaller than long-term
                            expert = TechnicalTradingExpert(strategy, n1=n1, n2=n2, ticker_universe=tickers)
                            experts.append(expert)
        
        self.experts = experts
        logger.info(f"Generated {len(experts)} experts")
        return experts
    
    def update_expert_weights(self):
        """Update the weights of experts based on their wealth performance"""
        if not self.experts:
            raise ValueError("Experts must be generated first")
        
        # Get wealth of each expert
        expert_wealth = np.array([expert.wealth for expert in self.experts])
        
        # Normalize expert weights to sum to 1
        expert_weights = expert_wealth / np.sum(expert_wealth)
        
        self.expert_weights = expert_weights
        return expert_weights
    
    def aggregate_portfolio_weights(self, expert_weights=None, expert_portfolios=None):
        """
        Aggregate portfolio weights from experts based on their weights.
        
        Parameters:
        -----------
        expert_weights : array-like
            Weights of each expert
        expert_portfolios : list of DataFrames
            Portfolio weights of each expert
            
        Returns:
        --------
        DataFrame
            Aggregated portfolio weights
        """
        if expert_weights is None:
            expert_weights = self.expert_weights
        
        if expert_portfolios is None:
            expert_portfolios = [expert.portfolio_weights for expert in self.experts]
        
        if expert_weights is None or expert_portfolios is None:
            raise ValueError("Expert weights and portfolios must be provided")
        
        # Find all unique tickers across expert portfolios
        all_tickers = set()
        for portfolio in expert_portfolios:
            if portfolio is not None:
                all_tickers.update(portfolio.columns)
        
        # Find common dates across all portfolios
        common_dates = None
        for portfolio in expert_portfolios:
            if portfolio is not None and len(portfolio) > 0:
                if common_dates is None:
                    common_dates = portfolio.index
                else:
                    common_dates = common_dates.intersection(portfolio.index)
        
        if not all_tickers or common_dates is None or len(common_dates) == 0:
            logger.warning("No common tickers or dates found across expert portfolios")
            return pd.DataFrame()
        
        # Initialize aggregated weights with zeros
        aggregated_weights = pd.DataFrame(0.0, index=common_dates, columns=list(all_tickers))
        
        # For each expert, add weighted contribution to aggregated weights
        for i, expert_portfolio in enumerate(expert_portfolios):
            if expert_portfolio is None or len(expert_portfolio) == 0:
                continue
                
            expert_contribution = expert_portfolio.reindex(index=common_dates, columns=list(all_tickers), fill_value=0.0)
            aggregated_weights = aggregated_weights + expert_contribution * expert_weights[i]
        
        # Normalize to ensure zero-cost and unit leverage
        for t in range(len(common_dates)):
            weights_t = aggregated_weights.iloc[t]
            long_sum = weights_t[weights_t > 0].sum()
            short_sum = abs(weights_t[weights_t < 0].sum())
            
            if long_sum > 0 and short_sum > 0:
                # Normalize to ensure long positions sum to 0.5 and short positions sum to -0.5
                scale_long = 0.5 / long_sum if long_sum > 0 else 0
                scale_short = 0.5 / short_sum if short_sum > 0 else 0
                
                for ticker in all_tickers:
                    if weights_t[ticker] > 0:
                        aggregated_weights.loc[common_dates[t], ticker] = weights_t[ticker] * scale_long
                    elif weights_t[ticker] < 0:
                        aggregated_weights.loc[common_dates[t], ticker] = weights_t[ticker] * scale_short
        
        self.portfolio_weights = aggregated_weights
        return aggregated_weights
    
    def update_wealth(self, weights, returns, transaction_costs=None):
        """
        Update portfolio wealth based on weights and returns.
        
        Parameters:
        -----------
        weights : DataFrame
            Portfolio weights for each asset
        returns : DataFrame
            Asset returns for each period
        transaction_costs : DataFrame
            Transaction costs for each period (if any)
            
        Returns:
        --------
        wealth : float
            Updated wealth
        """
        if weights is None or returns is None or len(weights) == 0:
            logger.warning("No weights or returns provided for wealth update")
            return self.wealth, []
        
        # Make sure weights and returns have aligned indices and columns
        common_index = weights.index.intersection(returns.index)
        common_columns = weights.columns.intersection(returns.columns)
        
        if len(common_index) == 0 or len(common_columns) == 0:
            logger.warning("No common dates or assets between weights and returns")
            return self.wealth, []
        
        weights = weights.loc[common_index, common_columns]
        returns = returns.loc[common_index, common_columns]
        
        # Calculate portfolio returns for each period
        portfolio_returns = (weights * returns).sum(axis=1)
        
        # Calculate profits and losses
        profits_and_losses = portfolio_returns.copy()
        
        # Subtract transaction costs if provided
        if transaction_costs is not None:
            transaction_costs = transaction_costs.loc[common_index] if hasattr(transaction_costs, 'loc') else transaction_costs
            portfolio_returns = portfolio_returns - transaction_costs
            
        # Compound returns to get wealth
        wealth_series = (1 + portfolio_returns).cumprod()
        
        # Update portfolio wealth
        if len(wealth_series) > 0:
            self.wealth = wealth_series.iloc[-1]
            
        # Store profits and losses
        self.profits_and_losses.extend(profits_and_losses.tolist())
        
        return self.wealth, profits_and_losses
    
    def calculate_transaction_costs(self, weights, spreads, volatilities, adv, shares_traded):
        """
        Calculate transaction costs using the square-root formula.
        
        Parameters:
        -----------
        weights : DataFrame
            Portfolio weights for each asset
        spreads : DataFrame
            Bid-ask spreads for each asset
        volatilities : DataFrame
            Volatility of each asset
        adv : DataFrame
            Average daily volume for each asset
        shares_traded : DataFrame
            Number of shares traded for each asset
            
        Returns:
        --------
        DataFrame
            Transaction costs for each period
        """
        if weights is None or len(weights) == 0:
            return pd.Series(0, index=weights.index if weights is not None else [])
            
        # Calculate weight changes
        weight_changes = weights.diff().abs()
        
        # Initialize transaction costs with zeros
        transaction_costs = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        
        # For each time period and asset, calculate transaction costs
        for t in range(1, len(weights.index)):
            tc_t = 0.0  # Total transaction cost for this period
            for ticker in weights.columns:
                # Only apply transaction costs if there's a change in weights
                if weight_changes.iloc[t, weight_changes.columns.get_loc(ticker)] > 0:
                    # Spread cost
                    spread_cost = spreads.iloc[t, spreads.columns.get_loc(ticker)] if spreads is not None else 0.0001
                    
                    # Price impact cost using square-root formula
                    impact_cost = 0.0
                    if volatilities is not None and adv is not None and shares_traded is not None:
                        vol = volatilities.iloc[t, volatilities.columns.get_loc(ticker)]
                        volume = adv.iloc[t, adv.columns.get_loc(ticker)]
                        shares = shares_traded.iloc[t, shares_traded.columns.get_loc(ticker)]
                        
                        if volume > 0:  # Avoid division by zero
                            impact_cost = vol * np.sqrt(shares / volume)
                    
                    # Total transaction cost for this ticker
                    tc_t += spread_cost + impact_cost
            
            # Set the total transaction cost for this period
            transaction_costs.iloc[t] = tc_t
        
        # Return the sum of transaction costs across all assets for each period
        return transaction_costs.sum(axis=1)
    
    def run_backtest(self, data, rf_data=None, transaction_costs=True):
        """
        Run a backtest of the online learning portfolio algorithm.
        
        Parameters:
        -----------
        data : DataFrame
            OHLCV data for all assets
        rf_data : DataFrame
            Risk-free rate data
        transaction_costs : bool
            Whether to include transaction costs
            
        Returns:
        --------
        dict
            Results of the backtest
        """
        if not self.experts:
            self.generate_experts()
        
        # Extract OHLCV data
        fields = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else ['PX_LAST']
        tickers = data.columns.levels[1] if isinstance(data.columns, pd.MultiIndex) else data.columns
        
        # Get closing prices and calculate returns
        if 'PX_LAST' in fields:
            close_data = data['PX_LAST']
        elif 'CLOSE' in fields:
            close_data = data['CLOSE']
        else:
            # If neither PX_LAST nor CLOSE is available, use the first field
            close_data = data[fields[0]]
        
        # Calculate returns
        returns = close_data.pct_change().fillna(0)
        
        # Save this for debugging
        returns.to_csv('data/returns.csv')
        
        # Initialize expert weights
        n_experts = len(self.experts)
        self.expert_weights = np.ones(n_experts) / n_experts
        
        # Initialize results
        expert_weights_history = []
        portfolio_weights_history = []
        wealth_history = [1.0]
        
        # For each time period
        for t in tqdm(range(1, len(data.index)), desc="Running backtest"):
            current_date = data.index[t]
            
            # Get data up to current date
            data_t = data.loc[:current_date]
            
            # Generate signals and weights for each expert
            valid_experts = []
            for i, expert in enumerate(self.experts):
                try:
                    expert.generate_signals(data_t)
                    expert.signals_to_weights(expert.signals)
                    valid_experts.append(i)
                except Exception as e:
                    logger.error(f"Error for expert {i} ({expert.strategy_type}): {e}")
                    continue
            
            if not valid_experts:
                logger.warning(f"No valid experts at {current_date}. Skipping update.")
                wealth_history.append(wealth_history[-1])  # Keep previous wealth
                continue
            
            # Update expert wealth based on returns
            current_returns = returns.loc[current_date:current_date]
            if len(current_returns) == 0:
                logger.warning(f"No returns data at {current_date}. Skipping update.")
                wealth_history.append(wealth_history[-1])  # Keep previous wealth
                continue
                
            for i in valid_experts:
                expert = self.experts[i]
                if expert.portfolio_weights is not None and len(expert.portfolio_weights) > 0:
                    expert.update_wealth(expert.portfolio_weights, current_returns)
            
            # Update expert weights based on performance
            self.update_expert_weights()
            
            # Aggregate portfolio weights
            try:
                self.aggregate_portfolio_weights()
            except Exception as e:
                logger.error(f"Error aggregating portfolio weights: {e}")
                self.portfolio_weights = pd.DataFrame(0, index=[current_date], columns=close_data.columns)
            
            # Store weights
            expert_weights_history.append(self.expert_weights.copy())
            if self.portfolio_weights is not None and len(self.portfolio_weights) > 0:
                portfolio_weights_history.append(self.portfolio_weights.loc[current_date].copy() if current_date in self.portfolio_weights.index else pd.Series(0, index=close_data.columns))
            else:
                portfolio_weights_history.append(pd.Series(0, index=close_data.columns))
            
            # Calculate transaction costs if needed
            tc = None
            if transaction_costs:
                try:
                    # Calculate spreads (simplified)
                    spreads = pd.DataFrame(0.0001, index=data.index, columns=close_data.columns)  # 1 bps spread
                    
                    # Calculate volatilities (22-day rolling window)
                    volatilities = returns.rolling(window=22).std() * np.sqrt(252)  # Annualized
                    
                    # Get volume data
                    if 'VOLUME' in fields:
                        volume_data = data['VOLUME']
                    else:
                        # If volume is not available, use a proxy
                        volume_data = pd.DataFrame(1000000, index=data.index, columns=close_data.columns)  # Default volume
                    
                    # Calculate ADV (22-day average volume)
                    adv = volume_data.rolling(window=22).mean()
                    
                    # Calculate shares traded (1 bps of ADV)
                    shares_traded = adv * 0.0001
                    
                    # Calculate transaction costs
                    if self.portfolio_weights is not None and len(self.portfolio_weights) > 0:
                        tc = 0.0001  # Default 1 bps if calculation fails
                    else:
                        tc = 0.0
                except Exception as e:
                    logger.error(f"Error calculating transaction costs: {e}")
                    tc = 0.0001  # Default 1 bps if calculation fails
            
            # Update portfolio wealth
            try:
                if self.portfolio_weights is not None and len(self.portfolio_weights) > 0:
                    new_wealth, new_pl = self.update_wealth(
                        self.portfolio_weights.loc[[current_date]] if current_date in self.portfolio_weights.index else pd.DataFrame(0, index=[current_date], columns=close_data.columns),
                        current_returns,
                        tc
                    )
                    wealth_history.append(new_wealth)
                else:
                    wealth_history.append(wealth_history[-1])  # Keep previous wealth
            except Exception as e:
                logger.error(f"Error updating wealth: {e}")
                wealth_history.append(wealth_history[-1])  # Keep previous wealth
        
        # Save the wealth history for debugging
        pd.Series(wealth_history, index=[data.index[0]] + list(data.index[1:])).to_csv('data/wealth_history.csv')
        
        # Compile results
        results = {
            'wealth': wealth_history,
            'expert_weights': expert_weights_history,
            'portfolio_weights': portfolio_weights_history,
            'profits_and_losses': self.profits_and_losses
        }
        
        return results
    
    def plot_results(self, results, benchmark=None):
        """
        Plot the results of the backtest.
        
        Parameters:
        -----------
        results : dict
            Results of the backtest
        benchmark : DataFrame
            Benchmark performance (if any)
        """
        # Plot wealth over time
        plt.figure(figsize=(12, 6))
        plt.plot(results['wealth'], label='Online Learning Portfolio')
        
        if benchmark is not None:
            plt.plot(benchmark, label='Benchmark')
        
        plt.title('Portfolio Wealth Over Time')
        plt.xlabel('Trading Periods')
        plt.ylabel('Wealth')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/portfolio_wealth.png')
        plt.close()
        
        # Plot profits and losses
        plt.figure(figsize=(12, 6))
        plt.plot(results['profits_and_losses'], label='Profits and Losses')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Profits and Losses')
        plt.xlabel('Trading Periods')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/profits_and_losses.png')
        plt.close()
        
        # Plot expert weights over time (first 10 experts)
        plt.figure(figsize=(12, 6))
        expert_weights = np.array(results['expert_weights'])
        for i in range(min(10, expert_weights.shape[1])):
            plt.plot(expert_weights[:, i], label=f'Expert {i+1}')
        plt.title('Expert Weights Over Time')
        plt.xlabel('Trading Periods')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/expert_weights.png')
        plt.close()
        
class StatisticalArbitrageTest:
    """
    Class to test for statistical arbitrage as described in the paper.
    Based on Jarrow et al. (2012).
    """
    
    def __init__(self, profits_and_losses):
        """
        Initialize the statistical arbitrage test.
        
        Parameters:
        -----------
        profits_and_losses : array-like
            The profits and losses of the trading strategy
        """
        self.profits_and_losses = np.array(profits_and_losses)
        self.incremental_profits = np.diff(np.append(0, self.profits_and_losses))
        
    def cm_model_mle(self):
        """
        Maximum Likelihood Estimation for the Constrained Mean (CM) model.
        
        Returns:
        --------
        tuple
            Estimated parameters (mu, lambda, sigma_squared)
        """
        # Initialize parameters
        params_init = np.array([0.001, 0.0, 0.001])  # [mu, lambda, sigma^2]
        
        # Define negative log-likelihood function for CM model
        def neg_loglikelihood(params):
            mu, lambda_, sigma_squared = params
            t = np.arange(1, len(self.incremental_profits) + 1)
            
            # Ensure sigma_squared is positive
            if sigma_squared <= 0:
                return 1e10
            
            # Calculate log-likelihood
            ll = -0.5 * np.sum(np.log(sigma_squared * t**(2*lambda_)))
            ll -= 0.5 * np.sum((self.incremental_profits - mu) ** 2 / (sigma_squared * t**(2*lambda_)))
            
            return -ll
        
        # Constraints: sigma_squared > 0
        constraints = ({'type': 'ineq', 'fun': lambda x: x[2]})
        
        # Optimize
        result = minimize(neg_loglikelihood, params_init, method='SLSQP', constraints=constraints)
        
        if not result.success:
            logger.warning("MLE optimization did not converge. Using initial estimates.")
            return params_init
        
        return result.x
    
    def calculate_standard_errors(self, params):
        """
        Calculate standard errors for the parameters.
        
        Parameters:
        -----------
        params : array-like
            Estimated parameters [mu, lambda, sigma_squared]
            
        Returns:
        --------
        array
            Standard errors for the parameters
        """
        mu, lambda_, sigma_squared = params
        t = np.arange(1, len(self.incremental_profits) + 1)
        
        # Calculate Fisher Information matrix numerically
        epsilon = 1e-6
        delta_mu = np.array([epsilon, 0, 0])
        delta_lambda = np.array([0, epsilon, 0])
        delta_sigma = np.array([0, 0, epsilon])
        
        # Define log-likelihood function
        def loglikelihood(params):
            mu, lambda_, sigma_squared = params
            ll = -0.5 * np.sum(np.log(sigma_squared * t**(2*lambda_)))
            ll -= 0.5 * np.sum((self.incremental_profits - mu) ** 2 / (sigma_squared * t**(2*lambda_)))
            return ll
        
        # Calculate derivatives
        d_mu = (loglikelihood(params + delta_mu) - loglikelihood(params - delta_mu)) / (2 * epsilon)
        d_lambda = (loglikelihood(params + delta_lambda) - loglikelihood(params - delta_lambda)) / (2 * epsilon)
        d_sigma = (loglikelihood(params + delta_sigma) - loglikelihood(params - delta_sigma)) / (2 * epsilon)
        
        # Calculate Fisher Information matrix
        fim = np.zeros((3, 3))
        fim[0, 0] = d_mu ** 2
        fim[1, 1] = d_lambda ** 2
        fim[2, 2] = d_sigma ** 2
        fim[0, 1] = fim[1, 0] = d_mu * d_lambda
        fim[0, 2] = fim[2, 0] = d_mu * d_sigma
        fim[1, 2] = fim[2, 1] = d_lambda * d_sigma
        
        # Calculate standard errors
        try:
            fim_inv = np.linalg.inv(fim)
            std_errors = np.sqrt(np.diag(fim_inv))
        except np.linalg.LinAlgError:
            logger.warning("Fisher Information Matrix is singular. Using approximate standard errors.")
            std_errors = np.array([0.01, 0.01, 0.01])
        
        return std_errors
    
    def min_t_statistic(self, params, std_errors):
        """
        Calculate the Min-t statistic for the CM model.
        
        Parameters:
        -----------
        params : array-like
            Estimated parameters [mu, lambda, sigma_squared]
        std_errors : array-like
            Standard errors for the parameters
            
        Returns:
        --------
        float
            Min-t statistic
        """
        mu, lambda_, sigma_squared = params
        se_mu, se_lambda, se_sigma = std_errors
        
        # Calculate t-statistics for each sub-hypothesis
        t_mu = mu / se_mu
        t_neg_lambda = -lambda_ / se_lambda
        
        # Min-t statistic (only considering mu and lambda for CM model)
        min_t = min(t_mu, t_neg_lambda)
        
        return min_t
    
    def simulate_critical_value(self, params, n_simulations=5000, alpha=0.05):
        """
        Simulate the critical value for the Min-t statistic.
        
        Parameters:
        -----------
        params : array-like
            Estimated parameters [mu, lambda, sigma_squared]
        n_simulations : int
            Number of simulations
        alpha : float
            Significance level
            
        Returns:
        --------
        float
            Critical value
        """
        _, _, sigma_squared = params
        n = len(self.incremental_profits)
        t = np.arange(1, n + 1)
        
        min_t_values = []
        
        for _ in range(n_simulations):
            # Simulate incremental profits under null hypothesis (mu=0, lambda=0)
            sim_increments = np.random.normal(0, np.sqrt(sigma_squared), n)
            
            # Create a temporary test object
            temp_test = StatisticalArbitrageTest(np.cumsum(sim_increments))
            
            # Estimate parameters
            sim_params = temp_test.cm_model_mle()
            
            # Calculate standard errors
            sim_std_errors = temp_test.calculate_standard_errors(sim_params)
            
            # Calculate Min-t statistic
            sim_min_t = temp_test.min_t_statistic(sim_params, sim_std_errors)
            
            min_t_values.append(sim_min_t)
        
        # Critical value is the (1-alpha) quantile of the simulated Min-t statistics
        critical_value = np.percentile(min_t_values, (1 - alpha) * 100)
        
        return critical_value, np.array(min_t_values)
    
    def probability_of_loss(self, params, num_periods=None):
        """
        Calculate the probability of the trading strategy generating a loss.
        
        Parameters:
        -----------
        params : array-like
            Estimated parameters [mu, lambda, sigma_squared]
        num_periods : int
            Number of periods to consider (None for all periods)
            
        Returns:
        --------
        array
            Probabilities of loss for each period
        """
        mu, lambda_, sigma_squared = params
        
        if num_periods is None:
            num_periods = len(self.incremental_profits)
        
        t = np.arange(1, num_periods + 1)
        
        # Calculate cumulative mean and standard deviation
        cum_mean = mu * np.sum(t ** 0)  # For CM model, theta=0
        cum_std = np.sqrt(sigma_squared * np.sum(t ** (2 * lambda_)))
        
        # Calculate probability of loss
        p_loss = stats.norm.cdf(-cum_mean / cum_std)
        
        return p_loss
    
    def test_statistical_arbitrage(self, alpha=0.05):
        """
        Test for statistical arbitrage.
        
        Parameters:
        -----------
        alpha : float
            Significance level
            
        Returns:
        --------
        dict
            Test results
        """
        # Check if profits_and_losses is sufficient for testing
        if len(self.profits_and_losses) < 10:
            logger.warning("Insufficient data for statistical arbitrage test")
            return {
                'parameters': [0, 0, 0.001],
                'standard_errors': [0.01, 0.01, 0.01],
                'min_t': 0,
                'critical_value': 0,
                'p_value': 1.0,
                'probability_of_loss': 0.5,
                'reject_null': False,
                'sim_min_t': np.zeros(100)
            }
        
        # Estimate parameters
        params = self.cm_model_mle()
        
        # Calculate standard errors
        std_errors = self.calculate_standard_errors(params)
        
        # Calculate Min-t statistic
        min_t = self.min_t_statistic(params, std_errors)
        
        # Simulate critical value
        critical_value, sim_min_t = self.simulate_critical_value(params, alpha=alpha)
        
        # Calculate p-value
        p_value = np.mean(sim_min_t <= min_t)
        
        # Calculate probability of loss
        p_loss = self.probability_of_loss(params)
        
        # Test result
        reject_null = min_t > critical_value
        
        results = {
            'parameters': params,
            'standard_errors': std_errors,
            'min_t': min_t,
            'critical_value': critical_value,
            'p_value': p_value,
            'probability_of_loss': p_loss,
            'reject_null': reject_null,
            'sim_min_t': sim_min_t
        }
        
        return results
    
    def plot_test_results(self, results):
        """
        Plot the results of the statistical arbitrage test.
        
        Parameters:
        -----------
        results : dict
            Test results
        """
        # Plot histogram of simulated Min-t statistics
        plt.figure(figsize=(10, 6))
        plt.hist(results['sim_min_t'], bins=50, alpha=0.7, color='gray', density=True)
        plt.axvline(x=results['min_t'], color='green', linestyle='--', label=f"Min-t = {results['min_t']:.4f}")
        plt.axvline(x=results['critical_value'], color='red', linestyle='--', label=f"Critical Value = {results['critical_value']:.4f}")
        plt.title('Distribution of Simulated Min-t Statistics')
        plt.xlabel('Min-t Statistic')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/min_t_distribution.png')
        plt.close()
        
        # Plot probability of loss over time
        n_periods = min(100, len(results['probability_of_loss']) if hasattr(results['probability_of_loss'], '__len__') else 1)
        if n_periods > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, n_periods+1), results['probability_of_loss'][:n_periods], color='blue')
            plt.axhline(y=0.05, color='red', linestyle='--', label='5% Threshold')
            plt.title('Probability of Loss Over Time')
            plt.xlabel('Trading Periods')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)
            plt.savefig('results/probability_of_loss.png')
            plt.close()

def main():
    """Main function to run the technical trading strategies"""
    
    # Create directories for results
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Configuration
    config = {
        'strategies': [
            'SMA_Crossover', 'EMA_Crossover', 'MACD', 'RSI', 
            'BB', 'Stochastic', 'Momentum', 'Ichimoku', 'Williams_R'
        ],
        'short_lookbacks': [5, 10, 15, 20],
        'long_lookbacks': [20, 40, 60, 100],
        'object_clusters': {
            'All': ['VTI US Equity', 'AGG US Equity', 'DBC US Equity', 'VIX INDEX'],
            'Equities': ['VTI US Equity'],
            'Bonds': ['AGG US Equity'],
            'Commodities': ['DBC US Equity'],
            'Volatility': ['VIX INDEX']
        },
        'transaction_cost': 0.0001,  # 1 bps
        'target_volatility': 0.10,  # 10% annual volatility
        'lookback_window': 50,
        'rebalance_frequency': 1  # Daily
    }
    
    try:
        # Initialize Bloomberg data fetcher
        bloomberg = BloombergDataFetcher()
        bloomberg.start_session()
        
        # Define date range
        start_date = dt.datetime(2018, 1, 1)
        end_date = dt.datetime(2023, 12, 31)
        
        # Fetch data
        logger.info(f"Fetching data from {start_date} to {end_date}...")
        data = bloomberg.fetch_historical_data(
            tickers=config['object_clusters']['All'],
            fields=['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'VOLUME'],
            start_date=start_date,
            end_date=end_date,
            period='DAILY'
        )
        
        # Fetch risk-free rate
        rf_data = bloomberg.fetch_risk_free_rate(
            start_date=start_date,
            end_date=end_date,
            region='US',
            period='DAILY'
        )
        
        # Stop Bloomberg session
        bloomberg.stop_session()
        
        # Save data
        data.to_pickle('data/market_data.pkl')
        rf_data.to_pickle('data/rf_data.pkl')
        
        logger.info("Data fetched and saved successfully.")
        
        # Print data information
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns}")
        logger.info(f"Data types: {data.dtypes}")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        logger.debug(traceback.format_exc())
        
        # Try to load data if fetching fails
        try:
            logger.info("Attempting to load data from files...")
            data = pd.read_pickle('data/market_data.pkl')
            rf_data = pd.read_pickle('data/rf_data.pkl')
            logger.info("Data loaded successfully.")
            
            # Print data information
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns}")
            logger.info(f"Data types: {data.dtypes}")
            
        except Exception as load_err:
            logger.error(f"Could not load data from files: {load_err}")
            logger.error("Generating synthetic data...")
            
            # Generate synthetic data
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            tickers = config['object_clusters']['All']
            
            # Initialize DataFrames for each field
            synthetic_data = {}
            
            for field in ['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'VOLUME']:
                synthetic_data[field] = pd.DataFrame(index=dates, columns=tickers)
            
            # Generate correlated prices
            np.random.seed(42)
            
            # Define parameters
            initial_prices = {
                'VTI US Equity': 100.0,  # Stock ETF
                'AGG US Equity': 100.0,  # Bond ETF
                'DBC US Equity': 100.0,  # Commodity ETF
                'VIX INDEX': 15.0,       # Volatility Index
            }
            
            annual_vols = {
                'VTI US Equity': 0.15,   # 15% annual volatility for stocks
                'AGG US Equity': 0.05,   # 5% annual volatility for bonds
                'DBC US Equity': 0.20,   # 20% annual volatility for commodities
                'VIX INDEX': 0.70,       # 70% annual volatility for VIX
            }
            
            annual_returns = {
                'VTI US Equity': 0.08,   # 8% annual return for stocks
                'AGG US Equity': 0.03,   # 3% annual return for bonds
                'DBC US Equity': 0.04,   # 4% annual return for commodities
                'VIX INDEX': 0.0,        # 0% annual return for VIX (mean-reverting)
            }
            
            # Correlation matrix
            correlation_matrix = np.array([
                [1.0, -0.2, 0.4, -0.7],  # VTI correlations
                [-0.2, 1.0, 0.1, 0.3],   # AGG correlations
                [0.4, 0.1, 1.0, 0.2],    # DBC correlations
                [-0.7, 0.3, 0.2, 1.0]    # VIX correlations
            ])
            
            # Calculate daily parameters
            daily_vols = {ticker: vol / np.sqrt(252) for ticker, vol in annual_vols.items()}
            daily_returns = {ticker: ret / 252 for ticker, ret in annual_returns.items()}
            
            # Calculate covariance matrix
            vol_vector = np.array([daily_vols[ticker] for ticker in tickers])
            cov_matrix = np.diag(vol_vector) @ correlation_matrix @ np.diag(vol_vector)
            
            # Generate correlated returns
            returns = np.random.multivariate_normal(
                mean=[daily_returns[ticker] for ticker in tickers],
                cov=cov_matrix,
                size=len(dates)
            )
            
            # Generate price series
            for i, ticker in enumerate(tickers):
                close_prices = np.zeros(len(dates))
                close_prices[0] = initial_prices[ticker]
                
                # Calculate close prices using cumulative returns
                for t in range(1, len(dates)):
                    close_prices[t] = close_prices[t-1] * (1 + returns[t, i])
                
                # Generate open, high, low prices
                open_prices = np.zeros(len(dates))
                high_prices = np.zeros(len(dates))
                low_prices = np.zeros(len(dates))
                
                # Generate volume
                volumes = np.random.lognormal(mean=np.log(1000000), sigma=0.5, size=len(dates))
                
                for t in range(len(dates)):
                    # Open price (previous close with small random variation)
                    if t == 0:
                        open_prices[t] = close_prices[t] * (1 + np.random.normal(0, daily_vols[ticker] / 2))
                    else:
                        open_prices[t] = close_prices[t-1] * (1 + np.random.normal(0, daily_vols[ticker] / 2))
                    
                    # Range for the day
                    daily_range = close_prices[t] * daily_vols[ticker] * 2
                    
                    # High and low prices
                    high_prices[t] = max(close_prices[t], open_prices[t]) + np.random.uniform(0, daily_range/2)
                    low_prices[t] = min(close_prices[t], open_prices[t]) - np.random.uniform(0, daily_range/2)
                
                # Store in DataFrames
                synthetic_data['PX_LAST'][ticker] = close_prices
                synthetic_data['PX_OPEN'][ticker] = open_prices
                synthetic_data['PX_HIGH'][ticker] = high_prices
                synthetic_data['PX_LOW'][ticker] = low_prices
                synthetic_data['VOLUME'][ticker] = volumes
            
            # Create multi-index DataFrame
            data = pd.concat({field: df for field, df in synthetic_data.items()}, axis=1)
            
            # Create risk-free rate data
            rf_data = pd.DataFrame(0.02 / 252, index=dates, columns=pd.MultiIndex.from_product([['PX_LAST'], ['US0001M Index']]))
            
            # Save synthetic data
            data.to_pickle('data/synthetic_market_data.pkl')
            rf_data.to_pickle('data/synthetic_rf_data.pkl')
            
            logger.info("Synthetic data generated and saved successfully.")
    
    # Initialize online learning portfolio
    portfolio = OnlineLearningPortfolio(config)
    
    # Generate experts
    portfolio.generate_experts()
    
    # Run backtest
    logger.info("Running backtest...")
    try:
        results_with_costs = portfolio.run_backtest(data, rf_data, transaction_costs=True)
        
        # Plot results
        portfolio.plot_results(results_with_costs)
        
        # Test for statistical arbitrage
        logger.info("Testing for statistical arbitrage...")
        stat_arb_test = StatisticalArbitrageTest(portfolio.profits_and_losses)
        test_results = stat_arb_test.test_statistical_arbitrage()
        
        # Print test results
        logger.info("Statistical Arbitrage Test Results:")
        logger.info(f"Parameters: mu={test_results['parameters'][0]:.6f}, lambda={test_results['parameters'][1]:.6f}, sigma^2={test_results['parameters'][2]:.6f}")
        logger.info(f"Min-t statistic: {test_results['min_t']:.6f}")
        logger.info(f"Critical value: {test_results['critical_value']:.6f}")
        logger.info(f"p-value: {test_results['p_value']:.6f}")
        logger.info(f"Reject null hypothesis of no statistical arbitrage: {test_results['reject_null']}")
        
        # Plot test results
        stat_arb_test.plot_test_results(test_results)
        
        # Run again without transaction costs for comparison
        logger.info("Running backtest without transaction costs...")
        portfolio_no_costs = OnlineLearningPortfolio(config)
        portfolio_no_costs.generate_experts()
        results_no_costs = portfolio_no_costs.run_backtest(data, rf_data, transaction_costs=False)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(results_with_costs['wealth'])), results_with_costs['wealth'], label='With Transaction Costs')
        plt.plot(range(len(results_no_costs['wealth'])), results_no_costs['wealth'], label='Without Transaction Costs')
        plt.title('Portfolio Wealth Comparison')
        plt.xlabel('Trading Periods')
        plt.ylabel('Wealth')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/wealth_comparison.png')
        plt.close()
        
        # Test for statistical arbitrage (no costs)
        stat_arb_test_no_costs = StatisticalArbitrageTest(portfolio_no_costs.profits_and_losses)
        test_results_no_costs = stat_arb_test_no_costs.test_statistical_arbitrage()
        
        logger.info("Statistical Arbitrage Test Results (No Costs):")
        logger.info(f"Parameters: mu={test_results_no_costs['parameters'][0]:.6f}, lambda={test_results_no_costs['parameters'][1]:.6f}, sigma^2={test_results_no_costs['parameters'][2]:.6f}")
        logger.info(f"Min-t statistic: {test_results_no_costs['min_t']:.6f}")
        logger.info(f"Critical value: {test_results_no_costs['critical_value']:.6f}")
        logger.info(f"p-value: {test_results_no_costs['p_value']:.6f}")
        logger.info(f"Reject null hypothesis of no statistical arbitrage: {test_results_no_costs['reject_null']}")
        
        logger.info("Analysis complete.")
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()