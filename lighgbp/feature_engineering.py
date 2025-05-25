"""
Feature Engineering Module

This module provides functions to create features for the LightGBM model based on
the research paper "Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM".
"""

import pandas as pd
import numpy as np
import talib  # Technical Analysis Library


class FeatureEngineer:
    """Class for creating features from OHLCV data."""
    
    def __init__(self, data):
        """
        Initialize with a DataFrame containing OHLCV data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with columns: open, high, low, close, volume
        """
        self.data = data.copy()
        # Ensure the data is sorted by date
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        
        # Create previous day columns for easier reference
        self.data['open_prev'] = self.data['open'].shift(1)
        self.data['high_prev'] = self.data['high'].shift(1)
        self.data['low_prev'] = self.data['low'].shift(1)
        self.data['close_prev'] = self.data['close'].shift(1)
        self.data['volume_prev'] = self.data['volume'].shift(1)
    
    def create_all_features(self):
        """
        Create all features mentioned in the paper.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all features
        """
        # Basic features
        self._add_typical_price()
        self._add_lag_features()
        self._add_statistical_features()
        
        # Technical indicators
        self._add_momentum_indicators()
        self._add_volatility_indicators()
        self._add_trend_indicators()
        
        # Novel features from the paper
        self._add_novel_features()
        
        # Drop rows with NaN values (due to lag features and indicators)
        self.data = self.data.dropna()
        
        return self.data
    
    def _add_typical_price(self):
        """Add typical price feature."""
        # Typical price = (High + Low + Close) / 3
        self.data['typical'] = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        self.data['typical_prev'] = self.data['typical'].shift(1)
    
    def _add_lag_features(self):
        """Add lag features for open, close, and typical price."""
        # Lag periods as mentioned in the paper: 1, 5, 30
        for lag in [1, 5, 30]:
            self.data[f'open_lag_{lag}'] = self.data['open'].shift(lag)
            self.data[f'close_prev_lag_{lag}'] = self.data['close_prev'].shift(lag)
            self.data[f'typical_lag_{lag}'] = self.data['typical'].shift(lag)
    
    def _add_statistical_features(self):
        """Add statistical features like SMA, rolling std, min/max."""
        # Windows for rolling calculations
        windows = [5, 10, 20, 50, 200]
        
        for window in windows:
            # Simple Moving Averages (SMA)
            self.data[f'sma_{window}_close'] = self.data['close_prev'].rolling(window=window).mean()
            self.data[f'sma_{window}_typical'] = self.data['typical_prev'].rolling(window=window).mean()
            
            # Rolling standard deviation
            self.data[f'std_{window}_close'] = self.data['close_prev'].rolling(window=window).std()
            self.data[f'std_{window}_typical'] = self.data['typical_prev'].rolling(window=window).std()
            
            # Rolling min/max
            self.data[f'min_{window}_close'] = self.data['close_prev'].rolling(window=window).min()
            self.data[f'max_{window}_close'] = self.data['close_prev'].rolling(window=window).max()
            
            # Volume features
            self.data[f'volume_sma_{window}'] = self.data['volume_prev'].rolling(window=window).mean()
            self.data[f'volume_std_{window}'] = self.data['volume_prev'].rolling(window=window).std()
    
    def _add_momentum_indicators(self):
        """Add momentum indicators mentioned in the paper."""
        # Relative Strength Index (RSI)
        self.data['rsi_14'] = talib.RSI(self.data['close_prev'].values, timeperiod=14)
        
        # Rate of Change (ROC)
        self.data['roc_10'] = talib.ROC(self.data['close_prev'].values, timeperiod=10)
        
        # Moving Average Convergence Divergence (MACD)
        macd, macd_signal, macd_hist = talib.MACD(
            self.data['close_prev'].values, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        self.data['macd'] = macd
        self.data['macd_signal'] = macd_signal
        self.data['macd_hist'] = macd_hist
        
        # Stochastic %K and %D
        self.data['stoch_k'], self.data['stoch_d'] = talib.STOCH(
            self.data['high_prev'].values,
            self.data['low_prev'].values,
            self.data['close_prev'].values,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # Commodity Channel Index (CCI)
        self.data['cci_20'] = talib.CCI(
            self.data['high_prev'].values,
            self.data['low_prev'].values,
            self.data['close_prev'].values,
            timeperiod=20
        )
        
        # Chande Momentum Oscillator (CMO)
        self.data['cmo_14'] = talib.CMO(self.data['close_prev'].values, timeperiod=14)
    
    def _add_volatility_indicators(self):
        """Add volatility indicators mentioned in the paper."""
        # Average True Range (ATR)
        self.data['atr_14'] = talib.ATR(
            self.data['high_prev'].values,
            self.data['low_prev'].values,
            self.data['close_prev'].values,
            timeperiod=14
        )
        
        # Chaikin Volatility
        # Chaikin Volatility is not directly available in talib
        # It's calculated as the rate of change of the ATR
        high_low_diff = self.data['high_prev'] - self.data['low_prev']
        ema_high_low = talib.EMA(high_low_diff.values, timeperiod=10)
        self.data['chaikin_volatility'] = talib.ROC(ema_high_low, timeperiod=10)
    
    def _add_trend_indicators(self):
        """Add trend indicators mentioned in the paper."""
        # Exponential Moving Average (EMA)
        for period in [5, 10, 20, 50, 200]:
            self.data[f'ema_{period}'] = talib.EMA(self.data['close_prev'].values, timeperiod=period)
        
        # Specifically add EMA 14 for novel features
        self.data['ema_14'] = talib.EMA(self.data['close_prev'].values, timeperiod=14)
    
    def _add_novel_features(self):
        """Add novel features mentioned in the paper."""
        # Indicator-price slope ratios
        indicators = ['rsi_14', 'macd', 'stoch_k', 'cci_20']
        for indicator in indicators:
            if indicator in self.data.columns:
                # Calculate slope of indicator (change over 5 days)
                indicator_slope = self.data[indicator] - self.data[indicator].shift(5)
                # Calculate slope of price (change over 5 days)
                price_slope = self.data['close_prev'] - self.data['close_prev'].shift(5)
                # Calculate ratio (handle division by zero)
                self.data[f'{indicator}_price_slope_ratio'] = np.where(
                    price_slope != 0,
                    indicator_slope / price_slope,
                    0
                )
        
        # Open-previous close price differences divided by 14-period EMA
        self.data['open_close_prev_diff_ema14_ratio'] = (self.data['open'] - self.data['close_prev']) / self.data['ema_14']
        
        # EMA difference ratios (as mentioned for target transformation, but also useful as features)
        self.data['close_prev_ema14_diff_ratio'] = (self.data['close_prev'] - self.data['ema_14']) / self.data['ema_14']


class DataTransformer:
    """Class for transforming target variables and features."""
    
    @staticmethod
    def returns(data, column='close'):
        """
        Calculate simple returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate returns for
            
        Returns:
        --------
        pandas.Series
            Series containing returns
        """
        return data[column].pct_change()
    
    @staticmethod
    def log_returns(data, column='close'):
        """
        Calculate logarithmic returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate log returns for
            
        Returns:
        --------
        pandas.Series
            Series containing log returns
        """
        return np.log(data[column] / data[column].shift(1))
    
    @staticmethod
    def ema_ratio(data, column='close', period=14):
        """
        Calculate ratio of price to its EMA.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate EMA ratio for
        period : int
            Period for EMA calculation
            
        Returns:
        --------
        pandas.Series
            Series containing EMA ratios
        """
        ema = talib.EMA(data[column].values, timeperiod=period)
        return data[column] / ema
    
    @staticmethod
    def standardized_returns(data, column='close', window=20):
        """
        Calculate standardized returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate standardized returns for
        window : int
            Window for rolling standardization
            
        Returns:
        --------
        pandas.Series
            Series containing standardized returns
        """
        returns = data[column].pct_change()
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return (returns - rolling_mean) / rolling_std
    
    @staticmethod
    def standardized_log_returns(data, column='close', window=20):
        """
        Calculate standardized log returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate standardized log returns for
        window : int
            Window for rolling standardization
            
        Returns:
        --------
        pandas.Series
            Series containing standardized log returns
        """
        log_returns = np.log(data[column] / data[column].shift(1))
        rolling_mean = log_returns.rolling(window=window).mean()
        rolling_std = log_returns.rolling(window=window).std()
        return (log_returns - rolling_mean) / rolling_std
    
    @staticmethod
    def standardized_ema_ratio(data, column='close', ema_period=14, window=20):
        """
        Calculate standardized EMA ratio.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate standardized EMA ratio for
        ema_period : int
            Period for EMA calculation
        window : int
            Window for rolling standardization
            
        Returns:
        --------
        pandas.Series
            Series containing standardized EMA ratios
        """
        ema = talib.EMA(data[column].values, timeperiod=ema_period)
        ema_ratio = data[column] / ema
        rolling_mean = ema_ratio.rolling(window=window).mean()
        rolling_std = ema_ratio.rolling(window=window).std()
        return (ema_ratio - rolling_mean) / rolling_std
    
    @staticmethod
    def ema_difference_ratio(data, column='close', period=14):
        """
        Calculate EMA difference ratio.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to calculate EMA difference ratio for
        period : int
            Period for EMA calculation
            
        Returns:
        --------
        pandas.Series
            Series containing EMA difference ratios
        """
        ema = talib.EMA(data[column].values, timeperiod=period)
        return (data[column] - ema) / ema


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.lognormal(15, 1, 100).astype(int)
    }, index=dates)
    
    # Ensure high >= open, close, low and low <= open, close
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Create features
    engineer = FeatureEngineer(data)
    featured_data = engineer.create_all_features()
    
    print(f"Original data shape: {data.shape}")
    print(f"Featured data shape: {featured_data.shape}")
    print(f"Number of features created: {featured_data.shape[1] - 5}")  # Subtract original OHLCV columns
