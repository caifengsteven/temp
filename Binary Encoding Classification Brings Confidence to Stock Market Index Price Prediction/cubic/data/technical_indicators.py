"""
Technical Indicators Calculator for CUBIC framework
Implements the 16 technical indicators mentioned in the paper
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import warnings

try:
    import ta
except ImportError:
    warnings.warn("ta package not found. Please install it for technical analysis.")
    ta = None

from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock market data
    Based on the 16 indicators mentioned in the CUBIC paper
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Technical Indicators Calculator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.indicator_config = self.config.get('data.technical_indicators', {})
        
        # Default parameters
        self.lookback_window = self.indicator_config.get('lookback_window', 5)
        self.sma_periods = self.indicator_config.get('sma_periods', [5, 10, 20])
        self.ema_periods = self.indicator_config.get('ema_periods', [5, 10, 20])
        self.rsi_period = self.indicator_config.get('rsi_period', 14)
        self.macd_fast = self.indicator_config.get('macd_fast', 12)
        self.macd_slow = self.indicator_config.get('macd_slow', 26)
        self.macd_signal = self.indicator_config.get('macd_signal', 9)
        self.atr_period = self.indicator_config.get('atr_period', 14)
        self.bb_period = self.indicator_config.get('bb_period', 20)
        self.bb_std = self.indicator_config.get('bb_std', 2)
        self.adx_period = self.indicator_config.get('adx_period', 14)
        self.mfi_period = self.indicator_config.get('mfi_period', 14)
    
    def calculate_arithmetic_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Arithmetic Ratio (Open/Close)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with arithmetic ratio values
        """
        if 'PX_OPEN' not in data.columns or 'PX_LAST' not in data.columns:
            raise ValueError("Data must contain 'PX_OPEN' and 'PX_LAST' columns")
        
        return data['PX_OPEN'] / data['PX_LAST']
    
    def calculate_sma(self, data: pd.DataFrame, column: str = 'PX_LAST', period: int = None) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: DataFrame with price data
            column: Column to calculate SMA for
            period: Period for SMA calculation
            
        Returns:
            Series with SMA values
        """
        if period is None:
            period = self.sma_periods[0]
        
        return data[column].rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.DataFrame, column: str = 'PX_LAST', period: int = None) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: DataFrame with price data
            column: Column to calculate EMA for
            period: Period for EMA calculation
            
        Returns:
            Series with EMA values
        """
        if period is None:
            period = self.ema_periods[0]
        
        return data[column].ewm(span=period).mean()
    
    def calculate_rsi(self, data: pd.DataFrame, column: str = 'PX_LAST') -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: DataFrame with price data
            column: Column to calculate RSI for
            
        Returns:
            Series with RSI values
        """
        if ta is not None:
            return ta.momentum.RSIIndicator(data[column], window=self.rsi_period).rsi()
        else:
            # Manual RSI calculation
            delta = data[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.DataFrame, column: str = 'PX_LAST') -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: DataFrame with price data
            column: Column to calculate MACD for
            
        Returns:
            Dictionary with MACD, Signal, and Histogram
        """
        if ta is not None:
            macd_indicator = ta.trend.MACD(
                data[column], 
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            return {
                'MACD': macd_indicator.macd(),
                'MACD_Signal': macd_indicator.macd_signal(),
                'MACD_Histogram': macd_indicator.macd_diff()
            }
        else:
            # Manual MACD calculation
            ema_fast = data[column].ewm(span=self.macd_fast).mean()
            ema_slow = data[column].ewm(span=self.macd_slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'MACD': macd_line,
                'MACD_Signal': signal_line,
                'MACD_Histogram': histogram
            }
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with ATR values
        """
        if ta is not None:
            return ta.volatility.AverageTrueRange(
                data['PX_HIGH'], 
                data['PX_LOW'], 
                data['PX_LAST'], 
                window=self.atr_period
            ).average_true_range()
        else:
            # Manual ATR calculation
            high_low = data['PX_HIGH'] - data['PX_LOW']
            high_close = np.abs(data['PX_HIGH'] - data['PX_LAST'].shift())
            low_close = np.abs(data['PX_LOW'] - data['PX_LAST'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=self.atr_period).mean()
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, column: str = 'PX_LAST') -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: DataFrame with price data
            column: Column to calculate Bollinger Bands for
            
        Returns:
            Dictionary with Upper, Middle, and Lower bands
        """
        if ta is not None:
            bb_indicator = ta.volatility.BollingerBands(
                data[column], 
                window=self.bb_period, 
                window_dev=self.bb_std
            )
            return {
                'BB_Upper': bb_indicator.bollinger_hband(),
                'BB_Middle': bb_indicator.bollinger_mavg(),
                'BB_Lower': bb_indicator.bollinger_lband()
            }
        else:
            # Manual Bollinger Bands calculation
            sma = data[column].rolling(window=self.bb_period).mean()
            std = data[column].rolling(window=self.bb_period).std()
            
            return {
                'BB_Upper': sma + (std * self.bb_std),
                'BB_Middle': sma,
                'BB_Lower': sma - (std * self.bb_std)
            }
    
    def calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with ADX values
        """
        if ta is not None:
            return ta.trend.ADXIndicator(
                data['PX_HIGH'], 
                data['PX_LOW'], 
                data['PX_LAST'], 
                window=self.adx_period
            ).adx()
        else:
            # Simplified ADX calculation (manual implementation is complex)
            logger.warning("ADX calculation requires 'ta' package for accurate results")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_mfi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Money Flow Index
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with MFI values
        """
        if ta is not None:
            return ta.volume.MFIIndicator(
                data['PX_HIGH'], 
                data['PX_LOW'], 
                data['PX_LAST'], 
                data['PX_VOLUME'], 
                window=self.mfi_period
            ).money_flow_index()
        else:
            # Manual MFI calculation
            typical_price = (data['PX_HIGH'] + data['PX_LOW'] + data['PX_LAST']) / 3
            money_flow = typical_price * data['PX_VOLUME']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
            
            positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
            negative_mf = negative_flow.rolling(window=self.mfi_period).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Args:
            data: DataFrame with price and volume data
            
        Returns:
            Series with OBV values
        """
        if ta is not None:
            return ta.volume.OnBalanceVolumeIndicator(
                data['PX_LAST'], 
                data['PX_VOLUME']
            ).on_balance_volume()
        else:
            # Manual OBV calculation
            price_change = data['PX_LAST'].diff()
            volume_direction = np.where(price_change > 0, data['PX_VOLUME'], 
                                      np.where(price_change < 0, -data['PX_VOLUME'], 0))
            return pd.Series(volume_direction, index=data.index).cumsum()
    
    def calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Stochastic %K

        Args:
            data: DataFrame with OHLC data
            period: Period for calculation

        Returns:
            Series with %K values
        """
        if ta is not None:
            return ta.momentum.StochasticOscillator(
                data['PX_HIGH'],
                data['PX_LOW'],
                data['PX_LAST'],
                window=period
            ).stoch()
        else:
            # Manual %K calculation
            lowest_low = data['PX_LOW'].rolling(window=period).min()
            highest_high = data['PX_HIGH'].rolling(window=period).max()
            k_percent = 100 * ((data['PX_LAST'] - lowest_low) / (highest_high - lowest_low))
            return k_percent

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 16 technical indicators mentioned in the paper

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Calculating all technical indicators...")

        indicators = pd.DataFrame(index=data.index)

        try:
            # Trend Indicators
            indicators['Arithmetic_Ratio'] = self.calculate_arithmetic_ratio(data)
            indicators['Open'] = data['PX_OPEN']
            indicators['Close'] = data['PX_LAST']
            indicators['Close_SMA'] = self.calculate_sma(data, 'PX_LAST', self.sma_periods[0])
            indicators['Volume_SMA'] = self.calculate_sma(data, 'PX_VOLUME', self.sma_periods[0])
            indicators['Close_EMA'] = self.calculate_ema(data, 'PX_LAST', self.ema_periods[0])
            indicators['Volume_EMA'] = self.calculate_ema(data, 'PX_VOLUME', self.ema_periods[0])
            indicators['ADX'] = self.calculate_adx(data)

            # Oscillator Indicators
            indicators['RSI'] = self.calculate_rsi(data)
            macd_data = self.calculate_macd(data)
            indicators['MACD'] = macd_data['MACD']
            indicators['MACD_Signal'] = macd_data['MACD_Signal']
            indicators['K'] = self.calculate_stochastic_k(data)
            indicators['MFI'] = self.calculate_mfi(data)

            # Volatility Indicators
            indicators['ATR'] = self.calculate_atr(data)
            bb_data = self.calculate_bollinger_bands(data)
            indicators['BB_Middle'] = bb_data['BB_Middle']
            indicators['OBV'] = self.calculate_obv(data)

            logger.info(f"Successfully calculated {len(indicators.columns)} technical indicators")

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

        return indicators

    def calculate_indicators_for_multiple_stocks(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators for multiple stocks

        Args:
            data_dict: Dictionary with stock tickers as keys and OHLCV data as values

        Returns:
            Dictionary with stock tickers as keys and indicator DataFrames as values
        """
        logger.info(f"Calculating indicators for {len(data_dict)} stocks...")

        indicators_dict = {}

        for ticker, data in data_dict.items():
            try:
                logger.debug(f"Processing indicators for {ticker}")
                indicators = self.calculate_all_indicators(data)
                indicators_dict[ticker] = indicators

            except Exception as e:
                logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
                # Create empty DataFrame with same index for consistency
                indicators_dict[ticker] = pd.DataFrame(index=data.index)

        logger.info(f"Successfully calculated indicators for {len(indicators_dict)} stocks")
        return indicators_dict

    def normalize_indicators(self, indicators: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize technical indicators

        Args:
            indicators: DataFrame with technical indicators
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            DataFrame with normalized indicators
        """
        logger.info(f"Normalizing indicators using {method} method...")

        normalized = indicators.copy()

        if method == 'zscore':
            # Z-score normalization
            normalized = (indicators - indicators.mean()) / indicators.std()
        elif method == 'minmax':
            # Min-max normalization
            normalized = (indicators - indicators.min()) / (indicators.max() - indicators.min())
        elif method == 'robust':
            # Robust normalization using median and IQR
            median = indicators.median()
            q75 = indicators.quantile(0.75)
            q25 = indicators.quantile(0.25)
            iqr = q75 - q25
            normalized = (indicators - median) / iqr
        else:
            logger.warning(f"Unknown normalization method: {method}. Using original data.")

        # Handle infinite and NaN values
        normalized = normalized.replace([np.inf, -np.inf], np.nan)
        normalized = normalized.fillna(0)

        logger.info("Normalization completed")
        return normalized
