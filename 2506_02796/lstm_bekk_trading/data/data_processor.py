"""
Data Processor Module

Handles data cleaning, preprocessing, and return calculations for the LSTM-BEKK system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats


class DataProcessor:
    """
    Data processor for financial time series.
    
    Implements data cleaning, return calculations, and preprocessing
    as specified in the LSTM-BEKK research paper.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def calculate_returns(self, prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with prices (dates x assets)
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            DataFrame with returns scaled by 100 (as in the paper)
        """
        self.logger.info(f"Calculating {method} returns for {len(prices.columns)} assets")
        
        if method == "log":
            # Log returns as in the paper: log(P_t / P_{t-1}) * 100
            returns = np.log(prices / prices.shift(1)) * 100
        elif method == "simple":
            # Simple returns: (P_t / P_{t-1} - 1) * 100
            returns = (prices / prices.shift(1) - 1) * 100
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Drop first row (NaN values)
        returns = returns.dropna()
        
        self.logger.info(f"Calculated returns for {len(returns)} periods")
        return returns
    
    def clean_data(self, data: pd.DataFrame, 
                   max_missing_pct: float = 0.05,
                   outlier_threshold: float = 5.0) -> pd.DataFrame:
        """
        Clean financial data by handling missing values and outliers.
        
        Args:
            data: Input DataFrame
            max_missing_pct: Maximum percentage of missing values allowed per asset
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        original_shape = data.shape
        
        # 1. Handle missing values
        data = self._handle_missing_values(data, max_missing_pct)
        
        # 2. Remove outliers
        data = self._remove_outliers(data, outlier_threshold)
        
        # 3. Ensure minimum data requirements
        data = self._ensure_minimum_data(data)
        
        final_shape = data.shape
        self.logger.info(f"Data cleaning complete: {original_shape} -> {final_shape}")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, max_missing_pct: float) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Calculate missing percentage per asset
        missing_pct = data.isnull().sum() / len(data)
        
        # Remove assets with too many missing values
        valid_assets = missing_pct[missing_pct <= max_missing_pct].index
        data = data[valid_assets]
        
        removed_count = len(missing_pct) - len(valid_assets)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} assets due to excessive missing data")
        
        # Forward fill remaining missing values
        data = data.fillna(method='ffill')
        
        # Drop any remaining rows with missing values
        data = data.dropna()
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        
        # Create mask for outliers
        outlier_mask = z_scores > threshold
        
        # Count outliers per asset
        outlier_counts = outlier_mask.sum()
        total_outliers = outlier_counts.sum()
        
        if total_outliers > 0:
            self.logger.info(f"Detected {total_outliers} outliers across all assets")
            
            # Replace outliers with NaN and then forward fill
            data_clean = data.copy()
            data_clean[outlier_mask] = np.nan
            data_clean = data_clean.fillna(method='ffill')
            data_clean = data_clean.fillna(method='bfill')
            
            return data_clean
        
        return data
    
    def _ensure_minimum_data(self, data: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
        """Ensure minimum data requirements."""
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data: {len(data)} < {min_periods} periods")
        
        # Remove assets with insufficient non-null data
        valid_counts = data.count()
        valid_assets = valid_counts[valid_counts >= min_periods].index
        
        if len(valid_assets) < len(data.columns):
            removed = len(data.columns) - len(valid_assets)
            self.logger.warning(f"Removed {removed} assets due to insufficient data")
            data = data[valid_assets]
        
        return data
    
    def demean_returns(self, returns: pd.DataFrame, method: str = "rolling", 
                      window: int = 252) -> pd.DataFrame:
        """
        De-mean returns as required by the LSTM-BEKK model.
        
        Args:
            returns: Return series
            method: De-meaning method ('rolling', 'expanding', or 'full_sample')
            window: Window size for rolling method
            
        Returns:
            De-meaned returns
        """
        self.logger.info(f"De-meaning returns using {method} method")
        
        if method == "rolling":
            # Rolling window mean
            rolling_mean = returns.rolling(window=window, min_periods=window//2).mean()
            demeaned = returns - rolling_mean
        elif method == "expanding":
            # Expanding window mean
            expanding_mean = returns.expanding(min_periods=window//2).mean()
            demeaned = returns - expanding_mean
        elif method == "full_sample":
            # Full sample mean
            demeaned = returns - returns.mean()
        else:
            raise ValueError(f"Unknown de-meaning method: {method}")
        
        # Drop initial NaN values
        demeaned = demeaned.dropna()
        
        return demeaned
    
    def create_train_test_split(self, data: pd.DataFrame, 
                               train_ratio: float = 0.7,
                               validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits for time series data.
        
        Args:
            data: Input data
            train_ratio: Proportion for training
            validation_ratio: Proportion for validation
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * validation_ratio)
        
        train_data = data.iloc[:n_train]
        val_data = data.iloc[n_train:n_train + n_val]
        test_data = data.iloc[n_train + n_val:]
        
        self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data

    def get_data_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for the dataset.

        Args:
            data: Input data

        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max()),
            'missing_values': data.isnull().sum().sum(),
            'mean_returns': data.mean(),
            'volatility': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'min_values': data.min(),
            'max_values': data.max(),
            'correlation_matrix': data.corr()
        }

        return stats_dict

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality for LSTM-BEKK modeling.

        Args:
            data: Input data

        Returns:
            Dictionary with validation results
        """
        validation = {
            'sufficient_length': len(data) >= 252,  # At least 1 year
            'sufficient_assets': len(data.columns) >= 5,  # At least 5 assets
            'no_missing_values': not data.isnull().any().any(),
            'no_infinite_values': not np.isinf(data).any().any(),
            'reasonable_volatility': (data.std() < 50).all(),  # Less than 50% daily vol
            'non_constant_series': (data.std() > 0.01).all()  # Some variation
        }

        return validation
