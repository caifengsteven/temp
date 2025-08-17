"""
Data Processor for CUBIC framework
Handles data preprocessing, feature engineering, and dataset creation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from .bloomberg_fetcher import BloombergDataFetcher
from .technical_indicators import TechnicalIndicators
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CUBICDataset(Dataset):
    """
    PyTorch Dataset for CUBIC framework
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback_window: int = 5):
        """
        Initialize CUBIC Dataset
        
        Args:
            features: Feature array of shape (n_samples, n_stocks, n_features)
            targets: Target array of shape (n_samples,)
            lookback_window: Number of time steps to look back
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback_window = lookback_window
        
    def __len__(self):
        return len(self.features) - self.lookback_window + 1
    
    def __getitem__(self, idx):
        # Get sequence of features
        feature_sequence = self.features[idx:idx + self.lookback_window]
        target = self.targets[idx + self.lookback_window - 1]

        # Ensure feature_sequence has the right shape: (seq_len, n_stocks, n_features)
        if len(feature_sequence.shape) == 2:
            # If 2D, add stock dimension
            feature_sequence = feature_sequence.unsqueeze(1)

        return feature_sequence, target


class DataProcessor:
    """
    Process and prepare data for CUBIC framework
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Data Processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.data_config = self.config.get('data', {})
        self.training_config = self.config.get('training', {})
        
        self.lookback_window = self.data_config.get('technical_indicators.lookback_window', 5)
        self.train_ratio = self.training_config.get('train_ratio', 0.7)
        self.val_ratio = self.training_config.get('val_ratio', 0.15)
        self.test_ratio = self.training_config.get('test_ratio', 0.15)
        
        self.scaler = StandardScaler()
        
    def calculate_market_return(self, index_data: pd.DataFrame) -> pd.Series:
        """
        Calculate market return as defined in the paper: (I_{t+1} - I_t) / I_t
        
        Args:
            index_data: DataFrame with index price data
            
        Returns:
            Series with market returns
        """
        if 'PX_LAST' not in index_data.columns:
            raise ValueError("Index data must contain 'PX_LAST' column")
        
        prices = index_data['PX_LAST']
        returns = (prices.shift(-1) - prices) / prices
        
        return returns.dropna()
    
    def prepare_features(self, constituent_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Prepare features from constituent stock data
        
        Args:
            constituent_data: Dictionary with stock indicators
            
        Returns:
            Feature array of shape (n_samples, n_stocks, n_features)
        """
        logger.info("Preparing features from constituent data...")
        
        # Get common date range across all stocks
        common_dates = None
        for ticker, data in constituent_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates found across constituent stocks")
        
        # Align all data to common dates
        aligned_data = {}
        for ticker, data in constituent_data.items():
            aligned_data[ticker] = data.loc[common_dates]
        
        # Stack features
        feature_list = []
        stock_names = []
        
        for ticker, data in aligned_data.items():
            if not data.empty:
                # Fill NaN values
                data_filled = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
                feature_list.append(data_filled.values)
                stock_names.append(ticker)
        
        if not feature_list:
            raise ValueError("No valid feature data found")
        
        # Create feature array: (n_samples, n_stocks, n_features)
        features = np.stack(feature_list, axis=1)
        
        logger.info(f"Prepared features shape: {features.shape}")
        logger.info(f"Number of stocks: {len(stock_names)}")
        logger.info(f"Number of features per stock: {features.shape[2]}")
        
        return features, stock_names, common_dates
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            features: Feature array
            targets: Target array
            
        Returns:
            Tuple of (sequence_features, sequence_targets)
        """
        logger.info(f"Creating sequences with lookback window: {self.lookback_window}")
        
        n_samples = len(features) - self.lookback_window + 1
        
        sequence_features = []
        sequence_targets = []
        
        for i in range(n_samples):
            # Get sequence of features
            seq_features = features[i:i + self.lookback_window]
            seq_target = targets[i + self.lookback_window - 1]
            
            sequence_features.append(seq_features)
            sequence_targets.append(seq_target)
        
        sequence_features = np.array(sequence_features)
        sequence_targets = np.array(sequence_targets)
        
        logger.info(f"Created {len(sequence_features)} sequences")
        logger.info(f"Sequence features shape: {sequence_features.shape}")
        
        return sequence_features, sequence_targets
    
    def split_data(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            features: Feature array
            targets: Target array
            
        Returns:
            Dictionary with train, val, and test splits
        """
        n_samples = len(features)
        
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        splits = {
            'train': (features[:train_end], targets[:train_end]),
            'val': (features[train_end:val_end], targets[train_end:val_end]),
            'test': (features[val_end:], targets[val_end:])
        }
        
        logger.info(f"Data split - Train: {len(splits['train'][0])}, "
                   f"Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
        
        return splits
    
    def normalize_features(self, train_features: np.ndarray, val_features: np.ndarray, 
                          test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using training set statistics
        
        Args:
            train_features: Training features
            val_features: Validation features
            test_features: Test features
            
        Returns:
            Tuple of normalized features
        """
        logger.info("Normalizing features...")
        
        # Reshape for normalization (flatten time and stock dimensions)
        original_shape = train_features.shape
        train_flat = train_features.reshape(-1, original_shape[-1])
        val_flat = val_features.reshape(-1, val_features.shape[-1])
        test_flat = test_features.reshape(-1, test_features.shape[-1])
        
        # Fit scaler on training data
        self.scaler.fit(train_flat)
        
        # Transform all splits
        train_normalized = self.scaler.transform(train_flat).reshape(original_shape)
        val_normalized = self.scaler.transform(val_flat).reshape(val_features.shape)
        test_normalized = self.scaler.transform(test_flat).reshape(test_features.shape)
        
        logger.info("Feature normalization completed")
        
        return train_normalized, val_normalized, test_normalized
    
    def create_dataloaders(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                          batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders
        
        Args:
            splits: Dictionary with data splits
            batch_size: Batch size for DataLoaders
            
        Returns:
            Dictionary with DataLoaders
        """
        logger.info("Creating DataLoaders...")
        
        dataloaders = {}
        
        for split_name, (features, targets) in splits.items():
            dataset = CUBICDataset(features, targets, self.lookback_window)
            
            shuffle = (split_name == 'train')
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=True
            )
            
            dataloaders[split_name] = dataloader
            logger.info(f"Created {split_name} DataLoader with {len(dataset)} samples")
        
        return dataloaders
    
    def process_index_data(self, index_name: str, start_date: str, end_date: str) -> Dict:
        """
        Complete data processing pipeline for an index
        
        Args:
            index_name: Name of the index to process
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with processed data and DataLoaders
        """
        logger.info(f"Processing data for index: {index_name}")
        
        # Initialize data fetcher and technical indicators
        data_fetcher = BloombergDataFetcher(self.config.config_path)
        tech_indicators = TechnicalIndicators(self.config.config_path)
        
        # Fetch data
        logger.info("Fetching data from Bloomberg...")
        raw_data = data_fetcher.get_index_data(index_name, start_date, end_date)
        
        if 'index' not in raw_data or 'constituents' not in raw_data:
            raise ValueError("Failed to fetch required data")
        
        # Calculate market returns (targets)
        logger.info("Calculating market returns...")
        market_returns = self.calculate_market_return(raw_data['index'])
        
        # Calculate technical indicators for constituents
        logger.info("Calculating technical indicators...")
        constituent_indicators = tech_indicators.calculate_indicators_for_multiple_stocks(
            raw_data['constituents']
        )
        
        # Normalize indicators
        for ticker in constituent_indicators:
            constituent_indicators[ticker] = tech_indicators.normalize_indicators(
                constituent_indicators[ticker]
            )
        
        # Prepare features
        features, stock_names, common_dates = self.prepare_features(constituent_indicators)
        
        # Align targets with features
        aligned_returns = market_returns.loc[common_dates]
        targets = aligned_returns.values
        
        # Create sequences
        seq_features, seq_targets = self.create_sequences(features, targets)
        
        # Split data
        splits = self.split_data(seq_features, seq_targets)
        
        # Normalize features
        train_features, val_features, test_features = self.normalize_features(
            splits['train'][0], splits['val'][0], splits['test'][0]
        )
        
        # Update splits with normalized features
        normalized_splits = {
            'train': (train_features, splits['train'][1]),
            'val': (val_features, splits['val'][1]),
            'test': (test_features, splits['test'][1])
        }
        
        # Create DataLoaders
        batch_size = self.training_config.get('batch_size', 32)
        dataloaders = self.create_dataloaders(normalized_splits, batch_size)
        
        result = {
            'dataloaders': dataloaders,
            'splits': normalized_splits,
            'stock_names': stock_names,
            'scaler': self.scaler,
            'raw_data': raw_data,
            'market_returns': market_returns
        }
        
        logger.info("Data processing completed successfully")
        return result
