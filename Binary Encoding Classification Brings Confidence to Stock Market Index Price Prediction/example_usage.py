#!/usr/bin/env python3
"""
Example usage of CUBIC framework with Bloomberg data
This script demonstrates how to use the CUBIC framework for stock market index prediction
"""

import os
import sys
import logging
import numpy as np
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cubic.data.bloomberg_fetcher import BloombergDataFetcher
from cubic.data.technical_indicators import TechnicalIndicators
from cubic.data.data_processor import DataProcessor
from cubic.models.cubic_model import CUBICModel
from cubic.training.trainer import CUBICTrainer
from cubic.evaluation.evaluator import CUBICEvaluator
from cubic.utils.config_manager import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_with_bloomberg_data():
    """
    Example using real Bloomberg data
    """
    logger.info("=== CUBIC Framework Example with Bloomberg Data ===")
    
    try:
        # Initialize configuration
        config = ConfigManager("config.yaml")
        
        # Step 1: Fetch data from Bloomberg
        logger.info("Step 1: Fetching data from Bloomberg...")
        data_fetcher = BloombergDataFetcher()
        
        # Fetch S&P 500 data
        index_data = data_fetcher.get_index_data(
            index_name='SPX',
            start_date='2020-01-01',
            end_date='2024-01-01',
            include_constituents=True
        )
        
        if index_data['index'].empty:
            logger.warning("No Bloomberg data available, switching to synthetic example")
            return example_with_synthetic_data()
        
        # Step 2: Calculate technical indicators
        logger.info("Step 2: Calculating technical indicators...")
        tech_indicators = TechnicalIndicators()
        
        # Calculate indicators for constituent stocks
        constituent_indicators = {}
        for ticker, data in index_data['constituents'].items():
            if not data.empty:
                indicators = tech_indicators.calculate_all_indicators(data)
                constituent_indicators[ticker] = indicators
        
        # Step 3: Process data for CUBIC
        logger.info("Step 3: Processing data for CUBIC framework...")
        data_processor = DataProcessor()
        
        # Prepare features and targets
        features, stock_names, dates = data_processor.prepare_features(constituent_indicators)
        market_returns = data_processor.calculate_market_return(index_data['index'])
        targets = market_returns.loc[dates].values
        
        # Create sequences and splits
        seq_features, seq_targets = data_processor.create_sequences(features, targets)
        splits = data_processor.split_data(seq_features, seq_targets)
        
        # Normalize features
        train_features, val_features, test_features = data_processor.normalize_features(
            splits['train'][0], splits['val'][0], splits['test'][0]
        )
        
        # Create dataloaders
        normalized_splits = {
            'train': (train_features, splits['train'][1]),
            'val': (val_features, splits['val'][1]),
            'test': (test_features, splits['test'][1])
        }
        dataloaders = data_processor.create_dataloaders(normalized_splits)
        
        # Step 4: Create CUBIC model
        logger.info("Step 4: Creating CUBIC model...")
        model = CUBICModel(
            input_dim=features.shape[-1],
            n_stocks=features.shape[1],
            backbone_type='lstm',
            backbone_config={'hidden_size': 128, 'num_layers': 2},
            fusion_config={'embedding_dim': 32, 'pooling_type': 'multi_head'},
            binary_config={'precision_bits': 15, 'value_range': (-1, 1)}
        )
        
        logger.info(f"Model info: {model.get_model_info()}")
        
        # Step 5: Train model
        logger.info("Step 5: Training CUBIC model...")
        trainer = CUBICTrainer(model)
        training_history = trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Step 6: Evaluate model
        logger.info("Step 6: Evaluating CUBIC model...")
        evaluator = CUBICEvaluator(model)
        results = evaluator.evaluate_model(dataloaders['test'])
        
        # Print results
        print("\n=== CUBIC Framework Results ===")
        print("Basic Metrics:")
        for metric, value in results['basic_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTrading Metrics (with confidence-guided trading):")
        for metric, value in results['trading_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nConfidence Statistics:")
        for metric, value in results['confidence_stats'].items():
            print(f"  {metric}: {value:.4f}")
        
        logger.info("Bloomberg data example completed successfully!")
        
    except Exception as e:
        logger.error(f"Bloomberg example failed: {str(e)}")
        logger.info("Falling back to synthetic data example...")
        return example_with_synthetic_data()


def example_with_synthetic_data():
    """
    Example using synthetic data for demonstration
    """
    logger.info("=== CUBIC Framework Example with Synthetic Data ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Step 1: Generate synthetic data
    logger.info("Step 1: Generating synthetic data...")
    
    n_samples = 1000
    n_stocks = 30
    n_features = 16
    sequence_length = 5
    
    # Generate synthetic features (technical indicators)
    features = np.random.randn(n_samples, sequence_length, n_stocks, n_features)
    
    # Generate synthetic targets (market returns)
    # Add some correlation with features for realistic behavior
    feature_sum = np.mean(features[:, -1, :, :], axis=(1, 2))  # Last timestep average
    targets = 0.02 * feature_sum + 0.01 * np.random.randn(n_samples)
    targets = np.clip(targets, -0.1, 0.1)  # Clip to reasonable return range
    
    # Step 2: Create data splits
    logger.info("Step 2: Creating data splits...")
    
    from cubic.data.data_processor import CUBICDataset
    from torch.utils.data import DataLoader
    
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    # Create datasets
    train_dataset = CUBICDataset(features[:train_end], targets[:train_end])
    val_dataset = CUBICDataset(features[train_end:val_end], targets[train_end:val_end])
    test_dataset = CUBICDataset(features[val_end:], targets[val_end:])
    
    # Create dataloaders
    batch_size = 32
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    # Step 3: Create CUBIC model
    logger.info("Step 3: Creating CUBIC model...")
    
    model = CUBICModel(
        input_dim=n_features,
        n_stocks=n_stocks,
        backbone_type='lstm',
        backbone_config={'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
        fusion_config={'embedding_dim': 32, 'pooling_type': 'multi_head'},
        binary_config={'precision_bits': 15, 'value_range': (-1, 1)},
        confidence_config={'weight': 0.1}
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 4: Train model (reduced epochs for demo)
    logger.info("Step 4: Training CUBIC model...")
    
    # Update config for faster training
    config = ConfigManager()
    config.set('training.num_epochs', 10)
    config.set('training.early_stopping_patience', 5)
    
    trainer = CUBICTrainer(model)
    trainer.num_epochs = 10  # Reduced for demo
    trainer.early_stopping_patience = 5
    
    training_history = trainer.train(dataloaders['train'], dataloaders['val'])
    
    # Step 5: Evaluate model
    logger.info("Step 5: Evaluating CUBIC model...")
    
    evaluator = CUBICEvaluator(model)
    results = evaluator.evaluate_model(dataloaders['test'])
    
    # Step 6: Display results
    print("\n" + "="*50)
    print("CUBIC Framework Results (Synthetic Data)")
    print("="*50)
    
    print("\nBasic Metrics:")
    for metric, value in results['basic_metrics'].items():
        print(f"  {metric:20s}: {value:8.4f}")
    
    print("\nTrading Metrics (Confidence-Guided):")
    for metric, value in results['trading_metrics'].items():
        print(f"  {metric:20s}: {value:8.4f}")
    
    print("\nConfidence Statistics:")
    for metric, value in results['confidence_stats'].items():
        print(f"  {metric:25s}: {value:8.4f}")
    
    print(f"\nModel Performance Summary:")
    print(f"  Number of test samples: {results['num_samples']}")
    print(f"  Number of trades: {results['num_trades']}")
    print(f"  Trade frequency: {results['num_trades']/results['num_samples']:.2%}")
    
    # Step 7: Demonstrate binary encoding
    logger.info("Step 6: Demonstrating binary encoding...")
    
    from cubic.utils.binary_encoder import BinaryEncoder
    
    encoder = BinaryEncoder(precision_bits=15, value_range=(-1, 1))
    
    # Test encoding/decoding
    test_values = [0.0, 0.5, -0.3, 0.8, -0.9]
    print(f"\nBinary Encoding Demonstration:")
    print(f"{'Original':>10s} {'Encoded':>20s} {'Decoded':>10s} {'Error':>10s}")
    print("-" * 55)
    
    for val in test_values:
        encoded = encoder.encode_value(val)
        decoded = encoder.decode_binary(encoded)
        error = abs(val - decoded)
        encoded_str = ''.join(map(str, encoded[:8])) + '...'  # Show first 8 bits
        print(f"{val:10.4f} {encoded_str:>20s} {decoded:10.4f} {error:10.6f}")
    
    logger.info("Synthetic data example completed successfully!")


def main():
    """Main function to run examples"""
    print("CUBIC Framework - Stock Market Index Prediction")
    print("Paper: 'Why Regression? Binary Encoding Classification Brings Confidence to Stock Market Index Price Prediction'")
    print()
    
    # Try Bloomberg data first, fall back to synthetic if not available
    try:
        example_with_bloomberg_data()
    except Exception as e:
        logger.warning(f"Bloomberg example failed: {str(e)}")
        logger.info("Running synthetic data example instead...")
        example_with_synthetic_data()


if __name__ == "__main__":
    main()
