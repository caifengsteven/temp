#!/usr/bin/env python3
"""
Main execution script for CUBIC framework
Implements the paper: "Why Regression? Binary Encoding Classification Brings Confidence to Stock Market Index Price Prediction"
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cubic.data.data_processor import DataProcessor
from cubic.models.cubic_model import CUBICModel
from cubic.training.trainer import CUBICTrainer
from cubic.evaluation.evaluator import CUBICEvaluator
from cubic.utils.config_manager import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cubic_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_experiment(config_path: str, index_name: str, backbone_type: str) -> dict:
    """
    Set up experiment configuration
    
    Args:
        config_path: Path to configuration file
        index_name: Name of the index to use
        backbone_type: Type of backbone model
        
    Returns:
        Dictionary with experiment setup
    """
    logger.info(f"Setting up experiment for {index_name} with {backbone_type} backbone")
    
    # Load configuration
    config = ConfigManager(config_path)
    
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Get data configuration
    data_config = config.get('data', {})
    start_date = data_config.get('date_range.start_date', '2010-01-01')
    end_date = data_config.get('date_range.end_date', '2024-12-31')
    
    # Check if index is configured
    indices_config = data_config.get('indices', {})
    if index_name not in indices_config:
        raise ValueError(f"Index {index_name} not found in configuration")
    
    return {
        'config': config,
        'index_name': index_name,
        'backbone_type': backbone_type,
        'start_date': start_date,
        'end_date': end_date,
        'seed': seed
    }


def prepare_data(experiment_setup: dict) -> dict:
    """
    Prepare data for training and evaluation
    
    Args:
        experiment_setup: Experiment setup dictionary
        
    Returns:
        Dictionary with prepared data
    """
    logger.info("Preparing data...")
    
    config = experiment_setup['config']
    index_name = experiment_setup['index_name']
    start_date = experiment_setup['start_date']
    end_date = experiment_setup['end_date']
    
    # Initialize data processor
    data_processor = DataProcessor(config.config_path)
    
    try:
        # Process index data
        processed_data = data_processor.process_index_data(index_name, start_date, end_date)
        
        logger.info(f"Data preparation completed successfully")
        logger.info(f"Training samples: {len(processed_data['dataloaders']['train'].dataset)}")
        logger.info(f"Validation samples: {len(processed_data['dataloaders']['val'].dataset)}")
        logger.info(f"Test samples: {len(processed_data['dataloaders']['test'].dataset)}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        logger.info("Using synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        return create_synthetic_data(config)


def create_synthetic_data(config: ConfigManager) -> dict:
    """
    Create synthetic data for demonstration purposes
    
    Args:
        config: Configuration manager
        
    Returns:
        Dictionary with synthetic data
    """
    logger.info("Creating synthetic data for demonstration...")
    
    # Synthetic data parameters
    n_samples = 1000
    n_stocks = 30
    n_features = 16
    sequence_length = 5
    
    # Generate synthetic features and targets
    np.random.seed(42)
    features = np.random.randn(n_samples, sequence_length, n_stocks, n_features)
    targets = np.random.randn(n_samples) * 0.02  # Small returns
    
    # Create train/val/test splits
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    from cubic.data.data_processor import CUBICDataset
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CUBICDataset(features[:train_end], targets[:train_end])
    val_dataset = CUBICDataset(features[train_end:val_end], targets[train_end:val_end])
    test_dataset = CUBICDataset(features[val_end:], targets[val_end:])
    
    # Create dataloaders
    batch_size = config.get('training.batch_size', 32)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    return {
        'dataloaders': dataloaders,
        'stock_names': [f'Stock_{i}' for i in range(n_stocks)],
        'n_features': n_features,
        'n_stocks': n_stocks
    }


def create_model(experiment_setup: dict, data_info: dict) -> CUBICModel:
    """
    Create CUBIC model
    
    Args:
        experiment_setup: Experiment setup dictionary
        data_info: Data information dictionary
        
    Returns:
        CUBIC model
    """
    logger.info("Creating CUBIC model...")
    
    config = experiment_setup['config']
    backbone_type = experiment_setup['backbone_type']
    
    # Model configuration
    input_dim = data_info.get('n_features', 16)
    n_stocks = data_info.get('n_stocks', 30)
    
    # Get configurations from config file
    model_config = config.get('model', {})
    backbone_config = model_config.get('architectures', {}).get(backbone_type, {})
    fusion_config = model_config.get('embedding', {})
    binary_config = model_config.get('binary_encoding', {})
    
    # Create model
    model = CUBICModel(
        input_dim=input_dim,
        n_stocks=n_stocks,
        backbone_type=backbone_type,
        backbone_config=backbone_config,
        fusion_config=fusion_config,
        binary_config=binary_config
    )
    
    logger.info(f"Model created: {model.get_model_info()}")
    return model


def train_model(model: CUBICModel, data_info: dict, config_path: str) -> CUBICTrainer:
    """
    Train CUBIC model
    
    Args:
        model: CUBIC model to train
        data_info: Data information dictionary
        config_path: Path to configuration file
        
    Returns:
        Trained model trainer
    """
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = CUBICTrainer(model, config_path)
    
    # Get dataloaders
    train_loader = data_info['dataloaders']['train']
    val_loader = data_info['dataloaders']['val']
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    logger.info("Model training completed")
    logger.info(f"Training summary: {trainer.get_training_summary()}")
    
    return trainer


def evaluate_model(model: CUBICModel, data_info: dict, config_path: str) -> dict:
    """
    Evaluate CUBIC model
    
    Args:
        model: Trained CUBIC model
        data_info: Data information dictionary
        config_path: Path to configuration file
        
    Returns:
        Evaluation results
    """
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = CUBICEvaluator(model, config_path)
    
    # Get test dataloader
    test_loader = data_info['dataloaders']['test']
    
    # Evaluate model
    results = evaluator.evaluate_model(test_loader)
    
    logger.info("Model evaluation completed")
    logger.info("Basic Metrics:")
    for metric, value in results['basic_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Trading Metrics:")
    for metric, value in results['trading_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='CUBIC Framework Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--index', type=str, default='SPX',
                       choices=['SPX', 'HSI', 'SX5E'],
                       help='Index to use for experiment')
    parser.add_argument('--backbone', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'mlp'],
                       help='Backbone architecture to use')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['data', 'train', 'eval', 'full'],
                       help='Execution mode')
    
    args = parser.parse_args()
    
    try:
        # Setup experiment
        experiment_setup = setup_experiment(args.config, args.index, args.backbone)
        
        if args.mode in ['data', 'full']:
            # Prepare data
            data_info = prepare_data(experiment_setup)
        
        if args.mode in ['train', 'full']:
            # Create and train model
            model = create_model(experiment_setup, data_info)
            trainer = train_model(model, data_info, args.config)
        
        if args.mode in ['eval', 'full']:
            # Evaluate model
            if args.mode == 'eval':
                # Load trained model for evaluation only
                model = create_model(experiment_setup, data_info)
                # TODO: Load trained weights
            
            results = evaluate_model(model, data_info, args.config)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
