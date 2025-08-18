"""
Training script for Delphyne model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import os
from datetime import datetime

from delphyne import DelphyneModel, DelphyneConfig, TrainingConfig
from delphyne.data import SyntheticDataset
from delphyne.training import DelphyneTrainer


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    # Create handlers with proper encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    file_handler = logging.FileHandler(
        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_datasets(config: TrainingConfig) -> tuple:
    """Create training and validation datasets."""
    # Training dataset
    train_dataset = SyntheticDataset(
        data_type="wavelet",
        num_samples=1000,
        seq_len=128,
        num_variates=1,
        correlated=False,
        forecast_length=32,
        missing_prob=0.1,
        seed=42
    )
    
    # Validation dataset
    val_dataset = SyntheticDataset(
        data_type="wavelet", 
        num_samples=200,
        seq_len=128,
        num_variates=1,
        correlated=False,
        forecast_length=32,
        missing_prob=0.1,
        seed=123
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_dataloader, val_dataloader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Delphyne model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Delphyne training")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create model configuration (smaller for testing)
    model_config = DelphyneConfig(
        num_layers=4,           # Smaller for faster training
        hidden_size=256,        # Smaller for faster training
        num_attention_heads=8,  # Smaller for faster training
        intermediate_size=1024, # Smaller for faster training
        patch_size=16,          # Smaller patches
        max_sequence_length=512, # Shorter sequences
        context_length=32,
        num_mixture_components=2,
        dropout_prob=0.1
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_train_steps=args.num_epochs * 100,  # Approximate steps per epoch
        warmup_steps=100,
        save_steps=200,
        eval_steps=100,
        logging_steps=50,
        use_mixed_precision=args.mixed_precision,
        device=str(device)
    )
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Create model
    model = DelphyneModel(model_config)
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Create datasets
    train_dataloader, val_dataloader = create_datasets(training_config)
    logger.info(f"Training batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")
    
    # Create trainer
    trainer = DelphyneTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        logger=logger
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Resumed training from {args.checkpoint}")
    
    # Train model
    try:
        results = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Final step: {results['final_step']}")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")
        
        # Print training history
        history = results['training_history']
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Test model inference
    logger.info("Testing model inference...")
    model.eval()
    
    # Get a sample batch
    sample_batch = next(iter(val_dataloader))
    sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in sample_batch.items()}
    
    with torch.no_grad():
        # Test forward pass
        outputs = model(
            time_series=sample_batch['time_series'],
            variate_ids=sample_batch['variate_ids'],
            forecast_mask=sample_batch['forecast_mask'],
            return_dict=True
        )
        
        logger.info(f"Inference successful! Distribution type: {type(outputs['distribution'])}")
        
        # Test forecast generation
        forecasts = model.generate_forecasts(
            time_series=sample_batch['time_series'],
            forecast_length=16,
            num_samples=10
        )
        
        logger.info(f"Forecast generation successful! Shape: {forecasts['samples'].shape}")
    
    logger.info("All tests passed! Success!")


if __name__ == "__main__":
    main()
