"""
Training pipeline for CUBIC framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import json
from datetime import datetime

from ..models.cubic_model import CUBICModel
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CUBICTrainer:
    """
    Trainer for CUBIC models
    """
    
    def __init__(self, model: CUBICModel, config_path: str = "config.yaml"):
        """
        Initialize CUBIC Trainer
        
        Args:
            model: CUBIC model to train
            config_path: Path to configuration file
        """
        self.model = model
        self.config = ConfigManager(config_path)
        self.training_config = self.config.get('training', {})
        
        # Training parameters
        self.learning_rate = float(self.training_config.get('learning_rate', 0.001))
        self.weight_decay = float(self.training_config.get('weight_decay', 1e-5))
        self.num_epochs = int(self.training_config.get('num_epochs', 100))
        self.early_stopping_patience = int(self.training_config.get('early_stopping_patience', 10))
        self.confidence_type = self.training_config.get('confidence_type', 'mean')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # Output directories
        self.model_dir = self.config.get('output.model_dir', 'models')
        self.results_dir = self.config.get('output.results_dir', 'results')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_conf_loss = 0.0
        total_recon_error = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            total_loss_batch, loss_components = self.model.calculate_loss(
                features, targets, self.confidence_type
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_ce_loss += loss_components['cross_entropy'].item()
            total_conf_loss += loss_components['confidence_regularization'].item()
            total_recon_error += loss_components['reconstruction_error'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss_batch.item():.4f}",
                'CE': f"{loss_components['cross_entropy'].item():.4f}",
                'Conf': f"{loss_components['confidence_regularization'].item():.4f}"
            })
        
        # Calculate average metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            'cross_entropy_loss': total_ce_loss / num_batches,
            'confidence_loss': total_conf_loss / num_batches,
            'reconstruction_error': total_recon_error / num_batches
        }
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_conf_loss = 0.0
        total_recon_error = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation", leave=False):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass and loss calculation
                total_loss_batch, loss_components = self.model.calculate_loss(
                    features, targets, self.confidence_type
                )
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                total_ce_loss += loss_components['cross_entropy'].item()
                total_conf_loss += loss_components['confidence_regularization'].item()
                total_recon_error += loss_components['reconstruction_error'].item()
                num_batches += 1
        
        # Calculate average metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            'cross_entropy_loss': total_ce_loss / num_batches,
            'confidence_loss': total_conf_loss / num_batches,
            'reconstruction_error': total_recon_error / num_batches
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.val_history.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val Loss: {val_metrics['total_loss']:.4f}, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")
        
        # Save training history
        self.save_training_history()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_config': self.model.get_model_info()
        }
        
        if is_best:
            filepath = os.path.join(self.model_dir, 'best_model.pth')
            logger.info(f"Saving best model to {filepath}")
        else:
            filepath = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Epoch number
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {epoch + 1}")
        
        return epoch
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'training_config': self.training_config,
            'model_info': self.model.get_model_info()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f'training_history_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {filepath}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        """Set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f"Learning rate set to {lr}")
    
    def get_training_summary(self) -> Dict:
        """Get training summary"""
        if not self.train_history or not self.val_history:
            return {}
        
        return {
            'total_epochs': len(self.train_history),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_history[-1]['total_loss'],
            'final_val_loss': self.val_history[-1]['total_loss'],
            'final_lr': self.get_learning_rate()
        }
