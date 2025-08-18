"""
Training loop implementation for Delphyne model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import os
from typing import Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import logging

from ..config import DelphyneConfig, TrainingConfig
from ..model.delphyne import DelphyneModel
from .utils import (
    create_optimizer, create_scheduler, compute_metrics, 
    compute_probabilistic_metrics, EarlyStopping, save_checkpoint, load_checkpoint
)


class DelphyneTrainer:
    """
    Trainer class for Delphyne model following the paper specifications.
    
    Implements:
    - Mixed precision training (bf16)
    - AdamW optimizer with cosine annealing
    - Linear warmup
    - Gradient accumulation
    - Checkpointing and logging
    """
    
    def __init__(
        self,
        model: DelphyneModel,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: Delphyne model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            device: Device to train on
            logger: Logger for training progress
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup device
        if device is None:
            self.device = torch.device(config.device)
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup logger
        if logger is None:
            logging.basicConfig(level=getattr(logging, config.log_level))
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_train_steps
        )
        
        # Mixed precision training
        self.use_mixed_precision = config.use_mixed_precision
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Early stopping
        if val_dataloader is not None:
            self.early_stopping = EarlyStopping(patience=10, mode='min')
        else:
            self.early_stopping = None
        
        self.logger.info(f"Trainer initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.epoch}",
            disable=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Validation
            if (self.val_dataloader is not None and 
                self.global_step % self.config.eval_steps == 0 and 
                self.global_step > 0):
                val_metrics = self.validate()
                self.logger.info(f"Validation metrics: {val_metrics}")
                
                # Early stopping check
                if self.early_stopping is not None:
                    if self.early_stopping(val_metrics['loss']):
                        self.logger.info("Early stopping triggered")
                        return {'loss': total_loss / num_batches}
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        return {'loss': total_loss / num_batches}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Extract data from batch
        time_series = batch['time_series']
        variate_ids = batch.get('variate_ids', None)
        missing_mask = batch.get('missing_mask', None)
        forecast_mask = batch.get('forecast_mask', None)
        
        # Use time_series as targets for self-supervised learning
        targets = time_series.clone()
        
        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    time_series=time_series,
                    variate_ids=variate_ids,
                    missing_mask=missing_mask,
                    forecast_mask=forecast_mask,
                    targets=targets,
                    return_dict=True
                )
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        else:
            # Standard precision training
            outputs = self.model(
                time_series=time_series,
                variate_ids=variate_ids,
                missing_mask=missing_mask,
                forecast_mask=forecast_mask,
                targets=targets,
                return_dict=True
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        self.global_step += 1
        return loss.item()
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                time_series = batch['time_series']
                variate_ids = batch.get('variate_ids', None)
                missing_mask = batch.get('missing_mask', None)
                forecast_mask = batch.get('forecast_mask', None)
                targets = time_series.clone()
                
                outputs = self.model(
                    time_series=time_series,
                    variate_ids=variate_ids,
                    missing_mask=missing_mask,
                    forecast_mask=forecast_mask,
                    targets=targets,
                    return_dict=True
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Compute additional metrics
                distribution = outputs['distribution']
                batch_metrics = compute_probabilistic_metrics(
                    distribution, targets, forecast_mask
                )
                all_metrics.append(batch_metrics)
        
        # Aggregate metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        avg_metrics['loss'] = total_loss / len(self.val_dataloader)
        
        self.model.train()
        return avg_metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        try:
            while self.global_step < self.config.num_train_steps:
                # Train epoch
                train_metrics = self.train_epoch()
                training_history['train_loss'].append(train_metrics['loss'])
                training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                self.logger.info(
                    f"Epoch {self.epoch} completed. "
                    f"Train loss: {train_metrics['loss']:.4f}"
                )
                
                # Final validation
                if self.val_dataloader is not None:
                    val_metrics = self.validate()
                    training_history['val_loss'].append(val_metrics['loss'])
                    self.logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
                
                self.epoch += 1
                
                # Check early stopping
                if self.early_stopping is not None and self.early_stopping.early_stop:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_history': training_history,
            'final_step': self.global_step,
            'training_time': training_time
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        filepath = os.path.join(os.getcwd(), filename)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=self.best_val_loss,
            filepath=filepath,
            config=self.config
        )
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = load_checkpoint(
            filepath=filepath,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('step', 0)
        self.best_val_loss = checkpoint.get('loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        self.logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")
