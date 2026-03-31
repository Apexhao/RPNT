"""
Professional Downstream Trainers for Neural Foundation Model Fine-tuning
-----------------------------------------------------------------------

This module provides comprehensive trainers for downstream tasks (regression/classification)
using pretrained foundation models with frozen or fine-tuned encoders.

Key Features:
- RegressionTrainer: For velocity/position prediction tasks
- ClassificationTrainer: For direction/action classification tasks
- Professional training loops with comprehensive logging
- Checkpoint management and model evaluation
- Integration with SingleSiteDownstreamDataset
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchmetrics

# Import our modules
from ..models import CrossSiteFoundationMAE
from ..models.downstream import SingleSiteDownstreamRegressor, SingleSiteClassifier
from ..data.downstream_dataset import SingleSiteDownstreamDataset
from ..utils.helpers import load_config, set_seed
from .enhanced_logger import EnhancedLogger


class BaseDownstreamTrainer:
    """
    Base trainer class for downstream tasks with common functionality.
    
    Features:
    - Pretrained model loading and encoder extraction
    - Professional logging and checkpointing
    - Common training utilities and metrics
    - Flexible training modes (frozen/fine-tuned encoders)
    """
    
    def __init__(self,
                 model: nn.Module,
                 dataset: SingleSiteDownstreamDataset,
                 config: Dict[str, Any],
                 task_type: str):
        """
        Initialize base downstream trainer.
        
        Args:
            model: Downstream model (regressor or classifier)
            dataset: SingleSiteDownstreamDataset instance
            config: Training configuration dictionary
            task_type: 'regression' or 'classification'
        """
        self.config = config
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = model.to(self.device)
        self.dataset = dataset
        
        # Extract configuration sections
        self.training_config = config.get('training', config.get('finetune', {}))
        self.paths_config = config.get('paths', {})
        
        # Initialize enhanced logging
        self._setup_enhanced_logging()
        
        # Initialize training components
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._setup_loss_function()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Log initialization
        self.enhanced_logger.logger.info(f"BaseDownstreamTrainer initialized for {task_type}")
        self._log_model_info()
        self._log_training_config()
        
        # Save config and model summary to files
        self._save_config_and_model_summary()
    
    def _setup_enhanced_logging(self):
        """Setup enhanced logging system."""
        experiment_name = self.paths_config.get('experiment_name', f"{self.task_type}_default")
        base_dir = self.paths_config.get('base_dir', './logs')
        
        # Initialize enhanced logger
        self.enhanced_logger = EnhancedLogger(
            experiment_name=experiment_name,
            base_dir=base_dir,
            top_k_checkpoints=3,
            local_rank=0
        )
        
        # Update paths config
        enhanced_paths = self.enhanced_logger.get_paths()
        self.paths_config.update(enhanced_paths)
        
        # Create legacy aliases
        self.logger = self.enhanced_logger.logger
        self.writer = self.enhanced_logger.writer
    
    def _setup_data_loaders(self):
        """Setup train/validation/test data loaders."""
        batch_size = self.training_config.get('batch_size', 32)
        
        # Determine output mode based on task type
        output_mode = 'regression' if self.task_type == 'regression' else 'classification'
        
        # Create data loaders
        self.train_loader = self.dataset.create_dataloader(
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            output_mode=output_mode
        )
        
        self.val_loader = self.dataset.create_dataloader(
            split='val',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            output_mode=output_mode
        )
        
        self.test_loader = self.dataset.create_dataloader(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            output_mode=output_mode
        )
        
        self.logger.info(f"Data loaders created - Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)} batches")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.get('learning_rate', 1e-3),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        decay_epochs = self.training_config.get('decay_epochs', 50)
        decay_rate = self.training_config.get('decay_rate', 0.9)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_epochs,
            gamma=decay_rate
        )
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        
        self.logger.info(f"Optimizer initialized - Total params: {total_params:,}, "
                        f"Trainable: {trainable_params_count:,} "
                        f"({100*trainable_params_count/total_params:.1f}%)")
    
    def _setup_loss_function(self):
        """Setup loss function (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _setup_loss_function")
    
    def _log_model_info(self):
        """Log detailed model information."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        model_info = {
            'task_type': self.task_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0,
            'device': str(self.device),
            'dataset_id': self.dataset.dataset_id,
            'training_mode': getattr(self.model, 'training_mode', 'unknown')
        }
        
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_text('Model/Info', 
                               f"Task: {self.task_type}\n"
                               f"Total Parameters: {total_params:,}\n"
                               f"Trainable Parameters: {trainable_params:,}\n"
                               f"Frozen Parameters: {frozen_params:,}\n"
                               f"Trainable Ratio: {100*model_info['trainable_ratio']:.1f}%\n"
                               f"Dataset: {self.dataset.dataset_id}")
    
    def _log_training_config(self):
        """Log training configuration."""
        config_str = json.dumps(self.training_config, indent=2)
        self.logger.info(f"Training Config: {config_str}")
        
        if self.writer:
            self.writer.add_text('Training/Config', config_str.replace('\n', '<br>'))
    
    def _save_config_and_model_summary(self):
        """Save config and model summary to files for comprehensive logging."""
        try:
            # Use the comprehensive log_training_start method
            self.enhanced_logger.log_training_start(self.config, self.model)
            
        except Exception as e:
            self.logger.warning(f"Failed to save config and model summary: {e}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': metrics,
            'task_type': self.task_type,
            'dataset_id': self.dataset.dataset_id,
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        }
        
        # Use enhanced logger's checkpoint manager
        saved_paths = self.enhanced_logger.save_checkpoint(checkpoint, epoch, val_loss)
        
        return saved_paths
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            self.logger.info(f"Checkpoint loaded: {filepath}")
            self.logger.info(f"Resuming from epoch {self.epoch}, step {self.step}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return False


class RegressionTrainer(BaseDownstreamTrainer):
    """
    Professional trainer for regression tasks (velocity/position prediction).
    
    Features:
    - MSE loss with optional L1 regularization
    - R² score and MAE metrics
    - Position and velocity prediction support
    - Comprehensive evaluation and logging
    """
    
    def __init__(self,
                 pretrained_temporal_encoder,
                 dataset: SingleSiteDownstreamDataset,
                 config: Dict[str, Any]):
        """
        Initialize RegressionTrainer.
        
        Args:
            pretrained_temporal_encoder: Pretrained temporal encoder from foundation model
            dataset: SingleSiteDownstreamDataset instance
            config: Training configuration dictionary
        """
        
        # Create regression model
        training_mode = config.get('training', config.get('finetune', {})).get('training_mode', 'frozen_encoder')
        output_dim = config.get('training', config.get('finetune', {})).get('output_dim', 2)
        prediction_layers = config.get('model', {}).get('prediction_layers', 2)
        dropout = config.get('model', {}).get('dropout', 0.1)
        
        model = SingleSiteDownstreamRegressor(
            pretrained_temporal_encoder=pretrained_temporal_encoder,
            output_dim=output_dim,
            training_mode=training_mode,
            prediction_layers=prediction_layers,
            dropout=dropout
        )
        
        super().__init__(model, dataset, config, 'regression')
        
        # Regression-specific configuration
        self.target_type = self.training_config.get('target_type', 'velocity')  # 'velocity' or 'position'
        self.use_l1_reg = self.training_config.get('use_l1_regularization', False)
        self.l1_weight = self.training_config.get('l1_weight', 0.01)
        
        self.logger.info(f"RegressionTrainer initialized - Target: {self.target_type}")
    
    def _setup_loss_function(self):
        """Setup MSE loss function."""
        self.criterion = nn.MSELoss()
        
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression metrics.
        Args:
            predictions: [B, T, 2] - predicted values
            targets: [B, T, 2] - target values
            
        Returns:
            Dictionary of metrics
        """
    
        r2_metric_x = torchmetrics.R2Score()
        r2_metric_y = torchmetrics.R2Score()
        
        # Update R2 metrics for each dimension separately
            # For each sample in the batch, update metrics across all timepoints
        for b in range(predictions.shape[0]):
                # Get predictions and targets for one sample across all time points
                # [time, 2] for both predictions and targets
                sample_pred = predictions[b]
                sample_target = targets[b]
                
                # Update separate metrics for x and y coordinates
                r2_metric_x.update(sample_pred[:, 0], sample_target[:, 0])
                r2_metric_y.update(sample_pred[:, 1], sample_target[:, 1])
        
        # Compute final metrics from accumulated state
        r2_x = r2_metric_x.compute().item()
        r2_y = r2_metric_y.compute().item()
        
        # Average metrics
        epoch_r2 = (r2_x + r2_y) / 2
        
        metrics = {
            'r2_x': r2_x,
            'r2_y': r2_y,
            'r2_mean': epoch_r2
        }   
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        epoch_stats = defaultdict(float)
        num_batches = len(self.train_loader)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            neural_data = batch_data[0].to(self.device)  # [B, 1, T, N]
            trajectories = batch_data[1].to(self.device)  # [B, T, 2]
            velocities = batch_data[2].to(self.device)   # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Get site coordinates
            B = neural_data.size(0)
            coords_3d = self.dataset.get_site_coordinates_batch(B).to(self.device)  # [B, 1, T, 2]
            
            # Forward pass
            predictions = self.model(neural_data.squeeze(1), coords_3d.squeeze(1))  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Add L1 regularization if enabled
            if self.use_l1_reg:
                l1_loss = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
                loss = loss + self.l1_weight * l1_loss
                epoch_stats['l1_loss'] += l1_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(predictions, targets)
            
            # Update statistics
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
                      
            self.step += 1
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    @torch.no_grad()
    def validate_epoch(self, split: str = 'val') -> Dict[str, float]:
        """Run one validation/test epoch."""
        self.model.eval()
        
        loader = self.val_loader if split == 'val' else self.test_loader
        epoch_stats = defaultdict(float)
        num_batches = len(loader)
        
        for batch_data in loader:
            neural_data = batch_data[0].to(self.device)  # [B, 1, T, N]
            trajectories = batch_data[1].to(self.device)  # [B, T, 2]
            velocities = batch_data[2].to(self.device)   # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Get site coordinates
            B = neural_data.size(0)
            coords_3d = self.dataset.get_site_coordinates_batch(B).to(self.device)  # [B, 1, T, 2]
            
            # Forward pass
            predictions = self.model(neural_data.squeeze(1), coords_3d.squeeze(1))  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Compute metrics
            metrics = self.compute_metrics(predictions, targets)
            
            # Update statistics
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    def train(self, num_epochs: int):
        """Main training loop."""
        self.logger.info(f"Starting regression training for {num_epochs} epochs")
        self.logger.info(f"Target type: {self.target_type}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch('val')
            test_stats = self.validate_epoch('test')
            
            # Learning rate step
            self.scheduler.step()
            
            # Enhanced logging
            self.enhanced_logger.log_epoch_stats(epoch, train_stats, val_stats, test_stats)
            
            # Checkpointing
            is_best = val_stats['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            metrics = {
                'train_r2': train_stats['r2_mean'],
                'val_r2': val_stats['r2_mean'],
                'test_r2': test_stats['r2_mean'],
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
                'test_loss': test_stats['loss']
            }
            
            saved_paths = self.save_checkpoint(epoch, val_stats['loss'], metrics, is_best)
            
            if is_best:
                self.logger.info(f"New best model saved (val_R²: {val_stats['r2_mean']:.4f})")
            
            # Early stopping
            early_stopping_patience = self.training_config.get('early_stopping_patience', 50)
            if self.epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completion
        self.enhanced_logger.log_training_end()
        self.enhanced_logger.close()


def load_pretrained_foundation_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load pretrained foundation model from checkpoint.
    
    Args:
        checkpoint_path: Path to pretrained model checkpoint
        device: Device to load model on
        
    Returns:
        Pretrained foundation model
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint['config']
    model_config = config['model']
    
    # Create foundation model
    from ..models import CrossSiteModelFactory
    
    model = CrossSiteModelFactory.create_mae_model(
        size=config['training']['model_size'],
        **model_config
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logging.info(f"Loaded pretrained model from {checkpoint_path}")
    logging.info(f"Model size: {config['training']['model_size']}")
    
    return model


def create_downstream_trainer(task_type: str,
                             pretrained_checkpoint_path: str,
                             dataset: SingleSiteDownstreamDataset,
                             config: Dict[str, Any]):
    """
    Factory function to create downstream trainers.
    
    Args:
        task_type: 'regression'
        pretrained_checkpoint_path: Path to pretrained foundation model
        dataset: SingleSiteDownstreamDataset instance
        config: Training configuration
        
    Returns:
        Configured trainer instance
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained foundation model
    foundation_model = load_pretrained_foundation_model(pretrained_checkpoint_path, device)
    
    # Extract temporal encoder
    pretrained_temporal_encoder = foundation_model.temporal_encoder
    
    # Create trainer based on task type
    if task_type == 'regression':
        trainer = RegressionTrainer(
            pretrained_temporal_encoder=pretrained_temporal_encoder,
            dataset=dataset,
            config=config
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'regression'")
    
    return trainer 