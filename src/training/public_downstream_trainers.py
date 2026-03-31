"""
Professional Public Dataset Downstream Trainers for Neural Foundation Model Fine-tuning
-----------------------------------------------------------------------------------

This module provides comprehensive trainers for downstream regression tasks using pretrained 
temporal-only foundation models on public dataset with session coordinates.

Key Features:
- PublicRegressionTrainer: For velocity/position prediction tasks with session coordinates
- Professional training loops with comprehensive logging and session-level evaluation
- TensorBoard integration for per-session monitoring during training
- Checkpoint management and model evaluation
- Integration with public downstream datasets and RoPE4D session coordinates
- Support for frozen and fine-tuned encoder modes
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
from ..models.public_transformer import PublicNeuralFoundationMAE
from ..models.public_downstream import PublicVelocityPredictor
from ..data.public_downstream_dataset import (
    PublicCrossSessionDataset, 
    PublicCrossSubjectCenterDataset, 
    PublicCrossSubjectRandomDataset,
    Public_No_T_Subject_Dataset,
    Public_Only_RT_Subject_Dataset,
    Public_Only_CO_Subject_Dataset,
)
from ..utils.helpers import load_config, set_seed
from .enhanced_logger import EnhancedLogger


class BasePublicDownstreamTrainer:
    """
    Base trainer class for public dataset downstream regression tasks with common functionality.
    
    Features:
    - Pretrained temporal-only model loading and encoder extraction
    - Professional logging and checkpointing with session-level evaluation
    - TensorBoard integration for per-session monitoring during training
    - Common training utilities and regression metrics
    - Session coordinate handling for RoPE4D
    - Flexible training modes (frozen/fine-tuned encoders)
    """
    
    def __init__(self,
                 model: nn.Module,
                 dataset,  # Public downstream dataset
                 config: Dict[str, Any],
                 foundation_config: Dict[str, Any]):
        """
        Initialize base public downstream trainer.
        
        Args:
            model: Public downstream regression model (velocity predictor)
            dataset: Public downstream dataset instance
            config: Downstream task configuration (training settings only)
            foundation_config: Original foundation model configuration (architecture)
        """
        self.downstream_config = config  # Task-specific settings
        self.foundation_config = foundation_config  # Foundation model architecture
        self.task_type = 'regression'  # Only regression supported for public dataset
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
        
        # Log initialization and save config/model summary
        self.enhanced_logger.logger.info(f"BasePublicDownstreamTrainer initialized for {self.task_type}")
        self._log_model_info()
        self._log_training_config()
        
        # Save configuration and model summary using enhanced logger
        self.enhanced_logger.log_training_start(config, self.model)
    
    def _setup_enhanced_logging(self):
        """Setup enhanced logging system."""
        experiment_name = self.paths_config.get('experiment_name', f"public_{self.task_type}_default")
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
        
        # Only regression is supported for public dataset
        output_mode = 'regression'
        
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
            'model_type': 'Public' + self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0,
            'device': str(self.device),
            'dataset_type': self.dataset.__class__.__name__,
            'training_mode': getattr(self.model, 'training_mode', 'unknown'),
            'architecture': 'temporal_only_with_session_coordinates'
        }
        
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_text('Model/Info', 
                               f"Task: {self.task_type}\n"
                               f"Architecture: Temporal-Only + Session Coordinates\n"
                               f"Total Parameters: {total_params:,}\n"
                               f"Trainable Parameters: {trainable_params:,}\n"
                               f"Frozen Parameters: {frozen_params:,}\n"
                               f"Trainable Ratio: {100*model_info['trainable_ratio']:.1f}%\n"
                               f"Dataset: {self.dataset.__class__.__name__}")
    
    def _log_training_config(self):
        """Log training configuration."""
        config_str = json.dumps(self.training_config, indent=2)
        self.logger.info(f"Training Config: {config_str}")
        
        if self.writer:
            self.writer.add_text('Training/Config', config_str.replace('\n', '<br>'))
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with clean config separation."""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'prediction_head_state_dict': self.model.prediction_head.state_dict(),
            'temporal_encoder_state_dict': self.model.temporal_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            
            # CLEAN SEPARATION: Save both configs separately
            'foundation_config': self.foundation_config,  # Original model architecture
            'downstream_config': self.downstream_config,  # Task training settings
            
            'metrics': metrics,
            'task_type': self.task_type,
            'dataset_type': self.dataset.__class__.__name__,
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


class PublicRegressionTrainer(BasePublicDownstreamTrainer):
    """
    Professional trainer for public dataset regression tasks velocity).
    
    Features:
    - MSE loss with optional L1 regularization
    - R² score metrics from torchmetrics
    - Position and velocity prediction support  
    - Session coordinate handling for RoPE4D
    - Comprehensive evaluation and logging
    """
    
    def __init__(self,
                 pretrained_foundation_model,
                 dataset,  # Public downstream dataset
                 config: Dict[str, Any],
                 foundation_config: Dict[str, Any]):
        """
        Initialize PublicRegressionTrainer.
        
        Args:
            pretrained_foundation_model: Pretrained PublicNeuralFoundationMAE model
            dataset: Public downstream dataset instance
            config: Downstream task configuration (training settings only)
            foundation_config: Original foundation model configuration (architecture)
        """
        
        # Create public regression model
        training_mode = config.get('training', config.get('finetune', {})).get('training_mode', 'frozen_encoder')
        
        model = PublicVelocityPredictor(
            pretrained_model=pretrained_foundation_model,
            training_mode=training_mode
        )
        
        super().__init__(model, dataset, config, foundation_config)
        
        # Regression-specific configuration
        self.target_type = self.training_config.get('target_type', 'velocity')  # 'velocity' or 'position'
        self.use_l1_reg = self.training_config.get('use_l1_regularization', False)
        self.l1_weight = self.training_config.get('l1_weight', 0.01)
        
        self.logger.info(f"PublicRegressionTrainer initialized - Target: {self.target_type}")
    
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
            neural_data = batch_data[0].to(self.device)      # [B, 1, T, N]
            session_coords = batch_data[1].to(self.device)   # [B, 1, 3] - session coordinates
            trajectories = batch_data[2].to(self.device)     # [B, T, 2]
            velocities = batch_data[3].to(self.device)       # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Forward pass
            predictions = self.model(neural_data, session_coords)  # [B, T, 2]
            
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
        
        # Rename 'loss' to 'total_loss' for enhanced logger compatibility
        if 'loss' in epoch_stats:
            epoch_stats['total_loss'] = epoch_stats['loss']
        
        # Add current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_stats['learning_rate'] = current_lr
        
        return dict(epoch_stats)
    
    @torch.no_grad()
    def validate_epoch(self, split: str = 'val') -> Dict[str, float]:
        """Run one validation/test epoch."""
        self.model.eval()
        
        loader = self.val_loader if split == 'val' else self.test_loader
        epoch_stats = defaultdict(float)
        num_batches = len(loader)
        
        for batch_idx, batch_data in enumerate(loader):
            neural_data = batch_data[0].to(self.device)      # [B, 1, T, N]
            session_coords = batch_data[1].to(self.device)   # [B, 1, 3]
            trajectories = batch_data[2].to(self.device)     # [B, T, 2]
            velocities = batch_data[3].to(self.device)       # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Forward pass
            predictions = self.model(neural_data, session_coords)  # [B, T, 2]
            
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
        
        # Rename 'loss' to 'total_loss' for enhanced logger compatibility
        if 'loss' in epoch_stats:
            epoch_stats['total_loss'] = epoch_stats['loss']
        
        # Add current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_stats['learning_rate'] = current_lr
        
        return dict(epoch_stats)
    
    def evaluate_per_session(self, split: str = 'test') -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on each session individually.
        
        Args:
            split: Data split to evaluate ('train', 'val', 'test')
            
        Returns:
            Dictionary with per-session metrics: {session_id: metrics}
        """
        self.model.eval()
        
        # Get session IDs from dataset
        try:
            session_ids = self.dataset.get_session_ids()
        except AttributeError:
            session_ids = getattr(self.dataset, 'session_ids', [])
        
        if not session_ids:
            self.logger.warning("No session IDs found in dataset, using aggregated evaluation")
            # Fallback to aggregated evaluation
            aggregated_stats = self.validate_epoch(split)
            return {'aggregated': aggregated_stats}
        
        session_results = {}
        
        with torch.no_grad():
            for session_id in session_ids:
                try:
                    # Get session-specific data
                    session_data = self.dataset.get_session_data(session_id, split)
                    
                    # Evaluate this session
                    session_stats = self._evaluate_single_session(session_id, session_data)
                    session_results[session_id] = session_stats
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate session {session_id}: {e}")
                    continue
        
        return session_results
    
    def _evaluate_single_session(self, session_id: str, session_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run model inference on a single session's data.
        
        Args:
            session_id: Session identifier
            session_data: Session-specific neural/behavioral data
            
        Returns:
            Dictionary with session metrics: {'r2_mean': 0.85, 'mae': 0.12, ...}
        """
        # Move data to device
        neural_data = session_data['neural_data'].to(self.device)      # [Session_Windows, 1, T, N]
        session_coords = session_data['coordinates'].to(self.device)   # [Session_Windows, 1, 3]
        trajectories = session_data['trajectories'].to(self.device)    # [Session_Windows, T, 2]
        velocities = session_data['velocities'].to(self.device)        # [Session_Windows, T, 2]
        
        # Choose target based on target_type
        targets = velocities if self.target_type == 'velocity' else trajectories
        
        n_windows = neural_data.shape[0]
        if n_windows == 0:
            self.logger.warning(f"Session {session_id} has no data windows")
            return {'r2_mean': 0.0, 'mae': float('inf'), 'mse': float('inf'), 'loss': float('inf')}
        
        # Forward pass through model
        predictions = self.model(neural_data, session_coords)  # [Session_Windows, T, 2]
        
        # Compute loss
        loss = self.criterion(predictions, targets)
        
        # Compute detailed metrics
        metrics = self.compute_metrics(predictions, targets)
        metrics['loss'] = loss.item()
        metrics['mse'] = loss.item()  # MSE is the same as our loss
        
        return metrics
    
    def compute_aggregated_session_metrics(self, session_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute aggregated metrics across all sessions.
        
        Args:
            session_results: Per-session results from evaluate_per_session()
            
        Returns:
            Aggregated metrics with mean, std, min, max across sessions
        """
        if not session_results:
            return {}
        
        # Get all metric names from first session
        metric_names = list(list(session_results.values())[0].keys())
        aggregated = {}
        
        for metric in metric_names:
            values = [session_data[metric] for session_data in session_results.values() 
                     if metric in session_data]
            
            if values:
                aggregated[f'{metric}_session_mean'] = sum(values) / len(values)
                aggregated[f'{metric}_session_std'] = (sum((x - aggregated[f'{metric}_session_mean'])**2 
                                                         for x in values) / len(values))**0.5 if len(values) > 1 else 0.0
                aggregated[f'{metric}_session_min'] = min(values)
                aggregated[f'{metric}_session_max'] = max(values)
                aggregated[f'{metric}_session_count'] = len(values)
        
        return aggregated
    
    def log_session_results_to_tensorboard(self, session_results: Dict[str, Dict[str, float]], 
                                          aggregated_results: Dict[str, float], epoch: int):
        """Log per-session and aggregated results to TensorBoard for easy monitoring."""
        
        if not self.writer:
            return
        
        # Log aggregated metrics across sessions
        if aggregated_results:
            self.writer.add_scalar('Session_Metrics/R2_Mean_Across_Sessions', 
                                 aggregated_results.get('r2_mean_session_mean', 0), epoch)
            self.writer.add_scalar('Session_Metrics/R2_Std_Across_Sessions', 
                                 aggregated_results.get('r2_mean_session_std', 0), epoch)
            self.writer.add_scalar('Session_Metrics/R2_Min_Across_Sessions', 
                                 aggregated_results.get('r2_mean_session_min', 0), epoch)
            self.writer.add_scalar('Session_Metrics/R2_Max_Across_Sessions', 
                                 aggregated_results.get('r2_mean_session_max', 0), epoch)
        
        # Log individual session performance
        for session_id, metrics in session_results.items():
            # Clean session ID for TensorBoard (replace special characters)
            clean_session_id = session_id.replace('/', '_').replace('\\', '_').replace('.', '_')
            
            self.writer.add_scalar(f'Individual_Sessions/R2_{clean_session_id}', 
                                 metrics.get('r2_mean', 0), epoch)
        
        # Log session count
        self.writer.add_scalar('Session_Metrics/Number_of_Sessions', len(session_results), epoch)
        
        self.logger.info(f"📊 Session-level metrics logged to TensorBoard for epoch {epoch}")

    def save_session_evaluation_results(self, session_results: Dict[str, Dict[str, float]], 
                                      aggregated_results: Dict[str, float]):
        """Save detailed session evaluation results to file."""
        
        results_dict = {
            'dataset_type': self.dataset.__class__.__name__,
            'task_type': self.task_type,
            'target_type': self.target_type,
            'training_mode': getattr(self.model, 'training_mode', 'unknown'),
            'num_sessions': len(session_results),
            'session_ids': list(session_results.keys()),
            'per_session_results': session_results,
            'aggregated_results': aggregated_results,
            'foundation_config': self.foundation_config,
            'downstream_config': self.downstream_config
        }
        
        # Save to enhanced logger's experiment directory
        results_path = Path(self.paths_config['checkpoint_dir']) / 'session_evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Session evaluation results saved to: {results_path}")
        
        return results_path
    
    def print_session_evaluation_summary(self, session_results: Dict[str, Dict[str, float]], 
                                        aggregated_results: Dict[str, float]):
        """Print comprehensive session evaluation summary."""
        
        self.logger.info(f"\n📊 SESSION-LEVEL EVALUATION SUMMARY")
        self.logger.info("="*60)
        
        # Aggregated metrics
        r2_mean = aggregated_results.get('r2_mean_session_mean', 0)
        r2_std = aggregated_results.get('r2_mean_session_std', 0)
        r2_min = aggregated_results.get('r2_mean_session_min', 0)
        r2_max = aggregated_results.get('r2_mean_session_max', 0)
        num_sessions = aggregated_results.get('r2_mean_session_count', 0)
        
        self.logger.info(f"🎯 Aggregated Metrics Across {num_sessions} Sessions:")
        self.logger.info(f"   R² Score (Mean): {r2_mean:.4f} ± {r2_std:.4f}")
        self.logger.info(f"   R² Score Range: [{r2_min:.4f}, {r2_max:.4f}]")
        
        # Per-session breakdown
        self.logger.info(f"\n📋 Per-Session Results:")
        self.logger.info(f"{'Session ID':<25} {'R² Score':<10}")
        self.logger.info("-" * 60)
        
        for session_id, metrics in session_results.items():
            r2 = metrics.get('r2_mean', 0)
            self.logger.info(f"{session_id:<25} {r2:<10.4f}")
        
        self.logger.info(f"\n✅ Session evaluation completed")

    def train(self, num_epochs: int):
        """Main training loop with session-level evaluation."""
        self.logger.info(f"Starting public regression training for {num_epochs} epochs")
        self.logger.info(f"Target type: {self.target_type}")
        self.logger.info(f"Dataset: {self.dataset.__class__.__name__}")
        
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
            
            # Session-level evaluation and logging
            session_results = self.evaluate_per_session('val')
            aggregated_results = self.compute_aggregated_session_metrics(session_results)
            
            # TensorBoard logging every epoch
            self.log_session_results_to_tensorboard(session_results, aggregated_results, epoch)
            
            # Detailed training log for each session every 5 epochs
            if epoch % 5 == 0:
                self.logger.info(f"\n🔍 INDIVIDUAL SESSION RESULTS (Epoch {epoch}):")
                self.logger.info("="*70)
                self.logger.info(f"{'Session ID':<30} {'R²':<8}")
                self.logger.info("-" * 70)
                
                for session_id, metrics in session_results.items():
                    r2 = metrics.get('r2_mean', 0)
                    self.logger.info(f"{session_id:<30} {r2:<8.4f}")
                self.logger.info(f"\n📊 Aggregated: R²={aggregated_results.get('r2_mean_session_mean', 0):.4f} ± "
                                f"{aggregated_results.get('r2_mean_session_std', 0):.4f}")
                self.logger.info("="*70)
            
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
        
        # Post-training session-level evaluation
        self.logger.info(f"\n🔍 Performing final session-level evaluation on test set...")
        
        # Evaluate on test set with session-level breakdown (same as training epochs)
        session_results = self.evaluate_per_session('test')
        aggregated_results = self.compute_aggregated_session_metrics(session_results)
        
        # Log final results to TensorBoard
        self.log_session_results_to_tensorboard(session_results, aggregated_results, self.epoch)
        
        # Print results using same format as training logs
        self.logger.info(f"\n🔍 FINAL SESSION RESULTS (Test Set):")
        self.logger.info("="*70)
        self.logger.info(f"{'Session ID':<30} {'R²':<8}")
        self.logger.info("-" * 70)
        
        for session_id, metrics in session_results.items():
            r2 = metrics.get('r2_mean', 0)
            self.logger.info(f"{session_id:<30} {r2:<8.4f}")
        
        self.logger.info(f"\n📊 Final Aggregated Results:")
        self.logger.info(f"   R² Score: {aggregated_results.get('r2_mean_session_mean', 0):.4f} ± {aggregated_results.get('r2_mean_session_std', 0):.4f}")
        self.logger.info(f"   Sessions evaluated: {len(session_results)}")
        self.logger.info("="*70)
        
        # Save detailed results
        self.save_session_evaluation_results(session_results, aggregated_results)
        
        # Training completion
        self.enhanced_logger.log_training_end()
        self.enhanced_logger.close()


def load_pretrained_public_foundation_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load pretrained public foundation model and return BOTH model and config.
    
    Args:
        checkpoint_path: Path to pretrained model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (foundation_model, foundation_config)
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get FOUNDATION config (the authoritative model architecture)
    foundation_config = checkpoint['config']
    model_config = foundation_config['model']
    
    # Create foundation model using ORIGINAL config
    model = PublicNeuralFoundationMAE(
        neural_dim=model_config.get('neural_dim', 50),
        d_model=model_config.get('d_model', 512),
        temporal_layers=model_config.get('temporal_layers', 6),
        heads=model_config.get('heads', 8),
        dropout=model_config.get('dropout', 0.1),
        max_seq_length=model_config.get('max_seq_length', 2000),
        kernel_size=model_config.get('kernel_size', [3, 3]),
        pos_encoding_type=model_config.get('pos_encoding_type', 'rope_4d'),
        session_scale=model_config.get('session_scale', 1.0),
        use_temporal_kernels=model_config.get('use_temporal_kernels', True),
        use_mae_decoder=model_config.get('use_mae_decoder', True),
        use_session_specific_heads=model_config.get('use_session_specific_heads', True)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logging.info(f"Loaded pretrained public model from {checkpoint_path}")
    logging.info(f"Model type: PublicNeuralFoundationMAE")
    
    return model, foundation_config  # Return BOTH


def create_public_downstream_trainer(pretrained_checkpoint_path: str,
                                   dataset,  # Public downstream dataset
                                   config: Dict[str, Any]):
    """
    Factory function to create public downstream regression trainer with clean config separation.
    
    Args:
        pretrained_checkpoint_path: Path to pretrained foundation model
        dataset: Public downstream dataset instance
        config: Downstream task configuration (training settings only)
        
    Returns:
        Configured PublicRegressionTrainer instance
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load foundation model AND its config
    foundation_model, foundation_config = load_pretrained_public_foundation_model(pretrained_checkpoint_path, device)
    
    # Create regression trainer with BOTH configs
    trainer = PublicRegressionTrainer(
        pretrained_foundation_model=foundation_model,
        dataset=dataset,
        config=config,  # Downstream config (task settings)
        foundation_config=foundation_config  # Foundation config (model architecture)
    )
    
    return trainer
