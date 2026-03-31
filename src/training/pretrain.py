"""
Professional Neural Foundation Model Pretraining Script

This module provides a comprehensive pretraining framework for cross-site neural foundation models:
- Standard Trainer class with professional training loop design
- TensorBoard logging for debugging and analysis
- Distributed Data Parallel (DDP) support for multi-GPU training
- Comprehensive argument parsing for hyperparameter tuning
- Checkpointing and resuming capabilities
- Integration with CrossSiteFoundationMAE and CausalMaskingEngine

Key Features:
- Professional trainer design following PyTorch best practices
- Full DDP support for A40/A100 multi-GPU training
- Comprehensive TensorBoard logging and monitoring
- Flexible configuration via YAML and command-line arguments
- Automatic experiment naming and path management
- Gradient clipping, learning rate scheduling, and early stopping
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging

# Import our modules
from ..models import CrossSiteFoundationMAE, CrossSiteModelFactory
from ..data import CrossSiteMonkeyDataset
from ..evaluation.loss_functions import compute_neural_mae_loss
from ..utils.masking import CausalMaskingEngine
from ..utils.helpers import load_config, set_seed, WarmupCosineSchedule
from .enhanced_logger import EnhancedLogger


class PretrainTrainer:
    """
    Professional trainer class for neural foundation model pretraining.
    
    Features:
    - Standard trainer design with train/validate/test loops
    - DDP support for multi-GPU training
    - Enhanced comprehensive logging system (terminal capture, top-K checkpoints, model summaries)
    - Automatic checkpointing and resuming
    - Learning rate scheduling and gradient clipping
    - Professional progress tracking and statistics
    """
    
    def __init__(self, 
                 model: CrossSiteFoundationMAE,
                 dataset: CrossSiteMonkeyDataset,
                 config: Dict[str, Any],
                 local_rank: int = 0,
                 world_size: int = 1,
                 is_distributed: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: CrossSiteFoundationMAE model
            dataset: CrossSiteMonkeyDataset for cross-site data
            config: Training configuration dictionary
            local_rank: Local GPU rank for DDP
            world_size: Total number of GPUs
            is_distributed: Whether using distributed training
        """
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Wrap with DDP if distributed, the find_unused_parameters is important for DDP
        if is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            
        self.dataset = dataset
        
        # Extract training configuration
        self.training_config = config['training']
        self.model_config = config['model']
        self.paths_config = config['paths']
        
        # Initialize enhanced logging system
        self._setup_enhanced_logging()
        
        # Initialize training components
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._setup_masking_engine()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        if self.local_rank == 0:
            self.enhanced_logger.logger.info("PretrainTrainer initialized successfully")
            self._log_model_info()
            self._log_training_config()
            
            # Log training start
            self.enhanced_logger.log_training_start(config, self.model)
    
    def _setup_enhanced_logging(self):
        """Setup enhanced logging system."""
        experiment_name = self.paths_config['experiment_name']
        base_dir = self.paths_config['base_dir']
        
        # Initialize enhanced logger
        self.enhanced_logger = EnhancedLogger(
            experiment_name=experiment_name,
            base_dir=base_dir,
            top_k_checkpoints=3,  # Keep top 3 + latest
            local_rank=self.local_rank
        )
        
        # Update paths config to use enhanced logger paths
        if self.local_rank == 0:
            enhanced_paths = self.enhanced_logger.get_paths()
            self.paths_config.update(enhanced_paths)
            
            # Create legacy aliases for backward compatibility
            self.logger = self.enhanced_logger.logger
            self.writer = self.enhanced_logger.writer
        else:
            # Non-primary processes use dummy logger
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.writer = None
    
    def _setup_data_loaders(self):
        """Setup train/validation/test data loaders with DDP support."""
        batch_size = self.training_config['batch_size']
        
        # Create samplers for DDP
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.dataset, 
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            shuffle = False  # DistributedSampler handles shuffling
        else:
            self.train_sampler = None
            shuffle = True
        
        # Training loader
        self.train_loader = self.dataset.create_dataloader(
            split='train',
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
        
        # Validation and test loaders (no DDP needed for evaluation)
        self.val_loader = self.dataset.create_dataloader(
            split='val',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.test_loader = self.dataset.create_dataloader(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        if self.local_rank == 0:
            self.logger.info(f"Data loaders created - Train: {len(self.train_loader)}, "
                           f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)} batches")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        num_epochs = self.training_config['num_epochs']
        warmup_epochs = self.training_config.get('warmup_epochs', 10)
        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = num_epochs * len(self.train_loader)
        
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.training_config.get('min_lr_ratio', 0.0)
        )
        
        if self.local_rank == 0:
            self.logger.info(f"Optimizer and scheduler initialized - LR: {self.training_config['learning_rate']}")
    
    def _setup_masking_engine(self):
        """Setup the causal masking engine."""
        masking_config = self.training_config.get('masking', {})
        
        self.masking_engine = CausalMaskingEngine(
            temporal_mask_ratio=masking_config.get('temporal_mask_ratio', [0.1, 0.3]),
            neuron_mask_ratio=masking_config.get('neuron_mask_ratio', [0.1, 0.25]),
            min_unmasked_timesteps=masking_config.get('min_unmasked_timesteps', 5),
            min_unmasked_neurons=masking_config.get('min_unmasked_neurons', 10)
        )
        
        if self.local_rank == 0:
            self.logger.info("Masking engine initialized")
    
    def _log_model_info(self):
        """Log detailed model information."""
        if self.local_rank != 0:
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pos_encoding_type': self.model_config['pos_encoding_type'],
            'use_temporal_kernels': self.model_config['use_temporal_kernels'],
            'kernel_size': self.model_config['kernel_size'],
            'model_size': self.training_config['model_size'],
            'd_model': self.model_config['d_model'],
            'neural_dim': self.model_config['neural_dim'],
            'device': str(self.device),
            'distributed': self.is_distributed,
            'world_size': self.world_size
        }
        
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_text('Model/Info', 
                               f"Total Parameters: {total_params:,}\n"
                               f"Trainable Parameters: {trainable_params:,}\n"
                               f"Pos Encoding Type: {self.model_config['pos_encoding_type']}\n"
                               f"Use Temporal Kernels: {self.model_config['use_temporal_kernels']}\n"
                               f"Kernel Size: {self.model_config['kernel_size']}\n"
                               f"Model Size: {self.training_config['model_size']}\n"
                               f"D Model: {self.model_config['d_model']}\n"
                               f"Neural Dim: {self.model_config['neural_dim']}")
    
    def _log_training_config(self):
        """Log training configuration."""
        if self.local_rank != 0:
            return
            
        config_str = json.dumps(self.training_config, indent=2)
        self.logger.info(f"Training Config: {config_str}")
        
        if self.writer:
            self.writer.add_text('Training/Config', config_str.replace('\n', '<br>'))
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        if self.is_distributed and self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)
        
        epoch_stats = defaultdict(float)
        num_batches = len(self.train_loader)
        
        for batch_idx, (neural_data,) in enumerate(self.train_loader):
            neural_data = neural_data.to(self.device)  # [B, S, T, N]
            
            # Get site coordinates
            site_coords = self.dataset.get_site_coordinates().to(self.device)  # [S, 2]
            
            # Forward pass and compute loss
            loss_dict = compute_neural_mae_loss(
                model=self.model,
                neural_data=neural_data,
                site_coords=site_coords,
                masking_engine=self.masking_engine,
                contrastive_weight=self.training_config.get('contrastive_weight', 0.1),
                reconstruction_weight=self.training_config.get('reconstruction_weight', 1.0)
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_stats[key] += value.item()
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        epoch_stats[f"{key}_{sub_key}"] += sub_value
            
            # Add additional metrics
            epoch_stats['learning_rate'] += self.scheduler.get_last_lr()[0]
            
            # Enhanced batch progress logging (every 50 batches)
            if self.local_rank == 0 and batch_idx % 50 == 0:
                progress_pct = 100.0 * batch_idx / num_batches
                self.logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{num_batches} ({progress_pct:.1f}%), "
                               f"Loss: {total_loss.item():.4f}, "
                               f"LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
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
        
        for neural_data, in loader:
            neural_data = neural_data.to(self.device)  # [B, S, T, N]
            
            # Get site coordinates
            site_coords = self.dataset.get_site_coordinates().to(self.device)  # [S, 2]
            
            # Forward pass and compute loss
            loss_dict = compute_neural_mae_loss(
                model=self.model,
                neural_data=neural_data,
                site_coords=site_coords,
                masking_engine=self.masking_engine,
                contrastive_weight=self.training_config.get('contrastive_weight', 0.1),
                reconstruction_weight=self.training_config.get('reconstruction_weight', 1.0)
            )
            
            # Update statistics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_stats[key] += value.item()
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        epoch_stats[f"{key}_{sub_key}"] += sub_value
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint using enhanced logging system."""
        if self.local_rank != 0:
            return
            
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_size': self.training_config['model_size'],
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
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            if self.local_rank == 0:
                self.logger.info(f"Checkpoint loaded: {filepath}")
                self.logger.info(f"Resuming from epoch {self.epoch}, step {self.step}")
            
            return True
            
        except Exception as e:
            if self.local_rank == 0:
                self.logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return False
    
    def log_model_gradients(self):
        """Log gradient statistics to TensorBoard."""
        if self.local_rank != 0 or not self.writer:
            return
        
        total_norm = 0
        grad_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_stats[name] = param_norm.item()
                
                # Log individual parameter gradients (every 10 epochs)
                if self.epoch % 10 == 0:
                    self.writer.add_scalar(f'Gradients/{name}', param_norm.item(), self.epoch)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Gradients/TotalNorm', total_norm, self.epoch)
        
        # Log gradient statistics
        if grad_stats:
            self.writer.add_scalar('Gradients/Mean', np.mean(list(grad_stats.values())), self.epoch)
            self.writer.add_scalar('Gradients/Max', np.max(list(grad_stats.values())), self.epoch)
            self.writer.add_scalar('Gradients/Min', np.min(list(grad_stats.values())), self.epoch)
    
    def train(self, num_epochs: int):
        """Main training loop with enhanced logging."""
        if self.local_rank == 0:
            self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch('val')
            test_stats = self.validate_epoch('test')

            # Enhanced logging
            if self.local_rank == 0:
                # Use enhanced logger for comprehensive epoch stats
                self.enhanced_logger.log_epoch_stats(epoch, train_stats, val_stats, test_stats)
                self.log_model_gradients()
            
            # Checkpointing with enhanced system
            is_best = val_stats['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['total_loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Simplified checkpoint saving: only best and last checkpoints
            if self.local_rank == 0:
                # Always save the latest checkpoint
                saved_paths = self.save_checkpoint(epoch, val_stats['total_loss'], is_best)
                
                # Log what was saved
                if 'best' in saved_paths:
                    self.logger.info(f"New best model saved (val_loss: {val_stats['total_loss']:.6f})")
                # Always save latest, so we don't need to log that every time
            
            # Early stopping
            early_stopping_patience = self.training_config.get('early_stopping_patience', 100)
            if self.epochs_without_improvement >= early_stopping_patience:
                if self.local_rank == 0:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completion
        if self.local_rank == 0:
            self.enhanced_logger.log_training_end()
            self.enhanced_logger.close()


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size, True
    else:
        return 0, 1, False


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """Generate experiment name based on configuration."""
    training_config = config['training']
    dataset_config = config['dataset']
    
    model_size = training_config['model_size']
    batch_size = training_config['batch_size']
    lr = training_config['learning_rate']
    
    # Masking ratios
    masking_config = training_config.get('masking', {})
    temporal_ratio = masking_config.get('temporal_mask_ratio', 0.15)
    neuron_ratio = masking_config.get('neuron_mask_ratio', 0.15)
    
    use_temporal_kernels = config['model']['use_temporal_kernels']
    kernel_size = config['model']['kernel_size']
    spatial_scale = config['model']['spatial_scale']
    use_site_specific_heads = config['model']['use_site_specific_heads']
    pos_encoding_type = config['model']['pos_encoding_type']
    
    # Format ratios (handle list/float)
    if isinstance(temporal_ratio, list):
        tr_str = f"{temporal_ratio[0]}-{temporal_ratio[1]}"
    else:
        tr_str = str(temporal_ratio)
    
    if isinstance(neuron_ratio, list):
        nr_str = f"{neuron_ratio[0]}-{neuron_ratio[1]}"
    else:
        nr_str = str(neuron_ratio)
    
    # Exclude IDs
    exclude_str = "_".join(str(id).replace('.0', '') for id in dataset_config.get('exclude_ids', []))
    
    # Contrastive weight
    cont_weight = training_config.get('contrastive_weight', 0.1)
    recon_weight = training_config.get('reconstruction_weight', 1.0)    
    
    name = f"mae_{model_size}_bs{batch_size}_lr{lr}_tr{tr_str}_nr{nr_str}_cw{cont_weight}_rw{recon_weight}_ex{exclude_str}_PE{pos_encoding_type}_Kernel_{use_temporal_kernels}_size{kernel_size}_SS_{spatial_scale}_SSP_{use_site_specific_heads}"
    
    return name


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Neural Foundation Model Pretraining')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/pretrain.yaml',
                       help='Path to configuration file')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large'],
                       help='Model size (overrides config)')
    parser.add_argument('--neural_dim', type=int, help='Neural dimension (overrides config)')
    parser.add_argument('--d_model', type=int, help='Model dimension (overrides config)')
    parser.add_argument('--temporal_layers', type=int, help='Temporal encoder layers (overrides config)')
    parser.add_argument('--spatial_layers', type=int, help='Spatial encoder layers (overrides config)')
    parser.add_argument('--heads', type=int, help='Number of attention heads (overrides config)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (overrides config)')
    
    # CRITICAL architectural parameters for research comparisons
    parser.add_argument('--pos_encoding_type', type=str, choices=['rope_3d', 'standard_rope', 'learnable', 'sinusoidal'],
                       help='Positional encoding type: rope_3d, standard_rope, learnable, sinusoidal (MAJOR architectural choice)')
    parser.add_argument('--use_temporal_kernels', type=str, default='true', choices=['true', 'false'],
                       help='Enable adaptive causal kernel attention')
    parser.add_argument('--kernel_size', type=int, nargs=2, metavar=('H', 'W'),
                       help='Kernel size for adaptive attention [height width]')
    parser.add_argument('--max_seq_length', type=int, help='Maximum sequence length (overrides config)')
    parser.add_argument('--spatial_scale', type=float, help='Spatial encoding scale factor (overrides config)')
    parser.add_argument('--use_site_specific_heads', action='store_true',
                       help='Use site-specific decoder heads')
    parser.add_argument('--no_site_specific_heads', action='store_true', 
                       help='Use shared decoder heads')
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (overrides config)')
    parser.add_argument('--warmup_epochs', type=int, help='Warmup epochs (overrides config)')
    parser.add_argument('--max_grad_norm', type=float, help='Max gradient norm (overrides config)')
    
    # Masking configuration
    parser.add_argument('--temporal_mask_ratio', type=float, nargs='+', 
                       help='Temporal mask ratio (float or min max) (overrides config)')
    parser.add_argument('--neuron_mask_ratio', type=float, nargs='+',
                       help='Neuron mask ratio (float or min max) (overrides config)')
    parser.add_argument('--contrastive_weight', type=float, help='Contrastive loss weight (overrides config)')
    parser.add_argument('--reconstruction_weight', type=float, help='Reconstruction loss weight (overrides config)')
    
    # Dataset configuration
    parser.add_argument('--exclude_ids', nargs='+', help='Dataset IDs to exclude (overrides config)')
    parser.add_argument('--target_neurons', type=int, help='Target neurons per site (overrides config)')
    parser.add_argument('--sample_times', type=int, help='Neuron sampling repetitions (overrides config)')
    parser.add_argument('--target_trials_per_site', type=int, help='Target trials per site (overrides config)')
    parser.add_argument('--min_val_test_trials', type=int, help='Minimum trials for val/test (overrides config)')
    
    # Paths and logging
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name (overrides auto-generation)')
    parser.add_argument('--base_dir', type=str, default='./logs_neuropixel', help='Base directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Other options
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience (overrides config)')
    
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, world_size, is_distributed = setup_distributed()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_size:
        config['training']['model_size'] = args.model_size
    if args.neural_dim:
        config['model']['neural_dim'] = args.neural_dim
    if args.d_model:
        config['model']['d_model'] = args.d_model
    if args.temporal_layers:
        config['model']['temporal_layers'] = args.temporal_layers
    if args.spatial_layers:
        config['model']['spatial_layers'] = args.spatial_layers
    if args.heads:
        config['model']['heads'] = args.heads
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    
    # CRITICAL architectural parameters for research comparisons
    if args.pos_encoding_type:
        config['model']['pos_encoding_type'] = args.pos_encoding_type
    if args.use_temporal_kernels is not None:
        config['model']['use_temporal_kernels'] = (args.use_temporal_kernels.lower() == 'true')
    if args.kernel_size:
        config['model']['kernel_size'] = args.kernel_size
    if args.max_seq_length:
        config['model']['max_seq_length'] = args.max_seq_length
    if args.spatial_scale is not None:
        config['model']['spatial_scale'] = args.spatial_scale
    if args.use_site_specific_heads:
        config['model']['use_site_specific_heads'] = True
    if args.no_site_specific_heads:
        config['model']['use_site_specific_heads'] = False
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.warmup_epochs is not None:
        config['training']['warmup_epochs'] = args.warmup_epochs
    if args.max_grad_norm:
        config['training']['max_grad_norm'] = args.max_grad_norm
    
    # Masking configuration
    if args.temporal_mask_ratio:
        if len(args.temporal_mask_ratio) == 1:
            config['training']['masking']['temporal_mask_ratio'] = args.temporal_mask_ratio[0]
        else:
            config['training']['masking']['temporal_mask_ratio'] = args.temporal_mask_ratio
    if args.neuron_mask_ratio:
        if len(args.neuron_mask_ratio) == 1:
            config['training']['masking']['neuron_mask_ratio'] = args.neuron_mask_ratio[0]
        else:
            config['training']['masking']['neuron_mask_ratio'] = args.neuron_mask_ratio
    if args.contrastive_weight:
        config['training']['contrastive_weight'] = args.contrastive_weight
    if args.reconstruction_weight is not None:
        config['training']['reconstruction_weight'] = args.reconstruction_weight
    
    # Dataset configuration
    if args.exclude_ids:
        config['dataset']['exclude_ids'] = args.exclude_ids
    if args.target_neurons:
        config['dataset']['target_neurons'] = args.target_neurons
    if args.sample_times:
        config['dataset']['sample_times'] = args.sample_times
    if args.target_trials_per_site:
        config['dataset']['target_trials_per_site'] = args.target_trials_per_site
    if args.min_val_test_trials:
        config['dataset']['min_val_test_trials'] = args.min_val_test_trials
    
    # Other options
    if args.base_dir:
        config['paths']['base_dir'] = args.base_dir
    if args.seed:
        config['training']['seed'] = args.seed
    if args.early_stopping_patience:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    # Generate experiment name and setup paths
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = generate_experiment_name(config)
    
    # Enhanced folder structure: logs/experiment_name/[training_logs, runs, checkpoints, config, model_summary]
    config['paths'] = {
        'experiment_name': experiment_name,
        'base_dir': args.base_dir,
        # These will be updated by EnhancedLogger to use the new structure
        'checkpoint_dir': f"{args.base_dir}/{experiment_name}/checkpoints",
        'tensorboard_dir': f"{args.base_dir}/{experiment_name}/runs",
        'log_dir': f"{args.base_dir}/{experiment_name}/training_logs"
    }
    
    # Set random seed
    set_seed(config['training'].get('seed', 42))
    
    if local_rank == 0:
        print("=" * 80)
        print("NEURAL FOUNDATION MODEL PRETRAINING")
        print("=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Model Size: {config['training']['model_size']}")
        print(f"Distributed Training: {is_distributed} (World Size: {world_size})")
        print(f"Device: cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print("=" * 80)
    
    try:
        # Initialize dataset
        if local_rank == 0:
            print("Initializing dataset...")
        
        dataset = CrossSiteMonkeyDataset(
            exclude_ids=config['dataset'].get('exclude_ids', []),
            split_ratios=tuple(config['dataset'].get('split_ratios', [0.8, 0.1, 0.1])),
            target_neurons=config['dataset'].get('target_neurons', 50),
            sample_times=config['dataset'].get('sample_times', 5),
            target_trials_per_site=config['dataset'].get('target_trials_per_site', 4000),
            min_val_test_trials=config['dataset'].get('min_val_test_trials', 100),
            width=config['dataset'].get('width', 0.02),
            sequence_length=config['dataset'].get('sequence_length', 50),
            random_seed=config['training'].get('seed', 42)
        )
        
        if local_rank == 0:
            print(f"Dataset loaded - Sites: {len(dataset.site_ids)}")
            train_data = dataset.get_split_data('train')
            val_data = dataset.get_split_data('val')
            test_data = dataset.get_split_data('test')
            print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        # Initialize model
        if local_rank == 0:
            print("Initializing model...")
        
        # Merge model size configuration
        model_size = config['training']['model_size']
        if 'model_sizes' in config and model_size in config['model_sizes']:
            # First apply model size defaults
            size_config = config['model_sizes'][model_size].copy()
            # Only update parameters that are null in the base config
            for param in ['d_model', 'temporal_layers', 'spatial_layers', 'heads']:
                if config['model'].get(param) is None:
                    config['model'][param] = size_config.get(param)
                # Note: If config['model'][param] is not None, it overrides the size config
        
        model = CrossSiteModelFactory.create_mae_model(
            size=model_size,
            neural_dim=config['model']['neural_dim'],
            d_model=config['model']['d_model'],
            n_sites=len(dataset.site_ids),
            temporal_layers=config['model'].get('temporal_layers', 6),
            spatial_layers=config['model'].get('spatial_layers', 4),
            heads=config['model'].get('heads', 8),
            dropout=config['model'].get('dropout', 0.1),
            
            # CRITICAL architectural parameters for research comparisons
            max_seq_length=config['model'].get('max_seq_length', 2000),
            pos_encoding_type=config['model'].get('pos_encoding_type', 'rope_3d'),
            spatial_scale=config['model'].get('spatial_scale', 0.1),
            use_temporal_kernels=config['model'].get('use_temporal_kernels', True),
            kernel_size=config['model'].get('kernel_size', [3, 3]),
            use_mae_decoder=config['model'].get('use_mae_decoder', True),
            use_site_specific_heads=config['model'].get('use_site_specific_heads', True)
        )
        
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model initialized - Total parameters: {total_params:,}")
        
        # Initialize trainer
        if local_rank == 0:
            print("Initializing trainer...")
        
        trainer = PretrainTrainer(
            model=model,
            dataset=dataset,
            config=config,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=is_distributed
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            if local_rank == 0:
                print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        if local_rank == 0:
            print("Starting training...")
        
        trainer.train(config['training']['num_epochs'])
        
        if local_rank == 0:
            print("Training completed successfully!")
    
    except Exception as e:
        if local_rank == 0:
            print(f"Error during training: {e}")
        raise e
    
    finally:
        if is_distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main() 