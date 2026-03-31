"""
Professional Public Dataset Pretraining Script for Temporal-Only Neural Foundation Model

This module provides a comprehensive pretraining framework for single-site temporal neural foundation models
adapted for the Perich-Miller 2018 public dataset:
- Standard Trainer class with professional training loop design
- TensorBoard logging for debugging and analysis
- Distributed Data Parallel (DDP) support for multi-GPU training
- Comprehensive argument parsing for hyperparameter tuning
- Checkpointing and resuming capabilities
- Integration with PublicNeuralFoundationMAE and temporal-only architecture

Key Features:
- Professional trainer design following PyTorch best practices
- Temporal-only architecture with RoPE4D session coordinates
- Single-site data format [B, 1, T, N] processing
- Session coordinate handling for (subject, time, task) embeddings
- Simplified loss function for temporal reconstruction only
- Full DDP support for A40/A100 multi-GPU training
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

# Import our modules
from ..models.public_transformer import PublicNeuralFoundationMAE
from ..data.public_pretrain_dataset import PublicDatasetForPretraining
from ..utils.helpers import load_config, set_seed, WarmupCosineSchedule
from .enhanced_logger import EnhancedLogger


def compute_public_mae_loss(model: PublicNeuralFoundationMAE,
                           neural_data: torch.Tensor,
                           session_coords: torch.Tensor,
                           temporal_mask_ratio: float = 0.3,
                           neuron_mask_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Compute temporal-only MAE loss for public dataset pretraining.
    
    Args:
        model: PublicNeuralFoundationMAE model
        neural_data: [B, 1, T, N] - single-site neural data
        session_coords: [B, 1, 3] - session coordinates (subject, time, task)
        temporal_mask_ratio: Ratio of temporal positions to mask
        neuron_mask_ratio: Ratio of neurons to mask
        
    Returns:
        Dictionary containing loss components
    """
    B, S, T, N = neural_data.shape
    assert S == 1, f"Expected single-site data (S=1), got S={S}"
    
    device = neural_data.device
    
    # Create temporal mask (causal masking)
    num_masked_timesteps = int(T * temporal_mask_ratio)
    temporal_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    
    for b in range(B):
        # Random causal masking: mask consecutive timesteps
        if num_masked_timesteps > 0:
            start_idx = torch.randint(0, T - num_masked_timesteps + 1, (1,)).item()
            temporal_mask[b, start_idx:start_idx + num_masked_timesteps] = True
    
    # Create neuron mask
    num_masked_neurons = int(N * neuron_mask_ratio)
    neuron_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    for b in range(B):
        if num_masked_neurons > 0:
            masked_neuron_indices = torch.randperm(N, device=device)[:num_masked_neurons]
            neuron_mask[b, masked_neuron_indices] = True
    
    # Apply masks to input data
    masked_neural_data = neural_data.clone()
    
    # Temporal masking: set masked timesteps to zero
    temporal_mask_expanded = temporal_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
    masked_neural_data = masked_neural_data.masked_fill(temporal_mask_expanded, 0.0)
    
    # Neuron masking: set masked neurons to zero
    neuron_mask_expanded = neuron_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
    masked_neural_data = masked_neural_data.masked_fill(neuron_mask_expanded, 0.0)
    
    # Forward pass with masking
    output = model(masked_neural_data, session_coords, return_mae_reconstruction=True)
    
    reconstruction = output['reconstruction']  # [B, 1, T, N]
    
    # Compute reconstruction loss only on masked positions
    combined_mask = temporal_mask.unsqueeze(1).unsqueeze(-1) | neuron_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, T, N]
    
    if combined_mask.sum() > 0:
        # Poisson reconstruction loss for neural spike data
        target_spikes = neural_data[combined_mask]
        predicted_rates = reconstruction[combined_mask]
        
        # Ensure positive rates for Poisson loss
        predicted_rates = torch.clamp(predicted_rates, min=1e-8)
        
        # Poisson negative log-likelihood
        poisson_loss = predicted_rates - target_spikes * torch.log(predicted_rates)
        reconstruction_loss = poisson_loss.mean()
    else:
        reconstruction_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
    
    # Total loss (no contrastive component for single-site data)
    total_loss = reconstruction_loss
    
    # Return loss dictionary
    loss_dict = {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'temporal_mask_ratio': torch.tensor(temporal_mask.float().mean().item()),
        'neuron_mask_ratio': torch.tensor(neuron_mask.float().mean().item()),
        'num_masked_positions': torch.tensor(combined_mask.sum().item())
    }
    
    return loss_dict


class PublicPretrainTrainer:
    """
    Professional trainer class for public dataset neural foundation model pretraining.
    
    Features:
    - Temporal-only architecture with single-site data processing
    - Session coordinate handling for RoPE4D positional encoding
    - Professional training loop with enhanced logging
    - DDP support for multi-GPU training
    - Automatic checkpointing and resuming
    - Learning rate scheduling and gradient clipping
    """
    
    def __init__(self, 
                 model: PublicNeuralFoundationMAE,
                 dataset: PublicDatasetForPretraining,
                 config: Dict[str, Any],
                 local_rank: int = 0,
                 world_size: int = 1,
                 is_distributed: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: PublicNeuralFoundationMAE model
            dataset: PublicDatasetForPretraining for public data
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
        
        # Wrap with DDP if distributed
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
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        if self.local_rank == 0:
            self.enhanced_logger.logger.info("PublicPretrainTrainer initialized successfully")
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
            # Note: PublicDatasetForPretraining doesn't use PyTorch Dataset interface directly
            # We'll handle DDP differently for our custom dataset
            shuffle = False
        else:
            shuffle = True
        
        # Training loader
        self.train_loader = self.dataset.create_dataloader(
            split='train',
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
        
        # Validation and test loaders
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
            'model_type': 'PublicNeuralFoundationMAE',
            'architecture': 'temporal_only',
            'd_model': self.model_config['d_model'],
            'neural_dim': self.model_config['neural_dim'],
            'temporal_layers': self.model_config.get('temporal_layers', 6),
            'heads': self.model_config.get('heads', 8),
            'dropout': self.model_config.get('dropout', 0.1),
            'pos_encoding_type': self.model_config.get('pos_encoding_type', 'rope_4d'),
            'use_temporal_kernels': self.model_config.get('use_temporal_kernels', True),
            'kernel_size': self.model_config.get('kernel_size', [3, 3]),
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
                               f"Architecture: Temporal-Only (Single-Site)\n"
                               f"D Model: {self.model_config['d_model']}\n"
                               f"Neural Dim: {self.model_config['neural_dim']}\n"
                               f"Temporal Layers: {self.model_config.get('temporal_layers', 6)}\n"
                               f"Heads: {self.model_config.get('heads', 8)}\n"
                               f"Dropout: {self.model_config.get('dropout', 0.1)}\n"
                               f"Pos Encoding Type: {self.model_config.get('pos_encoding_type', 'rope_4d')}\n"
                               f"Use Temporal Kernels: {self.model_config.get('use_temporal_kernels', True)}\n"
                               f"Kernel Size: {self.model_config.get('kernel_size', [3, 3])}")
    
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
        
        epoch_stats = defaultdict(float)
        num_batches = len(self.train_loader)
        
        for batch_idx, (neural_data, session_coords) in enumerate(self.train_loader):
            neural_data = neural_data.to(self.device)      # [B, 1, T, N]
            session_coords = session_coords.to(self.device)  # [B, 1, 3]
            
            # Get masking parameters
            masking_config = self.training_config.get('masking', {})
            temporal_mask_ratio = masking_config.get('temporal_mask_ratio', 0.3)
            neuron_mask_ratio = masking_config.get('neuron_mask_ratio', 0.2)
            
            # Handle dynamic masking ratios
            if isinstance(temporal_mask_ratio, list):
                temporal_mask_ratio = np.random.uniform(temporal_mask_ratio[0], temporal_mask_ratio[1])
            if isinstance(neuron_mask_ratio, list):
                neuron_mask_ratio = np.random.uniform(neuron_mask_ratio[0], neuron_mask_ratio[1])
            
            # Forward pass and compute loss
            loss_dict = compute_public_mae_loss(
                model=self.model,
                neural_data=neural_data,
                session_coords=session_coords,
                temporal_mask_ratio=temporal_mask_ratio,
                neuron_mask_ratio=neuron_mask_ratio
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
            
            # Add additional metrics
            epoch_stats['learning_rate'] += self.scheduler.get_last_lr()[0]
            
            # Enhanced batch progress logging (every 50 batches)
            if self.local_rank == 0 and batch_idx % 50 == 0:
                progress_pct = 100.0 * batch_idx / num_batches
                self.logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{num_batches} ({progress_pct:.1f}%), "
                               f"Loss: {total_loss.item():.4f}, "
                               f"LR: {self.scheduler.get_last_lr()[0]:.2e}, "
                               f"Temporal Mask: {temporal_mask_ratio:.3f}, "
                               f"Neuron Mask: {neuron_mask_ratio:.3f}")
            
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
        
        for neural_data, session_coords in loader:
            neural_data = neural_data.to(self.device)      # [B, 1, T, N]
            session_coords = session_coords.to(self.device)  # [B, 1, 3]
            
            # Get masking parameters
            masking_config = self.training_config.get('masking', {})
            temporal_mask_ratio = masking_config.get('temporal_mask_ratio', 0.3)
            neuron_mask_ratio = masking_config.get('neuron_mask_ratio', 0.2)
            
            # Use fixed masking ratios for validation
            if isinstance(temporal_mask_ratio, list):
                temporal_mask_ratio = np.mean(temporal_mask_ratio)
            if isinstance(neuron_mask_ratio, list):
                neuron_mask_ratio = np.mean(neuron_mask_ratio)
            
            # Forward pass and compute loss
            loss_dict = compute_public_mae_loss(
                model=self.model,
                neural_data=neural_data,
                session_coords=session_coords,
                temporal_mask_ratio=temporal_mask_ratio,
                neuron_mask_ratio=neuron_mask_ratio
            )
            
            # Update statistics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_stats[key] += value.item()
        
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
                'model_type': 'PublicNeuralFoundationMAE',
                'architecture': 'temporal_only',
                'd_model': self.model_config['d_model'],
                'neural_dim': self.model_config['neural_dim'],
                'temporal_layers': self.model_config.get('temporal_layers', 6),
                'heads': self.model_config.get('heads', 8),
                'dropout': self.model_config.get('dropout', 0.1),
                'pos_encoding_type': self.model_config.get('pos_encoding_type', 'rope_4d'),
                'use_temporal_kernels': self.model_config.get('use_temporal_kernels', True),
                'kernel_size': self.model_config.get('kernel_size', [3, 3])
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
    
    def train(self, num_epochs: int):
        """Main training loop with enhanced logging."""
        if self.local_rank == 0:
            self.logger.info(f"Starting public dataset pretraining for {num_epochs} epochs")
        
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
            
            # Checkpointing with enhanced system
            is_best = val_stats['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['total_loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if self.local_rank == 0:
                saved_paths = self.save_checkpoint(epoch, val_stats['total_loss'], is_best)
                
                if 'best' in saved_paths:
                    self.logger.info(f"New best model saved (val_loss: {val_stats['total_loss']:.6f})")
            
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


def create_public_model(config: Dict[str, Any]) -> PublicNeuralFoundationMAE:
    """
    Create PublicNeuralFoundationMAE model from configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        PublicNeuralFoundationMAE model
    """
    model_config = config['model']
    
    model = PublicNeuralFoundationMAE(
        neural_dim=model_config['neural_dim'],
        d_model=model_config['d_model'],
        temporal_layers=model_config.get('temporal_layers', 6),
        heads=model_config.get('heads', 8),
        dropout=model_config.get('dropout', 0.1),
        use_temporal_kernels=model_config.get('use_temporal_kernels', True),
        kernel_size=model_config.get('kernel_size', [3, 3]),
        pos_encoding_type=model_config.get('pos_encoding_type', 'rope_4d'),
        max_subjects=model_config.get('max_subjects', 10),
        max_time_periods=model_config.get('max_time_periods', 10),
        max_tasks=model_config.get('max_tasks', 10),
        rope_base=model_config.get('rope_base', 10000.0)
    )
    print(f"Model initialized - Positional encoding type: {model_config.get('pos_encoding_type', 'rope_4d')}")
    return model


def generate_public_experiment_name(config: Dict[str, Any]) -> str:
    """Generate experiment name for public dataset pretraining."""
    training_config = config['training']
    dataset_config = config['dataset']
    
    model_size = training_config.get('model_size', 'custom')
    d_model = config['model']['d_model']
    temporal_layers = config['model'].get('temporal_layers', 6)
    heads = config['model'].get('heads', 8)
    dropout = config['model'].get('dropout', 0.1)
    pos_encoding_type = config['model'].get('pos_encoding_type', 'rope_4d')
    use_temporal_kernels = config['model'].get('use_temporal_kernels', True)
    kernel_size = config['model'].get('kernel_size', [3, 3])
    batch_size = training_config['batch_size']
    lr = training_config['learning_rate']
    
    # Masking ratios
    masking_config = training_config.get('masking', {})
    temporal_ratio = masking_config.get('temporal_mask_ratio', 0.3)
    neuron_ratio = masking_config.get('neuron_mask_ratio', 0.2)
    
    # Format ratios (handle list/float)
    if isinstance(temporal_ratio, list):
        tr_str = f"{temporal_ratio[0]}-{temporal_ratio[1]}"
    else:
        tr_str = str(temporal_ratio)
    
    if isinstance(neuron_ratio, list):
        nr_str = f"{neuron_ratio[0]}-{neuron_ratio[1]}"
    else:
        nr_str = str(neuron_ratio)
    
    # Subjects
    subjects_str = "_".join(dataset_config.get('subjects', ['c', 'j', 'm']))
    
    name = f"public_mae_{model_size}_d{d_model}_l{temporal_layers}_h{heads}_bs{batch_size}_lr{lr}_tr{tr_str}_nr{nr_str}_subj{subjects_str}_PE{pos_encoding_type}_Kernel_{use_temporal_kernels}_size{kernel_size}"
    
    return name


def main():
    """Main training function for public dataset pretraining."""
    parser = argparse.ArgumentParser(description='Public Dataset Neural Foundation Model Pretraining')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/public_pretrain.yaml',
                       help='Path to configuration file')

    parser.add_argument('--data_root', type=str, help='Data root')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large', 'custom'],
                       help='Model size (overrides config)')
    parser.add_argument('--neural_dim', type=int, help='Neural dimension (overrides config)')
    
    parser.add_argument('--d_model', type=int, help='Model dimension (overrides config)')
    parser.add_argument('--temporal_layers', type=int, help='Temporal encoder layers (overrides config)')
    parser.add_argument('--heads', type=int, help='Number of attention heads (overrides config)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (overrides config)')
    parser.add_argument('--pos_encoding_type', type=str, choices=["rope_4d", "standard_rope", "learnable", "sinusoidal"],
                       help='Positional encoding type: ["rope_4d", "standard_rope", "learnable", "sinusoidal"]')
    parser.add_argument('--use_temporal_kernels', type=str, default='true', choices=['true', 'false'],
                       help='Enable adaptive causal kernel attention')
    parser.add_argument('--kernel_size', type=int, nargs=2, metavar=('H', 'W'),
                       help='Kernel size for adaptive attention [height width]')
    
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
    
    # Dataset configuration
    parser.add_argument('--subjects', nargs='+', help='Subjects to include (overrides config)')
    parser.add_argument('--target_neurons', type=int, help='Target neurons per session (overrides config)')
    parser.add_argument('--sample_times', type=int, help='Neuron sampling repetitions (overrides config)')
    parser.add_argument('--target_trials_per_site', type=int, help='Target trials per session (overrides config)')
    parser.add_argument('--min_val_test_trials', type=int, help='Minimum trials for val/test (overrides config)')
    
    # Paths and logging
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name (overrides auto-generation)')
    parser.add_argument('--base_dir', type=str, default='./logs_public', help='Base directory for logs and checkpoints')
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
    if args.data_root:
        config['dataset']['data_root'] = args.data_root
    if args.model_size:
        config['training']['model_size'] = args.model_size
    if args.neural_dim:
        config['model']['neural_dim'] = args.neural_dim
    if args.d_model:
        config['model']['d_model'] = args.d_model
    if args.temporal_layers:
        config['model']['temporal_layers'] = args.temporal_layers
    if args.heads:
        config['model']['heads'] = args.heads
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.pos_encoding_type:
        config['model']['pos_encoding_type'] = args.pos_encoding_type
    if args.use_temporal_kernels is not None:
        config['model']['use_temporal_kernels'] = (args.use_temporal_kernels.lower() == 'true')
    if args.kernel_size:
        config['model']['kernel_size'] = args.kernel_size
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
    
    # Dataset configuration
    if args.subjects:
        config['dataset']['subjects'] = args.subjects
    if args.target_neurons:
        config['dataset']['target_neurons'] = args.target_neurons
    if args.sample_times:
        config['dataset']['sample_times'] = args.sample_times
    if args.target_trials_per_site:
        config['dataset']['target_trials_per_site'] = args.target_trials_per_site
    if args.min_val_test_trials:
        config['dataset']['min_val_test_trials'] = args.min_val_test_trials
    
    # Other options
    if args.seed:
        config['training']['seed'] = args.seed
    if args.early_stopping_patience:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    # Generate experiment name and setup paths
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = generate_public_experiment_name(config)
    
    config['paths'] = {
        'experiment_name': experiment_name,
        'base_dir': args.base_dir,
        'checkpoint_dir': f"{args.base_dir}/{experiment_name}/checkpoints",
        'tensorboard_dir': f"{args.base_dir}/{experiment_name}/runs",
        'log_dir': f"{args.base_dir}/{experiment_name}/training_logs"
    }
    
    # Set random seed
    set_seed(config['training'].get('seed', 42))
    
    if local_rank == 0:
        print("=" * 80)
        print("PUBLIC DATASET NEURAL FOUNDATION MODEL PRETRAINING")
        print("=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Architecture: Temporal-Only (Single-Site)")
        print(f"Dataset: Perich-Miller 2018 Public Data")
        print(f"Distributed Training: {is_distributed} (World Size: {world_size})")
        print(f"Device: cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print("=" * 80)
    
    try:
        # Initialize dataset
        if local_rank == 0:
            print("Initializing public dataset...")
        
        dataset = PublicDatasetForPretraining(
            data_root=config['dataset'].get('data_root', "/data/Fang-analysis/causal-nfm/Data/processed_normalize_session"),
            subjects=config['dataset'].get('subjects', ['c', 'j', 'm']),
            target_neurons=config['dataset'].get('target_neurons', 50),
            sequence_length=config['dataset'].get('sequence_length', 50),
            sample_times=config['dataset'].get('sample_times', 1),
            target_trials_per_site=config['dataset'].get('target_trials_per_site', 100),
            min_val_test_trials=config['dataset'].get('min_val_test_trials', 10),
            random_seed=config['training'].get('seed', 42)
        )
        
        if local_rank == 0:
            print(f"Dataset loaded - Sessions: {len(dataset.session_ids)}")
            train_data, train_coords = dataset.get_split_data('train')
            val_data, val_coords = dataset.get_split_data('val')
            test_data, test_coords = dataset.get_split_data('test')
            print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        # Initialize model
        if local_rank == 0:
            print("Initializing temporal-only model...")
            
        # Merge model size configuration (if provided)
        model_size = config['training'].get('model_size', None)
        if model_size and 'model_sizes' in config and model_size in config['model_sizes']:
            config['model'].update(config['model_sizes'][model_size])
        
        model = create_public_model(config)
        
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model initialized - Total parameters: {total_params:,}")
        
        # Initialize trainer
        if local_rank == 0:
            print("Initializing trainer...")
        
        trainer = PublicPretrainTrainer(
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
            print("Starting pretraining...")
        
        trainer.train(config['training']['num_epochs'])
        
        if local_rank == 0:
            print("Pretraining completed successfully!")
    
    except Exception as e:
        if local_rank == 0:
            print(f"Error during training: {e}")
        raise e
    
    finally:
        if is_distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
