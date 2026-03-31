"""
Enhanced Logging System for Neural Foundation Model Training

This module provides comprehensive logging capabilities that capture all aspects
of training including:
- Structured Python logging with proper formatting
- TensorBoard logging with rich visualizations
- Simple checkpoint management (best.pth + last.pth)
- Configuration and model summary saving
- Organized folder structure under logs/experiment_name/

Key Features:
- Professional structured logging (training.log)
- Simple checkpoint management (keeps best + latest)
- Rich TensorBoard logging with model graphs
- Automatic config and model summary saving
- Professional folder organization
- Cross-platform compatibility
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml


class SimpleCheckpointManager:
    """
    Simple checkpoint manager that saves only best.pth and last.pth.
    
    Features:
    - Saves best model based on validation loss as 'best.pth'
    - Always saves latest model as 'last.pth' 
    - Automatic cleanup (only 2 files total)
    - Simple and reliable
    """
    
    def __init__(self, checkpoint_dir: Path, metric_name: str = 'val_loss', mode: str = 'min'):
        """
        Initialize simple checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Metric to track for best checkpoint
            mode: 'min' for lower is better, 'max' for higher is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best metric value
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Checkpoint paths
        self.best_path = self.checkpoint_dir / 'best.pth'
        self.last_path = self.checkpoint_dir / 'last.pth'
        
        # Load existing best metric if available
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        """Load existing checkpoint information."""
        metadata_file = self.checkpoint_dir / 'checkpoint_info.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.best_metric = metadata.get('best_metric', self.best_metric)
            except Exception as e:
                print(f"Warning: Could not load checkpoint info: {e}")
    
    def _save_checkpoint_info(self):
        """Save checkpoint information."""
        metadata = {
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
            'mode': self.mode,
            'last_updated': datetime.now().isoformat(),
            'best_checkpoint': str(self.best_path) if self.best_path.exists() else None,
            'last_checkpoint': str(self.last_path) if self.last_path.exists() else None
        }
        
        metadata_file = self.checkpoint_dir / 'checkpoint_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], epoch: int, metric_value: float) -> Dict[str, str]:
        """
        Save checkpoint as best.pth and/or last.pth.
        
        Args:
            checkpoint: Checkpoint dictionary to save
            epoch: Current epoch
            metric_value: Metric value for this checkpoint
        
        Returns:
            Dictionary with saved file paths
        """
        saved_paths = {}
        
        # Always save as latest
        torch.save(checkpoint, self.last_path)
        saved_paths['last'] = str(self.last_path)
        
        # Check if this is the best model
        is_best = False
        if self.mode == 'min' and metric_value < self.best_metric:
            is_best = True
            self.best_metric = metric_value
        elif self.mode == 'max' and metric_value > self.best_metric:
            is_best = True
            self.best_metric = metric_value
        
        # Save as best if improved
        if is_best:
            torch.save(checkpoint, self.best_path)
            saved_paths['best'] = str(self.best_path)
        
        # Update metadata
        self._save_checkpoint_info()
        
        return saved_paths
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints."""
        return {
            'best_metric': self.best_metric,
            'best_checkpoint': str(self.best_path) if self.best_path.exists() else None,
            'last_checkpoint': str(self.last_path) if self.last_path.exists() else None,
            'metric_name': self.metric_name,
            'mode': self.mode
        }
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        return str(self.best_path) if self.best_path.exists() else None
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        return str(self.last_path) if self.last_path.exists() else None



class EnhancedLogger:
    """
    Enhanced logging system for neural foundation model training.
    
    Features:
    - Complete terminal output capture
    - Top-K checkpoint management (for future use): Current implementation only saves the best and last checkpoints  
    - Rich TensorBoard logging
    - Configuration and model summary saving
    - Professional folder organization
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "./logs", 
                 top_k_checkpoints: int = 3, local_rank: int = 0):
        """
        Initialize enhanced logging system.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for all logs
            top_k_checkpoints: Ignored (kept for backward compatibility)
            local_rank: Local rank for distributed training
        """
        self.experiment_name = experiment_name
        self.local_rank = local_rank
        self.is_primary = (local_rank == 0)
        
        # Setup directory structure
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        
        if self.is_primary:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            self.training_logs_dir = self.experiment_dir / "training_logs"
            self.tensorboard_dir = self.experiment_dir / "runs"
            self.checkpoints_dir = self.experiment_dir / "checkpoints"
            self.config_dir = self.experiment_dir / "config"
            self.model_summary_dir = self.experiment_dir / "model_summary"
            
            for dir_path in [self.training_logs_dir, self.tensorboard_dir, 
                           self.checkpoints_dir, self.config_dir, self.model_summary_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize logging components
            self._setup_standard_logging()
            self._setup_tensorboard()
            self._setup_checkpoint_manager(top_k_checkpoints)
            
            # Initialize statistics tracking
            self.stats = defaultdict(list)
            self.start_time = datetime.now()
            
            self.logger.info("=" * 80)
            self.logger.info(f"ENHANCED LOGGING SYSTEM INITIALIZED")
            self.logger.info(f"Experiment: {experiment_name}")
            self.logger.info(f"Base Directory: {self.base_dir}")
            self.logger.info(f"Experiment Directory: {self.experiment_dir}")
            self.logger.info("=" * 80)
        
        else:
            # Non-primary processes use dummy components
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.writer = None
            self.checkpoint_manager = None
    
    def _setup_standard_logging(self):
        """Setup standard Python logging."""
        log_file = self.training_logs_dir / "training.log"
        
        # Create custom logger
        self.logger = logging.getLogger(f"trainer_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    

    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(self.tensorboard_dir)
    
    def _setup_checkpoint_manager(self, top_k: int):
        """Setup simple checkpoint manager."""
        self.checkpoint_manager = SimpleCheckpointManager(
            checkpoint_dir=self.checkpoints_dir,
            metric_name='val_loss',
            mode='min'
        )
    
    def save_config(self, config: Dict[str, Any]):
        """Save training configuration."""
        if not self.is_primary:
            return
        
        config_file = self.config_dir / "training_config.yaml"
        config_json_file = self.config_dir / "training_config.json"
        
        # Save as YAML
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Save as JSON for easier programmatic access
        with open(config_json_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_file} and {config_json_file}")
    
    def save_model_summary(self, model: nn.Module, input_shape: tuple = None):
        """Save detailed model summary and architecture."""
        if not self.is_primary:
            return
        
        summary_file = self.model_summary_dir / "model_summary.txt"
        architecture_file = self.model_summary_dir / "model_architecture.json"
        
        # Create detailed model summary
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEURAL FOUNDATION MODEL ARCHITECTURE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n\n")
            
            # Model architecture
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 40 + "\n")
            f.write(str(model))
            f.write("\n\n")
            
            # Parameter details by module
            f.write("PARAMETER BREAKDOWN BY MODULE:\n")
            f.write("-" * 40 + "\n")
            for name, module in model.named_modules():
                if len(list(module.parameters())) > 0:
                    module_params = sum(p.numel() for p in module.parameters())
                    f.write(f"{name}: {module_params:,} parameters\n")
        
        # Save architecture as JSON
        architecture_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_structure': str(model),
            'parameter_breakdown': {
                name: sum(p.numel() for p in module.parameters())
                for name, module in model.named_modules()
                if len(list(module.parameters())) > 0
            }
        }
        
        with open(architecture_file, 'w') as f:
            json.dump(architecture_info, f, indent=2)
        
        self.logger.info(f"Model summary saved to {summary_file}")
        self.logger.info(f"Model architecture saved to {architecture_file}")
        
        # Log to TensorBoard if possible
        if input_shape and hasattr(self, 'writer'):
            try:
                dummy_input = torch.randn(input_shape)
                self.writer.add_graph(model, dummy_input)
                self.logger.info("Model graph added to TensorBoard")
            except Exception as e:
                self.logger.warning(f"Could not add model graph to TensorBoard: {e}")
    
    def log_epoch_stats(self, epoch: int, train_stats: Dict[str, float], 
                       val_stats: Dict[str, float], test_stats: Dict[str, float]):
        """Log comprehensive epoch statistics."""
        if not self.is_primary:
            return
        
        # Console and file logging
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EPOCH {epoch} SUMMARY")
        self.logger.info(f"Learning Rate: {train_stats.get('learning_rate', 0):.2e}; Train Loss: {train_stats.get('total_loss', 0):.6f}; Val Loss: {val_stats.get('total_loss', 0):.6f}; Test Loss: {test_stats.get('total_loss', 0):.6f}")
        self.logger.info(f"Train R2: {train_stats.get('r2_mean', 0):.6f}; Val R2: {val_stats.get('r2_mean', 0):.6f}; Test R2: {test_stats.get('r2_mean', 0):.6f}")
        self.logger.info(f"\n{'='*80}")
        # TensorBoard logging - Individual loss curves for easy experiment comparison
        if self.writer:
            # Individual loss curves (9 total: 3 splits x 3 loss types)
            # This makes it much easier to compare experiments in TensorBoard
            
            # Total losses
            self.writer.add_scalar('Loss/train_total_loss', train_stats.get('total_loss', 0), epoch)
            self.writer.add_scalar('Loss/val_total_loss', val_stats.get('total_loss', 0), epoch)
            self.writer.add_scalar('Loss/test_total_loss', test_stats.get('total_loss', 0), epoch)
            
            # Contrastive losses
            self.writer.add_scalar('Loss/train_contrastive_loss', train_stats.get('contrastive_loss', 0), epoch)
            self.writer.add_scalar('Loss/val_contrastive_loss', val_stats.get('contrastive_loss', 0), epoch)
            self.writer.add_scalar('Loss/test_contrastive_loss', test_stats.get('contrastive_loss', 0), epoch)
            
            # Reconstruction losses (Poisson)
            self.writer.add_scalar('Loss/train_reconstruction_loss', train_stats.get('poisson_loss', 0), epoch)
            self.writer.add_scalar('Loss/val_reconstruction_loss', val_stats.get('poisson_loss', 0), epoch)
            self.writer.add_scalar('Loss/test_reconstruction_loss', test_stats.get('poisson_loss', 0), epoch)
            
            # Learning rate and other training metrics
            if 'learning_rate' in train_stats:
                self.writer.add_scalar('Training/learning_rate', train_stats['learning_rate'], epoch)
            
            # Additional metrics (if available)
            for key in train_stats.keys():
                if key not in ['total_loss', 'contrastive_loss', 'poisson_loss', 'learning_rate']:
                    self.writer.add_scalar(f'Train_Metrics/{key}', train_stats[key], epoch)
            
            for key in val_stats.keys():
                if key not in ['total_loss', 'contrastive_loss', 'poisson_loss']:
                    self.writer.add_scalar(f'Val_Metrics/{key}', val_stats[key], epoch)
            
            for key in test_stats.keys():
                if key not in ['total_loss', 'contrastive_loss', 'poisson_loss']:
                    self.writer.add_scalar(f'Test_Metrics/{key}', test_stats[key], epoch)
        
        # Update statistics
        for key, value in train_stats.items():
            self.stats[f'train_{key}'].append(value)
        for key, value in val_stats.items():
            self.stats[f'val_{key}'].append(value)
        for key, value in test_stats.items():
            self.stats[f'test_{key}'].append(value)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], epoch: int, val_loss: float) -> Dict[str, str]:
        """Save checkpoint using simple best/last management."""
        if not self.is_primary:
            return {}
        
        saved_paths = self.checkpoint_manager.save_checkpoint(
            checkpoint=checkpoint,
            epoch=epoch,
            metric_value=val_loss
        )
        
        # Log checkpoint info
        checkpoint_info = self.checkpoint_manager.get_checkpoint_info()
        self.logger.info(f"Checkpoint saved: {saved_paths}")
        self.logger.info(f"Best checkpoint: {checkpoint_info['best_checkpoint']}")
        self.logger.info(f"Latest checkpoint: {checkpoint_info['last_checkpoint']}")
        
        return saved_paths
    
    def log_training_start(self, config: Dict[str, Any], model: nn.Module):
        """Log training start information."""
        if not self.is_primary:
            return
        
        # Save config and model summary
        self.save_config(config)
        self.save_model_summary(model)
        
        # Log training info
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Experiment: {self.experiment_name}")
        
        # Handle different config structures (pretraining vs downstream)
        model_info = config['training'].get('model_size', config['training'].get('task_type', 'unknown'))
        self.logger.info(f"Model/Task: {model_info}")
        self.logger.info(f"Batch Size: {config['training']['batch_size']}")
        self.logger.info(f"Learning Rate: {config['training']['learning_rate']}")
        self.logger.info(f"Epochs: {config['training']['num_epochs']}")
        
        # Log additional info if available
        if 'training_mode' in config['training']:
            self.logger.info(f"Training Mode: {config['training']['training_mode']}")
        if 'target_type' in config['training']:
            self.logger.info(f"Target Type: {config['training']['target_type']}")
        
        self.logger.info(f"{'='*80}")
    
    def log_training_end(self):
        """Log training completion."""
        if not self.is_primary:
            return
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRAINING COMPLETED - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total Training Time: {duration}")
        self.logger.info(f"{'='*80}")
        
        # Save final statistics
        stats_file = self.training_logs_dir / "training_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        
        self.logger.info(f"Training statistics saved to {stats_file}")
        
        # Get final checkpoint info
        if self.checkpoint_manager:
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info()
            self.logger.info(f"Best checkpoint: {checkpoint_info['best_checkpoint']}")
            self.logger.info(f"Latest checkpoint: {checkpoint_info['last_checkpoint']}")
    
    def close(self):
        """Close all logging components."""
        if not self.is_primary:
            return
        
        if self.writer:
            self.writer.close()
        
        # Close logging handlers
        for handler in self.logger.handlers:
            handler.close()
    
    def get_paths(self) -> Dict[str, str]:
        """Get all logging paths for integration with existing trainer."""
        if not self.is_primary:
            return {}
        
        return {
            'experiment_name': self.experiment_name,
            'base_dir': str(self.base_dir),
            'experiment_dir': str(self.experiment_dir),
            'checkpoint_dir': str(self.checkpoints_dir),
            'tensorboard_dir': str(self.tensorboard_dir),
            'log_dir': str(self.training_logs_dir),
            'config_dir': str(self.config_dir),
            'model_summary_dir': str(self.model_summary_dir)
        } 