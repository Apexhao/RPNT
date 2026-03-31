"""
Cross-Dataset Downstream: Public Pretrained → Neuropixel Regression
===================================================================

This module adapts a public dataset pretrained encoder (with RoPE4D) for 
neuropixel single-site downstream regression by replacing the positional 
encoding with standard 1D RoPE (temporal-only).

Key Features:
- Load PublicNeuralFoundationMAE pretrained weights
- Replace RoPE4D with standard 1D temporal RoPE
- Fine-tune on neuropixel single-site data (te14116)
- Velocity/position prediction regression task
- Professional training loop with comprehensive logging

Usage:
    python src/training/downstream_trainer_with_public_pretrained.py \
        --config config/training/cross_dataset_public_to_neuropixel.yaml \
        --pretrained_path ./logs_public/pretrained_model/checkpoints/best.pth
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics

# Import our modules
from ..models.public_transformer import PublicNeuralFoundationMAE
from ..models.downstream import SingleSiteDownstreamRegressor
from ..data.downstream_dataset import SingleSiteDownstreamDataset
from ..utils.helpers import load_config, set_seed
from .enhanced_logger import EnhancedLogger
from ..models.attention import create_causal_mask


# ===========================================================================================
# Standard 1D RoPE (Temporal Only)
# ===========================================================================================

class StandardRoPE1D(nn.Module):
    """
    Standard 1D Rotary Position Embedding (RoPE) for temporal sequences.
    
    This is the vanilla RoPE used in models like LLaMA, adapted for our use case.
    Only encodes temporal position (no spatial/session coordinates).
    
    Design:
    - Input: temporal_indices [B, T] or [B, S, T]
    - Output: cos/sin embeddings for Q, K rotations in attention
    - Compatible with existing RoPE-based attention mechanisms
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 2000, base: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.rope_dim = d_model  # For compatibility with attention modules
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequency bands
        # Following standard RoPE: θ_i = base^(-2i/d) for i in [0, d/2)
        dim = d_model // 2  # Half dimensions for cos, half for sin
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Precompute cos and sin for all positions."""
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            # Position indices: [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies: [seq_len, d/4]
            freqs = torch.outer(t, self.inv_freq)
            
            # Duplicate for full head dimension: [seq_len, d/2]
            emb = torch.cat([freqs, freqs], dim=-1)
            
            # Compute cos and sin
            self._cos_cached = emb.cos()  # [seq_len, d/2]
            self._sin_cached = emb.sin()  # [seq_len, d/2]
            self._seq_len_cached = seq_len
        
        return self._cos_cached, self._sin_cached
    
    def forward(self, temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for temporal positions.
        
        Args:
            temporal_indices: [B, T] or [B, S, T] - temporal position indices
            
        Returns:
            cos_emb: [..., T, d/2] - cosine embeddings
            sin_emb: [..., T, d/2] - sine embeddings
        """
        device = temporal_indices.device
        
        # Handle different input shapes
        if temporal_indices.dim() == 2:  # [B, T]
            B, T = temporal_indices.shape
            expand_shape = (B, T, -1)
        elif temporal_indices.dim() == 3:  # [B, S, T]
            B, S, T = temporal_indices.shape
            expand_shape = (B, S, T, -1)
        else:
            raise ValueError(f"Expected 2D or 3D temporal_indices, got shape {temporal_indices.shape}")
        
        # Compute cos/sin for this sequence length
        cos, sin = self._compute_cos_sin(T, device)  # [T, d/2]
        
        # Expand to match input batch dimensions
        cos_emb = cos.unsqueeze(0).expand(*expand_shape)
        sin_emb = sin.unsqueeze(0).expand(*expand_shape)
        
        return cos_emb, sin_emb
    
    def compute_freqs_cis(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute complex frequency tensors for RoPE application.
        Required by PublicCausalAdaptiveKernelAttention.
        
        Args:
            time_indices: [batch, seq_len] - temporal position indices
            
        Returns:
            freqs_cis: [batch, seq_len, rope_dim//2] - complex frequency tensor
        """
        batch_size, seq_len = time_indices.shape
        device = time_indices.device
        
        # Compute frequencies: [seq_len] × [dim//4] -> [seq_len, dim//4]
        t = time_indices[0].float()  # Assume same positions across batch
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim//4]
        
        # Duplicate to get full rope_dim//2
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, rope_dim//2]
        
        # Convert to complex: e^(i*freqs) = cos(freqs) + i*sin(freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [seq_len, rope_dim//2]
        
        # Expand for batch: [seq_len, rope_dim//2] -> [batch, seq_len, rope_dim//2]
        freqs_cis = freqs_cis.unsqueeze(0).expand(batch_size, -1, -1)
        
        return freqs_cis
    
    def apply_rope(self,
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   session_coords: torch.Tensor,         # [batch, seq_len, 3] (IGNORED)
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 1D temporal RoPE to query and key tensors.
        Unified interface compatible with attention modules expecting RoPE4D.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - IGNORED (for interface compatibility)
            temporal_indices: [batch, seq_len] - temporal position indices
            
        Returns:
            q_rotated: [batch, seq_len, n_heads, head_dim] - rotated query
            k_rotated: [batch, seq_len, n_heads, head_dim] - rotated key
        """
        batch_size, seq_len, n_heads, head_dim = q.shape
        device = q.device
        
        # Compute cos/sin for this sequence length
        cos, sin = self._compute_cos_sin(seq_len, device)  # [seq_len, d/2]
        
        # Expand for batch and heads: [seq_len, d/2] -> [batch, seq_len, 1, d/2]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Determine how much of head_dim to apply RoPE to
        rope_head_dim = min(head_dim, self.d_model)
        rope_head_dim = (rope_head_dim // 2) * 2  # Ensure even number for complex pairs
        
        # Extract RoPE portions
        q_rope = q[..., :rope_head_dim]  # [batch, seq_len, n_heads, rope_head_dim]
        k_rope = k[..., :rope_head_dim]
        
        # Apply rotation using standard RoPE formula
        q_rotated = self._apply_rotation(q_rope, cos[..., :rope_head_dim//2], sin[..., :rope_head_dim//2])
        k_rotated = self._apply_rotation(k_rope, cos[..., :rope_head_dim//2], sin[..., :rope_head_dim//2])
        
        # Combine rotated and non-rotated parts
        if rope_head_dim < head_dim:
            q_pass = q[..., rope_head_dim:]
            k_pass = k[..., rope_head_dim:]
            q = torch.cat([q_rotated, q_pass], dim=-1)
            k = torch.cat([k_rotated, k_pass], dim=-1)
        else:
            q = q_rotated
            k = k_rotated
        
        return q, k
    
    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to tensor x.
        
        Args:
            x: [batch, seq_len, n_heads, rope_head_dim] - input tensor
            cos: [batch, seq_len, 1, rope_head_dim//2] - cosine values
            sin: [batch, seq_len, 1, rope_head_dim//2] - sine values
            
        Returns:
            rotated: [batch, seq_len, n_heads, rope_head_dim] - rotated tensor
        """
        # Split x into even and odd indices
        x1 = x[..., 0::2]  # [batch, seq_len, n_heads, rope_head_dim//2]
        x2 = x[..., 1::2]  # [batch, seq_len, n_heads, rope_head_dim//2]
        
        # Apply rotation: (x1, x2) * (cos, sin)
        # Rotation formula: [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,  # Even indices
            x1 * sin + x2 * cos   # Odd indices
        ], dim=-1).flatten(-2)  # Interleave back
        
        return x_rotated


# ===========================================================================================
# Adapted Public Encoder Regressor
# ===========================================================================================

class AdaptedPublicEncoderRegressor(nn.Module):
    """
    Regression model using adapted PublicSparseTemporalEncoder for neuropixel data.
    
    This wrapper is specifically designed for cross-dataset transfer where a
    public encoder (with StandardRoPE) is used for neuropixel downstream tasks.
    
    Key Features:
    - Accepts neural_data in [B, T, N] format (neuropixel style)
    - Converts to [B, 1, T, N] for PublicSparseTemporalEncoder
    - Creates dummy session_coords [B, 1, 3] (not used by StandardRoPE)
    - Returns predictions [B, T, 2] for velocity
    """
    
    def __init__(self,
                 adapted_temporal_encoder,  # PublicSparseTemporalEncoder with StandardRoPE
                 output_dim: int = 2,
                 prediction_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.temporal_encoder = adapted_temporal_encoder
        self.d_model = adapted_temporal_encoder.d_model
        self.training_mode = 'finetune_encoder'  # For compatibility
        
        # Prediction head for velocity/position
        if prediction_layers == 2:
            self.prediction_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model // 2, output_dim)
            )
        else:
            self.prediction_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, output_dim)
            )
    
    def forward(self, neural_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without coordinate inputs (StandardRoPE handles internally).
        
        Args:
            neural_data: [B, T, N] - single site neural data (neuropixel format)
            
        Returns:
            predictions: [B, T, 2] - velocity/position predictions
        """
        B, T, N = neural_data.shape
        device = neural_data.device
        
        # Add site dimension: [B, T, N] -> [B, 1, T, N] for PublicSparseTemporalEncoder
        neural_data_4d = neural_data.unsqueeze(1)
        
        # Create dummy session coords [B, 1, 3] - not used by StandardRoPE but required by signature
        dummy_session_coords = torch.zeros(B, 1, 3, device=device)
        
        # Create causal mask
        causal_mask = create_causal_mask(T, device)
        
        # Forward through adapted encoder
        encoded = self.temporal_encoder(
            neural_data=neural_data_4d,
            session_coords=dummy_session_coords,
            causal_mask=causal_mask,
            mask_data=None
        )  # [B, 1, T, D]
        
        # Remove site dimension: [B, 1, T, D] -> [B, T, D]
        encoded = encoded.squeeze(1)
        
        # Apply prediction head: [B, T, D] -> [B, T, output_dim]
        predictions = self.prediction_head(encoded)
        
        return predictions


# ===========================================================================================
# Encoder Adaptation
# ===========================================================================================

def replace_with_standard_rope(temporal_encoder: nn.Module, d_model: int, 
                               max_seq_length: int = 2000, base: float = 10000.0):
    """
    Replace RoPE4D/RoPE3D positional encoder with standard 1D RoPE.
    
    This function modifies the encoder in-place:
    1. Replaces positional_encoder module with StandardRoPE1D
    2. Keeps use_rope=True (attention mechanism unchanged)
    3. Updates attention layers' rope_module reference
    
    Args:
        temporal_encoder: PublicSparseTemporalEncoder or SparseTemporalEncoder
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        base: RoPE base frequency
        
    Returns:
        Modified encoder (in-place modification)
    """
    logging.info("🔄 Replacing positional encoding with Standard 1D RoPE...")
    
    # Replace positional encoder
    temporal_encoder.positional_encoder = StandardRoPE1D(
        d_model=d_model,
        max_seq_length=max_seq_length,
        base=base
    )
    
    # Ensure use_rope is True
    temporal_encoder.use_rope = True
    
    # Update attention layers' rope_module reference
    for layer in temporal_encoder.layers:
        if hasattr(layer['attention'], 'rope_module'):
            layer['attention'].rope_module = temporal_encoder.positional_encoder
            layer['attention'].use_rope = True
    
    logging.info("✅ Positional encoding replaced successfully")
    
    return temporal_encoder


def load_public_encoder_for_neuropixel(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load public pretrained encoder and adapt for neuropixel data.
    
    Steps:
    1. Load PublicNeuralFoundationMAE checkpoint
    2. Extract temporal_encoder
    3. Replace RoPE4D with StandardRoPE1D
    4. Return adapted encoder ready for neuropixel data
    
    Args:
        checkpoint_path: Path to public pretrained checkpoint
        device: Device to load model on
        
    Returns:
        Adapted temporal encoder with standard RoPE
    """
    logging.info(f"📦 Loading public pretrained encoder from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint['config']
    model_config = config['model']
    
    logging.info(f"   Model config: d_model={model_config.get('d_model', 512)}, "
                f"temporal_layers={model_config.get('temporal_layers', 6)}, "
                f"pos_encoding={model_config.get('pos_encoding_type', 'rope_4d')}")
    
    # Create PublicNeuralFoundationMAE
    foundation_model = PublicNeuralFoundationMAE(
        neural_dim=model_config.get('neural_dim', 50),
        d_model=model_config.get('d_model', 512),
        temporal_layers=model_config.get('temporal_layers', 6),
        heads=model_config.get('heads', 8),
        dropout=model_config.get('dropout', 0.1),
        kernel_size=model_config.get('kernel_size', [3, 3]),
        max_seq_length=model_config.get('max_seq_length', 2000),
        pos_encoding_type=model_config.get('pos_encoding_type', 'rope_4d'),
        session_scale=model_config.get('session_scale', 1.0),
        use_temporal_kernels=model_config.get('use_temporal_kernels', True),
        use_mae_decoder=model_config.get('use_mae_decoder', True),
        use_session_specific_heads=model_config.get('use_session_specific_heads', True)
    )
    
    # Load pretrained weights
    foundation_model.load_state_dict(checkpoint['model_state_dict'])
    foundation_model = foundation_model.to(device)
    
    logging.info("✅ Public foundation model loaded")
    
    # Extract temporal encoder
    temporal_encoder = foundation_model.temporal_encoder
    
    # Replace with standard RoPE
    d_model = model_config.get('d_model', 512)
    max_seq_length = model_config.get('max_seq_length', 2000)
    rope_base = model_config.get('rope_base', 10000.0)
    
    temporal_encoder = replace_with_standard_rope(
        temporal_encoder,
        d_model=d_model,
        max_seq_length=max_seq_length,
        base=rope_base
    )
    
    logging.info("🎯 Public encoder adapted for neuropixel data (Standard RoPE)")
    
    return temporal_encoder


# ===========================================================================================
# Regression Trainer (Adapted from downstream_trainers.py)
# ===========================================================================================

class CrossDatasetRegressionTrainer:
    """
    Regression trainer for cross-dataset generalization.
    
    Adapted from RegressionTrainer but with:
    - Public pretrained encoder (with standard RoPE)
    - No coordinate inputs in forward passes
    - Simplified data flow (temporal-only positional encoding)
    
    Features:
    - Fine-tune pretrained encoder
    - MSE loss with R² metrics
    - Professional logging and checkpointing
    - Velocity/position prediction
    """
    
    def __init__(self,
                 pretrained_temporal_encoder: nn.Module,
                 dataset: SingleSiteDownstreamDataset,
                 config: Dict[str, Any]):
        """
        Initialize CrossDatasetRegressionTrainer.
        
        Args:
            pretrained_temporal_encoder: Adapted public temporal encoder
            dataset: SingleSiteDownstreamDataset for neuropixel data
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        
        # Extract configuration
        self.training_config = config.get('training', {})
        self.paths_config = config.get('paths', {})
        
        # Create downstream model using AdaptedPublicEncoderRegressor
        output_dim = self.training_config.get('output_dim', 2)
        prediction_layers = config.get('model', {}).get('prediction_layers', 2)
        dropout = config.get('model', {}).get('dropout', 0.1)
        
        self.model = AdaptedPublicEncoderRegressor(
            adapted_temporal_encoder=pretrained_temporal_encoder,
            output_dim=output_dim,
            prediction_layers=prediction_layers,
            dropout=dropout
        ).to(self.device)
        
        # Setup components
        self._setup_enhanced_logging()
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._setup_loss_function()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Regression-specific config
        self.target_type = self.training_config.get('target_type', 'velocity')
        
        # Log initialization
        self.logger.info("CrossDatasetRegressionTrainer initialized (Public→Neuropixel)")
        self.logger.info(f"Training mode: finetune_encoder (hardcoded)")
        self.logger.info(f"Target type: {self.target_type}")
        self._log_model_info()
        self._save_config()
    
    def _setup_enhanced_logging(self):
        """Setup enhanced logging system."""
        experiment_name = self.paths_config.get('experiment_name') or f"cross_dataset_public_to_neuropixel_{self.dataset.dataset_id}"
        base_dir = self.paths_config.get('base_dir', './logs_cross_dataset')
        
        self.enhanced_logger = EnhancedLogger(
            experiment_name=experiment_name,
            base_dir=base_dir,
            top_k_checkpoints=3,
            local_rank=0
        )
        
        enhanced_paths = self.enhanced_logger.get_paths()
        self.paths_config.update(enhanced_paths)
        
        self.logger = self.enhanced_logger.logger
        self.writer = self.enhanced_logger.writer
    
    def _setup_data_loaders(self):
        """Setup train/validation/test data loaders."""
        batch_size = self.training_config.get('batch_size', 32)
        
        self.train_loader = self.dataset.create_dataloader(
            split='train', batch_size=batch_size, shuffle=True,
            num_workers=4, output_mode='regression'
        )
        
        self.val_loader = self.dataset.create_dataloader(
            split='val', batch_size=batch_size, shuffle=False,
            num_workers=4, output_mode='regression'
        )
        
        self.test_loader = self.dataset.create_dataloader(
            split='test', batch_size=batch_size, shuffle=False,
            num_workers=4, output_mode='regression'
        )
        
        self.logger.info(f"Data loaders created - Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)} batches")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.get('learning_rate', 1e-4),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        decay_epochs = self.training_config.get('decay_epochs', 50)
        decay_rate = self.training_config.get('decay_rate', 0.9)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_epochs, gamma=decay_rate
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        
        self.logger.info(f"Optimizer initialized - Total params: {total_params:,}, "
                        f"Trainable: {trainable_params_count:,} "
                        f"({100*trainable_params_count/total_params:.1f}%)")
    
    def _setup_loss_function(self):
        """Setup MSE loss function."""
        self.criterion = nn.MSELoss()
    
    def _log_model_info(self):
        """Log detailed model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_info = {
            'task_type': 'regression',
            'cross_dataset_type': 'public_to_neuropixel',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0,
            'device': str(self.device),
            'dataset_id': self.dataset.dataset_id,
            'training_mode': 'finetune_encoder'
        }
        
        self.logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
    
    def _save_config(self):
        """Save configuration and model summary."""
        try:
            self.enhanced_logger.log_training_start(self.config, self.model)
        except Exception as e:
            self.logger.warning(f"Failed to save config: {e}")
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute regression metrics (R² scores)."""
        r2_metric_x = torchmetrics.R2Score().to(self.device)
        r2_metric_y = torchmetrics.R2Score().to(self.device)
        
        # Update metrics for each sample
        for b in range(predictions.shape[0]):
            sample_pred = predictions[b]
            sample_target = targets[b]
            
            r2_metric_x.update(sample_pred[:, 0], sample_target[:, 0])
            r2_metric_y.update(sample_pred[:, 1], sample_target[:, 1])
        
        r2_x = r2_metric_x.compute().item()
        r2_y = r2_metric_y.compute().item()
        
        metrics = {
            'r2_x': r2_x,
            'r2_y': r2_y,
            'r2_mean': (r2_x + r2_y) / 2
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
            
            # Forward pass - AdaptedPublicEncoderRegressor handles everything internally
            predictions = self.model(neural_data.squeeze(1))  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(predictions.detach(), targets)
            
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
            
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Forward pass - AdaptedPublicEncoderRegressor handles everything internally
            predictions = self.model(neural_data.squeeze(1))  # [B, T, 2]
            
            # Compute loss and metrics
            loss = self.criterion(predictions, targets)
            metrics = self.compute_metrics(predictions, targets)
            
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
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
            'task_type': 'regression',
            'cross_dataset_type': 'public_to_neuropixel',
            'dataset_id': self.dataset.dataset_id
        }
        
        saved_paths = self.enhanced_logger.save_checkpoint(checkpoint, epoch, val_loss)
        return saved_paths
    
    def train(self, num_epochs: int):
        """Main training loop."""
        self.logger.info(f"Starting cross-dataset training for {num_epochs} epochs")
        self.logger.info(f"Public pretrained → Neuropixel {self.dataset.dataset_id}")
        
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
            
            saved_paths = self.save_checkpoint(epoch, val_stats['loss'], metrics)
            
            if is_best:
                self.logger.info(f"New best model saved (val_R²: {val_stats['r2_mean']:.4f})")
            
            # Early stopping
            early_stopping_patience = self.training_config.get('early_stopping_patience', 100)
            if self.epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completion
        self.enhanced_logger.log_training_end()
        self.enhanced_logger.close()


# ===========================================================================================
# Main Execution
# ===========================================================================================

def main():
    """Main training function for cross-dataset generalization."""
    parser = argparse.ArgumentParser(
        description='Cross-Dataset: Public Pretrained → Neuropixel Downstream Regression'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       default='config/training/cross_dataset_public_to_neuropixel.yaml',
                       help='Path to configuration file')
    
    # Paths
    parser.add_argument('--pretrained_path', type=str,
                       help='Path to public pretrained checkpoint (overrides config)')
    parser.add_argument('--dataset_id', type=str,
                       help='Neuropixel dataset ID (overrides config)')
    parser.add_argument('--base_dir', type=str,
                       help='Base directory for outputs (overrides config)')
    
    # Training settings
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of epochs (overrides config)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.pretrained_path:
        config['paths']['pretrained_path'] = args.pretrained_path
    if args.dataset_id:
        config['dataset']['dataset_id'] = args.dataset_id
    if args.base_dir:
        config['paths']['base_dir'] = args.base_dir
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    # Set random seed
    set_seed(args.seed)
    
    print("=" * 80)
    print("CROSS-DATASET DOWNSTREAM: PUBLIC → NEUROPIXEL")
    print("=" * 80)
    print(f"Pretrained: {config['paths']['pretrained_path']}")
    print(f"Dataset: {config['dataset']['dataset_id']}")
    print(f"Training mode: finetune_encoder (hardcoded)")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and adapt public encoder
        print("\n📦 Loading public pretrained encoder...")
        pretrained_encoder = load_public_encoder_for_neuropixel(
            checkpoint_path=config['paths']['pretrained_path'],
            device=device
        )
        
        # Initialize neuropixel dataset
        print("\n📊 Initializing neuropixel dataset...")
        dataset = SingleSiteDownstreamDataset(
            dataset_id=config['dataset']['dataset_id'],
            data_root=config['dataset']['data_root'],
            split_ratios=tuple(config['dataset']['split_ratios']),
            target_neurons=config['dataset']['target_neurons'],
            width=config['dataset']['width'],
            sequence_length=config['dataset']['sequence_length'],
            neuron_selection_strategy=config['dataset'].get('neuron_selection_strategy', 'first_n'),
            selected_neurons=config['dataset'].get('selected_neurons', 50),
            random_seed=config['dataset'].get('random_seed', 3407)
        )
        
        print(f"Dataset loaded: {dataset.dataset_id}")
        
        # Get split data to print shapes
        train_split = dataset.get_split_data('train')
        val_split = dataset.get_split_data('val')
        test_split = dataset.get_split_data('test')
        print(f"Train: {train_split['neural_data'].shape}, Val: {val_split['neural_data'].shape}, "
              f"Test: {test_split['neural_data'].shape}")
        
        # Initialize trainer
        print("\n🎯 Initializing cross-dataset trainer...")
        trainer = CrossDatasetRegressionTrainer(
            pretrained_temporal_encoder=pretrained_encoder,
            dataset=dataset,
            config=config
        )
        
        # Start training
        print("\n🚀 Starting training...")
        trainer.train(config['training']['num_epochs'])
        
        print("\n✅ Training completed successfully!")
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise e


if __name__ == '__main__':
    main()

