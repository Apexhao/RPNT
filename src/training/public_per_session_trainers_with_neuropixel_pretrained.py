"""
Cross-Dataset Per-Session: Neuropixel Pretrained → Public Sessions
==================================================================

This module adapts a neuropixel pretrained encoder (with RoPE3D) for 
public dataset per-session evaluation by replacing the positional 
encoding with standard 1D RoPE (temporal-only).

Key Features:
- Load CrossSiteFoundationMAE pretrained weights
- Replace RoPE3D with standard 1D temporal RoPE
- Evaluate on public dataset individual sessions
- Per-session regression with aggregated statistics
- Professional logging and comprehensive evaluation

Usage:
    python src/training/public_per_session_trainers_with_neuropixel_pretrained.py \
        --config config/training/cross_dataset_neuropixel_to_public.yaml \
        --pretrained_path ./logs_neuropixel/pretrained_model/checkpoints/best.pth \
        --scenario cross_session
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics

# Import our modules
from ..models.transformer import CrossSiteFoundationMAE
from ..models.public_downstream import PublicVelocityPredictor
from ..data.public_single_session_dataset import (
    PublicSingleSessionDataset,
    create_single_session_datasets
)
from ..utils.helpers import load_config, set_seed
from .enhanced_logger import EnhancedLogger


# ===========================================================================================
# Standard 1D RoPE (Temporal Only) - Same as in downstream_trainer_with_public_pretrained.py
# ===========================================================================================

class StandardRoPE1D(nn.Module):
    """
    Standard 1D Rotary Position Embedding (RoPE) for temporal sequences.
    
    This is the vanilla RoPE used in models like LLaMA, adapted for our use case.
    Only encodes temporal position (no spatial/session coordinates).
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 2000, base: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.rope_dim = d_model  # For compatibility with attention.py
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequency bands
        dim = d_model // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Precompute cos and sin for all positions."""
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
            self._seq_len_cached = seq_len
        
        return self._cos_cached, self._sin_cached
    
    def forward(self, temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for temporal positions.
        
        Args:
            temporal_indices: [B, T] or [B, S, T]
            
        Returns:
            cos_emb, sin_emb: [..., T, d/2]
        """
        device = temporal_indices.device
        
        if temporal_indices.dim() == 2:  # [B, T]
            B, T = temporal_indices.shape
            expand_shape = (B, T, -1)
        elif temporal_indices.dim() == 3:  # [B, S, T]
            B, S, T = temporal_indices.shape
            expand_shape = (B, S, T, -1)
        else:
            raise ValueError(f"Expected 2D or 3D temporal_indices, got shape {temporal_indices.shape}")
        
        cos, sin = self._compute_cos_sin(T, device)
        
        cos_emb = cos.unsqueeze(0).expand(*expand_shape)
        sin_emb = sin.unsqueeze(0).expand(*expand_shape)
        
        return cos_emb, sin_emb
    
    def compute_freqs_cis(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute complex frequency tensors for RoPE application.
        Required by CausalAdaptiveKernelAttention.
        
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
# Encoder Adaptation
# ===========================================================================================

def replace_with_standard_rope(temporal_encoder: nn.Module, d_model: int, 
                               max_seq_length: int = 2000, base: float = 10000.0):
    """
    Replace RoPE3D/RoPE4D positional encoder with standard 1D RoPE.
    
    Args:
        temporal_encoder: SparseTemporalEncoder or PublicSparseTemporalEncoder
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        base: RoPE base frequency
        
    Returns:
        Modified encoder (in-place)
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


def load_neuropixel_encoder_for_public(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load neuropixel pretrained encoder and adapt for public data.
    
    Steps:
    1. Load CrossSiteFoundationMAE checkpoint
    2. Extract temporal_encoder
    3. Replace RoPE3D with StandardRoPE1D
    4. Return adapted encoder and foundation config
    
    Args:
        checkpoint_path: Path to neuropixel pretrained checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (adapted_temporal_encoder, foundation_config)
    """
    logging.info(f"📦 Loading neuropixel pretrained encoder from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint['config']
    model_config = config['model']
    
    logging.info(f"   Model config: d_model={model_config.get('d_model', 512)}, "
                f"temporal_layers={model_config.get('temporal_layers', 6)}, "
                f"pos_encoding={model_config.get('pos_encoding_type', 'rope_3d')}")
    
    # Create CrossSiteFoundationMAE
    from ..models import CrossSiteModelFactory
    
    foundation_model = CrossSiteModelFactory.create_mae_model(
        size=config['training']['model_size'],
        neural_dim=model_config.get('neural_dim', 75),
        d_model=model_config.get('d_model', 512),
        n_sites=model_config.get('n_sites', 17),
        temporal_layers=model_config.get('temporal_layers', 6),
        spatial_layers=model_config.get('spatial_layers', 4),
        heads=model_config.get('heads', 8),
        dropout=model_config.get('dropout', 0.1),
        kernel_size=model_config.get('kernel_size', [3, 3]),
        max_seq_length=model_config.get('max_seq_length', 2000),
        pos_encoding_type=model_config.get('pos_encoding_type', 'rope_3d'),
        spatial_scale=model_config.get('spatial_scale', 0.1),
        use_temporal_kernels=model_config.get('use_temporal_kernels', True),
        use_mae_decoder=model_config.get('use_mae_decoder', True),
        use_site_specific_heads=model_config.get('use_site_specific_heads', True)
    )
    
    # Load pretrained weights
    foundation_model.load_state_dict(checkpoint['model_state_dict'])
    foundation_model = foundation_model.to(device)
    
    logging.info("✅ Neuropixel foundation model loaded")
    
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
    
    logging.info("🎯 Neuropixel encoder adapted for public data (Standard RoPE)")
    
    return temporal_encoder, config


# ===========================================================================================
# Cross-Dataset Model Wrapper
# ===========================================================================================

class CrossDatasetPublicModel(nn.Module):
    """
    Wrapper for neuropixel encoder adapted for public dataset.
    
    This provides a clean interface compatible with PublicVelocityPredictor
    but without requiring session coordinates.
    """
    
    def __init__(self, adapted_temporal_encoder: nn.Module, d_model: int = 512):
        super().__init__()
        
        self.temporal_encoder = adapted_temporal_encoder
        self.d_model = d_model
        
        # Prediction head for velocity
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)  # Output: [x, y] velocity
        )
    
    def forward(self, neural_data: torch.Tensor, session_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass without requiring session coordinates.
        
        Args:
            neural_data: [B, 1, T, N] - single-site neural data
            session_coords: Ignored (for interface compatibility)
            
        Returns:
            predictions: [B, T, 2] - velocity predictions
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Create causal mask
        from ..models.attention import create_causal_mask
        causal_mask = create_causal_mask(T, device)
        
        # Create dummy site_coords to satisfy SparseTemporalEncoder signature
        # Shape: [B, S=1, T, 2] - not used by StandardRoPE but required by forward()
        dummy_site_coords = torch.zeros(B, S, T, 2, device=device)
        
        # Forward through adapted encoder
        encoded = self.temporal_encoder(
            neural_data=neural_data,
            site_coords=dummy_site_coords,  # Dummy coords (ignored by StandardRoPE)
            causal_mask=causal_mask,
            mask_data=None
        )  # [B, S, T, D]
        
        # Prediction head
        predictions = self.prediction_head(encoded)  # [B, S, T, 2]
        
        # Remove site dimension (S=1)
        predictions = predictions.squeeze(1)  # [B, T, 2]
        
        return predictions


# ===========================================================================================
# Per-Session Evaluation Manager
# ===========================================================================================

class CrossDatasetPerSessionEvaluationManager:
    """
    Orchestrates per-session evaluation with neuropixel pretrained encoder.
    
    Adapted from PublicPerSessionEvaluationManager but using neuropixel encoder.
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 evaluation_scenario: str,
                 config: Dict[str, Any]):
        """
        Initialize evaluation manager.
        
        Args:
            checkpoint_path: Path to neuropixel pretrained checkpoint
            evaluation_scenario: 'cross_session', 'cross_subject_center', etc.
            config: Configuration dictionary
        """
        self.checkpoint_path = checkpoint_path
        self.evaluation_scenario = evaluation_scenario
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration
        self.training_config = config.get('training', {})
        self.paths_config = config.get('paths', {})
        self.dataset_config = config.get('dataset', {})
        self.model_config = config.get('model', {})
        
        # Setup logging
        self._setup_logging()
        
        # Load session datasets
        self.session_datasets = self._create_session_datasets()
        
        # Storage for results
        self.session_results = {}
        self.aggregated_results = {}
        
        # Progress tracking
        self.total_sessions = len(self.session_datasets)
        self.completed_sessions = 0
        
        self.logger.info(f"CrossDatasetPerSessionEvaluationManager initialized for {self.evaluation_scenario}")
        self.logger.info(f"Total sessions to evaluate: {self.total_sessions}")
        self.logger.info(f"Cross-dataset: Neuropixel pretrained → Public sessions")
        self.logger.info(f"Checkpoint path: {self.checkpoint_path}")
    
    def _setup_logging(self):
        """Setup logging system."""
        experiment_name = self.paths_config.get('experiment_name') or f"cross_dataset_neuropixel_to_public_{self.evaluation_scenario}"
        base_dir = self.paths_config.get('base_dir', './logs_cross_dataset')
        
        self.enhanced_logger = EnhancedLogger(
            experiment_name=experiment_name,
            base_dir=base_dir,
            top_k_checkpoints=1,
            local_rank=0
        )
        
        enhanced_paths = self.enhanced_logger.get_paths()
        self.paths_config.update(enhanced_paths)
        
        self.logger = self.enhanced_logger.logger
    
    def _create_session_datasets(self) -> List[PublicSingleSessionDataset]:
        """Create list of single session datasets."""
        dataset_kwargs = {
            'target_neurons': self.dataset_config.get('target_neurons', 50),
            'sequence_length': self.dataset_config.get('sequence_length', 50),
            'neuron_selection_strategy': self.dataset_config.get('neuron_selection_strategy', 'first_n'),
            'random_seed': self.dataset_config.get('random_seed', 42),
            'data_root': self.dataset_config.get('data_root', '/data/Fang-analysis/causal-nfm/Data/public_data')
        }
        
        try:
            session_datasets = create_single_session_datasets(
                scenario=self.evaluation_scenario,
                **dataset_kwargs
            )
            
            if not session_datasets:
                raise ValueError(f"No datasets created for scenario: {self.evaluation_scenario}")
            
            self.logger.info(f"Created {len(session_datasets)} session datasets")
            return session_datasets
            
        except Exception as e:
            self.logger.error(f"Failed to create session datasets: {str(e)}")
            raise e
    
    def run_per_session_evaluation(self) -> Dict[str, Any]:
        """
        Main evaluation loop: train and evaluate each session individually.
        
        Returns:
            Dictionary with session results and aggregated statistics
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STARTING PER-SESSION EVALUATION: {self.evaluation_scenario.upper()}")
        self.logger.info(f"Cross-Dataset: Neuropixel Pretrained → Public Sessions")
        self.logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        # Run evaluation for each session
        for i, session_dataset in enumerate(self.session_datasets):
            session_id = session_dataset.get_session_id()
            
            self.logger.info(f"\n🔄 Session {i+1}/{self.total_sessions}: {session_id}")
            self.logger.info("-" * 60)
            
            session_start_time = time.time()
            
            try:
                # Create fresh adapted model for this session
                model = self._create_adapted_model()
                
                # Create trainer for this session
                session_trainer = CrossDatasetSingleSessionTrainer(
                    model=model,
                    session_dataset=session_dataset,
                    config=self.config,
                    session_index=i,
                    total_sessions=self.total_sessions,
                    parent_logger=self.logger,
                    tensorboard_writer=self.enhanced_logger.writer,
                    session_checkpoint_dir=self.paths_config['checkpoint_dir']
                )
                
                # Train and evaluate
                session_result = session_trainer.train_and_evaluate()
                
                # Store results
                self.session_results[session_id] = session_result
                
                session_time = time.time() - session_start_time
                
                self.logger.info(f"✅ Session {i+1}/{self.total_sessions} completed in {session_time:.1f}s: "
                               f"{session_id} (R²: {session_result['r2_mean']:.4f})")
                
                self.completed_sessions += 1
                
            except Exception as e:
                self.logger.error(f"❌ Session {i+1}/{self.total_sessions} failed: {session_id}")
                self.logger.error(f"Error: {str(e)}")
                continue
        
        # Aggregate results
        self.aggregated_results = self._aggregate_session_results()
        self._log_aggregated_results_to_tensorboard()
        
        total_time = time.time() - start_time
        
        # Save and report results
        results_dict = self._save_final_results()
        self._print_final_summary(total_time)
        
        # Clean up
        self.enhanced_logger.log_training_end()
        self.enhanced_logger.close()
        
        return results_dict
    
    def _create_adapted_model(self) -> CrossDatasetPublicModel:
        """Create adapted model with neuropixel encoder."""
        # Load and adapt neuropixel encoder
        adapted_encoder, foundation_config = load_neuropixel_encoder_for_public(
            self.checkpoint_path,
            self.device
        )
        
        # Get d_model from config
        d_model = foundation_config['model'].get('d_model', 512)
        
        # Create cross-dataset model
        model = CrossDatasetPublicModel(
            adapted_temporal_encoder=adapted_encoder,
            d_model=d_model
        ).to(self.device)
        
        return model
    
    def _aggregate_session_results(self) -> Dict[str, float]:
        """Compute aggregated statistics across sessions."""
        if not self.session_results:
            self.logger.warning("No session results to aggregate")
            return {}
        
        metric_names = list(list(self.session_results.values())[0].keys())
        aggregated = {}
        
        self.logger.info(f"\n📊 Aggregating results across {len(self.session_results)} sessions")
        
        for metric in metric_names:
            values = [session_data[metric] for session_data in self.session_results.values() 
                     if metric in session_data and not math.isnan(session_data[metric])]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_count'] = len(values)
                
                self.logger.info(f"   {metric}: {aggregated[f'{metric}_mean']:.4f} ± {aggregated[f'{metric}_std']:.4f}")
        
        return aggregated
    
    def _log_aggregated_results_to_tensorboard(self):
        """Log aggregated results to TensorBoard."""
        if not self.enhanced_logger.writer or not self.aggregated_results:
            return
        
        writer = self.enhanced_logger.writer
        
        if 'r2_mean_mean' in self.aggregated_results:
            writer.add_scalar('Aggregated_Results/R2_Mean_Across_Sessions', 
                            self.aggregated_results['r2_mean_mean'], 0)
            writer.add_scalar('Aggregated_Results/R2_Std_Across_Sessions', 
                            self.aggregated_results['r2_mean_std'], 0)
        
        writer.add_scalar('Aggregated_Results/Number_of_Sessions', 
                        len(self.session_results), 0)
        
        # Log per-session results
        for session_id, metrics in self.session_results.items():
            r2_score = metrics.get('r2_mean', 0)
            clean_session_id = session_id.replace('/', '_').replace('\\', '_').replace('.', '_')
            writer.add_scalar(f'Final_Session_Results/R2_{clean_session_id}', r2_score, 0)
        
        self.logger.info("📊 Aggregated results logged to TensorBoard")
    
    def _save_final_results(self) -> Dict[str, Any]:
        """Save comprehensive evaluation results."""
        results_dict = {
            'evaluation_scenario': self.evaluation_scenario,
            'cross_dataset_type': 'neuropixel_to_public',
            'total_sessions': self.total_sessions,
            'completed_sessions': self.completed_sessions,
            'pretrained_model_path': self.checkpoint_path,
            'session_results': self.session_results,
            'aggregated_results': self.aggregated_results,
            'config': self.config,
            'session_ids': list(self.session_results.keys())
        }
        
        results_path = Path(self.paths_config['experiment_dir']) / 'per_session_evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to: {results_path}")
        
        return results_dict
    
    def _print_final_summary(self, total_time: float):
        """Print comprehensive evaluation summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 PER-SESSION EVALUATION SUMMARY")
        self.logger.info(f"{'='*80}")
        
        self.logger.info(f"🎯 Evaluation Scenario: {self.evaluation_scenario}")
        self.logger.info(f"🔄 Cross-Dataset: Neuropixel → Public")
        self.logger.info(f"📋 Sessions Completed: {self.completed_sessions}/{self.total_sessions}")
        self.logger.info(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        if self.aggregated_results:
            self.logger.info(f"\n📈 Aggregated Results:")
            
            r2_mean = self.aggregated_results.get('r2_mean_mean', 0)
            r2_std = self.aggregated_results.get('r2_mean_std', 0)
            r2_min = self.aggregated_results.get('r2_mean_min', 0)
            r2_max = self.aggregated_results.get('r2_mean_max', 0)
            
            self.logger.info(f"   R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
            self.logger.info(f"   R² Range: [{r2_min:.4f}, {r2_max:.4f}]")
        
        self.logger.info(f"\n✅ Per-session evaluation completed successfully!")
        self.logger.info(f"{'='*80}")


# ===========================================================================================
# Single Session Trainer
# ===========================================================================================

class CrossDatasetSingleSessionTrainer:
    """
    Trainer for a single session with neuropixel encoder.
    
    Adapted from PublicSingleSessionTrainer but using adapted neuropixel encoder.
    """
    
    def __init__(self,
                 model: CrossDatasetPublicModel,
                 session_dataset: PublicSingleSessionDataset,
                 config: Dict[str, Any],
                 session_index: int,
                 total_sessions: int,
                 parent_logger: logging.Logger,
                 tensorboard_writer=None,
                 session_checkpoint_dir: Optional[str] = None):
        """Initialize single session trainer."""
        self.model = model
        self.session_dataset = session_dataset
        self.session_id = session_dataset.get_session_id()
        self.config = config
        self.session_index = session_index
        self.total_sessions = total_sessions
        self.logger = parent_logger
        self.writer = tensorboard_writer
        self.session_checkpoint_dir = session_checkpoint_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration
        self.training_config = config.get('training', {})
        
        # Training configuration
        self.target_type = self.training_config.get('target_type', 'velocity')
        self.num_epochs = self.training_config.get('num_epochs', 50)
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 20)
        
        # Initialize components
        self._setup_training_components()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_r2_score = float('-inf')
        self.epochs_without_improvement = 0
        
        self.logger.info(f"   Single session trainer initialized for {self.session_id}")
        self.logger.info(f"   Training mode: finetune_encoder (hardcoded), Target: {self.target_type}")
    
    def _setup_training_components(self):
        """Setup training components."""
        batch_size = self.training_config.get('batch_size', 32)
        
        # Dataloaders
        self.train_loader = self.session_dataset.create_dataloader(
            split='train', batch_size=batch_size, shuffle=True,
            num_workers=2, output_mode='regression'
        )
        self.val_loader = self.session_dataset.create_dataloader(
            split='val', batch_size=batch_size, shuffle=False,
            num_workers=2, output_mode='regression'
        )
        self.test_loader = self.session_dataset.create_dataloader(
            split='test', batch_size=batch_size, shuffle=False,
            num_workers=2, output_mode='regression'
        )
        
        # Optimizer and scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.get('learning_rate', 1e-3),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        decay_epochs = self.training_config.get('decay_epochs', 1000)
        decay_rate = self.training_config.get('decay_rate', 0.9)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_epochs, gamma=decay_rate
        )
        
        # Loss and metrics
        self.criterion = nn.MSELoss()
        self.r2_metric_x = torchmetrics.R2Score().to(self.device)
        self.r2_metric_y = torchmetrics.R2Score().to(self.device)
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute regression metrics."""
        for b in range(predictions.shape[0]):
            sample_pred = predictions[b]
            sample_target = targets[b]
            
            self.r2_metric_x.update(sample_pred[:, 0], sample_target[:, 0])
            self.r2_metric_y.update(sample_pred[:, 1], sample_target[:, 1])
        
        r2_x = self.r2_metric_x.compute().item()
        r2_y = self.r2_metric_y.compute().item()
        
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
        
        for batch_data in self.train_loader:
            neural_data = batch_data[0].to(self.device)      # [B, 1, T, N]
            session_coords = batch_data[1].to(self.device)   # [B, 1, 3]
            trajectories = batch_data[2].to(self.device)     # [B, T, 2]
            velocities = batch_data[3].to(self.device)       # [B, T, 2]
            
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # ✅ KEY: Pass session_coords but they're ignored internally
            predictions = self.model(neural_data, session_coords)  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Metrics
            metrics = self.compute_metrics(predictions, targets)
            
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        # Average
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        epoch_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return dict(epoch_stats)
    
    @torch.no_grad()
    def validate_epoch(self, split: str = 'val') -> Dict[str, float]:
        """Run validation/test epoch."""
        self.model.eval()
        
        loader = self.val_loader if split == 'val' else self.test_loader
        epoch_stats = defaultdict(float)
        num_batches = len(loader)
        
        for batch_data in loader:
            neural_data = batch_data[0].to(self.device)
            session_coords = batch_data[1].to(self.device)
            trajectories = batch_data[2].to(self.device)
            velocities = batch_data[3].to(self.device)
            
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            predictions = self.model(neural_data, session_coords)
            
            loss = self.criterion(predictions, targets)
            metrics = self.compute_metrics(predictions, targets)
            
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    def train_and_evaluate(self) -> Dict[str, float]:
        """Train and evaluate on this session."""
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch('val')
            test_stats = self.validate_epoch('test')
            
            self.scheduler.step()
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                self.logger.info(f"     Epoch {epoch:3d}: Train R²={train_stats['r2_mean']:.4f}, "
                               f"Val R²={val_stats['r2_mean']:.4f}, Test R²={test_stats['r2_mean']:.4f}")
            
            # Early stopping
            is_best = val_stats['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"     Early stopping at epoch {epoch}")
                break
        
        return test_stats


# ===========================================================================================
# Main Execution
# ===========================================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Cross-Dataset: Neuropixel Pretrained → Public Per-Session Evaluation'
    )
    
    parser.add_argument('--config', type=str,
                       default='config/training/cross_dataset_neuropixel_to_public.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--pretrained_path', type=str,
                       help='Path to neuropixel pretrained checkpoint')
    parser.add_argument('--scenario', type=str,
                       choices=['cross_session', 'cross_subject_center', 'cross_subject_random'],
                       help='Evaluation scenario')
    parser.add_argument('--base_dir', type=str,
                       help='Base directory for outputs')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.pretrained_path:
        config['paths']['pretrained_path'] = args.pretrained_path
    if args.scenario:
        evaluation_scenario = args.scenario
    else:
        evaluation_scenario = config.get('evaluation', {}).get('scenario', 'cross_session')
    if args.base_dir:
        config['paths']['base_dir'] = args.base_dir
    
    # Set random seed
    set_seed(args.seed)
    
    print("=" * 80)
    print("CROSS-DATASET PER-SESSION: NEUROPIXEL → PUBLIC")
    print("=" * 80)
    print(f"Pretrained: {config['paths']['pretrained_path']}")
    print(f"Scenario: {evaluation_scenario}")
    print(f"Training mode: finetune_encoder (hardcoded)")
    print("=" * 80)
    
    try:
        # Create evaluation manager
        manager = CrossDatasetPerSessionEvaluationManager(
            checkpoint_path=config['paths']['pretrained_path'],
            evaluation_scenario=evaluation_scenario,
            config=config
        )
        
        # Run evaluation
        results = manager.run_per_session_evaluation()
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise e


if __name__ == '__main__':
    main()

