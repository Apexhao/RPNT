"""
Public Dataset Neural Foundation Model
------------------------------------

This module provides a temporal-only neural foundation model adapted for the public dataset.

Key Features:
- SparseTemporalEncoder ONLY (no spatial transformer for S=1)
- RoPE4D positional encoding for session-based coordinates
- Compatible with [B,S=1,T=50,N=50] input format
- Same MAE decoder for pretraining consistency
- Optimized for single-site temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List
import logging

# Import public dataset modules
from .public_positional_encoding import RoPE4D, create_public_positional_encoder, prepare_session_positional_data
# from .public_attention import PublicCausalAdaptiveKernelAttention, create_causal_mask
from .public_attention_fixed import PublicCausalAdaptiveKernelAttention, create_causal_mask



class PublicSparseTemporalEncoder(nn.Module):
    """
    Temporal Encoder for Public Dataset with RoPE4D Session Encoding.
    
    **DESIGN ADAPTATIONS**:
    - Single-site format: [B,S=1,T,N] -> [B,S=1,T,D]
    - RoPE4D integration for session-based positional encoding
    - Session coordinates instead of site coordinates
    - Same causal attention and temporal modeling as original
    
    **KEY FEATURES**:
    - Session-specific positional encoding with zero-shot generalization
    - Support for session coordinates (subject, time, task) + temporal positions
    - RoPE4D applied in attention, not as additive PE
    - Optimized for temporal-only processing (no spatial interactions)
    """
    
    def __init__(self,
                 neural_dim: int = 50,
                 d_model: int = 512,
                 num_layers: int = 6,
                 heads: int = 8,
                 kernel_size: List[int] = [3, 3],
                 dropout: float = 0.1,
                 max_seq_length: int = 2000,
                 pos_encoding_type: str = 'rope_4d',
                 session_scale: float = 1.0,
                 use_temporal_kernels: bool = True,
                 # Ablation study parameters
                 max_subjects: int = 10,
                 max_time_periods: int = 10,
                 max_tasks: int = 10,
                 rope_base: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.use_temporal_kernels = use_temporal_kernels
        self.pos_encoding_type = pos_encoding_type
        
        # Channel projection: neural_dim -> d_model
        self.channel_projection = nn.Linear(neural_dim, d_model)
        
        # Session-based positional encoder
        self.positional_encoder = create_public_positional_encoder(
            encoding_type=pos_encoding_type,
            d_model=d_model,
            max_seq_length=max_seq_length,
            session_scale=session_scale,
            max_subjects=max_subjects,
            max_time_periods=max_time_periods,
            max_tasks=max_tasks,
            base=rope_base
        )
        
        # Check if we're using RoPE (requires different handling)
        self.use_rope = (pos_encoding_type in ['rope_4d', 'standard_rope'])
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers with RoPE4D support
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': PublicCausalAdaptiveKernelAttention(
                    d_model=d_model, 
                    heads=heads, 
                    kernel_size=kernel_size, 
                    dropout=dropout,
                    use_kernel=use_temporal_kernels,
                    use_rope=self.use_rope,
                    rope_module=self.positional_encoder if self.use_rope else None
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
    
    def forward(self, 
                neural_data: torch.Tensor,        # [B, S=1, T, N]
                session_coords: torch.Tensor,     # [B, S=1, 3] - session coordinates
                causal_mask: torch.Tensor,        # [T, T]
                mask_data: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, S=1, T]
        """
        Forward pass with session-specific positional encoding.
        
        Args:
            neural_data: [B, S=1, T, N] - neural activity for single site
            session_coords: [B, S=1, 3] - (subject, time, task) coordinates
            causal_mask: [T, T] - causal attention mask
            mask_data: [B, S=1, T] - optional MAE mask
            
        Returns:
            encoded: [B, S=1, T, D] - temporally encoded features
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Validate single-site format
        if S != 1:
            raise ValueError(f"PublicSparseTemporalEncoder expects S=1, got S={S}")
        
        # Channel projection: [B, S=1, T, N] -> [B, S=1, T, D]
        x = self.channel_projection(neural_data)
        
        # Prepare session positional data
        coords_expanded, temporal_indices = prepare_session_positional_data(
            neural_data, session_coords, device
        )  # [B, S=1, T, 3], [B, S=1, T]
        
        # Handle positional encoding based on type
        if self.use_rope:
            # RoPE4D: Do NOT add positional encoding to embeddings!
            # RoPE will be applied directly in attention mechanism
            pass
        else:
            # Traditional positional encoding: add to embeddings for standard sinusoidal encoding and learnable multi-dimensional encoding
            pos_encoding = self.positional_encoder(coords_expanded, temporal_indices)
            x = x + pos_encoding
        
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            # Multi-head attention with residual connection
            attn_output = layer['attention'](
                x=layer['norm1'](x),
                session_coords=coords_expanded,
                temporal_indices=temporal_indices,
                causal_mask=causal_mask,
                historical_data=layer['norm1'](x),
                key_padding_mask=mask_data
            )
            x = x + attn_output
            
            # Feed-forward with residual connection
            ffn_output = layer['ffn'](layer['norm2'](x))
            x = x + ffn_output
        
        return x


class PublicLightweightMAEDecoder(nn.Module):
    """
    Lightweight MAE Decoder for Public Dataset Pretraining.
    
    **DESIGN ADAPTATIONS**:
    - Single-site format: [B,S=1,T,D] -> [B,S=1,T,N]
    - Session-specific reconstruction head
    - Poisson loss compatibility for neural spike data
    - Maintains compatibility with original decoder design
    """
    
    def __init__(self,
                 d_model: int = 512,
                 neural_dim: int = 50,
                 dropout: float = 0.1,
                 use_session_specific_heads: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.neural_dim = neural_dim
        self.use_session_specific_heads = use_session_specific_heads
        
        # Lightweight decoder layers
        self.decoder_layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Final reconstruction head
        if use_session_specific_heads:
            # Single session-specific head for S=1
            self.reconstruction_head = nn.Linear(d_model // 4, neural_dim)
        else:
            # Shared head
            self.reconstruction_head = nn.Linear(d_model // 4, neural_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MAE reconstruction.
        
        Args:
            encoded_features: [B, S=1, T, D] - encoded temporal features
            
        Returns:
            reconstruction: [B, S=1, T, N] - reconstructed neural activity
        """
        B, S, T, D = encoded_features.shape
        
        # Apply decoder layers: [B, S, T, D] -> [B, S, T, D//4]
        decoded = self.decoder_layers(encoded_features)
        
        # Reconstruction head: [B, S, T, D//4] -> [B, S, T, N]
        reconstruction = self.reconstruction_head(decoded)
        
        # Apply ReLU to ensure non-negative outputs (spike counts)
        reconstruction = F.relu(reconstruction)
        
        return reconstruction


class PublicNeuralFoundationMAE(nn.Module):
    """
    Public Dataset Neural Foundation Model with Temporal-Only Processing.
    
    **CORE ARCHITECTURE**:
    - Input: [B, S=1, T, N] (Batch, Single Site, Time, Neural_dim)
    - Stage 1: PublicSparseTemporalEncoder - processes with RoPE4D and causal attention
    - Stage 2: PublicLightweightMAEDecoder - Poisson reconstruction for pretraining
    
    **KEY ADAPTATIONS**:
    - NO spatial encoder (S=1, no cross-site processing needed)
    - RoPE4D session coordinates (subject, time, task, temporal)
    - Same MAE training paradigm as original model
    - Optimized for single-site temporal modeling
    """
    
    def __init__(self,
                 neural_dim: int = 50,          # N: neurons per site
                 d_model: int = 512,            # D: model hidden dimension
                 temporal_layers: int = 6,       # Temporal encoder depth
                 heads: int = 8,                # Attention heads
                 kernel_size: List[int] = [3, 3], # Adaptive kernel size
                 dropout: float = 0.1,
                 max_seq_length: int = 2000,    # T: max sequence length
                 pos_encoding_type: str = 'rope_4d',
                 session_scale: float = 1.0,
                 use_temporal_kernels: bool = True,
                 use_mae_decoder: bool = True,
                 use_session_specific_heads: bool = True,
                 # Ablation study parameters
                 max_subjects: int = 10,
                 max_time_periods: int = 10,
                 max_tasks: int = 10,
                 rope_base: float = 10000.0):
        super().__init__()
        
        self.neural_dim = neural_dim
        self.d_model = d_model
        self.use_mae_decoder = use_mae_decoder
        
        # Stage 1: Temporal Encoder (processes single site with causal attention)
        self.temporal_encoder = PublicSparseTemporalEncoder(
            neural_dim=neural_dim,
            d_model=d_model,
            num_layers=temporal_layers,
            heads=heads,
            kernel_size=kernel_size,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pos_encoding_type=pos_encoding_type,
            session_scale=session_scale,
            use_temporal_kernels=use_temporal_kernels,
            max_subjects=max_subjects,
            max_time_periods=max_time_periods,
            max_tasks=max_tasks,
            rope_base=rope_base
        )
        
        # Stage 2: MAE Decoder (for pretraining only)
        if use_mae_decoder:
            self.mae_decoder = PublicLightweightMAEDecoder(
                d_model=d_model,
                neural_dim=neural_dim,
                dropout=dropout,
                use_session_specific_heads=use_session_specific_heads
            )
        
        # For downstream tasks (alternative to MAE decoder)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, neural_dim)
        )
    
    def forward(self, 
                neural_data: torch.Tensor,           # [B, S=1, T, N]
                session_coords: torch.Tensor,        # [B, S=1, 3] - Session coordinates
                mask_data: Optional[torch.Tensor] = None,  # [B, S=1, T] - MAE mask
                return_mae_reconstruction: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Public Dataset Foundation MAE.
        
        Args:
            neural_data: [B, S=1, T, N] - neural activity data
            session_coords: [B, S=1, 3] - (subject, time, task) coordinates
            mask_data: [B, S=1, T] - binary mask for MAE (1=masked, 0=unmasked)
            return_mae_reconstruction: If True, return MAE reconstruction
            
        Returns:
            Dict containing:
            - 'representations': [B, S=1, T, D] - final temporal representations
            - 'reconstruction': [B, S=1, T, N] - MAE reconstruction (if requested)
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Validate single-site format
        if S != 1:
            raise ValueError(f"PublicNeuralFoundationMAE expects S=1, got S={S}")
        
        # Create causal mask for temporal modeling
        causal_mask = create_causal_mask(T, device)
        
        # Stage 1: Temporal encoding with session coordinates
        temporal_features = self.temporal_encoder(
            neural_data=neural_data,
            session_coords=session_coords,
            causal_mask=causal_mask,
            mask_data=mask_data
        )  # [B, S=1, T, D]
        
        output = {'representations': temporal_features}
        
        # Stage 2: MAE reconstruction (if enabled and requested)
        if self.use_mae_decoder and return_mae_reconstruction:
            reconstruction = self.mae_decoder(temporal_features)  # [B, S=1, T, N]
            output['reconstruction'] = reconstruction
        
        return output
    
    def get_temporal_representations(self, 
                                   neural_data: torch.Tensor,
                                   session_coords: torch.Tensor,
                                   mask_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get temporal representations for downstream tasks.
        
        Args:
            neural_data: [B, S=1, T, N] - neural activity
            session_coords: [B, S=1, 3] - session coordinates
            mask_data: [B, S=1, T] - optional mask
            
        Returns:
            representations: [B, S=1, T, D] - temporal representations
        """
        output = self.forward(
            neural_data=neural_data,
            session_coords=session_coords,
            mask_data=mask_data,
            return_mae_reconstruction=False
        )
        return output['representations']


# Test function
def test_public_neural_foundation_mae():
    """Test PublicNeuralFoundationMAE implementation."""
    
    print("🧪 Testing PublicNeuralFoundationMAE")
    print("=" * 50)
    
    # Test parameters
    neural_dim = 50
    d_model = 512
    batch_size = 2
    seq_len = 50
    
    # Create model
    model = PublicNeuralFoundationMAE(
        neural_dim=neural_dim,
        d_model=d_model,
        temporal_layers=3,  # Reduced for testing
        heads=8,
        use_mae_decoder=True
    )
    
    print(f"✅ Model created: neural_dim={neural_dim}, d_model={d_model}")
    
    # Create test data
    neural_data = torch.randn(batch_size, 1, seq_len, neural_dim)  # [B, S=1, T, N]
    session_coords = torch.tensor([
        [[0.0, 0.3, 0.0]],  # Subject c, mid-2013, center-out
        [[3.0, 0.8, 1.0]]   # Subject t, late-2013, random-target
    ]).expand(batch_size, 1, 3)  # [B, S=1, 3]
    
    print(f"Input shapes:")
    print(f"  neural_data: {neural_data.shape}")
    print(f"  session_coords: {session_coords.shape}")
    
    # Forward pass
    output = model(
        neural_data=neural_data,
        session_coords=session_coords,
        return_mae_reconstruction=True
    )
    
    print(f"Output shapes:")
    print(f"  representations: {output['representations'].shape}")
    if 'reconstruction' in output:
        print(f"  reconstruction: {output['reconstruction'].shape}")
    
    # Test temporal representations
    temporal_repr = model.get_temporal_representations(neural_data, session_coords)
    print(f"  temporal_representations: {temporal_repr.shape}")
    
    # Verify output properties
    expected_repr_shape = (batch_size, 1, seq_len, d_model)
    expected_recon_shape = (batch_size, 1, seq_len, neural_dim)
    
    success = True
    if output['representations'].shape != expected_repr_shape:
        print(f"❌ Representations shape mismatch: expected {expected_repr_shape}, got {output['representations'].shape}")
        success = False
    
    if 'reconstruction' in output and output['reconstruction'].shape != expected_recon_shape:
        print(f"❌ Reconstruction shape mismatch: expected {expected_recon_shape}, got {output['reconstruction'].shape}")
        success = False
    
    if temporal_repr.shape != expected_repr_shape:
        print(f"❌ Temporal representations shape mismatch: expected {expected_repr_shape}, got {temporal_repr.shape}")
        success = False
    
    if success:
        print("✅ PublicNeuralFoundationMAE test passed!")
        return True
    else:
        print("❌ PublicNeuralFoundationMAE test failed!")
        return False


if __name__ == "__main__":
    test_public_neural_foundation_mae()
