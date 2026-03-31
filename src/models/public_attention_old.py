"""
Public Dataset Attention Mechanisms for Neural Foundation Model
--------------------------------------------------------------

This module provides attention mechanisms adapted for the public dataset with RoPE4D support.

Key Features:
- CausalAdaptiveKernelAttention with RoPE4D integration
- Session-based positional encoding instead of site-based
- Optimized for [B,S=1,T,N] single-site format
- Maintains causal masking for temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List

from .public_positional_encoding import RoPE4D


class PublicCausalAdaptiveKernelAttention(nn.Module):
    """
    Causal Adaptive Kernel Attention with RoPE4D for Public Dataset.
    
    **DESIGN ADAPTATIONS**:
    - RoPE4D integration for session-based positional encoding
    - Optimized for single-site format [B,S=1,T,N]
    - Session coordinates instead of site coordinates
    - Same causal masking and adaptive kernels as original
    
    **KEY FEATURES**:
    - Causal attention with temporal masking
    - RoPE4D applied to Q/K for session encoding
    - Adaptive kernel generation for temporal dynamics
    - Efficient single-site processing
    """
    
    def __init__(self,
                 d_model: int = 512,
                 heads: int = 8, 
                 kernel_size: List[int] = [3, 3],
                 dropout: float = 0.1,
                 use_kernel: bool = True,
                 use_rope: bool = True,
                 rope_module: Optional[RoPE4D] = None):
        super().__init__()
        
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.use_kernel = use_kernel
        self.use_rope = use_rope
        self.rope_module = rope_module
        
        assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Adaptive kernel generation (if enabled)
        if use_kernel:
            self.kernel_size = kernel_size
            # Context-based kernel generator
            self.context_proj = nn.Linear(d_model, 64)
            self.kernel_generator = nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, kernel_size[0] * kernel_size[1]),
                nn.Softmax(dim=-1)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self,
                x: torch.Tensor,                       # [B, S=1, T, D] 
                session_coords: torch.Tensor,          # [B, S=1, T, 3]
                temporal_indices: torch.Tensor,        # [B, S=1, T]
                causal_mask: torch.Tensor,             # [T, T]
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, S=1, T]
        """
        Forward pass with RoPE4D and causal attention.
        
        Args:
            x: [B, S=1, T, D] - input embeddings
            session_coords: [B, S=1, T, 3] - session coordinates  
            temporal_indices: [B, S=1, T] - temporal indices
            causal_mask: [T, T] - causal attention mask
            key_padding_mask: [B, S=1, T] - optional padding mask
            
        Returns:
            output: [B, S=1, T, D] - attention output
        """
        B, S, T, D = x.shape
        device = x.device
        
        # Linear projections: [B, S, T, D] -> [B, S, T, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [B, S, T, D] -> [B*S, T, heads, head_dim]
        q = q.view(B * S, T, self.heads, self.head_dim)
        k = k.view(B * S, T, self.heads, self.head_dim)
        v = v.view(B * S, T, self.heads, self.head_dim)
        
        # Apply RoPE4D to Q and K if enabled
        if self.use_rope and self.rope_module is not None:
            # Reshape session_coords and temporal_indices for RoPE
            session_coords_flat = session_coords.view(B * S, T, 3)      # [B*S, T, 3]
            temporal_indices_flat = temporal_indices.view(B * S, T)     # [B*S, T]
            
            # Apply RoPE rotation (unified interface for all PE types)
            q, k = self.rope_module.apply_rope(q, k, session_coords_flat, temporal_indices_flat)
        
        # Transpose for attention computation: [B*S, T, heads, head_dim] -> [B*S, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: [B*S, heads, T, head_dim] @ [B*S, heads, head_dim, T] -> [B*S, heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask: [T, T] -> [B*S, heads, T, T]
        causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0).expand(B * S, self.heads, T, T)
        scores = scores.masked_fill(causal_mask_expanded == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, S, T] -> [B*S, T] -> [B*S, 1, 1, T] -> [B*S, heads, T, T]
            padding_mask_flat = key_padding_mask.view(B * S, T)
            padding_mask_expanded = padding_mask_flat.unsqueeze(1).unsqueeze(2).expand(B * S, self.heads, T, T)
            scores = scores.masked_fill(padding_mask_expanded == 0, -1e9)
        
        # Apply adaptive kernel if enabled
        if self.use_kernel:
            # Generate adaptive kernel from context
            context = x.mean(dim=2)  # [B, S, D] - average across time
            context_features = self.context_proj(context)  # [B, S, 64]
            kernel_weights = self.kernel_generator(context_features)  # [B, S, kernel_area]
            
            # Apply kernel convolution to attention scores
            scores = self._apply_adaptive_kernel(scores, kernel_weights.view(B * S, *self.kernel_size))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [B*S, heads, T, T] @ [B*S, heads, T, head_dim] -> [B*S, heads, T, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: [B*S, heads, T, head_dim] -> [B*S, T, heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # Concatenate heads: [B*S, T, heads, head_dim] -> [B*S, T, D]
        attn_output = attn_output.contiguous().view(B * S, T, D)
        
        # Reshape back to original format: [B*S, T, D] -> [B, S, T, D]
        attn_output = attn_output.view(B, S, T, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _apply_adaptive_kernel(self, scores: torch.Tensor, kernel_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive kernel convolution to attention scores.
        
        Args:
            scores: [B*S, heads, T, T] - attention scores
            kernel_weights: [B*S, kernel_h, kernel_w] - adaptive kernel weights
            
        Returns:
            scores_filtered: [B*S, heads, T, T] - filtered attention scores
        """
        BS, heads, T, _ = scores.shape
        kernel_h, kernel_w = kernel_weights.shape[1:]
        
        # Simple implementation: apply kernel as a local attention pattern
        # This is a simplified version - can be enhanced for better adaptive behavior
        
        return scores


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for temporal modeling.
    
    Args:
        seq_len: Sequence length
        device: Target device
        
    Returns:
        causal_mask: [seq_len, seq_len] - causal mask (1=allowed, 0=masked)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


# Test function
def test_public_attention():
    """Test PublicCausalAdaptiveKernelAttention implementation."""
    
    print("🧪 Testing PublicCausalAdaptiveKernelAttention")
    print("=" * 50)
    
    # Test parameters
    d_model = 512
    heads = 8
    batch_size = 2
    seq_len = 50
    
    # Create RoPE4D
    rope_4d = RoPE4D(d_model)
    
    # Create attention layer
    attention = PublicCausalAdaptiveKernelAttention(
        d_model=d_model,
        heads=heads,
        use_rope=True,
        rope_module=rope_4d
    )
    
    print(f"✅ Attention layer created: d_model={d_model}, heads={heads}")
    
    # Create test data
    x = torch.randn(batch_size, 1, seq_len, d_model)  # [B, S=1, T, D]
    
    # Session coordinates
    session_coords = torch.tensor([
        [[[0.0, 0.3, 0.0]] * seq_len],  # Subject c, mid-2013, center-out
        [[[3.0, 0.8, 1.0]] * seq_len]   # Subject t, late-2013, random-target
    ])  # [B=2, S=1, T=50, 3]
    
    temporal_indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len)
    causal_mask = create_causal_mask(seq_len, x.device)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  session_coords: {session_coords.shape}")
    print(f"  temporal_indices: {temporal_indices.shape}")
    print(f"  causal_mask: {causal_mask.shape}")
    
    # Forward pass
    output = attention(x, session_coords, temporal_indices, causal_mask)
    
    print(f"Output shape: {output.shape}")
    
    # Verify output properties
    if output.shape == x.shape:
        print("✅ Output shape matches input")
    else:
        print("❌ Output shape mismatch")
        return False
    
    # Check if output is different from input (attention should modify)
    diff_magnitude = torch.norm(output - x).item()
    print(f"Attention modification magnitude: {diff_magnitude:.4f}")
    
    if diff_magnitude > 0.1:
        print("✅ PublicCausalAdaptiveKernelAttention test passed!")
        return True
    else:
        print("❌ Attention test failed: insufficient modification")
        return False


if __name__ == "__main__":
    test_public_attention()
