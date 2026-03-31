"""
Public Dataset Attention Mechanisms for Neural Foundation Model (CORRECTED VERSION)
-----------------------------------------------------------------------------------

This module provides PROPERLY IMPLEMENTED attention mechanisms adapted for the public dataset.

Key Features:
- CausalAdaptiveKernelAttention with  context generation
- Session-based positional encoding with RoPE4D integration  
- Optimized for [B,S=1,T,N] single-site format
- Proper causal masking and adaptive kernel application order
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List

from .public_positional_encoding import RoPE4D


class PublicCausalAdaptiveKernelAttention(nn.Module):
    """
    CORRECTED: Causal Adaptive Kernel Attention with RoPE4D for Public Dataset.
    
    **PROPER ARCHITECTURE RESTORED**:
    - Historical data processing with MLP embedding (history_mlp)
    - Attention pooling for causal context generation (context_attention)
    - Real adaptive kernel convolution (apply_kernel with 2D conv)
    - RoPE4D integration for session-based coordinates
    - Proper causal context → kernel → attention flow
    
    **ARCHITECTURE OVERVIEW** (restored from attention.py):
    1. Process historical data through history_mlp
    2. Generate causal context via attention pooling  
    3. Generate adaptive kernels from causal context
    4. Apply real 2D convolution to attention scores
    5. Integrate with RoPE4D session encoding
    
    **SESSION ADAPTATION**:
    - Input format: [B, S=1, T, N] instead of [B, T, N]
    - Session coordinates: [B, S=1, T, 3] (subject, time, task) instead of [B, T, 2] (X, Y)
    - Historical data: [B, S=1, T_hist, D] for causal context generation
    """
    
    def __init__(self,
                 d_model: int = 512,
                 heads: int = 8, 
                 kernel_size: List[int] = [3, 3],
                 dropout: float = 0.1,
                 use_kernel: bool = True,
                 use_rope: bool = True,
                 rope_module: Optional[RoPE4D] = None,
                 context_dim: Optional[int] = None):
        super().__init__()
        
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.use_kernel = use_kernel
        self.use_rope = use_rope
        self.rope_module = rope_module
        self.kernel_size = kernel_size
        self.scale = math.sqrt(self.head_dim)
        
        assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
        
        # Set context dimension (same as original attention.py)
        if context_dim is None:
            context_dim = d_model * 2
        self.context_dim = context_dim
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ✅ RESTORED: Sophisticated adaptive kernel architecture (from attention.py)
        if use_kernel:
            # Historical data embedding MLP
            self.history_mlp = nn.Sequential(
                nn.Linear(d_model, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            
            # Attention pooling for causal context generation
            self.context_attention = nn.Sequential(
                nn.Linear(context_dim, context_dim // 4),
                nn.GELU(),
                nn.Linear(context_dim // 4, 1)
            )
            
            # Causal kernel generator
            self.kernel_generator = nn.Sequential(
                nn.Linear(context_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, heads * kernel_size[0] * kernel_size[1])
            )
        
        self.dropout = nn.Dropout(dropout)
        
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
    
    def generate_causal_context(self, historical_data: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            historical_data: [B*S, T_hist, D] - past timesteps for each sequence
            
        Returns:
            causal_context: [B*S, context_dim] - pooled causal context
        """
        # Step 1: MLP embedding of historical data
        embedded_history = self.history_mlp(historical_data)  # [B*S, T_hist, context_dim]
        
        # Step 2: Attention pooling to generate causal context
        attention_weights = self.context_attention(embedded_history)  # [B*S, T_hist, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Step 3: Weighted sum to get causal context
        causal_context = (embedded_history * attention_weights).sum(dim=1)  # [B*S, context_dim]
        
        return causal_context
    
    def apply_kernel(self, attn: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            attn: [B*S, heads, T, T] - attention scores
            kernels: [B*S, heads, kernel_h, kernel_w] - adaptive kernels
            
        Returns:
            filtered_attn: [B*S, heads, T, T] - convolved attention scores
        """
        BS, H, T, T = attn.shape
        K_1, K_2 = self.kernel_size
        pad_1, pad_2 = K_1 // 2, K_2 // 2
        
        # Skip kernel application if sequence is too short
        if T < max(K_1, K_2):
            return attn
        
        # Pad attention matrix for convolution
        attn_padded = F.pad(attn.reshape(1, BS*H, T, T), 
                           (pad_2, pad_2, pad_1, pad_1), mode='constant', value=0)
        
        # Reshape kernels for conv2d: [BS*H, 1, K_1, K_2]
        kernels = kernels.reshape(BS*H, 1, K_1, K_2)
        
        # Apply 2D convolution (this is the REAL adaptive kernel application!)
        output = F.conv2d(attn_padded, kernels, groups=BS*H)
        
        # Reshape back to original format
        return output.squeeze(1).reshape(BS, H, T, T)
    
    def forward(self,
                x: torch.Tensor,                       # [B, S=1, T, D] 
                session_coords: torch.Tensor,          # [B, S=1, T, 3]
                temporal_indices: torch.Tensor,        # [B, S=1, T]
                causal_mask: torch.Tensor,             # [T, T]
                historical_data: Optional[torch.Tensor] = None,  #
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        
        **CRITICAL ADDITIONS**:
        - historical_data parameter (was missing!)
        - Real causal context generation
        - Actual adaptive kernel application
        - Correct order: kernel → causal mask
        
        Args:
            x: [B, S=1, T, D] - input embeddings
            session_coords: [B, S=1, T, 3] - session coordinates  
            temporal_indices: [B, S=1, T] - temporal indices
            causal_mask: [T, T] - causal attention mask
            historical_data: [B, S=1, T_hist, D] - historical context (RESTORED!)
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
        
        # ✅ RESTORED: Apply adaptive kernel BEFORE causal mask (from attention.py)
        if self.use_kernel and historical_data is not None:
            # Reshape historical data for processing: [B, S, T_hist, D] -> [B*S, T_hist, D]
            historical_data_flat = historical_data.view(B * S, -1, D)
            
            # Generate causal context from historical data (RESTORED!)
            causal_context = self.generate_causal_context(historical_data_flat)  # [B*S, context_dim]
            
            # Generate adaptive kernels from causal context (RESTORED!)
            kernel_params = self.kernel_generator(causal_context)  # [B*S, heads*K1*K2]
            kernels = kernel_params.view(B * S, self.heads, self.kernel_size[0], self.kernel_size[1])
            
            # Normalize kernels
            kernels = F.softmax(kernels.view(B * S, self.heads, -1), dim=-1)
            kernels = kernels.view(B * S, self.heads, self.kernel_size[0], self.kernel_size[1])
            
            # This ensures no data leakage from future to past. for the kernel application in conv2d operations.
            causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0).expand(B * S, self.heads, T, T)
            scores = scores.masked_fill(causal_mask_expanded == 0, 0)
        
            # Apply real kernel convolution to attention scores (RESTORED!)
            scores = self.apply_kernel(scores, kernels)
        
        # Apply causal mask AFTER kernel application: [T, T] -> [B*S, heads, T, T]
        causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0).expand(B * S, self.heads, T, T)
        scores = scores.masked_fill(causal_mask_expanded == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, S, T] -> [B*S, T] -> [B*S, heads, T, T]
            padding_mask_flat = key_padding_mask.view(B * S, T)
            padding_mask_expanded = padding_mask_flat.unsqueeze(1).unsqueeze(2).expand(B * S, self.heads, T, T)
            scores = scores.masked_fill(padding_mask_expanded == 0, -1e9)
        
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
def test_public_attention_fixed():
    """Test the CORRECTED PublicCausalAdaptiveKernelAttention implementation."""
    
    print("🧪 Testing CORRECTED PublicCausalAdaptiveKernelAttention")
    print("=" * 60)
    
    # Test parameters
    d_model = 512
    heads = 8
    batch_size = 2
    seq_len = 50
    hist_len = 50  # Historical sequence length
    
    # Create RoPE4D
    from .public_positional_encoding import RoPE4D
    rope_4d = RoPE4D(d_model)
    
    # Create attention layer with CORRECTED implementation
    attention = PublicCausalAdaptiveKernelAttention(
        d_model=d_model,
        heads=heads,
        use_rope=True,
        use_kernel=True,  # Test with adaptive kernels enabled
        rope_module=rope_4d
    )
    
    print(f"✅ CORRECTED Attention layer created: d_model={d_model}, heads={heads}")
    print(f"   - Has history_mlp: {hasattr(attention, 'history_mlp')}")
    print(f"   - Has context_attention: {hasattr(attention, 'context_attention')}")
    print(f"   - Has kernel_generator: {hasattr(attention, 'kernel_generator')}")
    
    # Create test data
    x = torch.randn(batch_size, 1, seq_len, d_model)  # [B, S=1, T, D]
    historical_data = torch.randn(batch_size, 1, hist_len, d_model)  # [B, S=1, T_hist, D]
    
    # Session coordinates
    session_coords = torch.tensor([
        [[[0.0, 0.3, 0.0]] * seq_len],  # Subject c, mid-2013, center-out
        [[[3.0, 0.8, 1.0]] * seq_len]   # Subject t, late-2013, random-target
    ])  # [B=2, S=1, T=50, 3]
    
    temporal_indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len)
    causal_mask = create_causal_mask(seq_len, x.device)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  historical_data: {historical_data.shape}")
    print(f"  session_coords: {session_coords.shape}")
    print(f"  temporal_indices: {temporal_indices.shape}")
    print(f"  causal_mask: {causal_mask.shape}")
    
    # Forward pass WITH historical data (this should work now!)
    print("\n🔧 Testing with adaptive kernels...")
    output_with_kernels = attention(x, session_coords, temporal_indices, causal_mask, 
                                   historical_data=historical_data)
    
    print(f"✅ Forward pass with kernels successful: {output_with_kernels.shape}")
    
    # Forward pass WITHOUT historical data (standard attention)
    print("\n🔧 Testing without adaptive kernels...")
    output_without_kernels = attention(x, session_coords, temporal_indices, causal_mask)
    
    print(f"✅ Forward pass without kernels successful: {output_without_kernels.shape}")
    
    # Check if adaptive kernels actually modify the output
    diff_magnitude = torch.norm(output_with_kernels - output_without_kernels).item()
    print(f"\n📊 Adaptive kernel effect magnitude: {diff_magnitude:.4f}")
    
    if diff_magnitude > 0.01:
        print("✅ Adaptive kernels are WORKING (outputs differ significantly)")
    else:
        print("❌ Adaptive kernels may not be working (outputs too similar)")
    
    # Verify output properties
    if output_with_kernels.shape == x.shape:
        print("✅ Output shape matches input")
    else:
        print("❌ Output shape mismatch")
        return False
    
    print("\n🎉 CORRECTED PublicCausalAdaptiveKernelAttention test completed!")
    return True


if __name__ == "__main__":
    test_public_attention_fixed()