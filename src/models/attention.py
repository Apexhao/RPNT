import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CausalAdaptiveKernelAttention(nn.Module):
    """
    Simplified Causal Adaptive Kernel Attention with RoPE3D Support
    
    **ENHANCED FEATURES**:
    - Support for 3D Rotary Position Embedding (RoPE3D)
    - Proper integration of RoPE into attention computation
    - Maintains compatibility with other positional encoding methods
    
    **SIMPLIFIED APPROACH**:
    1. Get historical data
    2. Use MLP to embed it  
    3. Apply attention pooling to generate causal context
    4. Use causal context to generate causal kernel for sparse attention
    
    **COMPARISON MODE**: Can disable kernel application for standard masked attention
    """
    def __init__(self, 
                 dim: int, 
                 heads: int = 8, 
                 kernel_size: list = [3, 3], 
                 dropout: float = 0.1,
                 context_dim: int = None,
                 use_kernel: bool = True,
                 use_rope: bool = False,     # ✅ NEW: Enable RoPE integration
                 rope_module = None):         # ✅ NEW: RoPE3D module reference
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.use_kernel = use_kernel  # For comparison with standard attention
        self.use_rope = use_rope      # ✅ NEW: RoPE integration flag
        self.rope_module = rope_module  # ✅ NEW: Reference to RoPE3D instance
        
        if context_dim is None:
            context_dim = dim * 2
        
        # Standard attention projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Historical data embedding MLP
        self.history_mlp = nn.Sequential(
            nn.Linear(dim, context_dim),
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
        
        # Causal kernel generator (only if using kernels)
        if use_kernel:
            self.kernel_generator = nn.Sequential(
                nn.Linear(context_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, heads * kernel_size[0] * kernel_size[1])
            )
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values."""
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
        Generate causal context from historical data using simplified approach.
        
        Args:
            historical_data: [B, T_hist, D] - past timesteps only
            
        Returns:
            causal_context: [B, D_context] - pooled causal context
        """
        # Step 1: MLP embedding of historical data
        embedded_history = self.history_mlp(historical_data)  # [B, T_hist, D_context]
        
        # Step 2: Attention pooling to generate causal context
        attention_weights = self.context_attention(embedded_history)  # [B, T_hist, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Step 3: Weighted sum to get causal context
        causal_context = (embedded_history * attention_weights).sum(dim=1)  # [B, D_context]
        
        return causal_context
    
    def apply_kernel(self, attn: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """Apply 2D kernel to attention matrix."""
        B, H, N, N = attn.shape
        K_1, K_2 = self.kernel_size
        pad_1, pad_2 = K_1 // 2, K_2 // 2
        
        # Skip kernel application if sequence is too short
        if N < max(K_1, K_2):
            return attn
        
        # Pad attention matrix
        attn_padded = F.pad(attn.reshape(1, B*H, N, N), 
                           (pad_2, pad_2, pad_1, pad_1), mode='constant', value=0)
        
        # Reshape kernels for conv2d
        kernels = kernels.reshape(B*H, 1, K_1, K_2)
        
        # Apply convolution
        output = F.conv2d(attn_padded, kernels, groups=B*H)
        
        # Reshape back
        return output.squeeze(1).reshape(B, H, N, N)
    
    def forward(self, 
                x: torch.Tensor, 
                historical_data: torch.Tensor = None,
                causal_mask: Optional[torch.Tensor] = None,
                # ✅ NEW: RoPE3D integration parameters
                site_coords: Optional[torch.Tensor] = None,    # [B, T, 2] - (X, Y) coordinates
                time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, T] - time indices
        """
        Forward pass with simplified causal kernel attention and optional RoPE3D.
        
        Args:
            x: [B, T, D] - current input
            historical_data: [B, T_hist, D] - past context (if None, use standard attention)
            causal_mask: [T, T] - causal attention mask
            site_coords: [B, T, 2] - (X, Y) coordinates for RoPE3D (optional)
            time_indices: [B, T] - time indices for RoPE3D (optional)
            
        Returns:
            output: [B, T, D] - attended output
        """
        B, T, D = x.shape
        
        # Compute Q, K, V
        q = self.to_q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        k = self.to_k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        v = self.to_v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        
        # ✅ NEW: Apply RoPE rotations to Q and K if enabled
        if self.use_rope and self.rope_module is not None and time_indices is not None:
            # Import here to avoid circular imports
            from .positional_encoding import apply_rotary_emb_3d
            
            # Generate rotation frequencies based on RoPE type
            if hasattr(self.rope_module, 'compute_3d_freqs_cis'):
                # RoPE3D: requires site coordinates and time indices
                freqs_cis = self.rope_module.compute_3d_freqs_cis(site_coords, time_indices)  # [B, T, rope_dim//2]
            elif hasattr(self.rope_module, 'compute_freqs_cis'):
                # StandardRoPE: only requires time indices
                freqs_cis = self.rope_module.compute_freqs_cis(time_indices)  # [B, T, rope_dim//2]
            else:
                raise ValueError(f"Unknown RoPE module type: {type(self.rope_module)}")
            
            # Apply RoPE rotations to Q and K
            q, k = apply_rotary_emb_3d(
                q, k, freqs_cis, 
                rope_dim=self.rope_module.rope_dim
            )
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, T, T]
        
        # ✅ FIX: Apply kernel BEFORE causal mask to avoid -inf issues
        if self.use_kernel and historical_data is not None:
            # Generate causal context from historical data
            causal_context = self.generate_causal_context(historical_data)  # [B, D_context]
            
            # Generate causal kernels
            kernel_params = self.kernel_generator(causal_context)  # [B, H*K1*K2]
            kernels = kernel_params.view(B, self.heads, self.kernel_size[0], self.kernel_size[1])
            
            # Normalize kernels
            kernels = F.softmax(kernels.view(B, self.heads, -1), dim=-1)
            kernels = kernels.view(B, self.heads, self.kernel_size[0], self.kernel_size[1])
            
            # This ensures no data leakage from future to past. for the kernel application in conv2d operations.
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
            # Apply kernel to attention BEFORE causal masking
            attn = self.apply_kernel(attn, kernels)
        
        # Apply causal mask AFTER kernel application
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Standard attention computation
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        out = self.proj(out)
        
        return out


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for GPT-style autoregressive modeling.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        mask: [seq_len, seq_len] where True = masked position (should not attend)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  # True = masked (will be set to -inf in attention) 