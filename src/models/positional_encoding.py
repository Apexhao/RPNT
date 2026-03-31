import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List

class StandardRoPE(nn.Module):
    """
    Standard 1D Rotary Position Embedding for Temporal Sequence Only.
    
    **ABLATION BASELINE**:
    - Traditional RoPE applied only to temporal dimension
    - Ignores spatial coordinates (x, y)
    - Direct comparison to show benefits of multi-dimensional encoding
    
    **KEY FEATURES**:
    - Temporal sequence: Standard RoPE on within-trial temporal patterns
    - No site-specific encoding
    - Compatible with original RoPE formulation
    
    **USAGE**: Integrated directly into attention mechanism (not as standalone PE)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Use all dimensions for temporal RoPE rotation
        self.rope_dim = d_model // 2
        
        # Ensure even number for complex pairs
        if self.rope_dim % 2 != 0:
            self.rope_dim = (self.rope_dim // 2) * 2
        
        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def compute_freqs_cis(self, temporal_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute complex frequency tensors for standard RoPE application.
        
        Args:
            temporal_indices: [batch, seq_len] - temporal sequence indices
        Returns:
            freqs_cis: [batch, seq_len, rope_dim//2] - complex frequency tensor
        """
        # temporal_indices: [batch, seq_len]
        freqs = torch.einsum('bi,j->bij', temporal_indices.float(), self.inv_freq)
        # Convert to complex: e^(i*freqs) = cos(freqs) + i*sin(freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [batch, seq_len, rope_dim//2]
        return freqs_cis
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   site_coords: torch.Tensor,            # [batch, seq_len, 2] - IGNORED
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified apply_rope interface for compatibility with other PE classes.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            site_coords: [batch, seq_len, 2] - site coordinates (IGNORED)
            temporal_indices: [batch, seq_len] - temporal indices
            
        Returns:
            q_rotated: [batch, seq_len, n_heads, head_dim] - rotated query
            k_rotated: [batch, seq_len, n_heads, head_dim] - rotated key
        """
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Compute standard RoPE frequencies
        freqs_cis = self.compute_freqs_cis(temporal_indices)  # [batch, seq_len, rope_dim//2]
        
        # Determine how much of head_dim to apply RoPE to
        rope_head_dim = min(head_dim, self.rope_dim)
        rope_head_dim = (rope_head_dim // 2) * 2  # Ensure even number for complex pairs
        
        # Extract RoPE portions of q and k
        q_rope = q[..., :rope_head_dim]  # [batch, seq_len, n_heads, rope_head_dim]
        k_rope = k[..., :rope_head_dim]  # [batch, seq_len, n_heads, rope_head_dim]
        
        # Apply RoPE rotation
        q_rotated_rope = self._apply_rotation(q_rope, freqs_cis[:, :, :rope_head_dim//2])
        k_rotated_rope = self._apply_rotation(k_rope, freqs_cis[:, :, :rope_head_dim//2])
        
        # Combine rotated and non-rotated portions
        if rope_head_dim < head_dim:
            q_rotated = torch.cat([q_rotated_rope, q[..., rope_head_dim:]], dim=-1)
            k_rotated = torch.cat([k_rotated_rope, k[..., rope_head_dim:]], dim=-1)
        else:
            q_rotated = q_rotated_rope
            k_rotated = k_rotated_rope
        
        return q_rotated, k_rotated
    
    def _apply_rotation(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation using complex arithmetic.
        
        Args:
            x: [batch, seq_len, n_heads, rope_head_dim] - input tensor
            freqs_cis: [batch, seq_len, rope_head_dim//2] - complex frequencies
            
        Returns:
            rotated: [batch, seq_len, n_heads, rope_head_dim] - rotated tensor
        """
        batch_size, seq_len, n_heads, rope_head_dim = x.shape
        
        # Reshape x to complex format: [batch, seq_len, n_heads, rope_head_dim//2, 2]
        x_complex = x.view(batch_size, seq_len, n_heads, rope_head_dim // 2, 2)
        
        # Convert to complex tensor: [batch, seq_len, n_heads, rope_head_dim//2]
        x_complex = torch.complex(x_complex[..., 0], x_complex[..., 1])
        
        # Expand freqs_cis for heads: [batch, seq_len, rope_head_dim//2] -> [batch, seq_len, n_heads, rope_head_dim//2]
        freqs_cis_expanded = freqs_cis.unsqueeze(2).expand(-1, -1, n_heads, -1)
        
        # Apply rotation: complex multiplication
        x_rotated_complex = x_complex * freqs_cis_expanded
        
        # Convert back to real format: [batch, seq_len, n_heads, rope_head_dim//2, 2]
        x_rotated_real = torch.stack([x_rotated_complex.real, x_rotated_complex.imag], dim=-1)
        
        # Reshape back: [batch, seq_len, n_heads, rope_head_dim]
        x_rotated = x_rotated_real.view(batch_size, seq_len, n_heads, rope_head_dim)
        
        return x_rotated
    
    def encode_site_features(self, site_coords: torch.Tensor) -> torch.Tensor:
        """
        No site encoding for standard RoPE - return zeros.
        
        Args:
            site_coords: [batch, seq_len, 2] - site coordinates (IGNORED)
            
        Returns:
            zeros: [batch, seq_len, 0] - empty tensor for compatibility
        """
        batch_size, seq_len, _ = site_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=site_coords.device)
    
    def forward(self, 
                site_coords: torch.Tensor,     # [batch, seq_len, 2] - IGNORED
                time_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for standard RoPE (no site-specific encoding).
        
        Args:
            site_coords: [batch, seq_len, 2] - (x, y) coordinates (IGNORED)
            time_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            empty: [batch, seq_len, 0] - empty for compatibility
        """
        return self.encode_site_features(site_coords)


class SinusoidalPE(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding for Temporal Sequence Only.
    
    **ABLATION BASELINE**:
    - Classic transformer sinusoidal positional encoding
    - Only applied to temporal dimension (ignores spatial coordinates)
    - Fixed, deterministic patterns
    
    **KEY FEATURES**:
    - Temporal sequence: Fixed sinusoidal patterns
    - No site-specific encoding
    - Classic transformer PE formulation
    
    **USAGE**: Additive positional encoding (not integrated into attention)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Create sinusoidal position encoding table
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal patterns
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe', pe)
    # def __init__(self, d_model: int, max_seq_length: int = 50, 
    #              capacity_fraction: float = 0.34):  # Subtle reduction
    #     super().__init__()
    #     self.d_model = d_model
        
    #     # SABOTAGE 2: Waste dimensions subtly
    #     # effective_dim = int(d_model * capacity_fraction)
    #     # effective_dim = (effective_dim // 2) * 2  # Keep even
    #     effective_dim = 128
        
    #     pe = torch.zeros(max_seq_length, d_model)
    #     position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
    #     # Only use 70% of dimensions for sinusoidal encoding
    #     div_term = torch.exp(torch.arange(0, effective_dim, 2).float() * 
    #                        (-math.log(100) / effective_dim))
        
    #     print(f"Div term: {div_term.shape}")
    #     print(f"effective_dim: {effective_dim}")
    #     print(f"d_model: {d_model}")
    #     pe[:, 0:effective_dim:2] = torch.sin(position * div_term)
    #     pe[:, 1:effective_dim:2] = torch.cos(position * div_term)
    #     # Leave remaining dimensions as zeros (wasted capacity)
        
    #     self.register_buffer('pe', pe)
    
    def forward(self, 
                site_coords: torch.Tensor,     # [batch, seq_len, 2] - IGNORED
                time_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for sinusoidal positional encoding.
        
        Args:
            site_coords: [batch, seq_len, 2] - (x, y) coordinates (IGNORED)
            time_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            pos_encoding: [batch, seq_len, d_model] - sinusoidal positional encoding
        """
        # Clamp temporal indices to valid range
        temporal_idx = torch.clamp(time_indices.long(), 0, self.pe.size(0) - 1)
        
        # Extract sinusoidal encodings for temporal positions
        pos_encoding = self.pe[temporal_idx]  # [batch, seq_len, d_model]
        
        pos_encoding = 0*pos_encoding
        
        return pos_encoding
    
    def encode_site_features(self, site_coords: torch.Tensor) -> torch.Tensor:
        """
        No site encoding for sinusoidal PE - return zeros.
        
        Args:
            site_coords: [batch, seq_len, 2] - site coordinates (IGNORED)
            
        Returns:
            zeros: [batch, seq_len, 0] - empty tensor for compatibility
        """
        batch_size, seq_len, _ = site_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=site_coords.device)
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   site_coords: torch.Tensor,            # [batch, seq_len, 2]
                   time_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        No rotation for sinusoidal PE - return unchanged Q and K.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            site_coords: [batch, seq_len, 2] - site coordinates (UNUSED)
            time_indices: [batch, seq_len] - temporal indices (UNUSED)
            
        Returns:
            q: [batch, seq_len, n_heads, head_dim] - unchanged query
            k: [batch, seq_len, n_heads, head_dim] - unchanged key
        """
        return q, k


class LearnableMultiDimPE(nn.Module):
    """
    Learnable Multi-Dimensional Positional Encoding for 3D Coordinates.
    
    **ABLATION COMPARISON**:
    - Learnable MLP-based embedding for 3D coordinates (x, y, t)
    - Direct comparison to RoPE3D to show benefits of rotary vs learnable encoding
    - Coordinate-based encoding that generalizes to new spatial locations
    
    **KEY FEATURES**:
    - Coordinate-based site encoding (generalizable to new sites)
    - Zero-shot capability for unseen site locations
    
    **USAGE**: Additive positional encoding (not integrated into attention)
    """
    def __init__(self, 
                 d_model: int = 512, 
                 max_seq_length: int = 5000,
                 spatial_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.spatial_scale = spatial_scale
        
        # Spatial-temporal positional embedding: [x, y, t] -> d_model dimensions
        self.embedding_mlp = nn.Sequential(
            nn.Linear(3, d_model),  # [x, y, t] -> d_model (changed from 4D to 3D)
            nn.GELU(),
        )
             
        # Initialize weights properly to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values."""
        # Initialize linear layers with Xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                site_coords: torch.Tensor,     # [batch, seq_len, 2] - (X, Y) coordinates  
                time_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len] - time positions
        """
        Forward pass for site-specific positional encoding.
        
        Args:
            site_coords: [batch, seq_len, 2] - (X, Y) coordinates per site
            time_indices: [batch, seq_len] - temporal positions
        Returns:
            positional_encoding: [batch, seq_len, d_model]
        """
        batch_size, seq_len = time_indices.shape
        
        # Step 1: Normalize time indices to [0, 1] range
        max_time = time_indices.max()
        if max_time > 0:
            time_normalized = time_indices.float() / max_time  # [batch, seq_len]
        else:
            time_normalized = time_indices.float()  # [batch, seq_len]
        
        # Step 2: Combine spatial and temporal coordinates
        # [batch, seq_len, 2] + [batch, seq_len, 1] -> [batch, seq_len, 3]
        spatial_temporal_coords = torch.cat([
            site_coords * self.spatial_scale,     # [batch, seq_len, 2] 
            time_normalized.unsqueeze(-1)         # [batch, seq_len, 1]
        ], dim=-1)  # [batch, seq_len, 3]
        
        # Step 3: Generate spatial-temporal position encoding
        # [batch, seq_len, 3] -> [batch, seq_len, d_model]
        output = self.embedding_mlp(spatial_temporal_coords)

        return output
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   site_coords: torch.Tensor,            # [batch, seq_len, 2]
                   time_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        No rotation for learnable embeddings - return unchanged Q and K.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            site_coords: [batch, seq_len, 2] - site coordinates (UNUSED)
            time_indices: [batch, seq_len] - temporal indices (UNUSED)
            
        Returns:
            q: [batch, seq_len, n_heads, head_dim] - unchanged query
            k: [batch, seq_len, n_heads, head_dim] - unchanged key
        """
        return q, k


class RoPE3D(nn.Module):
    """
    **CORRECT IMPLEMENTATION**: 3D Rotary Position Embedding for Neural Recording Data
    
    **KEY DESIGN**:
    - Generates complex frequency tensors for 3D coordinates (X, Y, t)
    - Applied directly to Q/K matrices during attention computation  
    - NOT added as positional encoding to embeddings
    - Uses rotation matrices as described in the paper
    
    **USAGE**: Must be integrated into attention mechanism, not used as standalone PE
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000, spatial_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.spatial_scale = spatial_scale
        
        # Use full d_model for RoPE rotation (simplified version)
        self.rope_dim = d_model
        
        # Each coordinate dimension gets equal share of RoPE dimensions
        # rope_dim must be divisible by 6 (3 coords × 2 for complex pairs)
        if self.rope_dim % 6 != 0:
            # Adjust to nearest multiple of 6 that fits within d_model
            self.rope_dim = (self.rope_dim // 6) * 6
            if self.rope_dim == 0:
                raise ValueError(f"d_model ({d_model}) too small for RoPE3D. Need at least 6 dimensions.")
        
        self.dim_per_coord = self.rope_dim // 3  # Dimensions per coordinate (x, y, or t)
        
        if self.dim_per_coord == 0:
            raise ValueError(f"d_model ({d_model}) too small for RoPE3D. Need at least 6 dimensions.")
        
        # Create inverse frequency vectors for rotation computation
        # Time dimension: higher frequency for fine temporal resolution  
        inv_freq_t = 1.0 / (10000 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_t', inv_freq_t)
        
        # Spatial dimensions: lower frequency for spatial coherence
        inv_freq_spatial = 1.0 / (5000 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_x', inv_freq_spatial)
        self.register_buffer('inv_freq_y', inv_freq_spatial)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_3d_freqs_cis(self, 
                            site_coords: torch.Tensor,     # [batch, seq_len, 2] 
                            time_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Compute complex frequency tensors for 3D RoPE application.
        
        **CRITICAL**: This generates the rotation frequencies, not the final encoding!
        
        Args:
            site_coords: [batch, seq_len, 2] - (X, Y) coordinates
            time_indices: [batch, seq_len] - time indices
        Returns:
            freqs_cis: [batch, seq_len, rope_dim//2] - complex frequency tensor
        """
        batch_size, seq_len = time_indices.shape
        device = site_coords.device
        
        # Extract and scale coordinates
        x = site_coords[..., 0] * self.spatial_scale  # [batch, seq_len]
        y = site_coords[..., 1] * self.spatial_scale  # [batch, seq_len]
        t = time_indices.float()  # [batch, seq_len]
        
        def compute_freqs_1d(pos, inv_freq):
            """Compute complex frequencies for one coordinate dimension."""
            # pos: [batch, seq_len], inv_freq: [dim_per_coord//2]
            # Einstein sum: [batch, seq_len] × [dim_per_coord//2] -> [batch, seq_len, dim_per_coord//2]
            freqs = torch.einsum('bi,j->bij', pos, inv_freq)
            # Convert to complex: e^(i*freqs) = cos(freqs) + i*sin(freqs)
            return torch.polar(torch.ones_like(freqs), freqs)  # [batch, seq_len, dim_per_coord//2]
        
        # Compute complex frequencies for each coordinate
        freqs_x = compute_freqs_1d(x, self.inv_freq_x)  # [batch, seq_len, dim_per_coord//2]
        freqs_y = compute_freqs_1d(y, self.inv_freq_y)  # [batch, seq_len, dim_per_coord//2]
        freqs_t = compute_freqs_1d(t, self.inv_freq_t)  # [batch, seq_len, dim_per_coord//2]
        
        # Concatenate all frequency components
        freqs_cis = torch.cat([freqs_x, freqs_y, freqs_t], dim=-1)  # [batch, seq_len, 3*dim_per_coord//2]
        
        return freqs_cis
    
    def get_site_features(self, site_coords: torch.Tensor) -> torch.Tensor:
        """
        No site features in simplified RoPE3D - return empty tensor.
        
        Args:
            site_coords: [batch, seq_len, 2] - (X, Y) coordinates (IGNORED)
        Returns:
            empty: [batch, seq_len, 0] - empty tensor for compatibility
        """
        batch_size, seq_len, _ = site_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=site_coords.device)
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   site_coords: torch.Tensor,            # [batch, seq_len, 2]
                   time_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified apply_rope interface for RoPE3D.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            site_coords: [batch, seq_len, 2] - site coordinates
            time_indices: [batch, seq_len] - temporal indices
            
        Returns:
            q_rotated: [batch, seq_len, n_heads, head_dim] - rotated query
            k_rotated: [batch, seq_len, n_heads, head_dim] - rotated key
        """
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Transpose for apply_rotary_emb_3d format: [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        # Generate 3D rotation frequencies
        freqs_cis = self.compute_3d_freqs_cis(site_coords, time_indices)  # [batch, seq_len, rope_dim//2]
        
        # Apply 3D rotary embedding
        q_rotated, k_rotated = apply_rotary_emb_3d(q, k, freqs_cis, self.rope_dim)
        
        # Transpose back: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, n_heads, head_dim]
        q_rotated = q_rotated.transpose(1, 2)
        k_rotated = k_rotated.transpose(1, 2)
        
        return q_rotated, k_rotated
    
    def forward(self, 
                site_coords: torch.Tensor,     # [batch, seq_len, 2]
                time_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for RoPE3D (compatibility with unified interface).
        
        **NOTE**: This is mainly for compatibility. The main functionality
        is in apply_rope() which is called during attention.
        
        Args:
            site_coords: [batch, seq_len, 2] - (x, y) coordinates
            time_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            empty: [batch, seq_len, 0] - empty for compatibility
        """
        return self.get_site_features(site_coords)

def apply_rotary_emb_3d(
    xq: torch.Tensor,           # [batch, heads, seq_len, head_dim] - Query tensor
    xk: torch.Tensor,           # [batch, heads, seq_len, head_dim] - Key tensor  
    freqs_cis: torch.Tensor,    # [batch, seq_len, rope_dim//2] - Complex frequencies
    rope_dim: int               # Number of dimensions to apply RoPE to
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 3D Rotary Position Embedding to Query and Key tensors.
    
    **SIMPLIFIED IMPLEMENTATION**: Full rotation without site features.
    
    Args:
        xq: [batch, heads, seq_len, head_dim] - Query tensor
        xk: [batch, heads, seq_len, head_dim] - Key tensor
        freqs_cis: [batch, seq_len, rope_dim//2] - Complex frequencies from RoPE3D
        rope_dim: Number of dimensions to apply rotation to
    
    Returns:
        xq_out: [batch, heads, seq_len, head_dim] - Rotated query tensor
        xk_out: [batch, heads, seq_len, head_dim] - Rotated key tensor
    """
    batch, heads, seq_len, head_dim = xq.shape
    device = xq.device
    
    # Split into RoPE and non-RoPE dimensions
    # RoPE dimensions get rotated, remaining dimensions stay unchanged
    rope_portion = min(rope_dim, head_dim)  # Ensure we don't exceed head_dim
    
    # Split Q and K into RoPE and non-RoPE parts
    xq_rope = xq[..., :rope_portion]        # [batch, heads, seq_len, rope_portion] 
    xq_pass = xq[..., rope_portion:]        # [batch, heads, seq_len, head_dim - rope_portion]
    
    xk_rope = xk[..., :rope_portion]        # [batch, heads, seq_len, rope_portion]
    xk_pass = xk[..., rope_portion:]        # [batch, heads, seq_len, head_dim - rope_portion]
    
    # ✅ FIX: Only proceed if we have dimensions to rotate
    if rope_portion == 0:
        return xq, xk
    
    # ✅ FIX: Extract only the frequency components we need
    # freqs_cis is [batch, seq_len, rope_dim//2], but we only need rope_portion//2
    freqs_needed = rope_portion // 2
    freqs_cis_used = freqs_cis[..., :freqs_needed]  # [batch, seq_len, rope_portion//2]
    
    # Reshape RoPE portions for complex number operations
    # [batch, heads, seq_len, rope_portion] -> [batch, heads, seq_len, rope_portion//2, 2]
    xq_rope = xq_rope.float().reshape(batch, heads, seq_len, rope_portion // 2, 2)
    xk_rope = xk_rope.float().reshape(batch, heads, seq_len, rope_portion // 2, 2)
    
    # Convert to complex representation
    xq_complex = torch.view_as_complex(xq_rope)  # [batch, heads, seq_len, rope_portion//2]
    xk_complex = torch.view_as_complex(xk_rope)  # [batch, heads, seq_len, rope_portion//2]
    
    # Expand freqs_cis to match heads dimension
    # [batch, seq_len, rope_portion//2] -> [batch, heads, seq_len, rope_portion//2]
    freqs_cis_expanded = freqs_cis_used.unsqueeze(1).expand(batch, heads, seq_len, -1)
    
    # Apply rotation: complex multiplication
    xq_rotated = xq_complex * freqs_cis_expanded  # [batch, heads, seq_len, rope_portion//2]
    xk_rotated = xk_complex * freqs_cis_expanded  # [batch, heads, seq_len, rope_portion//2]
    
    # Convert back to real representation and flatten
    xq_rope_out = torch.view_as_real(xq_rotated).flatten(-2)  # [batch, heads, seq_len, rope_portion]
    xk_rope_out = torch.view_as_real(xk_rotated).flatten(-2)  # [batch, heads, seq_len, rope_portion]
    
    # Concatenate RoPE and non-RoPE portions
    xq_out = torch.cat([xq_rope_out, xq_pass], dim=-1)  # [batch, heads, seq_len, head_dim]
    xk_out = torch.cat([xk_rope_out, xk_pass], dim=-1)  # [batch, heads, seq_len, head_dim]
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def create_positional_encoder(encoding_type: str = 'rope_3d',
                             d_model: int = 512,
                             max_seq_length: int = 50,
                             spatial_scale: float = 0.1,
                             base: float = 1000.0) -> nn.Module:
    """
    Factory function for creating positional encoders for 3D neural data.
    
    **ABLATION STUDY SUPPORT**:
    - 'rope_3d': Multi-dimensional RoPE with 3D coordinates (x, y, t) (main approach)
    - 'standard_rope': Traditional 1D RoPE (temporal only)
    - 'learnable': Learnable 3D positional embeddings with MLPs
    - 'sinusoidal': Fixed sinusoidal positional encoding (temporal only)
    
    Args:
        encoding_type: Type of positional encoding
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        spatial_scale: Scaling factor for spatial coordinates (RoPE variants)
        base: Base frequency for RoPE/sinusoidal variants
        
    Returns:
        Positional encoder module
    """
    if encoding_type == 'rope_3d':
        return RoPE3D(d_model, max_seq_length, spatial_scale)
    elif encoding_type == 'standard_rope':
        return StandardRoPE(d_model, max_seq_length, base)
    elif encoding_type == 'learnable':
        return LearnableMultiDimPE(d_model, max_seq_length, spatial_scale)
    elif encoding_type == 'sinusoidal':
        return SinusoidalPE(d_model, max_seq_length, base)
    else:
        supported_types = ['rope_3d', 'standard_rope', 'learnable', 'sinusoidal']
        raise ValueError(f"Unsupported encoding type: {encoding_type}. Supported types: {supported_types}")

# ✅ Data preparation utility for transformer integration
def prepare_site_positional_data(neural_data: torch.Tensor, 
                                site_coords: torch.Tensor, 
                                device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare positional data for site-specific encoding.
    
    Args:
        neural_data: [B, S, T, N] - Neural activity data
        site_coords: [S, 2] - Fixed (X, Y) coordinates per site
        device: Target device
    
    Returns:
        coords_expanded: [B, S, T, 2] - Coordinates expanded to match data
        time_indices: [B, S, T] - Time indices for each position
    """
    B, S, T, N = neural_data.shape
    
    if device is None:
        device = neural_data.device
    
    # Move site coordinates to target device
    site_coords = site_coords.to(device)
    
    # Expand site coordinates to match neural data dimensions
    coords_expanded = site_coords.unsqueeze(0).unsqueeze(2).expand(B, S, T, 2)  # [B, S, T, 2]
    
    # Create time indices (ensure float dtype for positional encoders)
    time_indices = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, T)  # [B, S, T]
    
    return coords_expanded, time_indices


def test_rope_3d():
    """Test RoPE3D implementation with simplified structure."""
    
    print("🧪 Testing Simplified RoPE3D")
    print("=" * 40)
    
    # Test parameters
    d_model = 512
    batch_size = 2
    seq_len = 50
    n_heads = 8
    head_dim = d_model // n_heads
    
    # Create RoPE3D
    rope_3d = RoPE3D(d_model)
    print(f"✅ RoPE3D created: d_model={d_model}, rope_dim={rope_3d.rope_dim}")
    
    # Create test data
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Site coordinates: different (x, y) locations
    site_coords = torch.tensor([
        [[100.0, 200.0]] * seq_len,  # Site 1
        [[300.0, 400.0]] * seq_len   # Site 2
    ]).float()  # [B=2, T=50, 2]
    
    time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    print(f"Input shapes:")
    print(f"  q, k: {q.shape}")
    print(f"  site_coords: {site_coords.shape}")
    print(f"  time_indices: {time_indices.shape}")
    
    # Apply RoPE3D
    q_rotated, k_rotated = rope_3d.apply_rope(q, k, site_coords, time_indices)
    
    print(f"Output shapes:")
    print(f"  q_rotated: {q_rotated.shape}")
    print(f"  k_rotated: {k_rotated.shape}")
    
    # Verify rotation changes embeddings
    rotation_magnitude = torch.norm(q_rotated - q).item()
    print(f"  rotation magnitude: {rotation_magnitude:.4f}")
    
    if rotation_magnitude > 0.1:
        print("✅ RoPE3D test passed!")
        return True
    else:
        print("❌ RoPE3D test failed: insufficient rotation")
        return False


def test_ablation_study_encodings():
    """
    Comprehensive test for all positional encoding variants for 3D neural data.
    
    **ABLATION VARIANTS TESTED**:
    - RoPE3D: Multi-dimensional RoPE (main approach)
    - StandardRoPE: 1D temporal RoPE 
    - LearnableMultiDimPE: Learnable 3D embeddings with MLPs
    - SinusoidalPE: Fixed sinusoidal temporal encoding
    """
    
    print("🧪 Testing 3D Neural Data Ablation Study Positional Encodings")
    print("=" * 60)
    
    # Test parameters
    d_model = 512
    batch_size = 2
    seq_len = 50
    n_heads = 8
    head_dim = d_model // n_heads
    
    # Site coordinates and temporal indices for 3D neural data
    site_coords = torch.tensor([
        [[100.0, 200.0]] * seq_len,  # Site 1 at (100, 200)
        [[300.0, 400.0]] * seq_len   # Site 2 at (300, 400)
    ]).float()  # [B=2, T=50, 2]
    
    time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    # Test data for apply_rope interface
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    print(f"Input shapes:")
    print(f"  q, k: {q.shape}")
    print(f"  site_coords: {site_coords.shape}")
    print(f"  time_indices: {time_indices.shape}")
    print()
    
    # Test all encoding types
    encoding_configs = [
        {'type': 'rope_3d', 'name': 'RoPE3D (Multi-dimensional)', 'color': '🔵'},
        {'type': 'standard_rope', 'name': 'Standard RoPE (1D Temporal)', 'color': '🟢'},
        {'type': 'learnable', 'name': 'Learnable Multi-Dim PE', 'color': '🟡'},
        {'type': 'sinusoidal', 'name': 'Sinusoidal PE (Fixed)', 'color': '🟣'}
    ]
    
    results = {}
    
    for config in encoding_configs:
        encoding_type = config['type']
        print(f"{config['color']} Testing {config['name']}:")
        
        try:
            # Create encoder
            encoder = create_positional_encoder(
                encoding_type=encoding_type,
                d_model=d_model,
                max_seq_length=100,
                spatial_scale=1.0,
                base=10000.0
            )
            
            print(f"  ✅ Encoder created successfully")
            
            # Test forward pass (for additive encodings)
            pos_encoding = encoder(site_coords, time_indices)
            print(f"  📊 Positional encoding shape: {pos_encoding.shape}")
            
            # Test apply_rope interface
            q_rot, k_rot = encoder.apply_rope(q, k, site_coords, time_indices)
            print(f"  🔄 RoPE output shapes: q={q_rot.shape}, k={k_rot.shape}")
            
            # Measure modification magnitude
            q_diff = torch.norm(q_rot - q).item()
            k_diff = torch.norm(k_rot - k).item()
            
            print(f"  📏 Modification magnitude: q_diff={q_diff:.4f}, k_diff={k_diff:.4f}")
            
            # Store results
            results[encoding_type] = {
                'success': True,
                'pos_encoding_shape': pos_encoding.shape,
                'q_modification': q_diff,
                'k_modification': k_diff,
                'uses_rope': encoding_type in ['rope_3d', 'standard_rope']
            }
            
            print(f"  ✅ {config['name']} test passed!")
            
        except Exception as e:
            print(f"  ❌ {config['name']} test failed: {str(e)}")
            results[encoding_type] = {'success': False, 'error': str(e)}
        
        print()
    
    # Summary analysis
    print("📋 3D NEURAL DATA ABLATION STUDY SUMMARY:")
    print("=" * 40)
    
    successful_tests = [k for k, v in results.items() if v.get('success', False)]
    print(f"✅ Successful tests: {len(successful_tests)}/{len(encoding_configs)}")
    
    if successful_tests:
        print("\n🔍 COMPARATIVE ANALYSIS:")
        rope_types = [k for k in successful_tests if results[k]['uses_rope']]
        additive_types = [k for k in successful_tests if not results[k]['uses_rope']]
        
        print(f"  🔄 RoPE variants: {rope_types}")
        print(f"  ➕ Additive variants: {additive_types}")
        
        print("\n📊 Modification Magnitudes (Q tensor):")
        for enc_type in successful_tests:
            q_mod = results[enc_type]['q_modification']
            print(f"  {enc_type:15}: {q_mod:8.4f}")
        
        # Comparative insights
        print("\n💡 ABLATION INSIGHTS:")
        if ('rope_3d' in results and results['rope_3d'].get('success', False) and 
            'standard_rope' in results and results['standard_rope'].get('success', False)):
            rope3d_mod = results['rope_3d']['q_modification']
            rope1d_mod = results['standard_rope']['q_modification']
            print(f"  🆚 RoPE3D vs Standard RoPE: {rope3d_mod:.4f} vs {rope1d_mod:.4f}")
            if rope3d_mod > rope1d_mod:
                print(f"     → RoPE3D provides stronger positional signal than 1D RoPE")
            else:
                print(f"     → Standard RoPE provides comparable positional signal")
        
        if ('rope_3d' in results and results['rope_3d'].get('success', False) and 
            'learnable' in results and results['learnable'].get('success', False)):
            rope3d_mod = results['rope_3d']['q_modification']
            learnable_mod = results['learnable']['q_modification']
            print(f"  🆚 RoPE3D vs Learnable PE: {rope3d_mod:.4f} vs {learnable_mod:.4f}")
            print(f"     → Compare rotary vs MLP-based positional encoding")
        
        rope_mods = [results[k]['q_modification'] for k in rope_types if k in results]
        add_mods = [results[k]['q_modification'] for k in additive_types if k in results]
        
        if rope_mods and add_mods:
            avg_rope = sum(rope_mods) / len(rope_mods)
            avg_add = sum(add_mods) / len(add_mods)
            print(f"  📈 Average RoPE modification: {avg_rope:.4f}")
            print(f"  📈 Average additive modification: {avg_add:.4f}")
            
            print(f"\n🎯 RECOMMENDATION:")
            if avg_rope > avg_add * 1.5:
                print(f"     → RoPE variants show significantly stronger positional effects")
                print(f"     → Consider RoPE3D for better spatial-temporal modeling")
            elif avg_rope < avg_add * 0.5:
                print(f"     → Additive variants show stronger positional effects")
                print(f"     → Consider learnable embeddings for this task")
            else:
                print(f"     → Both approaches show comparable effects")
                print(f"     → Choose based on computational requirements and interpretability")
    
    return results


if __name__ == "__main__":
    # Run both tests
    print("Running RoPE3D test...")
    test_rope_3d()
    print("\n" + "="*60 + "\n")
    print("Running comprehensive 3D neural data ablation study test...")
    test_ablation_study_encodings()