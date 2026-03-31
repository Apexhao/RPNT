"""
Public Dataset Positional Encoding for Neural Foundation Model
-------------------------------------------------------------

This module provides RoPE4D positional encoding adapted for the public dataset sessions.

Key Features:
- RoPE4D: Handles 4D coordinates (subject, time, task, temporal_sequence)
- Session-based encoding instead of site-based encoding
- Compatible with [B,S=1,T,N] input format
- Zero-shot generalization to new session combinations
"""

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
    - Ignores session metadata (subject, time, task)
    - Direct comparison to show benefits of multi-dimensional encoding
    
    **KEY FEATURES**:
    - Temporal sequence: Standard RoPE on within-trial temporal patterns
    - No session-specific encoding
    - Compatible with original RoPE formulation
    
    **USAGE**: Integrated directly into attention mechanism (not as standalone PE)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, base: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Use all dimensions for temporal RoPE rotation
        self.rope_dim = d_model //2
        
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
                   session_coords: torch.Tensor,         # [batch, seq_len, 3]
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified apply_rope interface for compatibility with other PE classes.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - session coordinates
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
    
    def encode_session_features(self, session_coords: torch.Tensor) -> torch.Tensor:
        """
        No session encoding for standard RoPE - return zeros.
        
        Args:
            session_coords: [batch, seq_len, 3] - session coordinates (IGNORED)
            
        Returns:
            zeros: [batch, seq_len, 0] - empty tensor for compatibility
        """
        batch_size, seq_len, _ = session_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=session_coords.device)
    
    def forward(self, 
                session_coords: torch.Tensor,     # [batch, seq_len, 3] - IGNORED
                temporal_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for standard RoPE (no session-specific encoding).
        
        Args:
            session_coords: [batch, seq_len, 3] - (subject, time, task) coordinates (IGNORED)
            temporal_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            empty: [batch, seq_len, 0] - empty for compatibility
        """
        return self.encode_session_features(session_coords)


class LearnableMultiDimPE(nn.Module):
    """
    Learnable Multi-Dimensional Positional Encoding for 4D Coordinates.
    
    **ABLATION COMPARISON**:
    - Learnable embedding tables for each coordinate dimension
    - Direct comparison to RoPE4D to show benefits of rotary encoding
    - Standard additive positional encoding extended to 4D
    
    **KEY FEATURES**:
    - Subject embedding: Learnable embeddings for subjects
    - Time embedding: Learnable embeddings for time periods  
    - Task embedding: Learnable embeddings for task types
    - Temporal embedding: Learnable embeddings for temporal positions
    
    **USAGE**: Additive positional encoding (not integrated into attention)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, 
                 max_subjects: int = 4, max_time_periods: int = 10, max_tasks: int = 2):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Dimension allocation for each coordinate type
        # dim_per_coord = d_model // 4
        
        self.subject_dim = d_model //32
        self.time_dim = d_model //32  
        self.task_dim = d_model //32
        self.temporal_dim = d_model - 3 * (d_model // 32) # Give remainder to temporal
        
        # Learnable embedding tables for each coordinate dimension
        self.subject_embedding = nn.Embedding(max_subjects, self.subject_dim)
        self.time_embedding = nn.Embedding(max_time_periods, self.time_dim)
        self.task_embedding = nn.Embedding(max_tasks, self.task_dim)
        self.temporal_embedding = nn.Embedding(max_seq_length, self.temporal_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding tables with proper scaling."""
        for embedding in [self.subject_embedding, self.time_embedding, 
                         self.task_embedding, self.temporal_embedding]:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.5)
    
    def forward(self, 
                session_coords: torch.Tensor,     # [batch, seq_len, 3]
                temporal_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for learnable multi-dimensional positional encoding.
        
        Args:
            session_coords: [batch, seq_len, 3] - (subject, time, task) coordinates
            temporal_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            pos_encoding: [batch, seq_len, d_model] - multi-dimensional positional encoding
        """

        # batch_size, seq_len, _ = session_coords.shape
        # device = session_coords.device
        
        # Extract and convert coordinates to indices
        subject_idx = session_coords[..., 0].long()  # [batch, seq_len]
        time_idx = session_coords[..., 1].long()     # [batch, seq_len] 
        task_idx = session_coords[..., 2].long()     # [batch, seq_len]
        temporal_idx = temporal_indices.long()       # [batch, seq_len]
        
        # Clamp indices to valid ranges (safety)
        subject_idx = torch.clamp(subject_idx, 0, self.subject_embedding.num_embeddings - 1)
        time_idx = torch.clamp(time_idx, 0, self.time_embedding.num_embeddings - 1)
        task_idx = torch.clamp(task_idx, 0, self.task_embedding.num_embeddings - 1)
        temporal_idx = torch.clamp(temporal_idx, 0, self.temporal_embedding.num_embeddings - 1)
        
        # Get embeddings for each coordinate
        subject_emb = self.subject_embedding(subject_idx)    # [batch, seq_len, subject_dim]
        time_emb = self.time_embedding(time_idx)             # [batch, seq_len, time_dim]
        task_emb = self.task_embedding(task_idx)             # [batch, seq_len, task_dim]
        temporal_emb = self.temporal_embedding(temporal_idx) # [batch, seq_len, temporal_dim]
        
        # Concatenate all embeddings
        pos_encoding = torch.cat([subject_emb, time_emb, task_emb, temporal_emb], dim=-1)
        # [batch, seq_len, subject_dim + time_dim + task_dim + temporal_dim] = [batch, seq_len, d_model]
        
        return pos_encoding
    
    def encode_session_features(self, session_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode session-specific features (compatibility with RoPE interface).
        
        Args:
            session_coords: [batch, seq_len, 3] - session coordinates
            
        Returns:
            session_features: [batch, seq_len, 0] - empty for compatibility
        """
        batch_size, seq_len, _ = session_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=session_coords.device)
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   session_coords: torch.Tensor,         # [batch, seq_len, 3]
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        No rotation for learnable embeddings - return unchanged Q and K.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - session coordinates (UNUSED)
            temporal_indices: [batch, seq_len] - temporal indices (UNUSED)
            
        Returns:
            q: [batch, seq_len, n_heads, head_dim] - unchanged query
            k: [batch, seq_len, n_heads, head_dim] - unchanged key
        """
        return q, k


class SinusoidalPE(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding for Temporal Sequence Only.
    
    **ABLATION BASELINE**:
    - Classic transformer sinusoidal positional encoding
    - Only applied to temporal dimension (ignores session metadata)
    - Fixed, deterministic patterns
    
    **KEY FEATURES**:
    - Temporal sequence: Fixed sinusoidal patterns
    - No session-specific encoding
    - Classic transformer PE formulation
    
    **USAGE**: Additive positional encoding (not integrated into attention)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, base: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # Create sinusoidal position encoding table
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal patterns
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(base) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, 
                session_coords: torch.Tensor,     # [batch, seq_len, 3] - IGNORED
                temporal_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for sinusoidal positional encoding.
        
        Args:
            session_coords: [batch, seq_len, 3] - (subject, time, task) coordinates (IGNORED)
            temporal_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            pos_encoding: [batch, seq_len, d_model] - sinusoidal positional encoding
        """
        # batch_size, seq_len = temporal_indices.shape
        
        # Clamp temporal indices to valid range
        temporal_idx = torch.clamp(temporal_indices.long(), 0, self.pe.size(0) - 1)
        
        # Extract sinusoidal encodings for temporal positions
        pos_encoding = self.pe[temporal_idx]  # [batch, seq_len, d_model]
        
        return pos_encoding
    
    def encode_session_features(self, session_coords: torch.Tensor) -> torch.Tensor:
        """
        No session encoding for sinusoidal PE - return zeros.
        
        Args:
            session_coords: [batch, seq_len, 3] - session coordinates (IGNORED)
            
        Returns:
            zeros: [batch, seq_len, 0] - empty tensor for compatibility
        """
        batch_size, seq_len, _ = session_coords.shape
        return torch.zeros(batch_size, seq_len, 0, device=session_coords.device)
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   session_coords: torch.Tensor,         # [batch, seq_len, 3]
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        No rotation for sinusoidal PE - return unchanged Q and K.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - session coordinates (UNUSED)
            temporal_indices: [batch, seq_len] - temporal indices (UNUSED)
            
        Returns:
            q: [batch, seq_len, n_heads, head_dim] - unchanged query
            k: [batch, seq_len, n_heads, head_dim] - unchanged key
        """
        return q, k


class RoPE4D(nn.Module):
    """
    4D Rotary Position Embedding for Public Dataset Sessions.
    
    **DESIGN EVOLUTION**:
    - Extends RoPE3D (x, y, temporal) → RoPE4D (subject, time, task, temporal)
    - Session-based coordinates instead of spatial coordinates
    - Generalizable to new session combinations (subject/time/task)
    
    **KEY FEATURES**:
    - Subject embedding: Handles cross-subject generalization
    - Time embedding: Temporal patterns across recording sessions
    - Task embedding: Task-specific neural dynamics
    - Temporal sequence: Within-trial temporal patterns
    
    **USAGE**: Integrated directly into attention mechanism (not as standalone PE)
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, session_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.session_scale = session_scale
        
        self.rope_dim = d_model   # For 4D rotations (subject, time, task, temporal)
        
        # Each coordinate dimension gets equal share of RoPE dimensions
        # rope_dim must be divisible by 8 (4 coords × 2 for complex pairs)
        if self.rope_dim % 8 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 8 for RoPE4D. Need at least 16 dimensions.")
 
        self.dim_per_coord = self.rope_dim // 4  # Dimensions per coordinate (subject, time, task, temporal)
        
        if self.dim_per_coord == 0:
            raise ValueError(f"d_model ({d_model}) too small for RoPE4D. Need at least 16 dimensions.")
        
        # Create inverse frequency vectors for rotation computation
        
        # Subject dimension: lowest frequency for subject-level patterns
        inv_freq_subject = 1.0 / (100 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_subject', inv_freq_subject)
        
        # Time dimension: medium frequency for session-level temporal patterns
        inv_freq_time = 1.0 / (1000 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_time', inv_freq_time)
        
        # Task dimension: smallest frequency for task-specific patterns
        inv_freq_task = 1.0 / (10 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_task', inv_freq_task)
        
        # Temporal sequence: highest frequency for fine temporal resolution within trials
        inv_freq_temporal = 1.0 / (10000 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer('inv_freq_temporal', inv_freq_temporal)
         
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_4d_freqs_cis(self, 
                            session_coords: torch.Tensor,     # [batch, seq_len, 3] 
                            temporal_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Compute complex frequency tensors for 4D RoPE application.
        
        Args:
            session_coords: [batch, seq_len, 3] - (subject, time, task) coordinates
            temporal_indices: [batch, seq_len] - temporal sequence indices
        Returns:
            freqs_cis: [batch, seq_len, rope_dim//2] - complex frequency tensor
        """
        batch_size, seq_len = temporal_indices.shape
        device = session_coords.device
        
        # Extract session coordinates
        subject = session_coords[..., 0] * self.session_scale  # [batch, seq_len]
        time = session_coords[..., 1] * self.session_scale     # [batch, seq_len]
        task = session_coords[..., 2] * self.session_scale     # [batch, seq_len]
        temporal = temporal_indices.float()                    # [batch, seq_len]
        
        def compute_freqs_1d(pos, inv_freq):
            """Compute complex frequencies for one coordinate dimension."""
            # pos: [batch, seq_len], inv_freq: [dim_per_coord//2]
            freqs = torch.einsum('bi,j->bij', pos, inv_freq)
            # Convert to complex: e^(i*freqs) = cos(freqs) + i*sin(freqs)
            return torch.polar(torch.ones_like(freqs), freqs)  # [batch, seq_len, dim_per_coord//2]
        
        # Compute complex frequencies for each coordinate
        freqs_subject = compute_freqs_1d(subject, self.inv_freq_subject)    # [batch, seq_len, dim_per_coord//2]
        freqs_time = compute_freqs_1d(time, self.inv_freq_time)             # [batch, seq_len, dim_per_coord//2]
        freqs_task = compute_freqs_1d(task, self.inv_freq_task)             # [batch, seq_len, dim_per_coord//2]
        freqs_temporal = compute_freqs_1d(temporal, self.inv_freq_temporal) # [batch, seq_len, dim_per_coord//2]
        
        # Concatenate all frequency components
        freqs_cis = torch.cat([freqs_subject, freqs_time, freqs_task, freqs_temporal], dim=-1)
        # [batch, seq_len, 4 * dim_per_coord//2] = [batch, seq_len, rope_dim//2]
        
        return freqs_cis
    
    def apply_rope_4d(self, 
                     q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                     k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                     session_coords: torch.Tensor,         # [batch, seq_len, 3]
                     temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 4D RoPE to query and key tensors.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - session coordinates
            temporal_indices: [batch, seq_len] - temporal indices
            
        Returns:
            q_rotated: [batch, seq_len, n_heads, head_dim] - rotated query
            k_rotated: [batch, seq_len, n_heads, head_dim] - rotated key
        """
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Compute 4D rotation frequencies
        freqs_cis = self.compute_4d_freqs_cis(session_coords, temporal_indices)  # [batch, seq_len, rope_dim//2]
        
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
    
    def apply_rope(self, 
                   q: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   k: torch.Tensor,                      # [batch, seq_len, n_heads, head_dim]
                   session_coords: torch.Tensor,         # [batch, seq_len, 3]
                   temporal_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified apply_rope interface for compatibility with other PE classes.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim] - query tensor
            k: [batch, seq_len, n_heads, head_dim] - key tensor
            session_coords: [batch, seq_len, 3] - session coordinates
            temporal_indices: [batch, seq_len] - temporal indices
            
        Returns:
            q_rotated: [batch, seq_len, n_heads, head_dim] - rotated query
            k_rotated: [batch, seq_len, n_heads, head_dim] - rotated key
        """
        return self.apply_rope_4d(q, k, session_coords, temporal_indices)
    
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
    
    def encode_session_features(self, session_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode session-specific features for non-RoPE dimensions.
        
        Args:
            session_coords: [batch, seq_len, 3] - session coordinates
            
        Returns:
            session_features: [batch, seq_len, session_dim] - encoded session features
        """
        return self.session_encoder(session_coords)
    
    def forward(self, 
                session_coords: torch.Tensor,     # [batch, seq_len, 3]
                temporal_indices: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        """
        Forward pass for session-specific positional encoding.
        
        **NOTE**: This is mainly for compatibility. The main functionality
        is in apply_rope_4d() which is called during attention.
        
        Args:
            session_coords: [batch, seq_len, 3] - (subject, time, task) coordinates
            temporal_indices: [batch, seq_len] - temporal sequence positions
        Returns:
            session_encoding: [batch, seq_len, session_dim] - session-specific features
        """
        return self.encode_session_features(session_coords)


def create_public_positional_encoder(encoding_type: str = 'rope_4d',
                                    d_model: int = 512,
                                    max_seq_length: int = 5000,
                                    session_scale: float = 1.0,
                                    max_subjects: int = 10,
                                    max_time_periods: int = 10,
                                    max_tasks: int = 10,
                                    base: float = 10000.0) -> nn.Module:
    """
    Factory function for creating public dataset positional encoders.
    
    **ABLATION STUDY SUPPORT**:
    - 'rope_4d': Multi-dimensional RoPE with 4D coordinates (baseline)
    - 'standard_rope': Traditional 1D RoPE (temporal only)
    - 'learnable': Learnable 4D positional embeddings
    - 'sinusoidal': Fixed sinusoidal positional encoding (temporal only)
    
    Args:
        encoding_type: Type of positional encoding
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        session_scale: Scaling factor for session coordinates (RoPE variants)
        max_subjects: Maximum number of subjects (learnable only)
        max_time_periods: Maximum number of time periods (learnable only)
        max_tasks: Maximum number of tasks (learnable only)
        base: Base frequency for RoPE/sinusoidal (RoPE/sinusoidal variants)
        
    Returns:
        Positional encoder module
    """
    if encoding_type == 'rope_4d':
        return RoPE4D(d_model, max_seq_length, session_scale)
    elif encoding_type == 'standard_rope':
        return StandardRoPE(d_model, max_seq_length, base)
    elif encoding_type == 'learnable':
        return LearnableMultiDimPE(d_model, max_seq_length, max_subjects, max_time_periods, max_tasks)
    elif encoding_type == 'sinusoidal':
        return SinusoidalPE(d_model, max_seq_length, base)
    else:
        supported_types = ['rope_4d', 'standard_rope', 'learnable', 'sinusoidal']
        raise ValueError(f"Unsupported encoding type: {encoding_type}. Supported types: {supported_types}")


def prepare_session_positional_data(neural_data: torch.Tensor,        # [B, S=1, T, N]
                                   session_coords: torch.Tensor,       # [B, S=1, 3]
                                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare session-specific positional data for public dataset.
    
    Args:
        neural_data: [B, S=1, T, N] - neural activity data
        session_coords: [B, S=1, 3] - session coordinates per batch
        device: Target device
        
    Returns:
        coords_expanded: [B, S=1, T, 3] - session coordinates expanded for all timesteps
        temporal_indices: [B, S=1, T] - temporal indices
    """
    B, S, T, N = neural_data.shape
    
    # Create temporal indices
    temporal_indices = torch.arange(T, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, T)  # [B, S, T]
    
    # Expand session coordinates for all timesteps
    coords_expanded = session_coords.unsqueeze(2).expand(B, S, T, 3)  # [B, S=1, T, 3]
    
    return coords_expanded, temporal_indices


# Test function
def test_rope_4d():
    """Test RoPE4D implementation."""
    
    print("🧪 Testing RoPE4D")
    print("=" * 40)
    
    # Test parameters
    d_model = 512
    batch_size = 2
    seq_len = 50
    n_heads = 8
    head_dim = d_model // n_heads
    
    # Create RoPE4D
    rope_4d = RoPE4D(d_model)
    print(f"✅ RoPE4D created: d_model={d_model}, rope_dim={rope_4d.rope_dim}, session_dim={rope_4d.session_dim}")
    
    # Create test data
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Session coordinates: (subject=0-3, time=0-1, task=0-1)
    session_coords = torch.tensor([
        [[0.0, 0.3, 0.0]],  # Subject c, mid-2013, center-out
        [[3.0, 0.8, 1.0]]   # Subject t, late-2013, random-target
    ]).expand(batch_size, 1, 3).expand(batch_size, seq_len, 3)
    
    temporal_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    print(f"Input shapes:")
    print(f"  q, k: {q.shape}")
    print(f"  session_coords: {session_coords.shape}")
    print(f"  temporal_indices: {temporal_indices.shape}")
    
    # Apply RoPE4D (using unified interface)
    q_rotated, k_rotated = rope_4d.apply_rope(q, k, session_coords, temporal_indices)
    
    print(f"Output shapes:")
    print(f"  q_rotated: {q_rotated.shape}")
    print(f"  k_rotated: {k_rotated.shape}")
    
    # Test session encoding
    session_features = rope_4d.encode_session_features(session_coords)
    print(f"  session_features: {session_features.shape}")
    
    # Verify rotation changes embeddings
    rotation_magnitude = torch.norm(q_rotated - q).item()
    print(f"  rotation magnitude: {rotation_magnitude:.4f}")
    
    if rotation_magnitude > 0.1:
        print("✅ RoPE4D test passed!")
        return True
    else:
        print("❌ RoPE4D test failed: insufficient rotation")
        return False


def test_ablation_study_encodings():
    """
    Comprehensive test for all positional encoding variants in ablation study.
    
    **ABLATION VARIANTS TESTED**:
    - RoPE4D: Multi-dimensional RoPE (baseline)
    - StandardRoPE: 1D temporal RoPE 
    - LearnableMultiDimPE: Learnable 4D embeddings
    - SinusoidalPE: Fixed sinusoidal temporal encoding
    """
    
    print("🧪 Testing Ablation Study Positional Encodings")
    print("=" * 60)
    
    # Test parameters
    d_model = 512
    batch_size = 2
    seq_len = 50
    n_heads = 8
    head_dim = d_model // n_heads
    
    # Session coordinates and temporal indices
    session_coords = torch.tensor([
        [[0.0, 0.3, 0.0]],  # Subject c, mid-2013, center-out
        [[3.0, 0.8, 1.0]]   # Subject t, late-2013, random-target
    ]).expand(batch_size, 1, 3).expand(batch_size, seq_len, 3)
    
    temporal_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    
    # Test data for apply_rope interface
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    print(f"Input shapes:")
    print(f"  q, k: {q.shape}")
    print(f"  session_coords: {session_coords.shape}")
    print(f"  temporal_indices: {temporal_indices.shape}")
    print()
    
    # Test all encoding types
    encoding_configs = [
        {'type': 'rope_4d', 'name': 'RoPE4D (Multi-dimensional)', 'color': '🔵'},
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
            encoder = create_public_positional_encoder(
                encoding_type=encoding_type,
                d_model=d_model,
                max_seq_length=100,
                session_scale=1.0,
                max_subjects=5,
                max_time_periods=5,
                max_tasks=3
            )
            
            print(f"  ✅ Encoder created successfully")
            
            # Test forward pass (for additive encodings)
            pos_encoding = encoder(session_coords, temporal_indices)
            print(f"  📊 Positional encoding shape: {pos_encoding.shape}")
            
            # Test apply_rope interface
            q_rot, k_rot = encoder.apply_rope(q, k, session_coords, temporal_indices)
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
                'uses_rope': encoding_type in ['rope_4d', 'standard_rope']
            }
            
            print(f"  ✅ {config['name']} test passed!")
            
        except Exception as e:
            print(f"  ❌ {config['name']} test failed: {str(e)}")
            results[encoding_type] = {'success': False, 'error': str(e)}
        
        print()
    
    # Summary analysis
    print("📋 ABLATION STUDY SUMMARY:")
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
        if ('rope_4d' in results and results['rope_4d'].get('success', False) and 
            'standard_rope' in results and results['standard_rope'].get('success', False)):
            rope4d_mod = results['rope_4d']['q_modification']
            rope1d_mod = results['standard_rope']['q_modification']
            print(f"  🆚 RoPE4D vs Standard RoPE: {rope4d_mod:.4f} vs {rope1d_mod:.4f}")
        else:
            print(f"  ⚠️  Cannot compare RoPE variants (some tests failed)")
        
        rope_mods = [results[k]['q_modification'] for k in rope_types if k in results]
        add_mods = [results[k]['q_modification'] for k in additive_types if k in results]
        
        if rope_mods and add_mods:
            avg_rope = sum(rope_mods) / len(rope_mods)
            avg_add = sum(add_mods) / len(add_mods)
            print(f"  📈 Average RoPE modification: {avg_rope:.4f}")
            print(f"  📈 Average additive modification: {avg_add:.4f}")
    
    return results


if __name__ == "__main__":
    # Run both tests
    print("Running original RoPE4D test...")
    test_rope_4d()
    print("\n" + "="*60 + "\n")
    print("Running comprehensive ablation study test...")
    test_ablation_study_encodings()
