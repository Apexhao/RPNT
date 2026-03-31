"""
Neural foundation models for cross-site neural analysis.

This module contains:
- CrossSiteFoundationMAE: Main foundation model with causal MAE training
- SparseTemporalEncoder: Causal temporal encoder with simplified attention
- SpatialCrossAttentionEncoder: Timestep-wise spatial attention 
- LightweightMAEDecoder: Minimal MAE decoder for pretraining
- SingleSite downstream models: Test temporal encoder generalization
- Model factory and utilities
"""

# Core foundation model components
from .transformer import (
    CrossSiteFoundationMAE,
    SparseTemporalEncoder, 
    SpatialCrossAttentionEncoder,
    LightweightMAEDecoder,
    CrossSiteModelFactory
)

# Attention mechanisms
from .attention import (
    CausalAdaptiveKernelAttention,
    create_causal_mask
)

# Positional encoding
from .positional_encoding import (
    StandardRoPE,
    SinusoidalPE,
    LearnableMultiDimPE,
    RoPE3D,
    apply_rotary_emb_3d,  # ✅ Core RoPE3D function
    create_positional_encoder,
    prepare_site_positional_data
)

# Single-site downstream models (test temporal encoder generalization)
from .downstream import (
    SingleSiteDownstreamRegressor,
    SingleSiteClassifier,
    ParameterLoRAFinetuneRegressor,
    ParameterMatrixFactorizedRegressor,
    count_parameters
)

__all__ = [
    # Foundation model components
    'CrossSiteFoundationMAE',
    'SparseTemporalEncoder',
    'SpatialCrossAttentionEncoder', 
    'LightweightMAEDecoder',
    'CrossSiteModelFactory',
    
    # Attention mechanisms
    'CausalAdaptiveKernelAttention',
    'create_causal_mask',
    
    # Positional encoding
    'StandardRoPE',
    'SinusoidalPE', 
    'LearnableMultiDimPE',
    'RoPE3D',
    'apply_rotary_emb_3d',
    'create_positional_encoder',
    'prepare_site_positional_data',
    
    # Single-site downstream models
    'SingleSiteDownstreamRegressor',
    'SingleSiteClassifier',
    'ParameterLoRAFinetuneRegressor',
    'ParameterMatrixFactorizedRegressor',
    'count_parameters'
] 