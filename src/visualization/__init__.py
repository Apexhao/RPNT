"""
Visualization module for neural foundation model analysis.

This module provides tools for analyzing functional connectivity patterns
learned by the pretrained cross-site neural foundation model.

Key Analysis Methods:
- Attention-based connectivity from spatial encoder
- Noise replacement information flow analysis
- Temporal connectivity dynamics
- Anatomical visualization with site coordinates
"""

from .connectivity_analysis import ConnectivityAnalyzer
from .attention_maps import AttentionConnectivityExtractor
from .noise_replacement import NoiseReplacementAnalyzer
from .plotting_utils import ConnectivityPlotter

__all__ = [
    'ConnectivityAnalyzer',
    'AttentionConnectivityExtractor', 
    'NoiseReplacementAnalyzer',
    'ConnectivityPlotter'
]
