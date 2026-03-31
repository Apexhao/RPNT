"""
Attention-Based Connectivity Extraction Module

This module implements attention-based connectivity analysis from the spatial
encoder of the pretrained CrossSiteFoundationMAE model.

Mathematical Implementation:
C_{i→j}(attention) = (1/T·L) Σ_{t=1}^T Σ_{l=1}^L Attention_{i→j}^{(l,t)}

Key Features:
- Extract attention weights from SpatialCrossAttentionEncoder
- Temporal connectivity dynamics at specific timepoints
- Layer-wise attention analysis  
- Statistical summaries and confidence intervals
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from ..models import CrossSiteFoundationMAE


@dataclass
class AttentionAnalysisResults:
    """Results from attention-based connectivity analysis."""
    
    # Core connectivity matrices
    averaged_connectivity: torch.Tensor      # [S, S] - averaged over time/layers
    temporal_connectivity: torch.Tensor      # [T_samples, S, S] - at specific times
    layer_connectivity: torch.Tensor         # [L, S, S] - layer-wise connectivity
    
    # Raw attention weights
    raw_attention_weights: torch.Tensor      # [B, T, L, S, S] - full weights
    
    # Statistics
    attention_stats: Dict[str, torch.Tensor] # Mean, std, min, max across time/layers
    temporal_timepoints: List[int]           # Sampled timepoints
    
    # Metadata
    num_layers: int                          # Number of spatial layers
    sequence_length: int                     # Temporal sequence length
    num_sites: int                          # Number of recording sites


class AttentionConnectivityExtractor:
    """
    Extract functional connectivity from spatial encoder attention weights.
    
    **ATTENTION MECHANISM DETAILS**:
    - SpatialCrossAttentionEncoder processes timesteps independently
    - Each timestep has cross-site attention: Attention_{i→j}^{(l,t)}
    - Multiple layers provide hierarchical connectivity patterns
    - Temporal dynamics reveal how connectivity evolves during reaching
    """
    
    def __init__(self, 
                 model: CrossSiteFoundationMAE,
                 temporal_timepoints: List[int] = [0, 15, 30, 45]):
        """
        Initialize AttentionConnectivityExtractor.
        
        Args:
            model: Pretrained CrossSiteFoundationMAE model
            temporal_timepoints: Specific timepoints for temporal analysis
        """
        
        self.model = model
        self.temporal_timepoints = temporal_timepoints
        self.logger = logging.getLogger(__name__)
        
        # Verify model has spatial encoder
        if not hasattr(model, 'spatial_encoder'):
            raise ValueError("Model must have spatial_encoder attribute")
        
        if not hasattr(model.spatial_encoder, 'get_attention_weights'):
            raise ValueError("Spatial encoder must implement get_attention_weights()")
        
        self.num_layers = model.spatial_encoder.num_layers
        self.logger.info(f"AttentionConnectivityExtractor initialized - {self.num_layers} spatial layers")
    
    def extract_attention_weights(self, 
                                 neural_data: torch.Tensor,
                                 site_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Extract raw attention weights from spatial encoder.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coordinates: [S, 2] - site coordinates for positional encoding
            
        Returns:
            attention_weights: [B, T, L, S, S] - raw attention weights
        """
        
        self.logger.info("Extracting raw attention weights...")
        
        with torch.no_grad():
            # Forward pass through the model
            _ = self.model(neural_data, site_coordinates)
            
            # Extract attention weights from spatial encoder
            # Returns [B, T, num_layers, S, S]
            attention_weights = self.model.spatial_encoder.get_attention_weights()
            
            self.logger.info(f"Raw attention weights extracted: {attention_weights.shape}")
            
            return attention_weights
    
    def compute_averaged_connectivity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute connectivity averaged over time and layers (Equation 1).
        
        C_{i→j}(attention) = (1/T·L) Σ_{t=1}^T Σ_{l=1}^L Attention_{i→j}^{(l,t)}
        
        Args:
            attention_weights: [B, T, L, S, S] - raw attention weights
            
        Returns:
            averaged_connectivity: [S, S] - connectivity matrix
        """
        
        # Average across batch, time, and layers
        averaged_connectivity = attention_weights.mean(dim=[0, 1, 2])  # [S, S]
        
        self.logger.info(f"Averaged connectivity computed: {averaged_connectivity.shape}")
        
        return averaged_connectivity
    
    def compute_temporal_connectivity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute connectivity at specific temporal timepoints.
        
        Args:
            attention_weights: [B, T, L, S, S] - raw attention weights
            
        Returns:
            temporal_connectivity: [T_samples, S, S] - connectivity at timepoints
        """
        
        B, T, L, S, S = attention_weights.shape
        
        # Average across batch first
        batch_averaged = attention_weights.mean(dim=0)  # [T, L, S, S]
        
        temporal_connectivity = []
        
        for timepoint in self.temporal_timepoints:
            if timepoint < T:
                # Average across layers for this timepoint
                t_connectivity = batch_averaged[timepoint].mean(dim=0)  # [S, S]
                temporal_connectivity.append(t_connectivity)
            else:
                self.logger.warning(f"Timepoint {timepoint} exceeds sequence length {T}")
                # Use the last available timepoint
                t_connectivity = batch_averaged[-1].mean(dim=0)  # [S, S]
                temporal_connectivity.append(t_connectivity)
        
        temporal_connectivity = torch.stack(temporal_connectivity, dim=0)  # [T_samples, S, S]
        
        self.logger.info(f"Temporal connectivity computed: {temporal_connectivity.shape}")
        
        return temporal_connectivity
    
    def compute_layer_connectivity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute connectivity for each spatial layer separately.
        
        Args:
            attention_weights: [B, T, L, S, S] - raw attention weights
            
        Returns:
            layer_connectivity: [L, S, S] - connectivity per layer
        """
        
        # Average across batch and time for each layer
        layer_connectivity = attention_weights.mean(dim=[0, 1])  # [L, S, S]
        
        self.logger.info(f"Layer-wise connectivity computed: {layer_connectivity.shape}")
        
        return layer_connectivity
    
    def compute_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistical summaries of attention weights.
        
        Args:
            attention_weights: [B, T, L, S, S] - raw attention weights
            
        Returns:
            Dictionary with statistical summaries
        """
        
        # Flatten across batch, time, and layers for statistics
        flattened = attention_weights.view(-1, attention_weights.shape[-2], attention_weights.shape[-1])  # [B*T*L, S, S]
        
        stats = {
            'mean': flattened.mean(dim=0),      # [S, S]
            'std': flattened.std(dim=0),        # [S, S]
            'min': flattened.min(dim=0)[0],     # [S, S]
            'max': flattened.max(dim=0)[0],     # [S, S]
            'median': flattened.median(dim=0)[0] # [S, S]
        }
        
        self.logger.info("Attention statistics computed")
        
        return stats
    
    def analyze_attention_connectivity(self, 
                                     neural_data: torch.Tensor,
                                     site_coordinates: torch.Tensor) -> AttentionAnalysisResults:
        """
        Complete attention-based connectivity analysis.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            AttentionAnalysisResults with comprehensive analysis
        """
        
        self.logger.info("Starting complete attention connectivity analysis...")
        
        # 1. Extract raw attention weights
        attention_weights = self.extract_attention_weights(neural_data, site_coordinates)
        
        # 2. Compute different connectivity measures
        averaged_connectivity = self.compute_averaged_connectivity(attention_weights)
        temporal_connectivity = self.compute_temporal_connectivity(attention_weights)
        layer_connectivity = self.compute_layer_connectivity(attention_weights)
        
        # 3. Compute statistics
        attention_stats = self.compute_attention_statistics(attention_weights)
        
        # 4. Package results
        results = AttentionAnalysisResults(
            averaged_connectivity=averaged_connectivity,
            temporal_connectivity=temporal_connectivity,
            layer_connectivity=layer_connectivity,
            raw_attention_weights=attention_weights,
            attention_stats=attention_stats,
            temporal_timepoints=self.temporal_timepoints,
            num_layers=self.num_layers,
            sequence_length=attention_weights.shape[1],
            num_sites=attention_weights.shape[-1]
        )
        
        self.logger.info("✅ Attention connectivity analysis completed")
        self.logger.info(f"   Sites: {results.num_sites}")
        self.logger.info(f"   Layers: {results.num_layers}")  
        self.logger.info(f"   Sequence length: {results.sequence_length}")
        self.logger.info(f"   Temporal timepoints: {results.temporal_timepoints}")
        
        return results
    
    def get_connectivity_strength_ranking(self, connectivity_matrix: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Get ranked list of strongest connections.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            
        Returns:
            List of (source_site, target_site, strength) tuples, ranked by strength
        """
        
        S = connectivity_matrix.shape[0]
        connections = []
        
        for i in range(S):
            for j in range(S):
                if i != j:  # Exclude self-connections
                    strength = connectivity_matrix[i, j].item()
                    connections.append((i, j, strength))
        
        # Sort by strength (descending)
        connections.sort(key=lambda x: x[2], reverse=True)
        
        return connections
    
    def get_site_connectivity_summary(self, connectivity_matrix: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """
        Get connectivity summary for each site.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            
        Returns:
            Dictionary with per-site connectivity statistics
        """
        
        S = connectivity_matrix.shape[0]
        site_summary = {}
        
        for site in range(S):
            # Outgoing connections (influence on others)
            outgoing = connectivity_matrix[site, :]
            outgoing_strength = outgoing.sum().item() - outgoing[site].item()  # Exclude self
            
            # Incoming connections (influenced by others)
            incoming = connectivity_matrix[:, site]
            incoming_strength = incoming.sum().item() - incoming[site].item()  # Exclude self
            
            site_summary[site] = {
                'outgoing_strength': outgoing_strength,
                'incoming_strength': incoming_strength,
                'net_influence': outgoing_strength - incoming_strength,
                'total_connectivity': outgoing_strength + incoming_strength
            }
        
        return site_summary


def demo_attention_analysis():
    """Demo function for attention connectivity analysis."""
    
    print("🎯 Attention-Based Connectivity Analysis Demo")
    print("=" * 50)
    
    # This would normally use real model and data
    # For demo, create mock objects
    try:
        print("This demo requires a loaded model and data.")
        print("Use ConnectivityAnalyzer.analyze_connectivity() for full analysis.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_attention_analysis()
