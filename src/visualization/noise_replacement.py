"""
Noise Replacement Information Flow Analysis Module

This module implements noise replacement analysis to quantify information
dependencies between recording sites in the neural foundation model.

Mathematical Implementation:
C_{i→j}(N) = (L_j(X) - L_j(X_{i←N})) / L_j(X)

Where:
- L_j(X): Reconstruction loss at site j with original data
- L_j(X_{i←N}): Reconstruction loss at site j when site i is replaced with noise
- High values indicate site j depends on information from site i

Key Features:
- Gaussian noise replacement for each site
- Site-specific reconstruction loss computation  
- 0-1 normalized information dependency scores
- Asymmetric connectivity matrix revealing information flow
- Statistical significance testing with multiple noise realizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from ..models import CrossSiteFoundationMAE


@dataclass  
class NoiseAnalysisResults:
    """Results from noise replacement analysis."""
    
    # Core connectivity matrices
    raw_connectivity: torch.Tensor           # [S, S] - raw loss differences
    normalized_connectivity: torch.Tensor    # [S, S] - 0-1 normalized scores
    
    # Loss matrices
    original_losses: torch.Tensor            # [S] - original reconstruction losses per site
    noisy_losses: torch.Tensor              # [S, S] - losses when site i→noise affects site j
    
    # Statistics across noise realizations
    connectivity_mean: torch.Tensor          # [S, S] - mean across noise realizations
    connectivity_std: torch.Tensor           # [S, S] - std across noise realizations
    
    # Metadata
    num_noise_realizations: int              # Number of noise samples tested
    noise_std: float                        # Standard deviation of Gaussian noise
    normalization_method: str               # Method used for 0-1 normalization


class NoiseReplacementAnalyzer:
    """
    Analyze information flow using noise replacement methodology.
    
    **METHODOLOGY**:
    1. For each site i, replace its data with Gaussian noise
    2. Measure reconstruction loss degradation at all other sites j
    3. High degradation at j when i is noisy → site j depends on site i
    4. Creates asymmetric connectivity matrix revealing information flow
    5. Normalize to [0,1] range for interpretability
    """
    
    def __init__(self, 
                 model: CrossSiteFoundationMAE,
                 noise_std: float = 1.0,
                 num_noise_realizations: int = 5,
                 normalization_method: str = 'sigmoid'):
        """
        Initialize NoiseReplacementAnalyzer.
        
        Args:
            model: Pretrained CrossSiteFoundationMAE model
            noise_std: Standard deviation for Gaussian noise
            num_noise_realizations: Number of noise samples for robustness
            normalization_method: Method for 0-1 normalization ('sigmoid', 'minmax', 'clip')
        """
        
        self.model = model
        self.noise_std = noise_std
        self.num_noise_realizations = num_noise_realizations
        self.normalization_method = normalization_method
        self.logger = logging.getLogger(__name__)
        
        # Verify model has MAE decoder
        if not hasattr(model, 'mae_decoder') or model.mae_decoder is None:
            raise ValueError("Model must have mae_decoder for reconstruction loss computation")
        
        self.logger.info(f"NoiseReplacementAnalyzer initialized")
        self.logger.info(f"   Noise std: {noise_std}")
        self.logger.info(f"   Noise realizations: {num_noise_realizations}")
        self.logger.info(f"   Normalization: {normalization_method}")
    
    def compute_site_reconstruction_losses(self, 
                                         neural_data: torch.Tensor,
                                         site_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson reconstruction loss for each site separately.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            site_losses: [S] - Poisson reconstruction loss for each site
        """
        
        with torch.no_grad():
            # Forward pass to get reconstruction
            outputs = self.model(neural_data, site_coordinates, return_mae_reconstruction=True)
            reconstruction = outputs['reconstruction']  # [B, S, T, N]
            
            # Import Poisson loss function
            from ..evaluation.loss_functions import compute_poisson_loss_pytorch
            
            # Compute Poisson loss for each site separately
            site_losses = []
            B, S, T, N = neural_data.shape
            
            for site_idx in range(S):
                site_target = neural_data[:, site_idx, :, :]     # [B, T, N]
                site_recon = reconstruction[:, site_idx, :, :]   # [B, T, N]
                
                # Create mask (all positions are valid for loss computation)
                site_mask = torch.ones_like(site_target)  # [B, T, N]
                
                # Poisson loss for this site
                site_loss = compute_poisson_loss_pytorch(
                    predicted_rates=site_recon,  # [B, T, N]
                    target_spikes=site_target,   # [B, T, N]
                    mask=site_mask              # [B, T, N]
                )
                site_losses.append(site_loss)
            
            site_losses = torch.stack(site_losses)  # [S]
            
        return site_losses
    
    def replace_site_with_noise(self, 
                               neural_data: torch.Tensor,
                               site_idx: int,
                               noise_std: Optional[float] = None) -> torch.Tensor:
        """
        Replace specific site data with Gaussian noise.
        
        Args:
            neural_data: [B, S, T, N] - original neural data
            site_idx: Index of site to replace with noise
            noise_std: Noise standard deviation (uses self.noise_std if None)
            
        Returns:
            noisy_data: [B, S, T, N] - data with site_idx replaced by noise
        """
        
        if noise_std is None:
            noise_std = self.noise_std
        
        # Clone original data
        noisy_data = neural_data.clone()
        
        # Replace specified site with Gaussian noise
        B, S, T, N = neural_data.shape
        noise = torch.randn(B, T, N, device=neural_data.device) * noise_std
        
        noisy_data[:, site_idx, :, :] = noise
        
        return noisy_data
    
    def compute_information_dependency_matrix(self, 
                                            neural_data: torch.Tensor,
                                            site_coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full information dependency matrix using noise replacement.
        
        **MATHEMATICAL IMPLEMENTATION**:
        C_{i→j}(N) = (L_j(X) - L_j(X_{i←N})) / L_j(X)
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            Tuple of (raw_connectivity, original_losses, noisy_losses)
            - raw_connectivity: [S, S] - raw loss differences
            - original_losses: [S] - original losses per site
            - noisy_losses: [S, S] - losses when site i→noise affects site j
        """
        
        self.logger.info("Computing information dependency matrix...")
        
        B, S, T, N = neural_data.shape
        
        # 1. Compute original reconstruction losses
        original_losses = self.compute_site_reconstruction_losses(neural_data, site_coordinates)
        
        # 2. Initialize storage for noisy losses
        noisy_losses = torch.zeros(S, S, device=neural_data.device)
        
        # 3. For each site i, replace with noise and measure impact on all sites j
        for noise_site_i in range(S):
            self.logger.debug(f"Processing noise replacement for site {noise_site_i}/{S}")
            
            # Replace site i with noise
            noisy_data = self.replace_site_with_noise(neural_data, noise_site_i)
            
            # Compute reconstruction losses with noisy site i
            site_losses_noisy = self.compute_site_reconstruction_losses(noisy_data, site_coordinates)
            
            # Store losses for all affected sites j
            noisy_losses[noise_site_i, :] = site_losses_noisy
        
        # 4. Compute information dependency matrix (Equation 2)
        # C_{i→j} = (L_j(X) - L_j(X_{i←N})) / L_j(X)
        
        # Expand original_losses for broadcasting: [S] → [1, S]
        original_expanded = original_losses.unsqueeze(0)  # [1, S]
        
        # Compute raw differences: L_j(X_{i←N}) - L_j(X)
        # Note: We use (noisy - original) to get positive values when noise increases loss
        loss_differences = noisy_losses - original_expanded  # [S, S]
        
        # Normalize by original losses to get relative change
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        raw_connectivity = loss_differences / (original_expanded + epsilon)  # [S, S]
        
        self.logger.info(f"✅ Information dependency matrix computed: {raw_connectivity.shape}")
        
        return raw_connectivity, original_losses, noisy_losses
    
    def normalize_connectivity_to_01(self, raw_connectivity: torch.Tensor) -> torch.Tensor:
        """
        Normalize connectivity matrix to [0, 1] range.
        
        Args:
            raw_connectivity: [S, S] - raw connectivity values
            
        Returns:
            normalized_connectivity: [S, S] - values in [0, 1]
        """
        
        if self.normalization_method == 'sigmoid':
            # Sigmoid normalization: sigmoid(α * x)
            alpha = 2.0  # Scaling factor
            normalized = torch.sigmoid(alpha * raw_connectivity)
            
        elif self.normalization_method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            min_val = raw_connectivity.min()
            max_val = raw_connectivity.max()
            normalized = (raw_connectivity - min_val) / (max_val - min_val + 1e-8)
            
        elif self.normalization_method == 'standard':
            # Robust normalization using percentiles to handle outliers better
            # Use 10th and 90th percentiles to focus on majority of values
            p10 = torch.quantile(raw_connectivity, 0.20)
            p90 = torch.quantile(raw_connectivity, 0.80)
            
            # Clip extreme values to reduce outlier influence
            clipped_connectivity = torch.clamp(raw_connectivity, p10, p90)
            
            # Min-max normalization on clipped values for better spread
            min_val = clipped_connectivity.min()
            max_val = clipped_connectivity.max()
            normalized = (clipped_connectivity - min_val) / (max_val - min_val + 1e-8)
            
        elif self.normalization_method == 'clip':
            # Simple clipping to [0, 1]
            normalized = torch.clamp(raw_connectivity, 0.0, 1.0)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        self.logger.info(f"Connectivity normalized using {self.normalization_method} method")
        
        return normalized
    
    def compute_robust_connectivity(self, 
                                  neural_data: torch.Tensor,
                                  site_coordinates: torch.Tensor) -> NoiseAnalysisResults:
        """
        Compute robust connectivity with multiple noise realizations.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            NoiseAnalysisResults with robust connectivity estimates
        """
        
        self.logger.info(f"Computing robust connectivity with {self.num_noise_realizations} realizations...")
        
        S = neural_data.shape[1]
        
        # Storage for multiple realizations
        all_raw_connectivity = []
        all_original_losses = []
        all_noisy_losses = []
        
        # Compute connectivity for each noise realization
        for realization in range(self.num_noise_realizations):
            self.logger.debug(f"Noise realization {realization + 1}/{self.num_noise_realizations}")
            
            raw_conn, orig_losses, noisy_losses = self.compute_information_dependency_matrix(
                neural_data, site_coordinates
            )
            
            all_raw_connectivity.append(raw_conn)
            all_original_losses.append(orig_losses)
            all_noisy_losses.append(noisy_losses)
        
        # Stack results: [num_realizations, S, S]
        all_raw_connectivity = torch.stack(all_raw_connectivity, dim=0)
        all_original_losses = torch.stack(all_original_losses, dim=0)  # [num_realizations, S]
        all_noisy_losses = torch.stack(all_noisy_losses, dim=0)        # [num_realizations, S, S]
        
        # Compute statistics across realizations
        connectivity_mean = all_raw_connectivity.mean(dim=0)  # [S, S]
        connectivity_std = all_raw_connectivity.std(dim=0)    # [S, S]
        
        # Use mean connectivity for normalization
        normalized_connectivity = self.normalize_connectivity_to_01(connectivity_mean)
        
        # Package results
        results = NoiseAnalysisResults(
            raw_connectivity=connectivity_mean,
            normalized_connectivity=normalized_connectivity,
            original_losses=all_original_losses.mean(dim=0),  # [S]
            noisy_losses=all_noisy_losses.mean(dim=0),        # [S, S]
            connectivity_mean=connectivity_mean,
            connectivity_std=connectivity_std,
            num_noise_realizations=self.num_noise_realizations,
            noise_std=self.noise_std,
            normalization_method=self.normalization_method
        )
        
        self.logger.info("✅ Robust noise replacement analysis completed")
        
        return results
    
    def get_top_information_flows(self, 
                                connectivity_matrix: torch.Tensor, 
                                top_k: int = 10) -> List[Tuple[int, int, float]]:
        """
        Get top-k information flows from connectivity matrix.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            top_k: Number of top connections to return
            
        Returns:
            List of (source_site, target_site, flow_strength) tuples
        """
        
        S = connectivity_matrix.shape[0]
        flows = []
        
        for i in range(S):
            for j in range(S):
                if i != j:  # Exclude self-connections
                    flow_strength = connectivity_matrix[i, j].item()
                    flows.append((i, j, flow_strength))
        
        # Sort by flow strength (descending)
        flows.sort(key=lambda x: x[2], reverse=True)
        
        return flows[:top_k]
    
    def analyze_site_information_roles(self, connectivity_matrix: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """
        Analyze each site's role in information flow.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            
        Returns:
            Dictionary with per-site information flow characteristics
        """
        
        S = connectivity_matrix.shape[0]
        site_roles = {}
        
        for site in range(S):
            # Information provided to others (outgoing)
            info_provided = connectivity_matrix[site, :].sum().item() - connectivity_matrix[site, site].item()
            
            # Information received from others (incoming)  
            info_received = connectivity_matrix[:, site].sum().item() - connectivity_matrix[site, site].item()
            
            # Net information flow (positive = more provider, negative = more receiver)
            net_flow = info_provided - info_received
            
            site_roles[site] = {
                'information_provided': info_provided,
                'information_received': info_received,
                'net_information_flow': net_flow,
                'total_information_involvement': info_provided + info_received
            }
        
        return site_roles


def demo_noise_replacement_analysis():
    """Demo function for noise replacement analysis."""
    
    print("🔀 Noise Replacement Information Flow Analysis Demo")
    print("=" * 60)
    
    print("This demo requires a loaded model and data.")
    print("Use ConnectivityAnalyzer.analyze_connectivity() for full analysis.")


if __name__ == "__main__":
    demo_noise_replacement_analysis()
