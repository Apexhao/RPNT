"""
Main Connectivity Analysis Module for Neural Foundation Model

This module provides the ConnectivityAnalyzer class that coordinates:
1. Attention-based connectivity extraction
2. Noise replacement information flow analysis  
3. Temporal connectivity dynamics at specific timepoints
4. Correlation-based baseline for validation

Key Features:
- Load pretrained CrossSiteFoundationMAE model
- Extract connectivity from test data
- Support temporal analysis (0, 15, 30, 45 timepoints)
- Generate 16x16 connectivity matrices for motor cortex sites
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Import our modules
from ..models import CrossSiteFoundationMAE, CrossSiteModelFactory
from ..data import CrossSiteMonkeyDataset
from ..training.downstream_trainers import load_pretrained_foundation_model


@dataclass
class ConnectivityResults:
    """Data class to store connectivity analysis results."""
    
    # Attention-based connectivity
    attention_connectivity: torch.Tensor      # [S, S] - averaged over time/layers
    attention_temporal: torch.Tensor          # [T_samples, S, S] - temporal dynamics
    
    # Noise replacement connectivity  
    noise_connectivity: torch.Tensor          # [S, S] - information dependency
    noise_connectivity_normalized: torch.Tensor  # [S, S] - 0-1 normalized
    
    # Baseline correlation connectivity
    correlation_connectivity: torch.Tensor    # [S, S] - spike correlation
    
    # Metadata
    site_coordinates: torch.Tensor            # [S, 2] - (X, Y) coordinates
    site_ids: List[str]                       # Site identifiers
    temporal_timepoints: List[int]            # Sampled timepoints
    
    # Statistics
    reconstruction_losses: Dict[str, float]   # Original vs noisy losses


class ConnectivityAnalyzer:
    """
    Main analyzer for extracting functional connectivity from pretrained model.
    
    **ANALYSIS PIPELINE**:
    1. Load pretrained CrossSiteFoundationMAE model
    2. Extract attention weights from spatial encoder  
    3. Perform noise replacement analysis
    4. Compute correlation baseline
    5. Generate temporal connectivity dynamics
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'auto',
                 temporal_timepoints: List[int] = [0, 15, 30, 45]):
        """
        Initialize ConnectivityAnalyzer.
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
            temporal_timepoints: Timepoints to sample for temporal analysis
        """
        
        self.checkpoint_path = checkpoint_path
        self.temporal_timepoints = temporal_timepoints
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model = None
        self.dataset = None
        
        # Analysis results
        self.results = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ConnectivityAnalyzer initialized")
        self.logger.info(f"Checkpoint: {checkpoint_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Temporal timepoints: {temporal_timepoints}")
    
    def load_pretrained_model(self) -> CrossSiteFoundationMAE:
        """
        Load the pretrained CrossSiteFoundationMAE model from checkpoint.
        
        Returns:
            Loaded model in eval mode
        """
        
        self.logger.info("Loading pretrained model...")
        
        try:
            # Load using existing function from downstream_trainers
            self.model = load_pretrained_foundation_model(self.checkpoint_path, self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info("✅ Model loaded successfully")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise e
    
    def setup_dataset(self, 
                     data_root: str = "/data/Fang-analysis/causal-nfm/Data/Monkey_data_meta",
                     exclude_ids: List[str] = ['13122.0'],
                     **dataset_kwargs) -> CrossSiteMonkeyDataset:
        """
        Setup the CrossSiteMonkeyDataset for analysis.
        
        Args:
            data_root: Root directory for monkey data
            exclude_ids: Site IDs to exclude
            **dataset_kwargs: Additional dataset parameters
            
        Returns:
            Configured dataset
        """
        
        self.logger.info("Setting up dataset...")
        
        # Default dataset configuration for analysis
        default_config = {
            'split_ratios': (0.8, 0.1, 0.1),
            'target_neurons': 50,
            'sample_times': 1,  # No augmentation for analysis
            'target_trials_per_site': 1000,  # Smaller for analysis
            'min_val_test_trials': 100,
            'width': 0.02,
            'sequence_length': 50,
            'random_seed': 42
        }
        
        # Update with provided kwargs
        default_config.update(dataset_kwargs)
        
        try:
            self.dataset = CrossSiteMonkeyDataset(
                data_root=data_root,
                exclude_ids=exclude_ids,
                **default_config
            )
            
            self.logger.info(f"✅ Dataset loaded - {len(self.dataset.site_ids)} sites")
            self.logger.info(f"Sites: {self.dataset.site_ids}")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup dataset: {e}")
            raise e
    
    def get_test_data(self, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get test data for connectivity analysis.
        
        Args:
            batch_size: Batch size for analysis
            
        Returns:
            Tuple of (neural_data, site_coordinates)
            - neural_data: [B, S, T, N] - test neural activity
            - site_coordinates: [S, 2] - (X, Y) coordinates
        """
        
        if self.dataset is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        # Get test data
        test_data = self.dataset.get_split_data('test')  # [Trials, Sites, Time, Neurons]
        
        # Sample batch_size trials for analysis
        n_trials = test_data.shape[0]
        if n_trials > batch_size:
            indices = torch.randperm(n_trials)[:batch_size]
            test_data = test_data[indices]
        
        # Convert to tensor and move to device
        neural_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        
        # Get site coordinates
        site_coordinates = self.dataset.get_site_coordinates().to(self.device)  # [S, 2]
        
        self.logger.info(f"Test data prepared: {neural_data.shape}")
        self.logger.info(f"Site coordinates: {site_coordinates.shape}")
        
        return neural_data, site_coordinates
    
    def extract_attention_connectivity(self, 
                                     neural_data: torch.Tensor,
                                     site_coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention-based connectivity from spatial encoder.
        
        **MATHEMATICAL IMPLEMENTATION**:
        C_{i→j}(attention) = (1/T·L) Σ_{t=1}^T Σ_{l=1}^L Attention_{i→j}^{(l,t)}
        
        Args:
            neural_data: [B, S, T, N] - neural activity
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            Tuple of (averaged_connectivity, temporal_connectivity)
            - averaged_connectivity: [S, S] - averaged over time/layers
            - temporal_connectivity: [T_samples, S, S] - at specific timepoints
        """
        
        self.logger.info("Extracting attention-based connectivity...")
        
        with torch.no_grad():
            # Forward pass to get attention weights
            _ = self.model(neural_data, site_coordinates)
            
            # Extract attention weights: [B, T, num_layers, S, S]
            attention_weights = self.model.spatial_encoder.get_attention_weights()
            
            self.logger.info(f"Attention weights shape: {attention_weights.shape}")
            
            # Average across batch and compute connectivity
            attention_weights = attention_weights.mean(dim=0)  # [T, num_layers, S, S]
            
            # 1. Averaged connectivity (Equation 1)
            averaged_connectivity = attention_weights.mean(dim=[0, 1])  # [S, S]
            
            # 2. Temporal connectivity at specific timepoints
            temporal_connectivity = []
            for t in self.temporal_timepoints:
                if t < attention_weights.shape[0]:
                    # Average over layers for this timepoint
                    t_connectivity = attention_weights[t].mean(dim=0)  # [S, S]
                    temporal_connectivity.append(t_connectivity)
                else:
                    self.logger.warning(f"Timepoint {t} exceeds sequence length {attention_weights.shape[0]}")
            
            temporal_connectivity = torch.stack(temporal_connectivity, dim=0)  # [T_samples, S, S]
            
            self.logger.info(f"✅ Attention connectivity extracted")
            self.logger.info(f"Averaged: {averaged_connectivity.shape}, Temporal: {temporal_connectivity.shape}")
            
            return averaged_connectivity, temporal_connectivity
    
    def compute_correlation_baseline(self, neural_data: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation-based connectivity as baseline.
        
        Args:
            neural_data: [B, S, T, N] - neural activity
            
        Returns:
            correlation_matrix: [S, S] - correlation between sites
        """
        
        self.logger.info("Computing correlation baseline...")
        
        B, S, T, N = neural_data.shape
        
        # Average neural activity across neurons for each site: [B, S, T]
        site_activity = neural_data.mean(dim=-1)  # [B, S, T]
        
        # Flatten across batch and time: [B*T, S]
        flattened_activity = site_activity.transpose(1, 2).reshape(-1, S)  # [B*T, S]
        
        # Compute correlation matrix
        correlation_matrix = torch.corrcoef(flattened_activity.T)  # [S, S]
        
        # Handle NaN values (replace with 0)
        correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
        
        self.logger.info(f"✅ Correlation baseline computed: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def analyze_connectivity(self,
                           data_root: str = "/data/Fang-analysis/causal-nfm/Data/Monkey_data_meta",
                           batch_size: int = 16,
                           **dataset_kwargs) -> ConnectivityResults:
        """
        Run complete connectivity analysis pipeline.
        
        Args:
            data_root: Root directory for data
            batch_size: Batch size for analysis
            **dataset_kwargs: Additional dataset parameters
            
        Returns:
            ConnectivityResults with all analysis results
        """
        
        self.logger.info("🚀 Starting connectivity analysis pipeline...")
        
        # 1. Load model and dataset
        if self.model is None:
            self.load_pretrained_model()
        
        if self.dataset is None:
            self.setup_dataset(data_root=data_root, **dataset_kwargs)
        
        # 2. Get test data
        neural_data, site_coordinates = self.get_test_data(batch_size=batch_size)
        
        # 3. Extract attention connectivity
        attention_conn, attention_temporal = self.extract_attention_connectivity(
            neural_data, site_coordinates
        )
        
        # 4. Compute correlation baseline
        correlation_conn = self.compute_correlation_baseline(neural_data)
        
        # 5. Noise replacement analysis
        noise_conn, noise_conn_norm = self.extract_noise_replacement_connectivity(neural_data, site_coordinates)
        
        # 6. Package results
        self.results = ConnectivityResults(
            attention_connectivity=attention_conn.cpu(),
            attention_temporal=attention_temporal.cpu(),
            noise_connectivity=noise_conn,
            noise_connectivity_normalized=noise_conn_norm,
            correlation_connectivity=correlation_conn.cpu(),
            site_coordinates=site_coordinates.cpu(),
            site_ids=self.dataset.site_ids,
            temporal_timepoints=self.temporal_timepoints,
            reconstruction_losses={}  # Will fill with noise analysis
        )
        
        self.logger.info("✅ Connectivity analysis completed!")
        
        return self.results
    
    def extract_noise_replacement_connectivity(self, 
                                             neural_data: torch.Tensor, 
                                             site_coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract information flow connectivity using noise replacement analysis.
        
        **MATHEMATICAL IMPLEMENTATION**:
        C_{i→j}(N) = (L_j(X_{i←N}) - L_j(X)) / L_j(X)
        
        Args:
            neural_data: [B, S, T, N] - neural activity
            site_coordinates: [S, 2] - site coordinates
            
        Returns:
            Tuple of (raw_connectivity, normalized_connectivity)
            - raw_connectivity: [S, S] - raw information dependency scores
            - normalized_connectivity: [S, S] - 0-1 normalized scores
        """
        
        self.logger.info("Extracting noise replacement connectivity...")
        
        # Import here to avoid circular imports
        from .noise_replacement import NoiseReplacementAnalyzer
        
        # Initialize noise analyzer
        noise_analyzer = NoiseReplacementAnalyzer(
            model=self.model,
            noise_std=1.0,
            num_noise_realizations=3,  # Keep manageable for demo
            normalization_method='standard'  # Use standard normalization for better visualization
        )
        
        # Run robust connectivity analysis
        noise_results = noise_analyzer.compute_robust_connectivity(neural_data, site_coordinates)
        
        self.logger.info(f"✅ Noise replacement connectivity extracted")
        self.logger.info(f"Raw connectivity range: [{noise_results.raw_connectivity.min():.4f}, {noise_results.raw_connectivity.max():.4f}]")
        self.logger.info(f"Normalized connectivity range: [{noise_results.normalized_connectivity.min():.4f}, {noise_results.normalized_connectivity.max():.4f}]")
        
        return noise_results.raw_connectivity, noise_results.normalized_connectivity
    
    def save_results(self, output_path: str):
        """Save connectivity analysis results."""
        if self.results is None:
            raise ValueError("No results to save. Run analyze_connectivity() first.")
        
        torch.save(self.results, output_path)
        self.logger.info(f"Results saved to {output_path}")
    
    def load_results(self, results_path: str) -> ConnectivityResults:
        """Load previously computed results."""
        self.results = torch.load(results_path)
        self.logger.info(f"Results loaded from {results_path}")
        return self.results


# Example usage function
def demo_connectivity_analysis():
    """Demo function for connectivity analysis."""
    
    print("🧠 Neural Foundation Model - Connectivity Analysis Demo")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = "./logs_neuropixel/RoPE3D_UniformMasking_Kernel_11_11_d_model_384_TLL_4_SL_2_HL_4_h12/checkpoints/best.pth"
    
    try:
        # Initialize analyzer
        analyzer = ConnectivityAnalyzer(
            checkpoint_path=checkpoint_path,
            temporal_timepoints=[0, 15, 30, 45]
        )
        
        # Run analysis
        results = analyzer.analyze_connectivity(batch_size=8)
        
        # Print summary
        print(f"\n📊 Analysis Results:")
        print(f"   Sites analyzed: {len(results.site_ids)}")
        print(f"   Attention connectivity: {results.attention_connectivity.shape}")
        print(f"   Temporal dynamics: {results.attention_temporal.shape}")
        print(f"   Correlation baseline: {results.correlation_connectivity.shape}")
        
        print(f"\n✅ Demo completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    demo_connectivity_analysis()
