"""
Data utilities for neural foundation model training and testing.
"""

import torch
from typing import Tuple, Dict


def create_dummy_batch_data_4d(batch_size: int = 4,
                              n_sites: int = 17,
                              seq_len: int = 50,
                              neural_dim: int = 75,
                              device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy 4D batch data for testing the cross-site MAE models.
    
    Returns:
        neural_data: [B, S, T, N] - neural activity data
        coords_3d: [B, S, T, 3] - 3D electrode coordinates
    """
    # Neural data with realistic spike count statistics
    # Use Poisson-like data (mostly zeros with occasional spikes)
    neural_data = torch.poisson(torch.ones(batch_size, n_sites, seq_len, neural_dim) * 0.1)
    neural_data = neural_data.to(device)
    
    # 3D coordinates (simulate electrode positions)
    coords_3d = torch.zeros(batch_size, n_sites, seq_len, 3, device=device)
    
    for b in range(batch_size):
        for s in range(n_sites):
            # Each site has different spatial location (fixed across time)
            base_coords = torch.tensor([s * 0.5, (s % 3) * 0.3, (s % 2) * 0.4], device=device)
            site_coords = base_coords.unsqueeze(0).expand(seq_len, -1)  # [T, 3]
            # Add small random variations over time
            site_coords = site_coords + torch.randn_like(site_coords) * 0.05
            coords_3d[b, s, :, :] = site_coords
    
    return neural_data, coords_3d


def create_single_site_data(batch_size: int = 4,
                           seq_len: int = 50,
                           neural_dim: int = 75,
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy single site data for testing downstream models.
    
    Returns:
        neural_data: [B, T, N] - single site neural activity data
        coords_3d: [B, T, 3] - single site 3D electrode coordinates
    """
    # Neural data with realistic spike count statistics
    neural_data = torch.poisson(torch.ones(batch_size, seq_len, neural_dim) * 0.1)
    neural_data = neural_data.to(device)
    
    # 3D coordinates for single site
    coords_3d = torch.zeros(batch_size, seq_len, 3, device=device)
    
    for b in range(batch_size):
        # Single site location (fixed across time with small variations)
        base_coords = torch.tensor([1.0, 0.5, 0.2], device=device)
        site_coords = base_coords.unsqueeze(0).expand(seq_len, -1)  # [T, 3]
        # Add small random variations over time
        site_coords = site_coords + torch.randn_like(site_coords) * 0.05
        coords_3d[b, :, :] = site_coords
    
    return neural_data, coords_3d


def create_dummy_batch_data(batch_size: int = 4, 
                          seq_len: int = 50,
                          neural_dim: int = 75,
                          n_sites: int = 17,
                          device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create dummy batch data in legacy dict format for testing.
    
    **DEPRECATED**: Use create_dummy_batch_data_4d() for new 4D tensor format.
    
    Returns:
        batch_data: Dict mapping site_id -> [batch, time, neural_dim]
        coords_3d: Dict mapping site_id -> [batch, time, 3]
    """
    batch_data = {}
    coords_3d = {}
    
    for i in range(n_sites):
        site_id = f'site_{i+1}'
        
        # Neural data
        batch_data[site_id] = torch.randn(batch_size, seq_len, neural_dim, device=device)
        
        # 3D coordinates (simulate electrode positions)
        # Each site has different spatial location
        base_coords = torch.tensor([i * 0.5, (i % 3) * 0.3, (i % 2) * 0.4], device=device)
        site_coords = base_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        # Add small random variations
        site_coords = site_coords + torch.randn_like(site_coords) * 0.1
        coords_3d[site_id] = site_coords
    
    return batch_data, coords_3d


def convert_dict_to_4d_tensor(batch_data: Dict[str, torch.Tensor],
                             coords_3d: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert legacy dict format to 4D tensor format.
    
    Args:
        batch_data: Dict mapping site_id -> [B, T, N]
        coords_3d: Dict mapping site_id -> [B, T, 3]
        
    Returns:
        neural_data: [B, S, T, N]
        coords_4d: [B, S, T, 3]
    """
    # Get consistent site ordering
    site_names = sorted(batch_data.keys())
    
    # Stack data from all sites
    neural_data = torch.stack([batch_data[site_id] for site_id in site_names], dim=1)  # [B, S, T, N]
    coords_4d = torch.stack([coords_3d[site_id] for site_id in site_names], dim=1)     # [B, S, T, 3]
    
    return neural_data, coords_4d


def convert_4d_to_dict_tensor(neural_data: torch.Tensor,
                             coords_3d: torch.Tensor,
                             n_sites: int = 17) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convert 4D tensor format back to dict format for legacy compatibility.
    
    Args:
        neural_data: [B, S, T, N]
        coords_3d: [B, S, T, 3]
        n_sites: Number of sites
        
    Returns:
        batch_data: Dict mapping site_id -> [B, T, N]
        coords_dict: Dict mapping site_id -> [B, T, 3]
    """
    batch_data = {}
    coords_dict = {}
    
    for s in range(n_sites):
        site_id = f'site_{s+1}'
        batch_data[site_id] = neural_data[:, s, :, :]  # [B, T, N]
        coords_dict[site_id] = coords_3d[:, s, :, :]   # [B, T, 3]
    
    return batch_data, coords_dict 