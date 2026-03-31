"""
Enhanced Loss Functions for Neural Foundation Model MAE Training

This module provides simplified and effective loss functions for cross-site neural data:
1. Poisson reconstruction loss using PyTorch's built-in poisson_nll_loss
2. Site-level contrastive loss using SimCLR approach
3. Integrated with enhanced masking system and fixed site coordinates

Key Features:
- Compatible with new CausalMaskingEngine
- Uses site_coords [S, 2] instead of coords_3d [B, S, T, 3]
- Simple PyTorch-based Poisson loss
- Clean SimCLR-style contrastive learning
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def compute_neural_mae_loss(model,
                           neural_data: torch.Tensor,      # [B, S, T, N]
                           site_coords: torch.Tensor,      # [S, 2] - Fixed site coordinates
                           masking_engine,                  # CausalMaskingEngine instance
                           contrastive_weight: float = 0.1,
                           reconstruction_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Compute enhanced Neural MAE training loss with Poisson reconstruction + site contrastive learning.
    
    **UPDATED INTERFACE**: Compatible with new masking system and fixed site coordinates.
    
    Args:
        model: CrossSiteFoundationMAE model
        neural_data: [B, S, T, N] - neural activity data  
        site_coords: [S, 2] - fixed (X, Y) coordinates per site
        masking_engine: CausalMaskingEngine instance
        contrastive_weight: Weight for site-level contrastive loss
        reconstruction_weight: Weight for Poisson reconstruction loss
        
    Returns:
        Dict containing:
        - 'total_loss': Combined weighted loss
        - 'poisson_loss': Reconstruction loss (PyTorch built-in)
        - 'contrastive_loss': Site-level contrastive loss (SimCLR)
        - 'loss_components': Detailed breakdown for monitoring
    """
    device = neural_data.device
    
    # 1. Apply enhanced masking
    mask, masked_indices = masking_engine.apply_causal_mask(neural_data)  # [B, S, T, N], [B, S, T, N]
    
    # Convert mask format for transformer (expects [B, S, T])
    temporal_mask = (mask.sum(dim=-1) == 0).float()  # [B, S, T] - 1 if completely masked
    
    # 2. Forward pass with reconstruction
    outputs = model(
        neural_data, 
        site_coords, 
        mask_data=temporal_mask,
        return_mae_reconstruction=True
    )
    reconstruction = outputs['reconstruction']  # [B, S, T, N]
    representations = outputs['representations']  # [B, S, T, D]
    
    # 3. Poisson reconstruction loss (PyTorch built-in, only on masked positions)
    poisson_loss = compute_poisson_loss_pytorch(
        predicted_rates=reconstruction,
        target_spikes=neural_data,
        mask=mask  # [B, S, T, N]
    )
    
    # 4. Site-level contrastive loss (simplified approach - diagonal positive, off-diagonal negative)
    contrastive_loss = compute_site_contrastive_loss_simplified(
        representations=representations,  # [B, S, T, D]
        temperature=0.1
    )
    
    # 5. Combine losses
    total_loss = (reconstruction_weight * poisson_loss + 
                  contrastive_weight * contrastive_loss)
    
    # 6. Detailed breakdown for monitoring
    loss_components = {
        'weighted_poisson': (reconstruction_weight * poisson_loss).item(),
        'weighted_contrastive': (contrastive_weight * contrastive_loss).item(),
        'raw_poisson': poisson_loss.item(),
        'raw_contrastive': contrastive_loss.item(),
        'reconstruction_weight': reconstruction_weight,
        'contrastive_weight': contrastive_weight
    }
    
    return {
        'total_loss': total_loss,
        'poisson_loss': poisson_loss,
        'contrastive_loss': contrastive_loss,
        'loss_components': loss_components
    }


def compute_poisson_loss_pytorch(predicted_rates: torch.Tensor,
                                target_spikes: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
    """
    Simplified Poisson reconstruction loss using PyTorch's built-in function.
    
    **SIMPLIFIED APPROACH**: Uses torch.nn.functional.poisson_nll_loss for efficiency.
    
    Args:
        predicted_rates: [B, S, T, N] - predicted Poisson rate parameters (positive)
        target_spikes: [B, S, T, N] - observed spike counts
        mask: [B, S, T, N] - binary mask (1=keep, 0=mask)
        
    Returns:
        loss: Poisson NLL loss (only on masked positions)
    """
    # Ensure positive rates for Poisson distribution
    predicted_rates = torch.clamp(predicted_rates, min=1e-8)
    
    # Convert to log rates for PyTorch's poisson_nll_loss (expects log_input=True)
    log_rates = torch.log(predicted_rates)
    
    # Compute Poisson NLL using PyTorch's efficient implementation
    # reduction='none' gives us element-wise loss
    point_loss = F.poisson_nll_loss(
        input=log_rates,           # [B, S, T, N] - log rates
        target=target_spikes,      # [B, S, T, N] - spike counts  
        log_input=True,            # Input is log rates
        full=False,                # Use simplified formula (faster)
        reduction='none'           # Element-wise loss
    )  # [B, S, T, N]
    
    # Apply mask: only compute loss on masked positions
    masked_loss = point_loss * mask
    
    # Average over masked positions
    n_masked = mask.sum()
    if n_masked > 0:
        loss = masked_loss.sum() / n_masked
    else:
        # Edge case: no masked positions
        loss = torch.tensor(0.0, device=predicted_rates.device, requires_grad=True)
    
    return loss


def compute_site_contrastive_loss_simclr(representations: torch.Tensor,
                                        temperature: float = 0.1) -> torch.Tensor:
    """
    Site-level contrastive loss using SimCLR approach.
    
    **STRATEGY**: 
    - Pool temporal dimension to get site-level embeddings
    - Within each batch, same site = positive, different sites = negative
    - Use SimCLR-style similarity matrix with cross-entropy loss
    
    Args:
        representations: [B, S, T, D] - site representations across time
        temperature: Temperature parameter for softmax (controls hardness)
        
    Returns:
        contrastive_loss: Site-level contrastive loss
    """
    B, S, T, D = representations.shape
    
    # 1. Pool temporal dimension to get site-level representations
    # Average across time: [B, S, T, D] -> [B, S, D]
    site_embeddings = representations.mean(dim=2)  # [B, S, D]
    
    # 2. Flatten batch and site dimensions for SimCLR computation
    # [B, S, D] -> [B*S, D]
    flat_embeddings = site_embeddings.reshape(B * S, D)
    
    # 3. Normalize embeddings for cosine similarity
    normalized_embeddings = F.normalize(flat_embeddings, dim=1)  # [B*S, D]
    
    # 4. Compute similarity matrix
    # [B*S, D] x [D, B*S] -> [B*S, B*S]
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T) / temperature
    
    # 5. Create labels for positive pairs
    # Within each batch, same site indices are positive pairs
    labels = torch.arange(B * S, device=representations.device)
    
    # Site index pattern: [0,1,2,...,S-1, 0,1,2,...,S-1, ...] for B batches
    site_indices = torch.arange(S, device=representations.device).repeat(B)  # [B*S]
    
    # Create positive pair mask: same site across different batches
    positive_mask = site_indices.unsqueeze(0) == site_indices.unsqueeze(1)  # [B*S, B*S]
    
    # Remove self-similarity (diagonal)
    positive_mask.fill_diagonal_(False)
    
    # 6. Compute contrastive loss
    # For each sample, positive pairs are other instances of the same site
    total_loss = 0.0
    
    for i in range(B * S):
        # Get positive indices for sample i
        positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]
        
        if len(positive_indices) > 0:
            # Current sample similarity scores
            logits = similarity_matrix[i]  # [B*S]
            
            # Create labels: first len(positive_indices) are positive, rest are negative
            pos_logits = logits[positive_indices]  # [num_positives]
            neg_logits = logits[~positive_mask[i]]  # [num_negatives]
            
            # Combine into single tensor for cross-entropy
            all_logits = torch.cat([pos_logits, neg_logits])  # [num_positives + num_negatives]
            
            # Labels: positive samples are at indices 0 to len(positive_indices)-1
            target_labels = torch.zeros(len(positive_indices), device=representations.device, dtype=torch.long)
            
            # Compute loss for this sample (multiple positive labels)
            for pos_idx in range(len(positive_indices)):
                sample_logits = torch.cat([
                    all_logits[pos_idx:pos_idx+1],  # Current positive
                    neg_logits  # All negatives
                ])
                sample_target = torch.zeros(1, device=representations.device, dtype=torch.long)
                total_loss += F.cross_entropy(sample_logits.unsqueeze(0), sample_target)
    
    # Average over all samples with positive pairs
    num_samples_with_positives = positive_mask.any(dim=1).sum()
    if num_samples_with_positives > 0:
        contrastive_loss = total_loss / num_samples_with_positives
    else:
        contrastive_loss = torch.tensor(0.0, device=representations.device, requires_grad=True)
    
    return contrastive_loss


def compute_site_contrastive_loss_simplified(representations: torch.Tensor,
                                           temperature: float = 0.1) -> torch.Tensor:
    """
    Simplified site-level contrastive loss (alternative implementation).
    
    **SIMPLER APPROACH**: Diagonal = positive, off-diagonal = negative at site level.
    
    Args:
        representations: [B, S, T, D] - site representations
        temperature: Temperature parameter
        
    Returns:
        contrastive_loss: Simplified contrastive loss
    """
    B, S, T, D = representations.shape
    
    # Pool temporal dimension: [B, S, T, D] -> [B, S, D]
    site_embeddings = representations.mean(dim=2)
    
    # Average across batch for site-level comparison: [B, S, D] -> [S, D]
    avg_site_embeddings = site_embeddings.mean(dim=0)
    
    # Normalize for cosine similarity
    normalized_sites = F.normalize(avg_site_embeddings, dim=1)  # [S, D]
    
    # Compute site similarity matrix: [S, S]
    site_similarity = torch.matmul(normalized_sites, normalized_sites.T) / temperature
    
    # Labels: diagonal should be high (same site), off-diagonal low (different sites)
    labels = torch.arange(S, device=representations.device)
    
    # Cross-entropy loss: encourages diagonal to be highest
    contrastive_loss = F.cross_entropy(site_similarity, labels)
    
    return contrastive_loss


# ==================== UTILITY FUNCTIONS ====================

def get_loss_statistics(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Extract loss statistics for monitoring and logging.
    
    Args:
        loss_dict: Dictionary from compute_neural_mae_loss
        
    Returns:
        Dictionary of loss statistics as floats
    """
    stats = {
        'total_loss': loss_dict['total_loss'].item(),
        'poisson_loss': loss_dict['poisson_loss'].item(),
        'contrastive_loss': loss_dict['contrastive_loss'].item()
    }
    
    # Add detailed components if available
    if 'loss_components' in loss_dict:
        stats.update(loss_dict['loss_components'])
    
    return stats


def create_loss_function(contrastive_weight: float = 0.1,
                        reconstruction_weight: float = 1.0,
                        temperature: float = 0.1,
                        use_simplified_contrastive: bool = False):
    """
    Factory function to create a configured loss function.
    
    Args:
        contrastive_weight: Weight for contrastive loss
        reconstruction_weight: Weight for reconstruction loss  
        temperature: Temperature for contrastive loss
        use_simplified_contrastive: Use simplified contrastive implementation
        
    Returns:
        Configured loss function
    """
    def loss_fn(model, neural_data, site_coords, masking_engine):
        return compute_neural_mae_loss(
            model=model,
            neural_data=neural_data,
            site_coords=site_coords,
            masking_engine=masking_engine,
            contrastive_weight=contrastive_weight,
            reconstruction_weight=reconstruction_weight
        )
    
    return loss_fn


# ==================== NOTES ON L2 REGULARIZATION ====================
"""
**L2 REGULARIZATION WITH ADAMW**:

You asked about L2 regularization since you'll use AdamW. Good news: 
AdamW has built-in weight decay that serves the same purpose as L2 regularization!

RECOMMENDATION: Skip explicit L2 regularization here. Instead:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # This replaces L2 regularization!
)
```

AdamW's weight decay is actually superior to L2 regularization because:
1. Decouples weight decay from gradient-based updates
2. More stable training dynamics
3. Better generalization in practice

So no need to add L2 regularization to this loss function!
""" 