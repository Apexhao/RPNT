"""
Evaluation utilities for the neural foundation model.

This module contains:
- Enhanced loss computation functions for MAE training
- Evaluation metrics and validation utilities
- PyTorch-based Poisson loss and site contrastive learning
"""

from .loss_functions import (
    compute_neural_mae_loss,
    compute_poisson_loss_pytorch,
    compute_site_contrastive_loss_simclr,
    compute_site_contrastive_loss_simplified,
    get_loss_statistics,
    create_loss_function
)

__all__ = [
    'compute_neural_mae_loss',
    'compute_poisson_loss_pytorch',
    'compute_site_contrastive_loss_simclr',
    'compute_site_contrastive_loss_simplified',
    'get_loss_statistics',
    'create_loss_function'
] 