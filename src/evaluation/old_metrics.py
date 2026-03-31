import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import os
from typing import Tuple, Optional, Dict

class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for probability distribution targets.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # pred: [batch, num_classes]
        # target: [batch, num_classes] (one-hot or smoothed one-hot)
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()
        return loss

class NeuralMAELoss(nn.Module):
    """
    Combined loss for neural MAE:
    1. Poisson reconstruction loss using log firing rates
    2. Contrastive loss using encoder latent factors
    """
    def __init__(self, 
                 temperature: float = 0.1,
                 lambda_rec: float = 1.0,
                 lambda_con: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_rec = lambda_rec
        self.lambda_con = lambda_con
        
    def forward(self,
                log_rates: torch.Tensor,      # From decoder [batch, time, neurons]
                spike_counts: torch.Tensor,    # Target spikes [batch, time, neurons]
                mask: torch.Tensor,           # Masking pattern [batch, time, neurons]
                latent_1: torch.Tensor,       # Encoder factors view 1 [batch, time, dim]
                latent_2: torch.Tensor):      # Encoder factors view 2 [batch, time, dim]
        """
        Compute combined MAE reconstruction and contrastive loss
        """
        # 1. Reconstruction Loss (using log rates)
        rec_loss, rec_metrics = self._compute_reconstruction_loss(
            log_rates, spike_counts, mask)
            
        # 2. Contrastive Loss (using latent factors)
        con_loss, con_metrics = self._compute_contrastive_loss(
            latent_1, latent_2)
            
        # Combine losses
        total_loss = self.lambda_rec * rec_loss + self.lambda_con * con_loss
        
        # Combine metrics
        metrics = {
            'total_loss': total_loss.item(),
            'weighted_rec_loss': (self.lambda_rec * rec_loss).item(),
            'weighted_con_loss': (self.lambda_con * con_loss).item(),
            **rec_metrics,
            **con_metrics
        }
        
        return total_loss, metrics
    
    def _compute_reconstruction_loss(self, 
                                   log_rates: torch.Tensor,
                                   spike_counts: torch.Tensor,
                                   mask: torch.Tensor):
        """
        Poisson reconstruction loss using log firing rates
        
        Args:
            log_rates: Pre-clipped log firing rates [batch, time, neurons]
            spike_counts: Target spike counts [batch, time, neurons]
            mask: Binary mask [batch, time, neurons] (1 = keep, 0 = mask)
        """
        # Compute Poisson NLL loss
        point_loss = F.poisson_nll_loss(
            log_rates,
            spike_counts,
            log_input=True,
            full=True,
            reduction='none'
        )
        
        # Apply mask
        masked_loss = point_loss * mask
        rec_loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        # Compute metrics
        with torch.no_grad():
            rates = torch.exp(log_rates)
            metrics = {
                'rec_loss': rec_loss.item(),
                'mean_rate': rates[mask.bool()].mean().item(),
                'max_rate': rates[mask.bool()].max().item(),
                'mean_count': spike_counts[mask.bool()].mean().item(),
                'num_masked': mask.sum().item()
            }
            
        return rec_loss, metrics
        
    def _compute_contrastive_loss(self, latent_1: torch.Tensor, latent_2: torch.Tensor):
        """
        SimCLR-style contrastive loss using flattened temporal embeddings
        Args:
            latent_1: [batch, time, dim]
            latent_2: [batch, time, dim]
        """
        # Flatten temporal and feature dimensions
        B, T, D = latent_1.shape
        latent_1 = latent_1.reshape(B, -1)  # [batch, time*dim]
        latent_2 = latent_2.reshape(B, -1)  # [batch, time*dim]
        
        # Normalize embeddings
        latent_1 = F.normalize(latent_1, dim=-1)
        latent_2 = F.normalize(latent_2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(latent_1, latent_2.T) / self.temperature  # [batch, batch]
        
        # Positive pairs on diagonal
        labels = torch.arange(B, device=latent_1.device)
        con_loss = F.cross_entropy(similarity, labels)
        
        # Compute metrics
        with torch.no_grad():
            pos_sim = torch.diagonal(similarity).mean()
            neg_sim = (similarity - torch.eye(B, device=latent_1.device)).mean()
            metrics = {
                'con_loss': con_loss.item(),
                'positive_sim': pos_sim.item(),
                'negative_sim': neg_sim.item()
            }
            
        return con_loss, metrics

def test_neural_mae_loss():
    """
    Test the NeuralMAELoss implementation with various scenarios
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 32
    time_steps = 50
    n_neurons = 100
    latent_dim = 4
    
    # Initialize loss function
    criterion = NeuralMAELoss(
        temperature=0.1,
        threshold_rate=20.0,
        lambda_rec=1.0,
        lambda_con=0.5
    )
    
    # Create dummy data
    def create_test_data():
        # Log firing rates (decoder output)
        log_rates = torch.randn(batch_size, time_steps, n_neurons) * 2 - 1  # roughly between -3 and 1
        
        # Spike counts (target)
        rates = torch.exp(log_rates)
        spike_counts = torch.poisson(rates)
        
        # Masking pattern (75% masked)
        mask = torch.bernoulli(torch.ones(batch_size, time_steps, n_neurons) * 0.75)
        
        # Latent factors (encoder output)
        latent_1 = torch.randn(batch_size, time_steps, latent_dim)
        latent_2 = latent_1 + torch.randn_like(latent_1) * 0.1  # Similar but not identical
        
        return log_rates, spike_counts, mask, latent_1, latent_2
    
    # Test case 1: Basic functionality
    def test_basic_functionality():
        print("\nTest 1: Basic Functionality")
        log_rates, spike_counts, mask, latent_1, latent_2 = create_test_data()
        
        loss, metrics = criterion(log_rates, spike_counts, mask, latent_1, latent_2)
        
        print(f"Total Loss: {metrics['total_loss']:.4f}")
        print(f"Reconstruction Loss: {metrics['weighted_rec_loss']:.4f}")
        print(f"Contrastive Loss: {metrics['weighted_con_loss']:.4f}")
        print(f"Mean Firing Rate: {metrics['mean_rate']:.4f} Hz")
        print(f"Positive Similarity: {metrics['positive_sim']:.4f}")
        
        return loss, metrics
    
    # Test case 2: Edge cases
    def test_edge_cases():
        print("\nTest 2: Edge Cases")
        
        # Case 1: Very low firing rates
        log_rates = torch.ones(batch_size, time_steps, n_neurons) * -10
        spike_counts = torch.zeros_like(log_rates)
        mask = torch.ones_like(log_rates)
        latent_1 = torch.randn(batch_size, time_steps, latent_dim)
        latent_2 = latent_1.clone()
        
        loss1, metrics1 = criterion(log_rates, spike_counts, mask, latent_1, latent_2)
        print("Very low rates - Loss:", metrics1['total_loss'])
        
        # Case 2: Very high firing rates
        log_rates = torch.ones(batch_size, time_steps, n_neurons) * 10
        spike_counts = torch.ones_like(log_rates) * 100
        
        loss2, metrics2 = criterion(log_rates, spike_counts, mask, latent_1, latent_2)
        print("Very high rates - Loss:", metrics2['total_loss'])
        
        return (loss1, metrics1), (loss2, metrics2)
    
    # Test case 3: Visualization
    def visualize_results():
        print("\nTest 3: Visualization")
        log_rates, spike_counts, mask, latent_1, latent_2 = create_test_data()
        
        # Plot example data
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot log rates
        im1 = axes[0,0].imshow(log_rates[0].detach().numpy(), aspect='auto')
        axes[0,0].set_title('Log Firing Rates')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Plot spike counts
        im2 = axes[0,1].imshow(spike_counts[0].detach().numpy(), aspect='auto')
        axes[0,1].set_title('Spike Counts')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Plot mask
        im3 = axes[1,0].imshow(mask[0].detach().numpy(), aspect='auto')
        axes[1,0].set_title('Mask Pattern')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Plot similarity matrix
        B = latent_1.size(0)
        flat_latent_1 = F.normalize(latent_1.reshape(B, -1), dim=-1)
        flat_latent_2 = F.normalize(latent_2.reshape(B, -1), dim=-1)
        similarity = torch.matmul(flat_latent_1, flat_latent_2.T)
        im4 = axes[1,1].imshow(similarity.detach().numpy())
        axes[1,1].set_title('Batch Similarity Matrix')
        plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.show()
    
    # Run all tests
    basic_results = test_basic_functionality()
    edge_results = test_edge_cases()
    visualize_results()
    
    return {
        'basic_results': basic_results,
        'edge_results': edge_results
    }

# # # Run the tests
# if __name__ == "__main__":
#     results = test_neural_mae_loss()