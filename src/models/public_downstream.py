"""
Public Dataset Downstream Models for Neural Foundation Model
----------------------------------------------------------

This module provides downstream task models for the public dataset evaluation.

Key Features:
- Uses pretrained PublicNeuralFoundationMAE as backbone
- Velocity prediction (regression) and direction classification
- Compatible with [B,S=1,T,N] input format
- Professional fine-tuning modes: frozen, partial, full
- Cross-session and cross-subject evaluation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List
import logging

from .public_transformer import PublicNeuralFoundationMAE


class PublicVelocityPredictor(nn.Module):
    """
    Velocity Prediction Model using pretrained PublicNeuralFoundationMAE.
    
    **DESIGN GOALS**:
    - Regression task: predict cursor velocity from neural activity
    - Use pretrained temporal encoder with different fine-tuning modes
    - Compatible with [B,S=1,T,N] input format
    - Output: [B,T,2] velocity predictions
    
    **TRAINING MODES**:
    - frozen_encoder: Freeze temporal encoder, train prediction head only
    - partial_finetune: Train last few layers + prediction head
    - full_finetune: Train entire model end-to-end
    """
    
    def __init__(self,
                 pretrained_model: PublicNeuralFoundationMAE,
                 training_mode: str = 'frozen_encoder',
                 prediction_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.training_mode = training_mode
        self.d_model = pretrained_model.d_model
        
        # Load pretrained temporal encoder
        self.temporal_encoder = pretrained_model.temporal_encoder
        
        # Apply training mode
        self._apply_training_mode()
        
        # Velocity prediction head
        prediction_modules = []
        
        # First layer: aggregate temporal information
        prediction_modules.extend([
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 4),
            nn.GELU(),
            # nn.Dropout(dropout)
        ])
        
        # Additional prediction layers
        current_dim = self.d_model // 4
        for _ in range(prediction_layers - 1):
            prediction_modules.extend([
                nn.Linear(current_dim, current_dim // 4),
                nn.GELU(),
                # nn.Dropout(dropout)
            ])
            current_dim = current_dim // 4
        
        # Final output layer: [hidden] -> [2] for [vx, vy]
        prediction_modules.append(nn.Linear(current_dim, 2))
        
        self.prediction_head = nn.Sequential(*prediction_modules)
        
        # Initialize prediction head weights
        self._init_prediction_weights()
    
    def _apply_training_mode(self):
        """Apply training mode to temporal encoder."""
        
        if self.training_mode == 'frozen_encoder':
            # Freeze entire temporal encoder
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
            print(f"🔒 Temporal encoder frozen")
            
        elif self.training_mode == 'partial_finetune':
            # Freeze early layers, allow last 2 layers to train
            num_layers = len(self.temporal_encoder.layers)
            layers_to_freeze = max(0, num_layers - 2)
            
            # Freeze early layers
            for i in range(layers_to_freeze):
                for param in self.temporal_encoder.layers[i].parameters():
                    param.requires_grad = False
            
            # Allow later layers to train
            for i in range(layers_to_freeze, num_layers):
                for param in self.temporal_encoder.layers[i].parameters():
                    param.requires_grad = True
            
            print(f"🔓 Partial fine-tuning: last {num_layers - layers_to_freeze} layers trainable")
            
        elif self.training_mode == 'full_finetune':
            # Allow all parameters to train
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
            print(f"🔓 Full fine-tuning enabled")
            
        else:
            raise ValueError(f"Invalid training_mode: {self.training_mode}")
    
    def _init_prediction_weights(self):
        """Initialize prediction head weights."""
        for module in self.prediction_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                neural_data: torch.Tensor,      # [B, S=1, T, N]
                session_coords: torch.Tensor,   # [B, S=1, 3]
                mask_data: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, S=1, T]
        """
        Forward pass for velocity prediction.
        
        Args:
            neural_data: [B, S=1, T, N] - neural activity
            session_coords: [B, S=1, 3] - session coordinates
            mask_data: [B, S=1, T] - optional mask
            
        Returns:
            velocity_pred: [B, T, 2] - predicted velocities [vx, vy]
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Get temporal representations from pretrained encoder
        with torch.set_grad_enabled(self.training_mode != 'frozen_encoder'):
            causal_mask = torch.tril(torch.ones(T, T, device=device))
            temporal_features = self.temporal_encoder(
                neural_data=neural_data,
                session_coords=session_coords,
                causal_mask=causal_mask,
                mask_data=mask_data
            )  # [B, S=1, T, D]
        
        # Remove site dimension: [B, S=1, T, D] -> [B, T, D]
        temporal_features = temporal_features.squeeze(1)
        
        # Predict velocities: [B, T, D] -> [B, T, 2]
        velocity_pred = self.prediction_head(temporal_features)
        
        return velocity_pred


# Factory function for creating downstream models
def create_public_downstream_model(pretrained_model: PublicNeuralFoundationMAE,
                                 task_type: str = 'regression',
                                 training_mode: str = 'frozen_encoder',
                                 **kwargs) -> nn.Module:
    """
    Factory function for creating public dataset downstream models.
    
    Args:
        pretrained_model: Pretrained PublicNeuralFoundationMAE
        task_type: 'regression'
        training_mode: 'frozen_encoder', 'partial_finetune', or 'full_finetune'
        **kwargs: Additional arguments for specific model types
        
    Returns:
        Downstream model
    """
    if task_type == 'regression':
        return PublicVelocityPredictor(pretrained_model, training_mode=training_mode, **kwargs)
    else:
        raise ValueError(f"Invalid task_type: {task_type}")


# Test function
def test_public_downstream_models():
    """Test public downstream model implementations."""
    
    print("🧪 Testing Public Downstream Models")
    print("=" * 50)
    
    # Create pretrained model
    pretrained_model = PublicNeuralFoundationMAE(
        neural_dim=50,
        d_model=256,  # Smaller for testing
        temporal_layers=2,
        heads=4
    )
    
    # Test parameters
    batch_size = 2
    seq_len = 50
    neural_dim = 50
    
    # Create test data
    neural_data = torch.randn(batch_size, 1, seq_len, neural_dim)
    session_coords = torch.tensor([
        [[0.0, 0.3, 0.0]],  # Subject c, mid-2013, center-out
        [[3.0, 0.8, 1.0]]   # Subject t, late-2013, random-target
    ]).expand(batch_size, 1, 3)
    
    # Test velocity predictor
    print("\n📈 Testing Velocity Predictor:")
    velocity_model = PublicVelocityPredictor(pretrained_model, training_mode='frozen_encoder')
    velocity_pred = velocity_model(neural_data, session_coords)
    print(f"  Input: {neural_data.shape} -> Output: {velocity_pred.shape}")
    
    # Test factory function
    print("\n🏭 Testing Factory Function:")
    regression_model = create_public_downstream_model(pretrained_model, task_type='regression')
    print(f"  Regression model created: {type(regression_model).__name__}")
    
    print("\n✅ All downstream model tests passed!")
    return True


if __name__ == "__main__":
    test_public_downstream_models()
