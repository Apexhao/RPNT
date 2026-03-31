import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

# ==================== SINGLE SITE DOWNSTREAM MODELS ====================

class SingleSiteDownstreamRegressor(nn.Module):
    """
    Single-Site Regressor using only the SparseTemporalEncoder.
    
    **PURPOSE**: Test generalization ability of temporal encoder on new sites.
    During downstream tasks, we have only 1 new site data and use only
    the temporal encoder (no spatial encoder) to evaluate temporal encoding robustness.
    
    **KEY FEATURES**:
    - Works with single site data [B, T, N] format
    - Uses only SparseTemporalEncoder from pretrained foundation model
    - Tests temporal encoding generalization to unseen sites
    - Flexible training modes (frozen vs finetuning vs random)
    """
    def __init__(self,
                 pretrained_temporal_encoder,  # SparseTemporalEncoder from foundation model
                 output_dim: int = 2,
                 training_mode: str = 'frozen_encoder',  # 'frozen_encoder', 'finetune_encoder', 'random'
                 prediction_layers: int = 2,
                 dropout: float = 0.1,): 
        super().__init__()
        
        valid_modes = ['frozen_encoder', 'finetune_encoder', 'random']
        if training_mode not in valid_modes:
            raise ValueError(f"training_mode must be one of {valid_modes}")
        
        self.training_mode = training_mode
        self.d_model = pretrained_temporal_encoder.d_model
        
        # Extract temporal encoder (from pretrained foundation model)
        self.temporal_encoder = pretrained_temporal_encoder
        
        # Apply training mode
        self._apply_training_mode()

        # Progressive reduction regression head (inspired by public_downstream.py)
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
        
        # Final output layer: [hidden] -> [output_dim] for predictions
        prediction_modules.append(nn.Linear(current_dim, output_dim))
        
        self.regressor = nn.Sequential(*prediction_modules)
        
        # Initialize regression head weights
        self._init_regression_weights()
    
    def _apply_training_mode(self):
        """Apply training mode to temporal encoder."""
        
        if self.training_mode == 'frozen_encoder':
            # Freeze temporal encoder parameters
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
                
        elif self.training_mode == 'finetune_encoder':
            # Allow finetuning of temporal encoder
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
                
        elif self.training_mode == 'random':
            # Reinitialize temporal encoder with random weights
            self._reinitialize_temporal_encoder()
            # Allow training of randomly initialized encoder
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
        
    def _reinitialize_temporal_encoder(self):
        """Reinitialize temporal encoder with random weights for ablation study."""
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Reinitialize attention weights
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
        
        # Apply random initialization to all temporal encoder parameters
        self.temporal_encoder.apply(init_weights)
        
    def _init_regression_weights(self):
        """Initialize regression head weights."""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                neural_data: torch.Tensor,        # [B, T, N] - single site data
                coords_3d: torch.Tensor           # [B, T, 3] - single site coordinates
                ) -> torch.Tensor:
        """
        Forward pass for single-site regression.
        
        Args:
            neural_data: [B, T, N] - single site neural activity
            coords_3d: [B, T, 3] - single site 3D coordinates
            
        Returns:
            predictions: [B, T, output_dim] or [B, output_dim] - regression outputs
        """
        B, T, N = neural_data.shape
        device = neural_data.device
        
        # Convert single site data to multi-site format for temporal encoder
        # Add site dimension: [B, T, N] -> [B, 1, T, N]
        neural_data_4d = neural_data.unsqueeze(1)  # [B, 1, T, N]
        coords_3d_4d = coords_3d.unsqueeze(1)      # [B, 1, T, 3]
        
        # Create causal mask
        from .attention import create_causal_mask
        causal_mask = create_causal_mask(T, device)
        
        # Forward through temporal encoder (site dimension = 1)
        temporal_outputs = self.temporal_encoder(
            neural_data_4d, coords_3d_4d, causal_mask, mask_data=None
        )  # [B, 1, T, D]
        
        # Remove site dimension: [B, 1, T, D] -> [B, T, D]
        features = temporal_outputs.squeeze(1)  # [B, T, D]
        
        # Predict at each timestep
        predictions = self.regressor(features)  # [B, T, output_dim]  
        
        return predictions

class SingleSiteClassifier(nn.Module):
    """
    Single-Site Classifier using only the SparseTemporalEncoder.
    
    **PURPOSE**: Test temporal encoder generalization for classification tasks.
    
    **KEY FEATURES**:
    - Trial-level classification using single site data
    - Temporal pooling strategies for sequence-to-label prediction
    - Uses only temporal encoder (no spatial encoder needed)
    - Flexible training modes (frozen vs finetuning vs random)
    """
    def __init__(self,
                 pretrained_temporal_encoder,  # SparseTemporalEncoder from foundation model
                 num_classes: int = 8,
                 training_mode: str = 'frozen_encoder',
                 dropout: float = 0.1,
                 temporal_pooling: str = 'mean'):  # 'mean', 'last', 'attention'
        super().__init__()
        
        valid_modes = ['frozen_encoder', 'finetune_encoder', 'random']
        if training_mode not in valid_modes:
            raise ValueError(f"training_mode must be one of {valid_modes}")
        
        self.training_mode = training_mode
        self.temporal_pooling = temporal_pooling
        self.d_model = pretrained_temporal_encoder.d_model
        
        # Extract temporal encoder
        self.temporal_encoder = pretrained_temporal_encoder
        
        # Apply training mode
        self._apply_training_mode()
        
        # Attention-based pooling layer
        if temporal_pooling == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.Tanh(),
                nn.Linear(self.d_model // 4, 1),
                nn.Softmax(dim=1)  # Softmax over time dimension
            )
        
        # Classification head (always trainable)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def _apply_training_mode(self):
        """Apply training mode to temporal encoder."""
        
        if self.training_mode == 'frozen_encoder':
            # Freeze temporal encoder parameters
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
                
        elif self.training_mode == 'finetune_encoder':
            # Allow finetuning of temporal encoder
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
                
        elif self.training_mode == 'random':
            # Reinitialize temporal encoder with random weights
            self._reinitialize_temporal_encoder()
            # Allow training of randomly initialized encoder
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
        
    def _reinitialize_temporal_encoder(self):
        """Reinitialize temporal encoder with random weights for ablation study."""
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Reinitialize attention weights
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
        
        # Apply random initialization to all temporal encoder parameters
        self.temporal_encoder.apply(init_weights)
    
    def forward(self,
                neural_data: torch.Tensor,        # [B, T, N] - single site data
                coords_3d: torch.Tensor           # [B, T, 3] - single site coordinates
                ) -> torch.Tensor:
        """
        Forward pass for single-site classification.
        
        Args:
            neural_data: [B, T, N] - single site neural activity
            coords_3d: [B, T, 3] - single site 3D coordinates
            
        Returns:
            logits: [B, num_classes] - classification logits
        """
        B, T, N = neural_data.shape
        device = neural_data.device
        
        # Convert to multi-site format for temporal encoder
        neural_data_4d = neural_data.unsqueeze(1)  # [B, 1, T, N]
        coords_3d_4d = coords_3d.unsqueeze(1)      # [B, 1, T, 3]
        
        # Create causal mask
        from .attention import create_causal_mask
        causal_mask = create_causal_mask(T, device)
        
        # Forward through temporal encoder
        temporal_outputs = self.temporal_encoder(
            neural_data_4d, coords_3d_4d, causal_mask, mask_data=None
        )  # [B, 1, T, D]
        
        # Remove site dimension: [B, 1, T, D] -> [B, T, D]
        features = temporal_outputs.squeeze(1)  # [B, T, D]
        
        # Temporal pooling
        if self.temporal_pooling == 'mean':
            # Simple average across time
            pooled_features = features.mean(dim=1)  # [B, D]
            
        elif self.temporal_pooling == 'last':
            # Use last timestep
            pooled_features = features[:, -1, :]  # [B, D]
            
        elif self.temporal_pooling == 'attention':
            # Attention-weighted pooling across time
            attention_weights = self.temporal_attention(features)  # [B, T, 1]
            pooled_features = (features * attention_weights).sum(dim=1)  # [B, D]
            
        else:
            raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits

class ParameterMatrixFactorizedRegressor(nn.Module):
    """
    Parameter-efficient regressor using matrix factorization (DEFAULT APPROACH).
    
    **APPROACH**: Matrix factorization of regressor layers themselves.
    Instead of learning large weight matrices directly, we factorize them into 
    smaller matrices that are multiplied together.
    
    **KEY FEATURES**:
    - Factorizes regressor layers: W = A × B where A and B are much smaller
    - Dramatically reduces parameters: ~14K vs ~131K for traditional regressor
    - No "adaptation" concept - we learn the regressor from scratch via factorization
    - Most parameter-efficient approach for new downstream tasks
    
    **EXAMPLE**:
    Traditional Layer 1: [512 → 256] = 131K parameters
    Factorized Layer 1: [512 → 16] × [16 → 256] = 8K + 4K = 12K parameters
    """
    def __init__(self,
                 pretrained_temporal_encoder,  # SparseTemporalEncoder
                 output_dim: int = 2,
                 hidden_dim: int = 256,
                 rank1: int = 16,  # Factorization rank for layer 1
                 rank2: int = 8,   # Factorization rank for layer 2
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = pretrained_temporal_encoder.d_model
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Freeze temporal encoder (we only learn the regressor)
        self.temporal_encoder = pretrained_temporal_encoder
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
            
        # Factorized Layer 1: [512 → 256] = [512 → 16] × [16 → 256]
        self.layer1_A = nn.Linear(self.d_model, rank1, bias=False)   # [512 → 16]
        self.layer1_B = nn.Linear(rank1, hidden_dim, bias=True)      # [16 → 256]
        
        # Factorized Layer 2: [256 → output_dim] = [256 → 8] × [8 → output_dim]  
        self.layer2_A = nn.Linear(hidden_dim, rank2, bias=False)     # [256 → 8]
        self.layer2_B = nn.Linear(rank2, output_dim, bias=True)      # [8 → output_dim]
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Initialize factorized weights properly
        self._init_factorized_weights()
        
    def _init_factorized_weights(self):
        """Initialize factorized weights to approximate reasonable initialization."""
        # Initialize A matrices with Kaiming uniform, B matrices with smaller variance
        nn.init.kaiming_uniform_(self.layer1_A.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.layer1_B.weight, gain=0.5)
        
        nn.init.kaiming_uniform_(self.layer2_A.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.layer2_B.weight, gain=0.5)
        
    def get_representations(self, neural_data: torch.Tensor, coords_3d: torch.Tensor) -> torch.Tensor:
        """Get representations from frozen temporal encoder."""
        B, T, N = neural_data.shape
        device = neural_data.device
        
        # Convert to multi-site format for temporal encoder
        neural_data_4d = neural_data.unsqueeze(1)  # [B, 1, T, N]
        coords_3d_4d = coords_3d.unsqueeze(1)      # [B, 1, T, 3]
        
        # Create causal mask
        from .attention import create_causal_mask
        causal_mask = create_causal_mask(T, device)
        
        # Get temporal representations (frozen)
        temporal_outputs = self.temporal_encoder(
            neural_data_4d, coords_3d_4d, causal_mask, mask_data=None
        )  # [B, 1, T, D]
        
        # Remove site dimension: [B, 1, T, D] → [B, T, D]
        representations = temporal_outputs.squeeze(1)  # [B, T, D]
        
        return representations
        
    def forward(self, 
                neural_data: torch.Tensor,        # [B, T, N] - single site
                coords_3d: torch.Tensor           # [B, T, 3] - single site
                ) -> torch.Tensor:
        """
        Forward pass with factorized regressor layers.
        
        Args:
            neural_data: [B, T, N] - single site neural activity
            coords_3d: [B, T, 3] - single site 3D coordinates
            
        Returns:
            output: [B, output_dim] - regression predictions
        """
        # Get frozen representations
        representations = self.get_representations(neural_data, coords_3d)  # [B, T, D]
        
        # Factorized Layer 1: W1 = A1 × B1
        # [B, T, 512] → [B, T, 16] → [B, T, 256]
        # Factorized Layer 2: W2 = A2 × B2  
        # [B, T, 256] → [B, T, 8] → [B, T, output_dim]
        x = self.layer1_B(self.layer1_A(representations))  
        x = self.activation(x)
        x = self.dropout(x)
        output = self.layer2_B(self.layer2_A(x))  
        
        return output

class ParameterLoRAFinetuneRegressor(nn.Module):
    """
    Parameter-efficient finetuning using LoRA-style adaptation (ALTERNATIVE APPROACH).
    
    **APPROACH**: Adapts the frozen representations rather than factorizing the regressor.
    This is more similar to traditional LoRA from language models, where we adapt
    intermediate representations.
    
    **KEY FEATURES**:
    - Adapts representations via low-rank matrices: adapted = original + A×B×scaling
    - ~17K parameters (vs ~14K for matrix factorization)
    - More flexible for multi-task scenarios (same adaptation, different heads)
    - Additive adaptation without modifying temporal encoder weights
    """
    def __init__(self,
                 pretrained_temporal_encoder,  # SparseTemporalEncoder
                 task_type: str = 'regression',
                 output_dim: int = 2,
                 rank: int = 16,  # LoRA rank
                 alpha: float = 32):  # LoRA scaling
        super().__init__()
        
        self.task_type = task_type
        self.d_model = pretrained_temporal_encoder.d_model
        self.rank = rank
        self.scaling = alpha / rank
        
        # Freeze temporal encoder
        self.temporal_encoder = pretrained_temporal_encoder
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        
        # LoRA adaptation layers - adapt representations
        self.lora_A = nn.Linear(self.d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.d_model, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Task-specific head (simple, since adaptation happens before)
        if task_type == 'regression':
            self.head = nn.Linear(self.d_model, output_dim)

    
    def forward(self,
                neural_data: torch.Tensor,        # [B, T, N] - single site
                coords_3d: torch.Tensor           # [B, T, 3] - single site
                ) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation for single site data.
        """
        B, T, N = neural_data.shape
        device = neural_data.device
        
        # Convert to multi-site format for temporal encoder
        neural_data_4d = neural_data.unsqueeze(1)  # [B, 1, T, N]
        coords_3d_4d = coords_3d.unsqueeze(1)      # [B, 1, T, 3]
        
        # Create causal mask
        from .attention import create_causal_mask
        causal_mask = create_causal_mask(T, device)
        
        # Get temporal representations
        temporal_outputs = self.temporal_encoder(
            neural_data_4d, coords_3d_4d, causal_mask, mask_data=None
        )  # [B, 1, T, D]
        
        # Remove site dimension and flatten
        representations = temporal_outputs.squeeze(1)  # [B, T, D]
        repr_flat = representations.view(-1, self.d_model)  # [B*T, D]
        
        # Apply LoRA adaptation
        adaptation = self.lora_B(self.lora_A(repr_flat)) * self.scaling
        adapted_repr = repr_flat + adaptation
        adapted_repr = adapted_repr.view(B, T, self.d_model)
        
        # Task-specific prediction
        output = self.head(adapted_repr)
        
        return output

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
    }


# ==================== DEMO AND TESTING ====================

def demo_single_site_downstream_tasks():
    """
    Demonstration of single-site downstream task models.
    """
    print("🎯 Single-Site Downstream Tasks Demo")
    print("=" * 45)
    
    # Import required components
    from .transformer import CrossSiteModelFactory
    from ..utils import create_single_site_data
    
    # Create foundation model and extract temporal encoder
    foundation_model = CrossSiteModelFactory.create_mae_model(size='small')
    temporal_encoder = foundation_model.temporal_encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temporal_encoder = temporal_encoder.to(device)
    
    # Create single site dummy data
    neural_data, coords_3d = create_single_site_data(
        batch_size=2, seq_len=50, neural_dim=75, device=device
    )
    
    print(f"📊 Single site data: {neural_data.shape}")
    print(f"📊 Single site coords: {coords_3d.shape}")
    
    # Test 1: Traditional Single-Site Regression
    print(f"\n📈 Testing Traditional Single-Site Regression...")
    try:
        regressor = SingleSiteDownstreamRegressor(
            temporal_encoder, 
            output_dim=2, 
            training_mode='frozen_encoder',
        ).to(device)
        
        predictions = regressor(neural_data, coords_3d)
        print(f"✅ Traditional regression successful!")
        print(f"  Output shape: {predictions.shape}")
        
        # Count parameters
        param_counts = count_parameters(regressor)
        print(f"  Trainable params: {param_counts['trainable_parameters']:,}")
        print(f"  Trainable ratio: {param_counts['trainable_ratio']:.3f}")
        
    except Exception as e:
        print(f"❌ Traditional regression failed: {e}")
        
    # Test 2: Traditional Single-Site Classification
    print(f"\n📈 Testing Traditional Single-Site Classification...")
    try:
        classifier = SingleSiteClassifier(
            temporal_encoder, 
            num_classes=8, 
            training_mode='frozen_encoder',
        ).to(device)
        
        predictions = classifier(neural_data, coords_3d)
        print(f"✅ Traditional classification successful!")
        print(f"  Output shape: {predictions.shape}")
        
        # Count parameters
        param_counts = count_parameters(classifier)
        print(f"  Trainable params: {param_counts['trainable_parameters']:,}")
        print(f"  Trainable ratio: {param_counts['trainable_ratio']:.3f}")
        
    except Exception as e:
        print(f"❌ Traditional classification failed: {e}")
    
    # Test 3: Matrix Factorization Regression (DEFAULT/RECOMMENDED)
    print(f"\n⭐ Testing Matrix Factorization Regression (DEFAULT)...")
    try:
        factorized_regressor = ParameterMatrixFactorizedRegressor(
            temporal_encoder,
            output_dim=2,
            rank1=16,
            rank2=8
        ).to(device)
        
        predictions = factorized_regressor(neural_data, coords_3d)
        print(f"✅ Matrix factorization regression successful!")
        print(f"  Output shape: {predictions.shape}")
        
        # Count parameters
        param_counts = count_parameters(factorized_regressor)
        print(f"  Trainable params: {param_counts['trainable_parameters']:,}")
        print(f"  Trainable ratio: {param_counts['trainable_ratio']:.4f}")
        
    except Exception as e:
        print(f"❌ Matrix factorization regression failed: {e}")
    
    # Test 4: LoRA-style Adaptation (Alternative)
    print(f"\n⚡ Testing LoRA-style Adaptation (Alternative)...")
    try:
        lora_model = ParameterLoRAFinetuneRegressor(
            temporal_encoder,
            task_type='regression',
            output_dim=2,
            rank=16
        ).to(device)
        
        output = lora_model(neural_data, coords_3d)
        print(f"✅ LoRA-style adaptation successful!")
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        param_counts = count_parameters(lora_model)
        print(f"  Trainable params: {param_counts['trainable_parameters']:,}")
        print(f"  Trainable ratio: {param_counts['trainable_ratio']:.4f}")
        
    except Exception as e:
        print(f"❌ LoRA-style adaptation failed: {e}")
    
    print(f"\n🎉 All single-site downstream tests completed!")
    print(f"\n💡 Key Insights:")
    print(f"   • Matrix factorization (default) offers the most parameter efficiency")
    print(f"   • LoRA-style adaptation provides more flexibility for multi-task scenarios")
    print(f"   • Both approaches dramatically reduce parameters vs traditional methods")
    print(f"   • These models test temporal encoder generalization to unseen sites")


if __name__ == "__main__":
    demo_single_site_downstream_tasks() 
    # python -c "from src.models.downstream import demo_single_site_downstream_tasks; demo_single_site_downstream_tasks()"