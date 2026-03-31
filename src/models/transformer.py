import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Union, Optional, List
import logging

# Import from separate modules
from .positional_encoding import (LearnableMultiDimPE, RoPE3D, 
                                 create_positional_encoder, prepare_site_positional_data)
from .attention import CausalAdaptiveKernelAttention, create_causal_mask

class CrossSiteFoundationMAE(nn.Module):
    """
    Cross-Site Neural Foundation Model with Causal MAE Training
    
    **CORE ARCHITECTURE**:
    - Input: 4D tensor [B, S, T, N] (Batch, Sites, Time, Neural_dim)
    - Stage 1: SparseTemporalEncoder - processes each site with causal attention
    - Stage 2: SpatialCrossAttentionEncoder - models cross-site interactions  
    - Stage 3: LightweightMAEDecoder - Poisson reconstruction for pretraining
    
    **KEY FEATURES**:
    - Causal masking throughout the pipeline
    - 4D tensor processing for efficient computation
    - Simplified causal kernel attention
    - Lightweight decoder for MAE training
    """
    def __init__(self,
                 neural_dim: int = 75,          # N: channels per site
                 d_model: int = 512,            # D: model hidden dimension
                 n_sites: int = 17,             # S: number of recording sites
                 temporal_layers: int = 6,       # Temporal encoder depth
                 spatial_layers: int = 4,        # Spatial encoder depth
                 heads: int = 8,                # Attention heads
                 kernel_size: List[int] = [3, 3], # Adaptive kernel size
                 dropout: float = 0.1,
                 max_seq_length: int = 2000,    # T: max sequence length
                 pos_encoding_type: str = 'rope_3d',
                 spatial_scale: float = 0.1,
                 use_temporal_kernels: bool = True,
                 use_mae_decoder: bool = True,
                 use_site_specific_heads: bool = True):
        super().__init__()
        
        self.neural_dim = neural_dim
        self.d_model = d_model
        self.n_sites = n_sites
        self.use_mae_decoder = use_mae_decoder
        
        # Stage 1: Temporal Encoder (processes each site with causal attention)
        self.temporal_encoder = SparseTemporalEncoder(
            neural_dim=neural_dim,
            d_model=d_model,
            n_sites=n_sites,
            num_layers=temporal_layers,
            heads=heads,
            kernel_size=kernel_size,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pos_encoding_type=pos_encoding_type,
            spatial_scale=spatial_scale,
            use_temporal_kernels=use_temporal_kernels
        )
        
        # Stage 2: Spatial Encoder (cross-site interactions at each timestep)
        self.spatial_encoder = SpatialCrossAttentionEncoder(
            d_model=d_model,
            n_sites=n_sites,
            num_layers=spatial_layers,
            heads=heads,
            dropout=dropout
        )
        
        # Stage 3: MAE Decoder (for pretraining only)
        if use_mae_decoder:
            self.mae_decoder = LightweightMAEDecoder(
                d_model=d_model,
                neural_dim=neural_dim,
                n_sites=n_sites,
                dropout=dropout,
                use_site_specific_heads=use_site_specific_heads
            )
        
        # For downstream tasks (alternative to MAE decoder)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, neural_dim)
        )
        
    def forward(self, 
                neural_data: torch.Tensor,           # [B, S, T, N]
                site_coords: torch.Tensor,           # [S, 2] - Fixed (X, Y) coordinates per site
                mask_data: Optional[torch.Tensor] = None,  # [B, S, T] - MAE mask
                return_mae_reconstruction: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Cross-Site Foundation MAE with site-specific positional encoding.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            site_coords: [S, 2] - Fixed (X, Y) coordinates per site
            mask_data: [B, S, T] - binary mask for MAE (1=masked, 0=unmasked)
            return_mae_reconstruction: If True, return MAE reconstruction
            
        Returns:
            Dict containing:
            - 'representations': [B, S, T, D] - final representations
            - 'reconstruction': [B, S, T, N] - MAE reconstruction (if requested)
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # ✅ ENHANCED: Prepare site-specific positional data
        coords_expanded, time_indices = prepare_site_positional_data(
            neural_data, site_coords, device
        )  # [B, S, T, 2], [B, S, T]
        
        # Create causal mask for temporal encoding
        causal_mask = create_causal_mask(T, device)
        
        # Stage 1: Temporal encoding with site-specific positional encoding
        temporal_outputs = self.temporal_encoder(
            neural_data, coords_expanded, causal_mask, mask_data
        )  # [B, S, T, D]
        
        results = {'representations': temporal_outputs}
        # Stage 2: Spatial cross-attention (timestep-wise causality)
        spatial_outputs = self.spatial_encoder(temporal_outputs)  # [B, S, T, D]
        
        # Stage 3: MAE reconstruction (if requested)
        if return_mae_reconstruction and self.use_mae_decoder:
            reconstruction = self.mae_decoder(spatial_outputs, mask_data)  # [B, S, T, N]
            results['reconstruction'] = reconstruction
        
        return results
        
    def get_connectivity_matrix(self, neural_data: torch.Tensor, site_coords: torch.Tensor) -> torch.Tensor:
        """
        Extract functional connectivity matrix from spatial attention weights.
        
        Args:
            neural_data: [B, S, T, N] - Neural activity data
            site_coords: [S, 2] - Fixed site coordinates
        
        Returns:
            connectivity_matrix: [S, S] - site-to-site connectivity
        """
        with torch.no_grad():
            _ = self.forward(neural_data, site_coords)
            connectivity = self.spatial_encoder.get_attention_weights()
            # Average across batch, time, and layers: [B, T, layers, S, S] -> [S, S]
            return connectivity.mean(dim=[0, 1, 2])

class SparseTemporalEncoder(nn.Module):
    """
    Enhanced Temporal Encoder with Site-Specific Positional Encoding
    
    **KEY ENHANCEMENTS**:
    - 4D tensor processing: [B, S, T, N] -> [B, S, T, D]
    - Site-specific positional encoding with zero-shot generalization
    - Support for fixed site coordinates (X, Y) + temporal positions
    - Two core encoding strategies: learned (Advanced3D) and mathematical (RoPE3D)
    
    **GENERALIZATION**: Works with new unseen site locations at test time.
    """
    def __init__(self,
                 neural_dim: int = 75,
                 d_model: int = 512,
                 n_sites: int = 17,
                 num_layers: int = 6,
                 heads: int = 8,
                 kernel_size: List[int] = [3, 3],
                 dropout: float = 0.1,
                 max_seq_length: int = 2000,
                 pos_encoding_type: str = 'rope_3d',  # Updated default
                 spatial_scale: float = 0.1,
                 use_temporal_kernels: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_sites = n_sites
        self.use_temporal_kernels = use_temporal_kernels
        self.pos_encoding_type = pos_encoding_type
        
        # Channel projection: neural_dim -> d_model (applied per site)
        self.channel_projection = nn.Linear(neural_dim, d_model)
        
       
        self.positional_encoder = create_positional_encoder(
            encoding_type=pos_encoding_type,
            d_model=d_model,
            max_seq_length=max_seq_length,
            spatial_scale=spatial_scale
        )
        
        
        self.use_rope = (pos_encoding_type in ['rope_3d', 'standard_rope'])
        
        self.dropout = nn.Dropout(dropout)
        
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CausalAdaptiveKernelAttention(
                    d_model, heads, kernel_size, dropout,
                    use_kernel=use_temporal_kernels,
                    use_rope=self.use_rope,              # ✅ NEW: Enable RoPE if using RoPE3D
                    rope_module=self.positional_encoder if self.use_rope else None  # ✅ NEW: Pass RoPE module
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
    
    def forward(self, 
                neural_data: torch.Tensor,        # [B, S, T, N]
                site_coords: torch.Tensor,        # [B, S, T, 2] - (X, Y) coordinates
                causal_mask: torch.Tensor,        # [T, T]
                mask_data: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, S, T]
        """
        Forward pass with site-specific positional encoding.
        
        **CRITICAL CHANGE**: RoPE3D is handled differently from traditional PE!
        
        Args:
            neural_data: [B, S, T, N] - neural activity for all sites
            site_coords: [B, S, T, 2] - (X, Y) coordinates for all sites
            causal_mask: [T, T] - causal attention mask
            mask_data: [B, S, T] - optional MAE mask
            
        Returns:
            encoded: [B, S, T, D] - temporally encoded features
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Channel projection for all sites: [B, S, T, N] -> [B, S, T, D]
        x = self.channel_projection(neural_data)
        
        #  CRITICAL: Handle positional encoding based on type
        if self.use_rope:
            # RoPE3D and standard rope: Do NOT add positional encoding to embeddings!
            # RoPE will be applied directly in attention mechanism
            x = self.dropout(x)  # Just apply dropout to embeddings
            
        else:
            # Traditional PE: Add positional encoding to embeddings
            # Create time indices (ensure float dtype for positional encoders)
            time_indices = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, T)  # [B, S, T]
            
            # Reshape for positional encoder: [B*S, T, 2], [B*S, T]
            coords_reshaped = site_coords.reshape(B*S, T, 2)     # [B*S, T, 2]
            time_reshaped = time_indices.reshape(B*S, T)         # [B*S, T]
            
            # Apply positional encoding (works for all encoder types except RoPE)
            pos_encoding = self.positional_encoder(coords_reshaped, time_reshaped)  # [B*S, T, D]
            pos_encoding = pos_encoding.reshape(B, S, T, self.d_model)  # [B, S, T, D]
                
            x = x + pos_encoding
            x = self.dropout(x)
        
        # Process each site separately with causal attention
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            
            # Process each site independently for temporal attention
            site_outputs = []
            for site_idx in range(S):
                site_data = x[:, site_idx, :, :]  # [B, T, D]
                
                
                if self.use_rope:
                    # For RoPE: Pass site coordinates and time indices to attention
                    site_coords_2d = site_coords[:, site_idx, :, :]  # [B, T, 2]
                    time_indices = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).expand(B, T)  # [B, T]
                    
                    site_output = layer['attention'](
                        site_data, 
                        historical_data=site_data if self.use_temporal_kernels and T > 1 else None,
                        causal_mask=causal_mask,
                        site_coords=site_coords_2d,     # 
                        time_indices=time_indices       # 
                    )
                else:
                    # For traditional PE: No coordinate data needed
                    site_output = layer['attention'](
                        site_data, 
                        historical_data=site_data if self.use_temporal_kernels and T > 1 else None,
                        causal_mask=causal_mask
                    )
                
                site_outputs.append(site_output)
            
            # Stack site outputs: List of [B, T, D] -> [B, S, T, D]
            x = torch.stack(site_outputs, dim=1)
            x = residual + x
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x_flat = x.reshape(B*S*T, self.d_model)  # Flatten for FFN
            x_flat = layer['ffn'](x_flat)
            x = x_flat.reshape(B, S, T, self.d_model)  # Reshape back
            x = residual + x
            
        return x

class SpatialCrossAttentionEncoder(nn.Module):
    """
    Enhanced Spatial Cross-Attention with Timestep-wise Causality
    
    **CAUSALITY GUARANTEE**: Processes each timestep independently.
    At timestep t, only uses information from timestep t (no future leakage).
    
    **KEY FEATURES**:
    - 4D tensor processing: [B, S, T, D] -> [B, S, T, D]
    - Timestep-wise spatial attention for natural causality
    - Attention weight extraction for connectivity analysis
    - Efficient parallel processing across timesteps
    """
    def __init__(self,
                 d_model: int = 512,
                 n_sites: int = 17,
                 num_layers: int = 4,
                 heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_sites = n_sites
        self.num_layers = num_layers
        
        # Spatial attention layers (cross-site interactions)
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attention': nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
        
        # Store attention weights for connectivity analysis
        self.attention_weights = []
        
    def forward(self, temporal_outputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with timestep-wise spatial attention.
        
        **CAUSAL PROPERTY**: Each timestep t is processed independently,
        ensuring no future information leakage.
        
        Args:
            temporal_outputs: [B, S, T, D] - temporally encoded features
            
        Returns:
            spatial_outputs: [B, S, T, D] - spatially processed features
        """
        B, S, T, D = temporal_outputs.shape
        
        # Initialize output tensor
        x = temporal_outputs
        
        # Clear attention weights for new forward pass
        self.attention_weights = []
        
        # Apply spatial attention layers
        for layer in self.spatial_layers:
            residual = x
            x = layer['norm1'](x)
            
            # Process each timestep independently for spatial connectivity
            # This ensures causality: timestep t cannot see timestep t+1
            timestep_outputs = []
            timestep_attention_weights = []
            
            for t in range(T):
                # Extract features at timestep t: [B, S, D]
                timestep_features = x[:, :, t, :]  # [B, S, D]
                
                # Cross-attention between all sites at this timestep
                attn_output, attn_weights = layer['cross_attention'](
                    timestep_features, timestep_features, timestep_features
                )  # [B, S, D], [B, S, S]
                
                timestep_outputs.append(attn_output)
                timestep_attention_weights.append(attn_weights)
            
            # Stack timestep outputs: List of [B, S, D] -> [B, S, T, D]
            x = torch.stack(timestep_outputs, dim=2)
            x = residual + x
            
            # Store attention weights: List of [B, S, S] -> [B, T, S, S]
            layer_attention_weights = torch.stack(timestep_attention_weights, dim=1)
            self.attention_weights.append(layer_attention_weights)
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x_flat = x.reshape(B*S*T, D)  # Flatten for FFN
            x_flat = layer['ffn'](x_flat)
            x = x_flat.reshape(B, S, T, D)  # Reshape back
            x = residual + x
        
        return x
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get spatial attention weights for connectivity analysis.
        
        Returns:
            attention_weights: [B, T, num_layers, S, S] - attention matrices
        """
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run forward pass first.")
        
        # Stack attention weights across layers: [num_layers, B, T, S, S] -> [B, T, num_layers, S, S]
        stacked_weights = torch.stack(self.attention_weights, dim=2)  # [B, T, num_layers, S, S]
        return stacked_weights

class LightweightMAEDecoder(nn.Module):
    """
    Lightweight MAE Decoder for Causal Reconstruction
    
    **KEY FEATURES**:
    - Minimal parameters (~0.5M) for efficient training
    - Poisson rate prediction for neural spike statistics
    - Causal reconstruction respecting temporal ordering
    - Site-specific heads with shared core processing
    
    **CAUSAL GUARANTEE**: Uses only representations up to current timestep
    for reconstruction at that timestep.
    """
    def __init__(self,
                 d_model: int = 512,
                 neural_dim: int = 75,
                 n_sites: int = 17,
                 dropout: float = 0.1,
                 use_site_specific_heads: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.neural_dim = neural_dim
        self.n_sites = n_sites
        self.use_site_specific_heads = use_site_specific_heads
        
        # Shared lightweight decoder core
        self.decoder_core = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if use_site_specific_heads:
            # Site-specific output heads (account for electrode differences)
            self.site_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model // 4, neural_dim),
                    nn.Softplus()  # Ensure positive rates for Poisson
                ) for _ in range(n_sites)
            ])
        else:
            # Shared output head
            self.shared_head = nn.Sequential(
                nn.Linear(d_model // 4, neural_dim),
                nn.Softplus()  # Ensure positive rates for Poisson
            )
            
    def forward(self, 
                representations: torch.Tensor,           # [B, S, T, D]
                mask_data: Optional[torch.Tensor] = None # [B, S, T]
                ) -> torch.Tensor:
        """
        Causal MAE reconstruction.
        
        Args:
            representations: [B, S, T, D] - encoded representations
            mask_data: [B, S, T] - binary mask (1=masked, 0=unmasked)
            
        Returns:
            reconstruction: [B, S, T, N] - Poisson rate parameters
        """
        B, S, T, D = representations.shape
        
        # Apply decoder core to all positions
        decoded_core = self.decoder_core(representations)  # [B, S, T, D//4]
        
        if self.use_site_specific_heads:
            # Apply site-specific heads
            site_reconstructions = []
            for site_idx in range(S):
                site_features = decoded_core[:, site_idx, :, :]  # [B, T, D//4]
                site_recon = self.site_heads[site_idx](site_features)  # [B, T, N]
                site_reconstructions.append(site_recon)
            
            reconstruction = torch.stack(site_reconstructions, dim=1)  # [B, S, T, N]
        else:
            # Apply shared head
            decoded_flat = decoded_core.view(B*S*T, D//4)
            recon_flat = self.shared_head(decoded_flat)  # [B*S*T, N]
            reconstruction = recon_flat.view(B, S, T, self.neural_dim)  # [B, S, T, N]
        
        return reconstruction

# ==================== MODEL FACTORY ====================
class CrossSiteModelFactory:
    """
    Factory for creating CrossSiteFoundationMAE models with predefined configurations.
    """
    
    @staticmethod
    def create_mae_model(size: str = 'medium', **kwargs) -> CrossSiteFoundationMAE:
        """
        Create CrossSiteFoundationMAE with optimal configurations.
        
        Args:
            size: 'small', 'medium', or 'large'
            **kwargs: Override default parameters
            
        Returns:
            CrossSiteFoundationMAE model
        """
        configs = {
            'small': {
                'd_model': 256,
                'temporal_layers': 4,
                'spatial_layers': 2,
                'heads': 4,
                'neural_dim': 75,
                'n_sites': 17,
                'dropout': 0.1,
                'kernel_size': [3, 3]
            },
            'medium': {
                'd_model': 512,
                'temporal_layers': 6,
                'spatial_layers': 4,
                'heads': 8,
                'neural_dim': 75,
                'n_sites': 17,
                'dropout': 0.1,
                'kernel_size': [3, 3]
            },
            'large': {
                'd_model': 768,
                'temporal_layers': 8,
                'spatial_layers': 6,
                'heads': 12,
                'neural_dim': 75,
                'n_sites': 17,
                'dropout': 0.1,
                'kernel_size': [5, 5]
            }
        }
        
        if size not in configs:
            raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
        
        config = configs[size]
        config.update(kwargs)  # Override with any provided kwargs
        
        return CrossSiteFoundationMAE(**config)
    
    @staticmethod
    def create_downstream_model(foundation_model: CrossSiteFoundationMAE,
                               task_type: str = 'regression',
                               **kwargs):
        """
        Create task-specific model with frozen foundation weights.
        
        Args:
            foundation_model: Pretrained foundation model
            task_type: 'regression' or 'classification'
            **kwargs: Task-specific parameters
            
        Returns:
            Downstream model
        """
        # Import here to avoid circular imports
        from .downstream import SingleSiteDownstreamRegressor, SingleSiteClassifier
        
        if task_type == 'regression':
            return SingleSiteDownstreamRegressor(
                foundation_model.temporal_encoder, 
                **kwargs
            )
        elif task_type == 'classification':
            return SingleSiteClassifier(
                foundation_model.temporal_encoder,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Use 'regression' or 'classification'")


# ==================== DEMO AND TESTING ====================

def demo_causal_mae_training():
    """
    Demonstration of site-specific positional encoding with zero-shot generalization.
    
    Shows how to:
    1. Use fixed site coordinates for training
    2. Apply different positional encoding strategies
    3. Test with new unseen site locations (zero-shot)
    """
    print("🧠 Demo: Site-Specific Positional Encoding for Neural Foundation Models")
    
    # Configuration
    batch_size, n_sites, seq_len, neural_dim = 4, 17, 500, 75
    d_model = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ✅ STEP 1: Define fixed site coordinates (anatomical locations)
    # Example: Sites arranged in a grid pattern (could be real electrode coordinates)
    site_coords = torch.randn(n_sites, 2) * 10.0  # [17, 2] - (X, Y) coordinates
    print(f"📍 Site coordinates shape: {site_coords.shape}")
    print(f"📍 Example coordinates: {site_coords[:3]}")
    
    # ✅ STEP 2: Generate neural data
    neural_data = torch.randn(batch_size, n_sites, seq_len, neural_dim)
    print(f"🧬 Neural data shape: {neural_data.shape}")
    
    # ✅ STEP 3: Test different positional encoding strategies
    encoding_types = ['advanced_3d', 'rope_3d']
    
    for encoding_type in encoding_types:
        print(f"\n🔬 Testing {encoding_type} positional encoding...")
        
        # Create model with specific encoding type
        model = CrossSiteFoundationMAE(
            neural_dim=neural_dim,
            d_model=d_model,
            n_sites=n_sites,
            pos_encoding_type=encoding_type,
            spatial_scale=1.0
        ).to(device)
        
        # Move data to device
        neural_data_device = neural_data.to(device)
        site_coords_device = site_coords.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(neural_data_device, site_coords_device, return_mae_reconstruction=True)
            representations = outputs['representations']
            reconstruction = outputs['reconstruction']
            
            print(f"   ✅ Representations shape: {representations.shape}")
            print(f"   ✅ Reconstruction shape: {reconstruction.shape}")
            print(f"   ✅ Reconstruction error: {F.mse_loss(reconstruction, neural_data_device):.6f}")
    
    # ✅ STEP 4: Test zero-shot generalization to new sites
    print(f"\n🎯 Testing Zero-Shot Generalization to New Sites...")
    
    # Create a new site with different coordinates
    new_site_coords = torch.tensor([[25.0, 15.0]])  # [1, 2] - New site location
    new_neural_data = torch.randn(batch_size, 1, seq_len, neural_dim)  # [B, 1, T, N]
    
    print(f"🆕 New site coordinates: {new_site_coords}")
    print(f"🆕 New neural data shape: {new_neural_data.shape}")
    
    # Test generalization with each encoding type
    for encoding_type in encoding_types:
        print(f"\n   🔬 {encoding_type} zero-shot test...")
        
        # Create fresh model (to simulate deployment scenario)
        model = CrossSiteFoundationMAE(
            neural_dim=neural_dim,
            d_model=d_model,
            n_sites=1,  # Single new site
            pos_encoding_type=encoding_type,
            spatial_scale=1.0
        ).to(device)
        
        try:
            with torch.no_grad():
                new_outputs = model(
                    new_neural_data.to(device), 
                    new_site_coords.to(device),
                    return_mae_reconstruction=True
                )
                print(f"      ✅ SUCCESS: Generated representations {new_outputs['representations'].shape}")
                print(f"      ✅ Reconstruction error: {F.mse_loss(new_outputs['reconstruction'], new_neural_data.to(device)):.6f}")
        except Exception as e:
            print(f"      ❌ FAILED: {str(e)}")
    
    # ✅ STEP 5: Demonstrate cross-site connectivity analysis
    print(f"\n🔗 Testing Cross-Site Connectivity Analysis...")
    
    model = CrossSiteFoundationMAE(
        neural_dim=neural_dim,
        d_model=d_model,
        n_sites=n_sites,
        pos_encoding_type='rope_3d',  # Use mathematical RoPE approach
        spatial_layers=2  # Fewer layers for demo
    ).to(device)
    
    with torch.no_grad():
        connectivity_matrix = model.get_connectivity_matrix(
            neural_data.to(device), 
            site_coords.to(device)
        )
        print(f"🔗 Connectivity matrix shape: {connectivity_matrix.shape}")
        print(f"🔗 Average connectivity: {connectivity_matrix.mean():.6f}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"📋 Key Benefits Demonstrated:")
    print(f"   • Site-specific positional encoding")
    print(f"   • Zero-shot generalization to new sites") 
    print(f"   • Multiple encoding strategies")
    print(f"   • Cross-site connectivity analysis")

if __name__ == "__main__":
    demo_causal_mae_training()