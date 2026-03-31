#!/usr/bin/env python3
"""
Debug script specifically for SparseTemporalEncoder NaN issues.
"""

import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.transformer import SparseTemporalEncoder
from src.models.positional_encoding import prepare_site_positional_data
from src.models.attention import create_causal_mask

def debug_sparse_temporal_encoder():
    """Debug the SparseTemporalEncoder step by step."""
    print("🔍 Debugging SparseTemporalEncoder Step by Step")
    print("=" * 60)
    
    # Simple configuration for debugging
    batch_size, n_sites, seq_len, neural_dim = 2, 3, 10, 75
    d_model = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📊 Configuration: B={batch_size}, S={n_sites}, T={seq_len}, N={neural_dim}, D={d_model}")
    print(f"🖥️  Device: {device}")
    
    # Create test data
    neural_data = torch.randn(batch_size, n_sites, seq_len, neural_dim) * 0.1  # Small values
    site_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # [3, 2] fixed coordinates
    
    print(f"\n📊 Input data:")
    print(f"  Neural data: {neural_data.shape}, mean={neural_data.mean():.6f}, std={neural_data.std():.6f}")
    print(f"  Site coords: {site_coords.shape}")
    print(f"  Site coords: {site_coords}")
    
    # Prepare site-specific data
    coords_expanded, time_indices = prepare_site_positional_data(
        neural_data, site_coords, device
    )
    
    print(f"\n📊 Prepared data:")
    print(f"  Coords expanded: {coords_expanded.shape}")
    print(f"  Time indices: {time_indices.shape}")
    print(f"  Coords sample: {coords_expanded[0, 0, 0]}")
    print(f"  Time sample: {time_indices[0, 0, :5]}")
    
    # Create causal mask
    causal_mask = create_causal_mask(seq_len, device)
    print(f"  Causal mask: {causal_mask.shape}")
    
    # Test different positional encoding types
    for encoding_type in ['advanced_3d', 'rope_3d', 'generalizable']:
        print(f"\n🧪 Testing {encoding_type}...")
        print("-" * 40)
        
        try:
            # Create encoder
            encoder = SparseTemporalEncoder(
                neural_dim=neural_dim,
                d_model=d_model,
                n_sites=n_sites,
                num_layers=1,  # Single layer for debugging
                heads=2,       # Fewer heads for simplicity
                pos_encoding_type=encoding_type,
                spatial_scale=1.0,
                use_temporal_kernels=False  # Disable kernels for simplicity
            ).to(device)
            
            print(f"✅ Encoder created")
            
            # Move data to device
            neural_data_device = neural_data.to(device)
            coords_expanded_device = coords_expanded.to(device)
            
            # Manual step-by-step forward pass
            print("\n🔍 Step-by-step forward pass:")
            
            # Step 1: Channel projection
            B, S, T, N = neural_data_device.shape
            x = encoder.channel_projection(neural_data_device)
            print(f"1. Channel projection: {x.shape}")
            print(f"   Stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected in channel projection!")
                return False
            
            # Step 2: Create time indices
            time_indices_device = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, T)
            print(f"2. Time indices: {time_indices_device.shape}")
            print(f"   Range: {time_indices_device.min().item()}-{time_indices_device.max().item()}")
            
            # Step 3: Reshape for positional encoder
            coords_reshaped = coords_expanded_device.reshape(B*S, T, 2)
            time_reshaped = time_indices_device.reshape(B*S, T)
            print(f"3. Reshaped for PE: coords={coords_reshaped.shape}, time={time_reshaped.shape}")
            print(f"   Coords stats: mean={coords_reshaped.mean().item():.6f}, std={coords_reshaped.std().item():.6f}")
            print(f"   Time stats: mean={time_reshaped.mean().item():.6f}, std={time_reshaped.std().item():.6f}")
            
            # Step 4: Apply positional encoding
            pos_encoding = encoder.positional_encoder(coords_reshaped, time_reshaped)
            print(f"4. Positional encoding: {pos_encoding.shape}")
            print(f"   Stats: mean={pos_encoding.mean().item():.6f}, std={pos_encoding.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(pos_encoding).any().item()}")
            
            if torch.isnan(pos_encoding).any():
                print(f"❌ NaN detected in positional encoding!")
                return False
            
            # Step 5: Reshape positional encoding
            pos_encoding = pos_encoding.reshape(B, S, T, d_model)
            print(f"5. PE reshaped: {pos_encoding.shape}")
            
            # Step 6: Add positional encoding
            x = x + pos_encoding
            print(f"6. After adding PE: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after adding positional encoding!")
                return False
            
            # Step 7: Apply dropout
            x = encoder.dropout(x)
            print(f"7. After dropout: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after dropout!")
                return False
            
            # Step 8: Process through attention layer
            layer = encoder.layers[0]
            residual = x
            x = layer['norm1'](x)
            print(f"8. After norm1: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after norm1!")
                return False
            
            # Process each site independently
            site_outputs = []
            for site_idx in range(S):
                site_data = x[:, site_idx, :, :]  # [B, T, D]
                print(f"   Site {site_idx}: {site_data.shape}, mean={site_data.mean().item():.6f}")
                
                # Apply attention (without kernels for simplicity)
                site_output = layer['attention'](
                    site_data, 
                    historical_data=None, 
                    causal_mask=causal_mask
                )
                print(f"   Site {site_idx} after attention: mean={site_output.mean().item():.6f}")
                print(f"   Has NaN: {torch.isnan(site_output).any().item()}")
                
                if torch.isnan(site_output).any():
                    print(f"❌ NaN detected in site {site_idx} attention!")
                    return False
                
                site_outputs.append(site_output)
            
            # Stack and add residual
            x = torch.stack(site_outputs, dim=1)
            x = residual + x
            print(f"9. After attention + residual: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"   Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after attention residual!")
                return False
            
            # Apply FFN
            residual = x
            x = layer['norm2'](x)
            print(f"10. After norm2: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"    Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after norm2!")
                return False
            
            # FFN
            x_flat = x.reshape(B*S*T, d_model)
            x_flat = layer['ffn'](x_flat)
            x = x_flat.reshape(B, S, T, d_model)
            x = residual + x
            
            print(f"11. After FFN: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            print(f"    Has NaN: {torch.isnan(x).any().item()}")
            
            if torch.isnan(x).any():
                print(f"❌ NaN detected after FFN!")
                return False
            
            print(f"✅ {encoding_type} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in {encoding_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n✅ All encoders completed successfully!")
    return True

def test_isolated_positional_encoding():
    """Test positional encoders in isolation with transformer-like data."""
    print("\n🔍 Testing Positional Encoders with Transformer Data")
    print("=" * 60)
    
    # Configuration matching transformer usage
    batch_size, n_sites, seq_len = 2, 3, 10
    d_model = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data exactly as transformer does
    neural_data = torch.randn(batch_size, n_sites, seq_len, 75)
    site_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Prepare data exactly as transformer does
    coords_expanded, _ = prepare_site_positional_data(neural_data, site_coords, device)
    
    B, S, T = batch_size, n_sites, seq_len
    time_indices = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, T)
    
    # Reshape exactly as transformer does
    coords_reshaped = coords_expanded.reshape(B*S, T, 2)
    time_reshaped = time_indices.reshape(B*S, T)
    
    print(f"📊 Transformer-style inputs:")
    print(f"  coords_reshaped: {coords_reshaped.shape}")
    print(f"  time_reshaped: {time_reshaped.shape}")
    print(f"  coords sample: {coords_reshaped[0, 0]}")
    print(f"  time sample: {time_reshaped[0, :5]}")
    
    from src.models.positional_encoding import create_positional_encoder
    
    for encoding_type in ['advanced_3d', 'rope_3d', 'generalizable']:
        print(f"\n🧪 Testing {encoding_type} with transformer data...")
        
        try:
            encoder = create_positional_encoder(
                encoding_type=encoding_type,
                d_model=d_model,
                spatial_scale=1.0
            ).to(device)
            
            with torch.no_grad():
                output = encoder(coords_reshaped.to(device), time_reshaped.to(device))
                
                print(f"   ✅ Output: {output.shape}")
                print(f"   📊 Stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
                print(f"   🔍 Has NaN: {torch.isnan(output).any().item()}")
                
                if torch.isnan(output).any():
                    print(f"   ❌ NaN detected!")
                    return False
                else:
                    print(f"   ✅ No NaN values")
                    
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    print(f"\n✅ All positional encoders work with transformer data!")
    return True

def main():
    """Run all debug tests."""
    print("🔍 SparseTemporalEncoder Debug Suite")
    print("=" * 70)
    
    success1 = test_isolated_positional_encoding()
    success2 = debug_sparse_temporal_encoder()
    
    print("\n" + "=" * 70)
    print("📊 DEBUG SUMMARY")
    print("=" * 70)
    print(f"Isolated PE Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Full Encoder Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n✅ All tests passed - SparseTemporalEncoder should work!")
    else:
        print("\n❌ Issues detected - need to investigate further")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 