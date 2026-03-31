#!/usr/bin/env python3
"""
Debug script for positional encoding NaN issues.
"""

import torch
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.positional_encoding import (
    Advanced3DPositionalEncoding,
    RoPE3D,
    GeneralizablePositionalEncoding,
    create_positional_encoder,
    prepare_site_positional_data
)

def debug_positional_encoder(encoding_type: str):
    """Debug a specific positional encoder."""
    print(f"\n🔍 Debugging {encoding_type} positional encoder...")
    print("-" * 60)
    
    # Simple test configuration
    batch_size, seq_len = 2, 10  # Small sizes for debugging
    d_model = 64  # Small d_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple test data
    site_coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # [2, 2] - two sites
    time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)  # [2, 10]
    
    print(f"📊 Input shapes:")
    print(f"  site_coords: {site_coords.shape} - {site_coords}")
    print(f"  time_indices: {time_indices.shape} - range {time_indices.min()}-{time_indices.max()}")
    
    try:
        # Create encoder
        encoder = create_positional_encoder(
            encoding_type=encoding_type,
            d_model=d_model,
            spatial_scale=1.0
        ).to(device)
        
        print(f"✅ Encoder created successfully")
        print(f"📊 Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
        
        # Create proper input shapes for the positional encoder
        # The encoder expects [batch, seq_len, 2] and [batch, seq_len]
        # Let's test with a simpler case: batch_size=2, seq_len=10
        
        # Expand site coordinates to match batch and sequence dimensions
        # site_coords: [2, 2] -> [batch_size, seq_len, 2] 
        coords_reshaped = site_coords[0].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, 2)  # Use first site for simplicity
        time_reshaped = time_indices  # [batch_size, seq_len]
        
        print(f"📊 Reshaped inputs:")
        print(f"  coords_reshaped: {coords_reshaped.shape}")
        print(f"  time_reshaped: {time_reshaped.shape}")
        print(f"  coords sample: {coords_reshaped[0, 0]}")
        print(f"  time sample: {time_reshaped[0, :5]}")
        
        # Move to device
        coords_reshaped = coords_reshaped.to(device)
        time_reshaped = time_reshaped.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = encoder(coords_reshaped, time_reshaped)
            
            print(f"✅ Forward pass successful")
            print(f"📊 Output shape: {output.shape}")
            print(f"📊 Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            print(f"📊 Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
            print(f"🔍 Has NaN: {torch.isnan(output).any().item()}")
            print(f"🔍 Has Inf: {torch.isinf(output).any().item()}")
            
            if torch.isnan(output).any():
                print(f"❌ NaN detected!")
                
                # Find where NaNs are
                nan_mask = torch.isnan(output)
                nan_positions = torch.where(nan_mask)
                print(f"🔍 NaN positions: {len(nan_positions[0])} total")
                if len(nan_positions[0]) > 0:
                    print(f"🔍 First few NaN positions: {list(zip(nan_positions[0][:5].cpu().numpy(), nan_positions[1][:5].cpu().numpy(), nan_positions[2][:5].cpu().numpy()))}")
                
                # Check intermediate calculations for specific encoders
                if encoding_type == 'advanced_3d':
                    debug_advanced_3d_encoder(encoder, coords_reshaped, time_reshaped)
                elif encoding_type == 'rope_3d':
                    debug_rope_3d_encoder(encoder, coords_reshaped, time_reshaped)
                elif encoding_type == 'generalizable':
                    debug_generalizable_encoder(encoder, coords_reshaped, time_reshaped)
                
                return False
            else:
                print(f"✅ No NaN values detected")
                return True
                
    except Exception as e:
        print(f"❌ Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_advanced_3d_encoder(encoder, coords, time_indices):
    """Debug Advanced3DPositionalEncoding internals."""
    print(f"\n🔬 Debugging Advanced3DPositionalEncoding internals...")
    
    with torch.no_grad():
        batch_size, seq_len = time_indices.shape
        
        # Step 1: Time normalization
        max_time = time_indices.max()
        print(f"📊 Max time: {max_time}")
        
        if max_time > 0:
            time_normalized = time_indices.float() / max_time
        else:
            time_normalized = time_indices.float()
        
        print(f"📊 Time normalized: {time_normalized.mean():.6f} ± {time_normalized.std():.6f}")
        print(f"🔍 Time normalized has NaN: {torch.isnan(time_normalized).any()}")
        
        # Step 2: Spatial-temporal coordinates
        spatial_temporal_coords = torch.cat([
            coords * encoder.spatial_scale,
            time_normalized.unsqueeze(-1)
        ], dim=-1)
        
        print(f"📊 Spatial-temporal coords: {spatial_temporal_coords.shape}")
        print(f"📊 ST coords stats: mean={spatial_temporal_coords.mean():.6f}, std={spatial_temporal_coords.std():.6f}")
        print(f"🔍 ST coords has NaN: {torch.isnan(spatial_temporal_coords).any()}")
        
        # Step 3: Position MLP
        pos_encoding = encoder.position_mlp(spatial_temporal_coords)
        print(f"📊 Position encoding: {pos_encoding.shape}")
        print(f"📊 Position encoding stats: mean={pos_encoding.mean():.6f}, std={pos_encoding.std():.6f}")
        print(f"🔍 Position encoding has NaN: {torch.isnan(pos_encoding).any()}")
        
        # Step 4: Site encoding
        site_emb = encoder.site_encoder(coords * encoder.spatial_scale)
        print(f"📊 Site embedding: {site_emb.shape}")
        print(f"📊 Site embedding stats: mean={site_emb.mean():.6f}, std={site_emb.std():.6f}")
        print(f"🔍 Site embedding has NaN: {torch.isnan(site_emb).any()}")
        
        # Step 5: Concatenation
        full_encoding = torch.cat([pos_encoding, site_emb], dim=-1)
        print(f"📊 Full encoding: {full_encoding.shape}")
        print(f"🔍 Full encoding has NaN: {torch.isnan(full_encoding).any()}")
        
        # Step 6: Final projection
        output = encoder.final_proj(full_encoding)
        print(f"📊 Final output: {output.shape}")
        print(f"🔍 Final output has NaN: {torch.isnan(output).any()}")

def debug_rope_3d_encoder(encoder, coords, time_indices):
    """Debug RoPE3D internals."""
    print(f"\n🔬 Debugging RoPE3D internals...")
    
    with torch.no_grad():
        batch_size, seq_len = time_indices.shape
        
        # Extract coordinates
        x = coords[..., 0] * encoder.spatial_scale
        y = coords[..., 1] * encoder.spatial_scale
        t = time_indices.float()
        
        print(f"📊 X coords: {x.mean():.6f} ± {x.std():.6f}")
        print(f"📊 Y coords: {y.mean():.6f} ± {y.std():.6f}")
        print(f"📊 T coords: {t.mean():.6f} ± {t.std():.6f}")
        print(f"🔍 X has NaN: {torch.isnan(x).any()}")
        print(f"🔍 Y has NaN: {torch.isnan(y).any()}")
        print(f"🔍 T has NaN: {torch.isnan(t).any()}")
        
        # Check frequency computations
        print(f"📊 inv_freq_x: {encoder.inv_freq_x}")
        print(f"📊 inv_freq_y: {encoder.inv_freq_y}")
        print(f"📊 inv_freq_t: {encoder.inv_freq_t}")
        
        # Test sinusoidal computation for X
        sinusoid_inp_x = torch.einsum('bi,j->bij', x, encoder.inv_freq_x)
        print(f"📊 Sinusoid input X: {sinusoid_inp_x.mean():.6f} ± {sinusoid_inp_x.std():.6f}")
        print(f"🔍 Sinusoid input X has NaN: {torch.isnan(sinusoid_inp_x).any()}")
        
        sin_emb_x = torch.sin(sinusoid_inp_x)
        cos_emb_x = torch.cos(sinusoid_inp_x)
        print(f"🔍 Sin X has NaN: {torch.isnan(sin_emb_x).any()}")
        print(f"🔍 Cos X has NaN: {torch.isnan(cos_emb_x).any()}")

def debug_generalizable_encoder(encoder, coords, time_indices):
    """Debug GeneralizablePositionalEncoding internals."""
    print(f"\n🔬 Debugging GeneralizablePositionalEncoding internals...")
    
    with torch.no_grad():
        batch_size, seq_len = time_indices.shape
        
        # Coordinate scaling
        scaled_coords = coords * encoder.spatial_scale
        print(f"📊 Scaled coords: {scaled_coords.mean():.6f} ± {scaled_coords.std():.6f}")
        print(f"🔍 Scaled coords has NaN: {torch.isnan(scaled_coords).any()}")
        
        # Time normalization
        max_time = time_indices.max()
        if max_time > 0:
            time_normalized = time_indices.float() / max_time
        else:
            time_normalized = time_indices.float()
        
        print(f"📊 Time normalized: {time_normalized.mean():.6f} ± {time_normalized.std():.6f}")
        print(f"🔍 Time normalized has NaN: {torch.isnan(time_normalized).any()}")
        
        # Spatial encoding
        spatial_features = encoder.spatial_encoder(scaled_coords)
        print(f"📊 Spatial features: {spatial_features.shape}")
        print(f"📊 Spatial features stats: mean={spatial_features.mean():.6f}, std={spatial_features.std():.6f}")
        print(f"🔍 Spatial features has NaN: {torch.isnan(spatial_features).any()}")
        
        # Temporal encoding
        if encoder.use_fourier_features:
            time_expanded = time_normalized.unsqueeze(-1)
            freq_input = time_expanded * encoder.temporal_freqs.unsqueeze(0).unsqueeze(0)
            phase_input = freq_input + encoder.temporal_phases.unsqueeze(0).unsqueeze(0)
            
            print(f"📊 Temporal freqs: {encoder.temporal_freqs}")
            print(f"📊 Temporal phases: {encoder.temporal_phases}")
            print(f"📊 Freq input: {freq_input.mean():.6f} ± {freq_input.std():.6f}")
            print(f"🔍 Freq input has NaN: {torch.isnan(freq_input).any()}")
            
            sin_features = torch.sin(phase_input)
            cos_features = torch.cos(phase_input)
            
            print(f"🔍 Sin features has NaN: {torch.isnan(sin_features).any()}")
            print(f"🔍 Cos features has NaN: {torch.isnan(cos_features).any()}")

def main():
    """Run debug tests for all positional encoders."""
    print("🔍 Positional Encoding Debug Suite")
    print("=" * 70)
    
    encoding_types = ['advanced_3d', 'rope_3d', 'generalizable']
    results = {}
    
    for encoding_type in encoding_types:
        success = debug_positional_encoder(encoding_type)
        results[encoding_type] = success
    
    print("\n" + "=" * 70)
    print("📊 DEBUG SUMMARY")
    print("=" * 70)
    
    for encoding_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{encoding_type:>15}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All positional encoders working correctly!")
    else:
        print("\n❌ Some positional encoders have issues!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 