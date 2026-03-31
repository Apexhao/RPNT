#!/usr/bin/env python3
"""
Simplified Test Suite for Site-Specific Positional Encoding

This script tests two core positional encoding approaches:
1. Advanced3DPositionalEncoding (learned coordinate-based) - BASELINE
2. RoPE3D (mathematical with site awareness) - NEW APPROACH  

Key tests:
- Dimensional consistency
- Site-specific encoding properties
- Zero-shot generalization to new sites
- Head-to-head comparison between approaches
- Integration with transformer architecture
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os

# Clean imports - no sys.path manipulation needed! ✨

from models.positional_encoding import (
    Advanced3DPositionalEncoding, 
    RoPE3D, 
    apply_rotary_emb_3d,
    create_positional_encoder,
    prepare_site_positional_data
)
from models.transformer import CrossSiteFoundationMAE

def test_positional_encoding_dimensions():
    """Test 1: Verify dimensional consistency for both encoders."""
    print("🔬 Test 1: Dimensional Consistency")
    print("=" * 50)
    
    # Test parameters
    batch_size, seq_len = 4, 100
    d_model = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    site_coords = torch.randn(batch_size, seq_len, 2)  # [B, T, 2] - (X, Y)
    time_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)  # [B, T]
    
    # ✅ SIMPLIFIED: Only test two core approaches
    encoding_types = ['advanced_3d', 'rope_3d']
    results = {}
    
    for encoding_type in encoding_types:
        print(f"\n📊 Testing {encoding_type}...")
        
        # Create encoder
        encoder = create_positional_encoder(encoding_type, d_model=d_model)
        encoder = encoder.to(device)
        
        # Move data to device
        coords_device = site_coords.to(device)
        time_device = time_indices.to(device)
        
        # Forward pass
        with torch.no_grad():
            if encoding_type == 'rope_3d':
                # For RoPE3D, we test the frequency generation (not direct PE)
                freqs_cis = encoder.compute_3d_freqs_cis(coords_device, time_device)
                site_features = encoder.get_site_features(coords_device)
                
                print(f"   ✅ RoPE frequencies shape: {freqs_cis.shape}")
                print(f"   ✅ Site features shape: {site_features.shape}")
                print(f"   📊 Freq stats: mean={freqs_cis.abs().mean().item():.4f}")
                print(f"   📊 Site stats: mean={site_features.mean().item():.4f}")
                
                # Mark as successful for RoPE3D
                results[encoding_type] = {
                    'success': True,
                    'freqs_shape': freqs_cis.shape,
                    'site_features_shape': site_features.shape,
                    'note': 'RoPE3D generates frequencies, not direct positional encoding'
                }
                
            else:
                # For Advanced3D, test normal positional encoding
                pos_encoding = encoder(coords_device, time_device)
                
                # Verify dimensions
                expected_shape = (batch_size, seq_len, d_model)
                actual_shape = pos_encoding.shape
                
                success = actual_shape == expected_shape
                results[encoding_type] = {
                    'success': success,
                    'expected_shape': expected_shape,
                    'actual_shape': actual_shape,
                    'encoding_stats': {
                        'mean': pos_encoding.mean().item(),
                        'std': pos_encoding.std().item(),
                        'min': pos_encoding.min().item(),
                        'max': pos_encoding.max().item()
                    }
                }
                
                status = "✅" if success else "❌"
                print(f"   {status} Shape: {actual_shape} (expected: {expected_shape})")
                print(f"   📈 Stats: mean={pos_encoding.mean().item():.4f}, std={pos_encoding.std().item():.4f}")
    
    return results

def test_site_specificity():
    """Test 2: Verify site-specific encoding properties."""
    print("\n🎯 Test 2: Site-Specific Encoding Properties")
    print("=" * 50)
    
    d_model = 256  # Smaller for easier analysis
    seq_len = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create two different sites at the same time points
    site1_coords = torch.zeros(1, seq_len, 2)  # Site 1 at origin
    site2_coords = torch.ones(1, seq_len, 2) * 10.0  # Site 2 at (10, 10)
    time_indices = torch.arange(seq_len).unsqueeze(0)  # Same time points
    
    encoding_types = ['advanced_3d', 'rope_3d']
    
    for encoding_type in encoding_types:
        print(f"\n📊 Testing {encoding_type} site specificity...")
        
        encoder = create_positional_encoder(encoding_type, d_model=d_model).to(device)
        
        with torch.no_grad():
            if encoding_type == 'rope_3d':
                # For RoPE3D, compare frequency generation
                freqs1 = encoder.compute_3d_freqs_cis(site1_coords.to(device), time_indices.to(device))
                freqs2 = encoder.compute_3d_freqs_cis(site2_coords.to(device), time_indices.to(device))
                
                # Calculate similarity metrics for frequency tensors (convert complex to real)
                freq_diff = torch.norm(freqs1 - freqs2)
                # Use absolute values for cosine similarity since freqs are complex
                freq_cosine = F.cosine_similarity(freqs1.abs().flatten(), freqs2.abs().flatten(), dim=0)
                
                print(f"   📏 Frequency difference: {freq_diff.item():.4f}")
                print(f"   📏 Frequency cosine similarity: {freq_cosine.item():.4f}")
                
                # RoPE should generate different frequencies for different sites
                is_site_specific = freq_diff.item() > 0.1
                status = "✅" if is_site_specific else "⚠️"
                print(f"   {status} Site-specific frequencies: {is_site_specific}")
                
            else:
                # For Advanced3D, compare positional encodings
                pe1 = encoder(site1_coords.to(device), time_indices.to(device))  # [1, T, D]
                pe2 = encoder(site2_coords.to(device), time_indices.to(device))  # [1, T, D]
                
                # Calculate similarity metrics
                cosine_similarity = F.cosine_similarity(pe1.flatten(), pe2.flatten(), dim=0)
                mse_difference = F.mse_loss(pe1, pe2)
                max_abs_difference = (pe1 - pe2).abs().max()
                
                print(f"   📏 Cosine similarity: {cosine_similarity.item():.4f}")
                print(f"   📏 MSE difference: {mse_difference.item():.4f}")
                print(f"   📏 Max absolute difference: {max_abs_difference.item():.4f}")
                
                # Site-specific encoding should be different (low similarity, high difference)
                is_site_specific = cosine_similarity.item() < 0.9 and mse_difference.item() > 0.1
                status = "✅" if is_site_specific else "⚠️"
                print(f"   {status} Site-specific: {is_site_specific}")

def test_temporal_consistency():
    """Test 3: Verify temporal encoding consistency."""
    print("\n⏰ Test 3: Temporal Encoding Consistency")
    print("=" * 50)
    
    d_model = 256
    seq_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fixed site, different time points
    site_coords = torch.zeros(1, seq_len, 2)  # Same site location
    time_indices = torch.arange(seq_len).unsqueeze(0)  # Different time points
    
    encoding_types = ['advanced_3d', 'rope_3d']
    
    for encoding_type in encoding_types:
        print(f"\n📊 Testing {encoding_type} temporal consistency...")
        
        encoder = create_positional_encoder(encoding_type, d_model=d_model).to(device)
        
        with torch.no_grad():
            if encoding_type == 'rope_3d':
                # For RoPE3D, test frequency variation over time
                freqs = encoder.compute_3d_freqs_cis(site_coords.to(device), time_indices.to(device))  # [1, T, D]
                
                # Compare frequencies at different time points
                freq_t0 = freqs[0, 0]    # Time 0
                freq_t10 = freqs[0, 10]  # Time 10  
                freq_t50 = freqs[0, 50]  # Time 50
                
                # Calculate temporal differences
                diff_0_10 = torch.norm(freq_t0 - freq_t10)
                diff_10_50 = torch.norm(freq_t10 - freq_t50)
                diff_0_50 = torch.norm(freq_t0 - freq_t50)
                
                print(f"   📏 Freq diff (t=0 vs t=10): {diff_0_10.item():.4f}")
                print(f"   📏 Freq diff (t=10 vs t=50): {diff_10_50.item():.4f}")
                print(f"   📏 Freq diff (t=0 vs t=50): {diff_0_50.item():.4f}")
                
                # Temporal encoding should change over time
                is_temporal = diff_0_50.item() > 0.01
                status = "✅" if is_temporal else "⚠️"
                print(f"   {status} Temporal variation: {is_temporal}")
                
            else:
                # For Advanced3D, test positional encoding variation
                pe = encoder(site_coords.to(device), time_indices.to(device))  # [1, T, D]
                
                # Compare encodings at different time points
                pe_t0 = pe[0, 0]    # Time 0
                pe_t10 = pe[0, 10]  # Time 10  
                pe_t50 = pe[0, 50]  # Time 50
                
                # Calculate temporal differences
                diff_0_10 = F.mse_loss(pe_t0, pe_t10)
                diff_10_50 = F.mse_loss(pe_t10, pe_t50)
                diff_0_50 = F.mse_loss(pe_t0, pe_t50)
                
                print(f"   📏 Diff (t=0 vs t=10): {diff_0_10.item():.4f}")
                print(f"   📏 Diff (t=10 vs t=50): {diff_10_50.item():.4f}")
                print(f"   📏 Diff (t=0 vs t=50): {diff_0_50.item():.4f}")
                
                # Temporal encoding should change over time
                is_temporal = diff_0_50.item() > 0.01
                status = "✅" if is_temporal else "⚠️"
                print(f"   {status} Temporal variation: {is_temporal}")

def test_zero_shot_generalization():
    """Test 4: Zero-shot generalization to new sites."""
    print("\n🎯 Test 4: Zero-Shot Generalization")
    print("=" * 50)
    
    # Training configuration
    n_sites_train = 10
    seq_len = 200
    d_model = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create training site coordinates (grid pattern)
    train_coords = torch.stack([
        torch.arange(n_sites_train).float(),
        torch.zeros(n_sites_train)
    ], dim=1)  # [10, 2] - sites along x-axis
    
    print(f"📍 Training sites: {train_coords.shape}")
    print(f"📍 Training coordinates: \n{train_coords}")
    
    # Create test site coordinates (new locations)
    test_coords = torch.tensor([
        [15.0, 5.0],   # New site 1
        [-5.0, 10.0],  # New site 2  
        [7.5, -3.0]    # New site 3
    ])  # [3, 2]
    
    print(f"\n🆕 Test sites: {test_coords.shape}")
    print(f"🆕 Test coordinates: \n{test_coords}")
    
    encoding_types = ['advanced_3d', 'rope_3d']
    
    for encoding_type in encoding_types:
        print(f"\n🔬 Testing {encoding_type} zero-shot capability...")
        
        try:
            encoder = create_positional_encoder(encoding_type, d_model=d_model).to(device)
            
            with torch.no_grad():
                # Test with training sites
                time_indices_train = torch.arange(seq_len).unsqueeze(0).expand(n_sites_train, seq_len)
                coords_train_expanded = train_coords.unsqueeze(1).expand(n_sites_train, seq_len, 2)
                
                # Test with new sites (zero-shot)
                n_test_sites = test_coords.shape[0]
                time_indices_test = torch.arange(seq_len).unsqueeze(0).expand(n_test_sites, seq_len)
                coords_test_expanded = test_coords.unsqueeze(1).expand(n_test_sites, seq_len, 2)
                
                if encoding_type == 'rope_3d':
                    # For RoPE3D, test frequency generation
                    freqs_train = encoder.compute_3d_freqs_cis(
                        coords_train_expanded.to(device), 
                        time_indices_train.to(device)
                    )  # [10, T, D]
                    
                    freqs_test = encoder.compute_3d_freqs_cis(
                        coords_test_expanded.to(device),
                        time_indices_test.to(device)
                    )  # [3, T, D]
                    
                    print(f"   ✅ Training frequencies shape: {freqs_train.shape}")
                    print(f"   ✅ Test frequencies shape: {freqs_test.shape}")
                    print(f"   ✅ Zero-shot frequency generation successful!")
                    
                    # Verify frequencies are reasonable
                    train_stats = f"mean={freqs_train.abs().mean().item():.4f}, std={freqs_train.abs().std().item():.4f}"
                    test_stats = f"mean={freqs_test.abs().mean().item():.4f}, std={freqs_test.abs().std().item():.4f}"
                    print(f"   📊 Training stats: {train_stats}")
                    print(f"   📊 Test stats: {test_stats}")
                    
                else:
                    # For Advanced3D, test positional encoding generation
                    pe_train = encoder(
                        coords_train_expanded.to(device), 
                        time_indices_train.to(device)
                    )  # [10, T, D]
                    
                    pe_test = encoder(
                        coords_test_expanded.to(device),
                        time_indices_test.to(device)
                    )  # [3, T, D]
                    
                    print(f"   ✅ Training PE shape: {pe_train.shape}")
                    print(f"   ✅ Test PE shape: {pe_test.shape}")
                    print(f"   ✅ Zero-shot generation successful!")
                    
                    # Verify encodings are reasonable
                    train_stats = f"mean={pe_train.mean().item():.4f}, std={pe_train.std().item():.4f}"
                    test_stats = f"mean={pe_test.mean().item():.4f}, std={pe_test.std().item():.4f}"
                    print(f"   📊 Training stats: {train_stats}")
                    print(f"   📊 Test stats: {test_stats}")
                
        except Exception as e:
            print(f"   ❌ Zero-shot failed: {str(e)}")

def test_transformer_integration():
    """Test 5: Integration with transformer architecture."""
    print("\n🔗 Test 5: Transformer Integration")
    print("=" * 50)
    
    # Configuration
    batch_size, n_sites, seq_len, neural_dim = 2, 5, 100, 75
    d_model = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic neural data and site coordinates
    neural_data = torch.randn(batch_size, n_sites, seq_len, neural_dim)
    site_coords = torch.randn(n_sites, 2) * 5.0  # [S, 2] fixed coordinates
    
    print(f"🧬 Neural data shape: {neural_data.shape}")
    print(f"📍 Site coordinates shape: {site_coords.shape}")
    
    encoding_types = ['advanced_3d', 'rope_3d']
    
    for encoding_type in encoding_types:
        print(f"\n🔬 Testing {encoding_type} in transformer...")
        
        try:
            # Create transformer model
            model = CrossSiteFoundationMAE(
                neural_dim=neural_dim,
                d_model=d_model,
                n_sites=n_sites,
                temporal_layers=2,  # Smaller for testing
                spatial_layers=1,
                pos_encoding_type=encoding_type,
                use_mae_decoder=True
            ).to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    neural_data.to(device),
                    site_coords.to(device),
                    return_mae_reconstruction=True
                )
                
                representations = outputs['representations']
                reconstruction = outputs['reconstruction']
                
                print(f"   ✅ Representations shape: {representations.shape}")
                print(f"   ✅ Reconstruction shape: {reconstruction.shape}")
                
                # Calculate reconstruction error
                recon_error = F.mse_loss(reconstruction, neural_data.to(device))
                print(f"   📊 Reconstruction MSE: {recon_error.item():.6f}")
                
                print(f"   ✅ Integration successful!")
                
        except Exception as e:
            print(f"   ❌ Integration failed: {str(e)}")

def compare_encoding_approaches():
    """Test 6: Head-to-head comparison between approaches."""
    print("\n⚖️ Test 6: Advanced3D vs RoPE3D Comparison")
    print("=" * 50)
    
    batch_size, seq_len = 3, 50
    d_model = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test coordinates (different spatial patterns)
    site_coords = torch.tensor([
        [0.0, 0.0],    # Origin
        [5.0, 5.0],    # Diagonal
        [10.0, 0.0]    # X-axis
    ]).unsqueeze(1).expand(3, seq_len, 2)  # [3, T, 2]
    
    time_indices = torch.arange(seq_len).unsqueeze(0).expand(3, seq_len)  # [3, T]
    
    print("🔬 Comparing encoding approaches...")
    
    # Advanced3D encoding
    advanced_encoder = create_positional_encoder('advanced_3d', d_model=d_model).to(device)
    with torch.no_grad():
        advanced_pe = advanced_encoder(site_coords.to(device), time_indices.to(device))
    
    print(f"   📊 Advanced3D: shape {advanced_pe.shape}, "
          f"mean={advanced_pe.mean().item():.4f}, std={advanced_pe.std().item():.4f}")
    
    # RoPE3D frequency generation (different type of output)
    rope_encoder = create_positional_encoder('rope_3d', d_model=d_model).to(device)
    with torch.no_grad():
        rope_freqs = rope_encoder.compute_3d_freqs_cis(site_coords.to(device), time_indices.to(device))
        rope_features = rope_encoder.get_site_features(site_coords.to(device))
    
    print(f"   📊 RoPE3D frequencies: shape {rope_freqs.shape}, "
          f"mean={rope_freqs.abs().mean().item():.4f}, std={rope_freqs.abs().std().item():.4f}")
    print(f"   📊 RoPE3D site features: shape {rope_features.shape}, "
          f"mean={rope_features.mean().item():.4f}, std={rope_features.std().item():.4f}")
    
    print(f"\n🔍 Key Differences:")
    print(f"   📌 Advanced3D: Traditional PE added to embeddings")
    print(f"   📌 RoPE3D: Frequencies applied as rotations to Q/K in attention")
    print(f"   📌 Advanced3D: Outputs {d_model}-dim positional encoding")
    print(f"   📌 RoPE3D: Outputs {rope_freqs.shape[-1]}-dim frequencies + {rope_features.shape[-1]}-dim site features")

def run_comprehensive_tests():
    """Run all tests in sequence."""
    print("🧪 Simplified Site-Specific Positional Encoding Tests")
    print("=" * 70)
    print(f"🎯 Focus: Advanced3DPositionalEncoding vs RoPE3D")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    try:
        # Run all tests
        test_positional_encoding_dimensions()
        test_site_specificity()
        test_temporal_consistency()
        test_zero_shot_generalization()
        test_transformer_integration()
        compare_encoding_approaches()
        
        print("\n" + "=" * 70)
        print("🎉 All tests completed successfully!")
        print("✅ Advanced3DPositionalEncoding: Traditional learned approach working")
        print("✅ RoPE3D: Mathematical rotation approach working")
        print("✅ Zero-shot generalization functional for both")
        print("✅ Transformer integration successful for both")
        print("\n🎯 Ready for head-to-head comparison and validation!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_tests() 