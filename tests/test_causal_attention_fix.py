#!/usr/bin/env python3
"""
Test CausalAdaptiveKernelAttention Fix

Quick test to verify that the fix for temporal kernels resolves NaN issues.
"""

import torch
import sys
import os

# Clean imports - no sys.path manipulation needed! ✨

from models.attention import CausalAdaptiveKernelAttention, create_causal_mask

def check_for_nan_inf(tensor, name):
    """Check tensor for NaN/Inf and print statistics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    mean_val = tensor.mean().item() if not (has_nan or has_inf) else float('nan')
    std_val = tensor.std().item() if not (has_nan or has_inf) else float('nan')
    
    status = "❌" if (has_nan or has_inf) else "✅"
    print(f"  {status} {name}: shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, NaN={has_nan}, Inf={has_inf}")
    
    return has_nan or has_inf

def test_causal_attention_fix():
    """Test the fixed CausalAdaptiveKernelAttention."""
    print("🧪 Testing Fixed CausalAdaptiveKernelAttention")
    print("=" * 60)
    
    # Configuration
    batch_size, seq_len, dim = 2, 50, 128
    heads = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📊 Configuration: B={batch_size}, T={seq_len}, D={dim}, H={heads}")
    print(f"🖥️  Device: {device}")
    
    # Create test data with small magnitude
    x = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    historical_data = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    causal_mask = create_causal_mask(seq_len, device)
    
    # Test with kernels ENABLED (previously caused NaN)
    print(f"\n🧪 Testing with temporal kernels ENABLED...")
    
    try:
        attention_with_kernel = CausalAdaptiveKernelAttention(
            dim=dim,
            heads=heads,
            kernel_size=[3, 3],
            dropout=0.1,
            use_kernel=True
        ).to(device)
        
        with torch.no_grad():
            output_with_kernel = attention_with_kernel(x, historical_data, causal_mask)
            
            has_issues = check_for_nan_inf(output_with_kernel, "Output (with kernels)")
            
            if has_issues:
                print("❌ FAILED: Still has NaN/Inf issues!")
                return False
            else:
                print("✅ SUCCESS: No NaN/Inf issues!")
                
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        return False
    
    # Test with kernels DISABLED (should still work)
    print(f"\n🧪 Testing with temporal kernels DISABLED...")
    
    try:
        attention_no_kernel = CausalAdaptiveKernelAttention(
            dim=dim,
            heads=heads,
            kernel_size=[3, 3],
            dropout=0.1,
            use_kernel=False
        ).to(device)
        
        with torch.no_grad():
            output_no_kernel = attention_no_kernel(x, historical_data, causal_mask)
            
            has_issues = check_for_nan_inf(output_no_kernel, "Output (no kernels)")
            
            if has_issues:
                print("❌ FAILED: No-kernel version has issues!")
                return False
            else:
                print("✅ SUCCESS: No-kernel version works!")
                
    except Exception as e:
        print(f"❌ FAILED: Exception in no-kernel version: {e}")
        return False
    
    # Test different input magnitudes to verify robustness
    print(f"\n🧪 Testing different input magnitudes...")
    
    magnitudes = [0.001, 0.01, 0.1, 1.0]
    
    for mag in magnitudes:
        x_mag = torch.randn(batch_size, seq_len, dim, device=device) * mag
        hist_mag = torch.randn(batch_size, seq_len, dim, device=device) * mag
        
        try:
            with torch.no_grad():
                output_mag = attention_with_kernel(x_mag, hist_mag, causal_mask)
                
                has_issues = check_for_nan_inf(output_mag, f"Output (mag={mag})")
                
                if has_issues:
                    print(f"❌ FAILED: Magnitude {mag} causes issues!")
                    return False
                    
        except Exception as e:
            print(f"❌ FAILED: Magnitude {mag} caused exception: {e}")
            return False
    
    print(f"\n✅ All tests passed! The fix works!")
    return True

if __name__ == "__main__":
    success = test_causal_attention_fix()
    if success:
        print("\n🎉 CausalAdaptiveKernelAttention fix is successful!")
        print("🔧 Temporal kernels now work without NaN issues")
    else:
        print("\n❌ Fix did not resolve the issues")
    
    exit(0 if success else 1) 