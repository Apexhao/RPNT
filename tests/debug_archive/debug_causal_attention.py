#!/usr/bin/env python3
"""
Debug CausalAdaptiveKernelAttention NaN Issues

This script systematically tests each component of the CausalAdaptiveKernelAttention
to identify exactly where NaN values originate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.attention import CausalAdaptiveKernelAttention, create_causal_mask

def check_for_nan_inf(tensor, name):
    """Check tensor for NaN/Inf and print statistics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    mean_val = tensor.mean().item() if not (has_nan or has_inf) else float('nan')
    std_val = tensor.std().item() if not (has_nan or has_inf) else float('nan')
    
    status = "❌" if (has_nan or has_inf) else "✅"
    print(f"  {status} {name}: shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, NaN={has_nan}, Inf={has_inf}")
    
    if has_nan or has_inf:
        # Find where NaN/Inf occur
        nan_positions = torch.isnan(tensor).nonzero()
        inf_positions = torch.isinf(tensor).nonzero()
        if len(nan_positions) > 0:
            print(f"    NaN positions (first 5): {nan_positions[:5]}")
        if len(inf_positions) > 0:
            print(f"    Inf positions (first 5): {inf_positions[:5]}")
    
    return has_nan or has_inf

def test_individual_components():
    """Test each component of CausalAdaptiveKernelAttention individually."""
    print("🔬 Testing Individual Components of CausalAdaptiveKernelAttention")
    print("=" * 70)
    
    # Configuration
    batch_size, seq_len, dim = 2, 50, 128
    heads = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data with very small magnitude
    x = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    historical_data = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    
    print(f"📊 Input Configuration:")
    print(f"  Batch size: {batch_size}, Seq len: {seq_len}, Dim: {dim}, Heads: {heads}")
    print(f"  Device: {device}")
    
    # Check input data
    print(f"\n🔍 Input Data Check:")
    input_issues = check_for_nan_inf(x, "Input x")
    hist_issues = check_for_nan_inf(historical_data, "Historical data")
    
    if input_issues or hist_issues:
        print("❌ Input data has issues!")
        return False
    
    # Create attention module
    print(f"\n🏗️ Creating CausalAdaptiveKernelAttention...")
    try:
        attention = CausalAdaptiveKernelAttention(
            dim=dim,
            heads=heads,
            kernel_size=[3, 3],
            dropout=0.1,
            use_kernel=True
        ).to(device)
        print("✅ Module created successfully")
    except Exception as e:
        print(f"❌ Failed to create module: {e}")
        return False
    
    # Test 1: Linear projections (Q, K, V)
    print(f"\n🧪 Test 1: Linear Projections (Q, K, V)")
    try:
        with torch.no_grad():
            q = attention.to_q(x)
            k = attention.to_k(x) 
            v = attention.to_v(x)
            
            q_issues = check_for_nan_inf(q, "Q projection")
            k_issues = check_for_nan_inf(k, "K projection")
            v_issues = check_for_nan_inf(v, "V projection")
            
            if q_issues or k_issues or v_issues:
                print("❌ Linear projections have issues!")
                return False
            else:
                print("✅ Linear projections OK")
                
    except Exception as e:
        print(f"❌ Linear projections failed: {e}")
        return False
    
    # Test 2: Historical data embedding MLP
    print(f"\n🧪 Test 2: Historical Data Embedding MLP")
    try:
        with torch.no_grad():
            embedded_history = attention.history_mlp(historical_data)
            
            if check_for_nan_inf(embedded_history, "Embedded history"):
                print("❌ History MLP has issues!")
                # Check each layer individually
                print("  Debugging history MLP layers:")
                temp = historical_data
                for i, layer in enumerate(attention.history_mlp):
                    temp = layer(temp)
                    check_for_nan_inf(temp, f"    Layer {i} ({type(layer).__name__})")
                return False
            else:
                print("✅ History MLP OK")
                
    except Exception as e:
        print(f"❌ History MLP failed: {e}")
        return False
    
    # Test 3: Context attention pooling
    print(f"\n🧪 Test 3: Context Attention Pooling")
    try:
        with torch.no_grad():
            # Use the embedded history from previous test
            attention_weights = attention.context_attention(embedded_history)
            
            if check_for_nan_inf(attention_weights, "Attention weights (raw)"):
                print("❌ Context attention weights have issues!")
                return False
            
            # Apply softmax
            attention_weights_norm = F.softmax(attention_weights, dim=1)
            
            if check_for_nan_inf(attention_weights_norm, "Attention weights (normalized)"):
                print("❌ Normalized attention weights have issues!")
                return False
            
            # Compute weighted sum
            causal_context = (embedded_history * attention_weights_norm).sum(dim=1)
            
            if check_for_nan_inf(causal_context, "Causal context"):
                print("❌ Causal context has issues!")
                return False
            else:
                print("✅ Context attention pooling OK")
                
    except Exception as e:
        print(f"❌ Context attention pooling failed: {e}")
        return False
    
    # Test 4: Kernel generation
    print(f"\n🧪 Test 4: Kernel Generation")
    try:
        with torch.no_grad():
            # Use the causal context from previous test
            kernel_params = attention.kernel_generator(causal_context)
            
            if check_for_nan_inf(kernel_params, "Kernel params (raw)"):
                print("❌ Raw kernel params have issues!")
                # Debug kernel generator layers
                print("  Debugging kernel generator layers:")
                temp = causal_context
                for i, layer in enumerate(attention.kernel_generator):
                    temp = layer(temp)
                    check_for_nan_inf(temp, f"    Layer {i} ({type(layer).__name__})")
                return False
            
            # Reshape kernels
            B, H = batch_size, heads
            K_1, K_2 = 3, 3
            kernels = kernel_params.view(B, H, K_1, K_2)
            
            if check_for_nan_inf(kernels, "Kernels (reshaped)"):
                print("❌ Reshaped kernels have issues!")
                return False
            
            # Normalize kernels
            kernels_flat = kernels.view(B, H, -1)
            kernels_norm_flat = F.softmax(kernels_flat, dim=-1)
            kernels_norm = kernels_norm_flat.view(B, H, K_1, K_2)
            
            if check_for_nan_inf(kernels_norm, "Kernels (normalized)"):
                print("❌ Normalized kernels have issues!")
                return False
            else:
                print("✅ Kernel generation OK")
                
    except Exception as e:
        print(f"❌ Kernel generation failed: {e}")
        return False
    
    # Test 5: Attention computation (before kernel application)
    print(f"\n🧪 Test 5: Attention Computation (before kernel)")
    try:
        with torch.no_grad():
            # Reshape Q, K for attention computation
            q_reshaped = q.view(batch_size, seq_len, heads, dim//heads).transpose(1, 2)
            k_reshaped = k.view(batch_size, seq_len, heads, dim//heads).transpose(1, 2)
            
            # Compute attention scores
            scale = (dim // heads) ** -0.5
            attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-1, -2)) * scale
            
            if check_for_nan_inf(attn_scores, "Attention scores (raw)"):
                print("❌ Raw attention scores have issues!")
                return False
            
            # Apply causal mask
            causal_mask = create_causal_mask(seq_len, device)
            attn_masked = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            if check_for_nan_inf(attn_masked, "Attention scores (masked)"):
                print("❌ Masked attention scores have issues!")
                return False
            else:
                print("✅ Attention computation (before kernel) OK")
                
    except Exception as e:
        print(f"❌ Attention computation failed: {e}")
        return False
    
    # Test 6: Kernel application
    print(f"\n🧪 Test 6: Kernel Application")
    try:
        with torch.no_grad():
            # Use normalized kernels from Test 4 and masked attention from Test 5
            
            # Apply kernel using the module's method
            attn_with_kernel = attention.apply_kernel(attn_masked, kernels_norm)
            
            if check_for_nan_inf(attn_with_kernel, "Attention with kernel"):
                print("❌ Kernel application has issues!")
                
                # Debug kernel application step by step
                print("  Debugging kernel application:")
                B, H, N, N = attn_masked.shape
                K_1, K_2 = 3, 3
                pad_1, pad_2 = K_1 // 2, K_2 // 2
                
                # Check if sequence is too short
                if N < max(K_1, K_2):
                    print(f"    Sequence too short: {N} < {max(K_1, K_2)}")
                    return False
                
                # Pad attention matrix
                attn_padded = F.pad(attn_masked.reshape(1, B*H, N, N), 
                                   (pad_2, pad_2, pad_1, pad_1), mode='constant', value=0)
                check_for_nan_inf(attn_padded, "    Padded attention")
                
                # Reshape kernels for conv2d
                kernels_conv = kernels_norm.reshape(B*H, 1, K_1, K_2)
                check_for_nan_inf(kernels_conv, "    Kernels for conv2d")
                
                # Apply convolution
                try:
                    output = F.conv2d(attn_padded, kernels_conv, groups=B*H)
                    check_for_nan_inf(output, "    Conv2d output")
                except Exception as conv_e:
                    print(f"    ❌ Conv2d failed: {conv_e}")
                
                return False
            else:
                print("✅ Kernel application OK")
                
    except Exception as e:
        print(f"❌ Kernel application failed: {e}")
        return False
    
    # Test 7: Final attention softmax and output
    print(f"\n🧪 Test 7: Final Attention Softmax and Output")
    try:
        with torch.no_grad():
            # Apply softmax to attention with kernel
            attn_final = attn_with_kernel.softmax(dim=-1)
            
            if check_for_nan_inf(attn_final, "Final attention (softmax)"):
                print("❌ Final attention softmax has issues!")
                return False
            
            # Apply dropout (but skip in debug mode)
            # attn_final = attention.dropout(attn_final)
            
            # Compute output
            v_reshaped = v.view(batch_size, seq_len, heads, dim//heads).transpose(1, 2)
            out = torch.matmul(attn_final, v_reshaped)
            
            if check_for_nan_inf(out, "Attention output (before reshape)"):
                print("❌ Attention output has issues!")
                return False
            
            # Reshape and project
            out_reshaped = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            out_final = attention.proj(out_reshaped)
            
            if check_for_nan_inf(out_final, "Final output"):
                print("❌ Final output has issues!")
                return False
            else:
                print("✅ Final attention and output OK")
                
    except Exception as e:
        print(f"❌ Final attention computation failed: {e}")
        return False
    
    print(f"\n✅ All individual components passed!")
    return True

def test_full_forward_pass():
    """Test the full forward pass to see where NaN emerges."""
    print(f"\n🔬 Testing Full Forward Pass")
    print("=" * 70)
    
    # Configuration
    batch_size, seq_len, dim = 2, 50, 128
    heads = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    historical_data = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
    causal_mask = create_causal_mask(seq_len, device)
    
    # Test with kernels enabled
    print(f"\n🧪 Testing with kernels ENABLED...")
    try:
        attention_with_kernel = CausalAdaptiveKernelAttention(
            dim=dim, heads=heads, use_kernel=True
        ).to(device)
        
        with torch.no_grad():
            output_with_kernel = attention_with_kernel(x, historical_data, causal_mask)
            
            if check_for_nan_inf(output_with_kernel, "Output (with kernel)"):
                print("❌ Full forward pass with kernels has NaN!")
                return False
            else:
                print("✅ Full forward pass with kernels OK")
                
    except Exception as e:
        print(f"❌ Full forward pass with kernels failed: {e}")
        return False
    
    # Test with kernels disabled
    print(f"\n🧪 Testing with kernels DISABLED...")
    try:
        attention_no_kernel = CausalAdaptiveKernelAttention(
            dim=dim, heads=heads, use_kernel=False
        ).to(device)
        
        with torch.no_grad():
            output_no_kernel = attention_no_kernel(x, historical_data, causal_mask)
            
            if check_for_nan_inf(output_no_kernel, "Output (no kernel)"):
                print("❌ Full forward pass without kernels has NaN!")
                return False
            else:
                print("✅ Full forward pass without kernels OK")
                
    except Exception as e:
        print(f"❌ Full forward pass without kernels failed: {e}")
        return False
    
    return True

def test_parameter_sensitivity():
    """Test sensitivity to different input magnitudes and parameters."""
    print(f"\n🔬 Testing Parameter Sensitivity")
    print("=" * 70)
    
    # Configuration
    batch_size, seq_len, dim = 2, 20, 64  # Smaller for sensitivity testing
    heads = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different input magnitudes
    magnitudes = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    for mag in magnitudes:
        print(f"\n🧪 Testing input magnitude: {mag}")
        
        x = torch.randn(batch_size, seq_len, dim, device=device) * mag
        historical_data = torch.randn(batch_size, seq_len, dim, device=device) * mag
        causal_mask = create_causal_mask(seq_len, device)
        
        try:
            attention = CausalAdaptiveKernelAttention(
                dim=dim, heads=heads, use_kernel=True
            ).to(device)
            
            with torch.no_grad():
                output = attention(x, historical_data, causal_mask)
                
                has_issues = check_for_nan_inf(output, f"Output (mag={mag})")
                if has_issues:
                    print(f"❌ Magnitude {mag} causes issues!")
                    break
                else:
                    print(f"✅ Magnitude {mag} OK")
                    
        except Exception as e:
            print(f"❌ Magnitude {mag} failed: {e}")
            break
    
    return True

def run_causal_attention_debug():
    """Run all debugging tests."""
    print("🐛 CausalAdaptiveKernelAttention Debugging Suite")
    print("=" * 70)
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    tests = [
        ("Individual Components", test_individual_components),
        ("Full Forward Pass", test_full_forward_pass),
        ("Parameter Sensitivity", test_parameter_sensitivity)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n" + "=" * 70)
            success = test_func()
            results[test_name] = success
            
            if not success:
                print(f"\n❌ {test_name} FAILED - stopping here for analysis")
                break
                
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            break
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 DEBUGGING SUMMARY")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:>20}: {status}")
    
    print("=" * 70)
    
    return all(results.values())

if __name__ == "__main__":
    success = run_causal_attention_debug()
    exit(0 if success else 1) 