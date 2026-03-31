#!/usr/bin/env python3
"""
Test Script for Connectivity Analysis

This script tests the connectivity analysis pipeline with the pretrained model.
Run this to verify everything works before generating publication figures.

Usage:
    python test_connectivity_analysis.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.visualization.run_analysis import run_connectivity_analysis, setup_logging
from src.utils.helpers import set_seed


def test_connectivity_analysis():
    """Test the connectivity analysis with the actual pretrained model."""
    
    print("🧪 Testing Neural Foundation Model Connectivity Analysis")
    print("=" * 70)
    
    # Setup logging
    setup_logging('INFO')
    set_seed(42)
    
    # Configuration
    checkpoint_path = "./logs_neuropixel/RoPE3D_UniformMasking_Kernel_11_11_d_model_384_TLL_4_SL_2_HL_4_h12/checkpoints/best.pth"
    output_dir = "./connectivity_analysis_results"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please verify the checkpoint path and try again.")
        return False
    
    try:
        # Run connectivity analysis
        results = run_connectivity_analysis(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            batch_size=4,  # Small batch for testing
            temporal_timepoints=[0, 15, 30, 45]
        )
        
        print("\n✅ Connectivity Analysis Test Completed Successfully!")
        print(f"📊 Results Summary:")
        print(f"   - Sites analyzed: {len(results.site_ids)}")
        print(f"   - Attention connectivity: {results.attention_connectivity.shape}")
        print(f"   - Temporal dynamics: {results.attention_temporal.shape}")
        print(f"   - Noise connectivity: {results.noise_connectivity.shape}")
        print(f"   - Site coordinates: {results.site_coordinates.shape}")
        
        print(f"\n📁 Output saved to: {output_dir}")
        print(f"   - Results: {output_dir}/results/")
        print(f"   - Figures: {output_dir}/figures/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connectivity_analysis()
    sys.exit(0 if success else 1)
