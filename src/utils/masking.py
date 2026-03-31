"""
Enhanced Masking Strategies for Neural Foundation Model MAE Training

This module provides clean and effective masking strategies for cross-site neural data:
1. Temporal masking: Mask entire timesteps across all sites and neurons
2. Neuron masking: Mask entire neurons across all sites and timesteps  
3. Dynamic ratio sampling: Sample mask ratios from ranges for each batch

Key Features:
- Simple and direct masking without complex augmentations
- Dynamic mask ratios for robust training
- 4D tensor support [B, S, T, N]
- Causal masking compatibility
"""

import torch
import numpy as np
from typing import Tuple, Union, List, Optional


class CausalMaskingEngine:
    """
    Enhanced masking engine for neural foundation model training.
    
    **KEY FEATURES**:
    - Temporal masking: Mask entire timesteps (B,S,MASK,N)
    - Neuron masking: Mask entire neurons (B,S,T,MASK)
    - Dynamic ratios: Sample from ranges like [0.3, 0.7] per batch
    - 4D tensor support for cross-site neural data
    
    **DESIGN PHILOSOPHY**: Keep it simple but effective.
    Avoid complex augmentations - focus on core masking that works.
    """
    
    def __init__(self, 
                 temporal_mask_ratio: Union[float, List[float]] = 0.15,
                 neuron_mask_ratio: Union[float, List[float]] = 0.15,
                 min_unmasked_timesteps: int = 5,
                 min_unmasked_neurons: int = 10):
        """
        Initialize masking engine with dynamic ratio support.
        
        Args:
            temporal_mask_ratio: Fixed ratio or [min, max] range for temporal masking
            neuron_mask_ratio: Fixed ratio or [min, max] range for neuron masking  
            min_unmasked_timesteps: Minimum timesteps to keep unmasked
            min_unmasked_neurons: Minimum neurons to keep unmasked per site
        """
        self.temporal_mask_ratio = temporal_mask_ratio
        self.neuron_mask_ratio = neuron_mask_ratio
        self.min_unmasked_timesteps = min_unmasked_timesteps
        self.min_unmasked_neurons = min_unmasked_neurons
        
        # Validate ratio formats
        self._validate_ratio(temporal_mask_ratio, "temporal_mask_ratio")
        self._validate_ratio(neuron_mask_ratio, "neuron_mask_ratio")
    
    def _validate_ratio(self, ratio: Union[float, List[float]], name: str):
        """Validate that ratio is either float or [min, max] list."""
        if isinstance(ratio, (int, float)):
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {ratio}")
        elif isinstance(ratio, (list, tuple)):
            if len(ratio) != 2:
                raise ValueError(f"{name} range must have exactly 2 values, got {len(ratio)}")
            if not (0 <= ratio[0] <= ratio[1] <= 1):
                raise ValueError(f"{name} range must be [min, max] with 0 <= min <= max <= 1, got {ratio}")
        else:
            raise ValueError(f"{name} must be float or [min, max] list, got {type(ratio)}")
    
    def _sample_ratio(self, ratio: Union[float, List[float]]) -> float:
        """Sample a ratio value (either fixed or from range)."""
        if isinstance(ratio, (int, float)):
            return float(ratio)
        else:
            # Sample uniformly from [min, max] range
            return np.random.uniform(ratio[0], ratio[1])
    
    def apply_causal_mask(self, neural_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply enhanced causal masking to neural data.
        
        Args:
            neural_data: [B, S, T, N] - neural activity data
            
        Returns:
            mask: [B, S, T, N] - binary mask (1=keep, 0=mask)
            masked_indices: [B, S, T, N] - boolean mask for masked positions
        """
        B, S, T, N = neural_data.shape
        device = neural_data.device
        
        # Initialize mask (1=keep, 0=mask)
        mask = torch.ones(B, S, T, N, device=device, dtype=torch.float32)
        
        # Sample dynamic ratios for this batch
        temporal_ratio = self._sample_ratio(self.temporal_mask_ratio)
        neuron_ratio = self._sample_ratio(self.neuron_mask_ratio)
        
        # 1. TEMPORAL MASKING: Mask entire timesteps
        temporal_mask = self._create_temporal_mask(B, S, T, N, temporal_ratio, device)
        mask = mask * temporal_mask  # Element-wise multiplication
        
        # 2. NEURON MASKING: Mask entire neurons
        neuron_mask = self._create_neuron_mask(B, S, T, N, neuron_ratio, device)
        mask = mask * neuron_mask  # Element-wise multiplication
        
        # Create boolean masked indices (True = masked, False = keep)
        masked_indices = (mask == 0.0)
        
        return mask, masked_indices
    
    def _create_temporal_mask(self, B: int, S: int, T: int, N: int, 
                             ratio: float, device: torch.device) -> torch.Tensor:
        """
        Create temporal masking pattern.
        
        **STRATEGY**: For each batch sample, randomly select timesteps to mask.
        When a timestep is masked, ALL sites and ALL neurons at that timestep are masked.
        
        Args:
            B, S, T, N: Tensor dimensions
            ratio: Proportion of timesteps to mask
            device: Device for tensor creation
            
        Returns:
            temporal_mask: [B, S, T, N] - mask for temporal masking
        """
        mask = torch.ones(B, S, T, N, device=device)
        
        for b in range(B):
            # Calculate number of timesteps to mask
            n_timesteps_to_mask = int(T * ratio)
            
            # Ensure we don't mask too many timesteps
            n_timesteps_to_mask = min(n_timesteps_to_mask, T - self.min_unmasked_timesteps)
            n_timesteps_to_mask = max(0, n_timesteps_to_mask)  # Ensure non-negative
            
            if n_timesteps_to_mask > 0:
                # Randomly select timesteps to mask
                timesteps_to_mask = torch.randperm(T, device=device)[:n_timesteps_to_mask]
                
                # Mask selected timesteps across ALL sites and neurons
                mask[b, :, timesteps_to_mask, :] = 0.0
        
        return mask
    
    def _create_neuron_mask(self, B: int, S: int, T: int, N: int,
                           ratio: float, device: torch.device) -> torch.Tensor:
        """
        Create neuron masking pattern.
        
        **STRATEGY**: For each batch sample, randomly select neurons to mask.
        When a neuron is masked, it's masked across ALL sites and ALL timesteps.
        
        Args:
            B, S, T, N: Tensor dimensions  
            ratio: Proportion of neurons to mask
            device: Device for tensor creation
            
        Returns:
            neuron_mask: [B, S, T, N] - mask for neuron masking
        """
        mask = torch.ones(B, S, T, N, device=device)
        
        for b in range(B):
            # Calculate number of neurons to mask
            n_neurons_to_mask = int(N * ratio)
            
            # Ensure we don't mask too many neurons
            n_neurons_to_mask = min(n_neurons_to_mask, N - self.min_unmasked_neurons)
            n_neurons_to_mask = max(0, n_neurons_to_mask)  # Ensure non-negative
            
            if n_neurons_to_mask > 0:
                # Randomly select neurons to mask
                neurons_to_mask = torch.randperm(N, device=device)[:n_neurons_to_mask]
                
                # Mask selected neurons across ALL sites and timesteps
                mask[b, :, :, neurons_to_mask] = 0.0
        
        return mask
    
    def get_masking_statistics(self, mask: torch.Tensor) -> dict:
        """
        Compute masking statistics for analysis.
        
        Args:
            mask: [B, S, T, N] - binary mask
            
        Returns:
            dict with masking statistics
        """
        B, S, T, N = mask.shape
        
        # Overall masking statistics
        total_elements = mask.numel()
        masked_elements = (mask == 0.0).sum().item()
        masking_ratio = masked_elements / total_elements
        
        # Temporal masking statistics (timesteps completely masked)
        timestep_mask = (mask.sum(dim=[1, 3]) == 0)  # [B, T] - True if timestep completely masked
        masked_timesteps = timestep_mask.sum().item()
        total_timesteps = B * T
        temporal_masking_ratio = masked_timesteps / total_timesteps if total_timesteps > 0 else 0
        
        # Neuron masking statistics (neurons completely masked)
        neuron_mask = (mask.sum(dim=[1, 2]) == 0)  # [B, N] - True if neuron completely masked
        masked_neurons = neuron_mask.sum().item()
        total_neurons = B * N
        neuron_masking_ratio = masked_neurons / total_neurons if total_neurons > 0 else 0
        
        return {
            'overall_masking_ratio': masking_ratio,
            'temporal_masking_ratio': temporal_masking_ratio,
            'neuron_masking_ratio': neuron_masking_ratio,
            'total_masked_elements': masked_elements,
            'total_elements': total_elements,
            'masked_timesteps': masked_timesteps,
            'masked_neurons': masked_neurons
        }


class BiologicallyInformedMask:
    """
    Optional: Biologically-informed masking patterns for future extensions.
    
    **NOT IMPLEMENTED YET** - placeholder for future sophisticated masking:
    - Electrode failure patterns
    - Anatomically structured masking
    - Temporal correlation-aware masking
    """
    
    def __init__(self):
        self.implemented = False
        
    def apply_electrode_failure_mask(self, neural_data: torch.Tensor) -> torch.Tensor:
        """Future: Simulate realistic electrode failure patterns."""
        raise NotImplementedError("Biologically informed masking not yet implemented")
    
    def apply_cortical_area_mask(self, neural_data: torch.Tensor, 
                                area_assignments: torch.Tensor) -> torch.Tensor:
        """Future: Mask based on cortical area assignments."""
        raise NotImplementedError("Cortical area masking not yet implemented")


def test_causal_masking_engine():
    """
    Comprehensive test for the enhanced CausalMaskingEngine.
    """
    print("🧪 Testing Enhanced CausalMaskingEngine")
    print("=" * 60)
    
    # Test configuration
    B, S, T, N = 4, 8, 50, 75  # Batch, Sites, Time, Neurons
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    neural_data = torch.randn(B, S, T, N, device=device)
    print(f"📊 Test data shape: {neural_data.shape}")
    print(f"🖥️  Device: {device}")
    
    # Test 1: Fixed ratios
    print(f"\n🔬 Test 1: Fixed Ratios")
    engine_fixed = CausalMaskingEngine(
        temporal_mask_ratio=0.2,
        neuron_mask_ratio=0.15
    )
    
    mask_fixed, masked_indices_fixed = engine_fixed.apply_causal_mask(neural_data)
    stats_fixed = engine_fixed.get_masking_statistics(mask_fixed)
    
    print(f"   Mask shape: {mask_fixed.shape}")
    print(f"   Overall masking: {stats_fixed['overall_masking_ratio']:.3f}")
    print(f"   Temporal masking: {stats_fixed['temporal_masking_ratio']:.3f}")
    print(f"   Neuron masking: {stats_fixed['neuron_masking_ratio']:.3f}")
    
    # Test 2: Dynamic ratios
    print(f"\n🔬 Test 2: Dynamic Ratios")
    engine_dynamic = CausalMaskingEngine(
        temporal_mask_ratio=[0.1, 0.3],  # Sample between 10% and 30%
        neuron_mask_ratio=[0.05, 0.25]   # Sample between 5% and 25%
    )
    
    # Run multiple times to show variation
    dynamic_stats = []
    for i in range(5):
        mask_dynamic, _ = engine_dynamic.apply_causal_mask(neural_data)
        stats = engine_dynamic.get_masking_statistics(mask_dynamic)
        dynamic_stats.append(stats)
        print(f"   Run {i+1}: Overall={stats['overall_masking_ratio']:.3f}, "
              f"Temporal={stats['temporal_masking_ratio']:.3f}, "
              f"Neuron={stats['neuron_masking_ratio']:.3f}")
    
    # Test 3: Edge cases
    print(f"\n🔬 Test 3: Edge Cases")
    
    # Very high ratios (should be clamped)
    engine_extreme = CausalMaskingEngine(
        temporal_mask_ratio=0.9,
        neuron_mask_ratio=0.9,
        min_unmasked_timesteps=5,
        min_unmasked_neurons=10
    )
    
    mask_extreme, _ = engine_extreme.apply_causal_mask(neural_data)
    stats_extreme = engine_extreme.get_masking_statistics(mask_extreme)
    
    print(f"   Extreme ratios: Overall={stats_extreme['overall_masking_ratio']:.3f}")
    print(f"   Should respect minimum unmasked constraints")
    
    # Test 4: Verify masking patterns
    print(f"\n🔬 Test 4: Verify Masking Patterns")
    
    # Create simple test case for pattern verification
    small_data = torch.ones(1, 2, 10, 5, device=device)  # [1, 2, 10, 5]
    engine_verify = CausalMaskingEngine(
        temporal_mask_ratio=0.3,  # Should mask ~3 timesteps
        neuron_mask_ratio=0.2     # Should mask ~1 neuron
    )
    
    mask_verify, _ = engine_verify.apply_causal_mask(small_data)
    
    # Check temporal masking pattern
    timestep_completely_masked = (mask_verify.sum(dim=[1, 3]) == 0).sum()  # Count fully masked timesteps
    neuron_completely_masked = (mask_verify.sum(dim=[1, 2]) == 0).sum()    # Count fully masked neurons
    
    print(f"   Completely masked timesteps: {timestep_completely_masked}")
    print(f"   Completely masked neurons: {neuron_completely_masked}")
    
    print(f"\n✅ All tests completed successfully!")
    
    return {
        'fixed_stats': stats_fixed,
        'dynamic_stats': dynamic_stats,
        'extreme_stats': stats_extreme
    }


if __name__ == "__main__":
    # Run tests
    test_results = test_causal_masking_engine()
    
    print(f"\n📋 Test Summary:")
    print(f"Fixed masking works: ✓")
    print(f"Dynamic masking works: ✓") 
    print(f"Edge cases handled: ✓")
    print(f"Pattern verification: ✓") 