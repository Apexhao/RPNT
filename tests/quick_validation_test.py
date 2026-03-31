#!/usr/bin/env python3
"""
Quick validation test for the pretraining system
Tests the complete pipeline with minimal data and 1 epoch
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_validation_test():
    """Run a super quick validation test with minimal data."""
    print("🚀 QUICK VALIDATION TEST")
    print("="*50)
    
    try:
        # Import all required components
        print("📦 Importing components...")
        from src.training.pretrain import PretrainTrainer
        from src.data.cross_site_dataset import CrossSiteMonkeyDataset
        from src.models.transformer import CrossSiteFoundationMAE
        from src.utils.helpers import load_config, set_seed
        
        print("✅ All imports successful!")
        
        # Load and configure settings
        print("⚙️  Loading configuration...")
        config = load_config("config/training/foundation_pretrain.yaml")
        
        # Override for ultra-quick test
        config['training']['model_size'] = 'small'
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['training']['learning_rate'] = 0.001
        config['training']['save_freq'] = 1
        
        # Merge small model config manually
        small_config = config['model_sizes']['small']
        config['model'].update(small_config)
        
        # Override dataset settings for speed
        config['dataset']['target_trials_per_site'] = 50  # Very small
        config['dataset']['exclude_ids'] = ['13122.0']     # Exclude one site
        
        # Setup experiment paths manually
        config['paths']['experiment_name'] = 'quick_validation_test'
        config['paths']['checkpoint_dir'] = './logs/checkpoints/quick_validation_test'
        config['paths']['tensorboard_dir'] = './logs/runs/quick_validation_test'
        
        print(f"✅ Configuration ready!")
        print(f"   Model: {config['training']['model_size']}")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create dataset with minimal settings
        print("📊 Creating dataset (this may take a moment)...")
        dataset = CrossSiteMonkeyDataset(
            target_trials_per_site=50,      # Minimal
            min_val_test_trials=10,         # Minimal
            target_neurons=50,              # Standard
            sample_times=1,                 # No multi-sampling
            split_ratios=(0.6, 0.2, 0.2),  # Standard
            exclude_ids=['13122.0']         # Exclude one site
        )
        
        print(f"✅ Dataset created!")
        print(f"   Sites: {len(dataset.available_sites)}")
        print(f"   Train: {dataset.get_split_data('train').shape}")
        print(f"   Val: {dataset.get_split_data('val').shape}")
        
        # Create model
        print("🏗️  Creating model...")
        model = CrossSiteFoundationMAE(
            neural_dim=config['model']['neural_dim'],
            d_model=config['model']['d_model'],
            temporal_layers=config['model']['temporal_layers'],
            spatial_layers=config['model']['spatial_layers'],
            heads=config['model']['heads'],
            dropout=config['model']['dropout'],
            kernel_size=config['model']['kernel_size'],
            pos_encoding_type=config['model']['pos_encoding_type'],
            use_temporal_kernels=config['model']['use_temporal_kernels'],
            use_site_specific_heads=config['model']['use_site_specific_heads']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created! ({total_params:,} parameters)")
        
        # Create trainer
        print("🏋️  Creating trainer...")
        trainer = PretrainTrainer(
            model=model,
            dataset=dataset,
            config=config
        )
        
        print(f"✅ Trainer created!")
        
        # Test single training step
        print("🔄 Testing single training step...")
        train_metrics = trainer.train_epoch()
        
        print(f"✅ Training step successful!")
        print(f"   Total loss: {train_metrics['total_loss']:.4f}")
        print(f"   Poisson loss: {train_metrics.get('poisson_loss', 'N/A')}")
        print(f"   Contrastive loss: {train_metrics.get('contrastive_loss', 'N/A')}")
        
        # Test validation step
        print("📊 Testing validation step...")
        val_metrics = trainer.validate_epoch()
        
        print(f"✅ Validation step successful!")
        print(f"   Val total loss: {val_metrics['total_loss']:.4f}")
        
        # Test save checkpoint
        print("💾 Testing checkpoint save...")
        trainer.save_checkpoint('quick_test.pth')
        
        print(f"✅ Checkpoint saved!")
        
        # Clean up
        print("🧹 Cleaning up...")
        import shutil
        if os.path.exists('./logs/checkpoints/quick_validation_test'):
            shutil.rmtree('./logs/checkpoints/quick_validation_test')
        if os.path.exists('./logs/runs/quick_validation_test'):
            shutil.rmtree('./logs/runs/quick_validation_test')
        
        print("="*50)
        print("🎉 QUICK VALIDATION TEST PASSED!")
        print("✅ All components working correctly!")
        print("🚀 Ready for full training!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Quick validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_validation_test()
    
    if success:
        print("\n💡 Next steps:")
        print("1. Run debug preset: python scripts/run_foundation_pretrain.py --preset debug")
        print("2. Monitor with TensorBoard: tensorboard --logdir ./logs/runs")
        print("3. Run full training: python scripts/run_foundation_pretrain.py --preset quick_test")
    else:
        print("\n❌ Please fix issues before proceeding") 