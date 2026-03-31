#!/usr/bin/env python3
"""
Test Script for Neural Foundation Model Pretraining System

This script validates that our complete pretraining pipeline works correctly:
1. Configuration loading and validation
2. Model creation and parameter counting
3. Dataset loading and formatting
4. Trainer initialization
5. Short training run to verify everything integrates

Run this before starting long training runs to catch issues early.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup basic logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing imports...")
    
    try:
        # Core training components
        from src.training.pretrain import PretrainTrainer
        from src.data.cross_site_dataset import CrossSiteMonkeyDataset
        from src.models.transformer import CrossSiteFoundationMAE
        from src.utils.masking import CausalMaskingEngine
        from src.evaluation.loss_functions import compute_neural_mae_loss
        from src.utils.helpers import load_config
        
        logger.info("✅ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading and validation."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing configuration loading...")
    
    try:
        from src.utils.helpers import load_config
        
        config_path = "config/training/foundation_pretrain.yaml"
        if not os.path.exists(config_path):
            logger.error(f"❌ Config file not found: {config_path}")
            return False
            
        config = load_config(config_path)
        
        # Validate required sections
        required_sections = ['training', 'model', 'dataset', 'masking', 'paths']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing config section: {section}")
                return False
                
        logger.info(f"✅ Configuration loaded successfully!")
        logger.info(f"   Model sizes available: {list(config.get('model_sizes', {}).keys())}")
        logger.info(f"   Default model size: {config['training']['model_size']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation with minimal parameters."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing dataset creation...")
    
    try:
        from src.data.cross_site_dataset import CrossSiteMonkeyDataset
        
        # Create dataset with minimal parameters for testing
        dataset = CrossSiteMonkeyDataset(
            target_trials_per_site=100,      # Small for testing
            min_val_test_trials=20,          # Small for testing
            target_neurons=50,               # Standard
            sample_times=1,                  # Minimal sampling
            split_ratios=(0.6, 0.2, 0.2)    # Balanced for testing
        )
        
        # Check data shapes
        train_data = dataset.get_split_data('train')
        val_data = dataset.get_split_data('val')
        test_data = dataset.get_split_data('test')
        site_coords = dataset.get_site_coordinates()
        
        logger.info(f"✅ Dataset created successfully!")
        logger.info(f"   Train data: {train_data.shape}")
        logger.info(f"   Val data: {val_data.shape}")
        logger.info(f"   Test data: {test_data.shape}")
        logger.info(f"   Site coords: {site_coords.shape}")
        logger.info(f"   Number of sites: {len(dataset.available_sites)}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        return None

def test_model_creation():
    """Test model creation with different sizes."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing model creation...")
    
    try:
        from src.models.transformer import CrossSiteFoundationMAE
        from src.utils.helpers import load_config
        
        config = load_config("config/training/foundation_pretrain.yaml")
        
        # Test small model for speed
        model_config = config['model_sizes']['small'].copy()
        model_config.update(config['model'])
        
        model = CrossSiteFoundationMAE(
            neural_dim=model_config['neural_dim'],
            d_model=model_config['d_model'],
            temporal_layers=model_config['temporal_layers'],
            spatial_layers=model_config['spatial_layers'],
            heads=model_config['heads'],
            dropout=model_config['dropout'],
            kernel_size=model_config['kernel_size'],
            pos_encoding_type=model_config['pos_encoding_type'],
            use_temporal_kernels=model_config['use_temporal_kernels'],
            use_site_specific_heads=model_config['use_site_specific_heads']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model created successfully!")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {model_config['d_model']}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return None

def test_masking_engine():
    """Test masking engine functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing masking engine...")
    
    try:
        from src.utils.masking import CausalMaskingEngine
        
        # Create masking engine with moderate settings
        masking_engine = CausalMaskingEngine(
            temporal_mask_ratio=[0.1, 0.2],  # Dynamic ratio
            neuron_mask_ratio=[0.1, 0.2]     # Dynamic ratio
        )
        
        # Test with dummy data
        B, S, T, N = 4, 16, 50, 50
        neural_data = torch.randn(B, S, T, N)
        
        mask, masked_indices = masking_engine.apply_causal_mask(neural_data)
        stats = masking_engine.get_masking_statistics(mask)
        
        logger.info(f"✅ Masking engine working!")
        logger.info(f"   Overall masking ratio: {stats['overall_masking_ratio']:.3f}")
        logger.info(f"   Temporal masking ratio: {stats['temporal_masking_ratio']:.3f}")
        logger.info(f"   Neuron masking ratio: {stats['neuron_masking_ratio']:.3f}")
        
        return masking_engine
        
    except Exception as e:
        logger.error(f"❌ Masking engine failed: {e}")
        return None

def test_loss_computation(model, dataset, masking_engine):
    """Test loss computation functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing loss computation...")
    
    try:
        from src.evaluation.loss_functions import compute_neural_mae_loss
        
        # Get a small batch of data
        dataloader = dataset.create_dataloader('train', batch_size=2, shuffle=False)
        neural_data = next(iter(dataloader))[0]  # Get first batch
        site_coords = dataset.get_site_coordinates()
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        neural_data = neural_data.to(device)
        site_coords = site_coords.to(device)
        model = model.to(device)
        
        # Compute loss
        loss_dict = compute_neural_mae_loss(
            model=model,
            neural_data=neural_data,
            site_coords=site_coords,
            masking_engine=masking_engine,
            contrastive_weight=0.1,
            reconstruction_weight=1.0
        )
        
        logger.info(f"✅ Loss computation successful!")
        logger.info(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
        logger.info(f"   Poisson loss: {loss_dict['poisson_loss'].item():.4f}")
        logger.info(f"   Contrastive loss: {loss_dict['contrastive_loss'].item():.4f}")
        logger.info(f"   Device: {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss computation failed: {e}")
        return False

def test_trainer_initialization(model, dataset):
    """Test trainer initialization."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing trainer initialization...")
    
    try:
        from src.training.pretrain import PretrainTrainer
        from src.utils.helpers import load_config
        
        # Load config and modify for testing
        config = load_config("config/training/foundation_pretrain.yaml")
        
        # Use small model config
        config['training']['model_size'] = 'small'
        config['training'].update(config['model_sizes']['small'])
        
        # Set minimal training parameters for testing
        config['training']['num_epochs'] = 2
        config['training']['batch_size'] = 2
        config['training']['learning_rate'] = 1e-4
        
        # Set up test paths
        config['paths']['experiment_name'] = 'test_run'
        config['paths']['base_dir'] = './test_logs'
        
        # Create trainer
        trainer = PretrainTrainer(
            model=model,
            dataset=dataset,
            config=config
        )
        
        logger.info(f"✅ Trainer initialized successfully!")
        logger.info(f"   Experiment name: {config['paths']['experiment_name']}")
        logger.info(f"   Batch size: {config['training']['batch_size']}")
        logger.info(f"   Learning rate: {config['training']['learning_rate']}")
        
        return trainer, config
        
    except Exception as e:
        logger.error(f"❌ Trainer initialization failed: {e}")
        return None, None

def test_training_step(trainer):
    """Test a single training step."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing single training step...")
    
    try:
        # Run one training epoch
        train_metrics = trainer.train_epoch()
        
        logger.info(f"✅ Training step successful!")
        logger.info(f"   Total loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"   Learning rate: {train_metrics.get('learning_rate', 'N/A')}")
        
        # Test validation step
        val_metrics = trainer.validate_epoch()
        
        logger.info(f"✅ Validation step successful!")
        logger.info(f"   Val total loss: {val_metrics['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        return False

def run_short_training_test():
    """Run a very short training test (2 epochs) to validate the complete pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running short training test...")
    
    try:
        # Use the launcher for a quick test
        import subprocess
        
        cmd = [
            "python", "scripts/run_foundation_pretrain.py",
            "--preset", "debug",  # Use debug preset
            "--experiment_name", "test_short_training"
        ]
        
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info("   Starting 2-epoch training test...")
        
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Short training test completed successfully!")
            logger.info("   Check ./logs/ for training outputs")
            return True
        else:
            logger.error(f"❌ Short training test failed!")
            logger.error(f"   Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"   Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training test timed out (5 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    logger = logging.getLogger(__name__)
    logger.info("🧹 Cleaning up test files...")
    
    import shutil
    
    test_dirs = ['./test_logs', './logs/checkpoints/test_run', './logs/runs/test_run']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                logger.info(f"   Removed: {test_dir}")
            except Exception as e:
                logger.warning(f"   Could not remove {test_dir}: {e}")

def main():
    """Run comprehensive pretraining system test."""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("🧪 NEURAL FOUNDATION MODEL PRETRAINING SYSTEM TEST")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        logger.error("❌ Cannot proceed without proper imports")
        return False
    
    # Test 2: Configuration
    if test_config_loading():
        tests_passed += 1
    else:
        logger.error("❌ Cannot proceed without valid configuration")
        return False
    
    # Test 3: Dataset
    dataset = test_dataset_creation()
    if dataset is not None:
        tests_passed += 1
    else:
        logger.error("❌ Cannot proceed without dataset")
        return False
    
    # Test 4: Model
    model = test_model_creation()
    if model is not None:
        tests_passed += 1
    else:
        logger.error("❌ Cannot proceed without model")
        return False
    
    # Test 5: Masking
    masking_engine = test_masking_engine()
    if masking_engine is not None:
        tests_passed += 1
    else:
        logger.error("❌ Cannot proceed without masking engine")
        return False
    
    # Test 6: Loss computation
    if test_loss_computation(model, dataset, masking_engine):
        tests_passed += 1
    
    # Test 7: Trainer
    trainer, config = test_trainer_initialization(model, dataset)
    if trainer is not None:
        tests_passed += 1
    
    # Test 8: Training step
    if trainer is not None and test_training_step(trainer):
        tests_passed += 1
    
    # Summary
    logger.info("=" * 70)
    logger.info(f"📊 TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! System is ready for training.")
        
        # Ask user if they want to run short training test
        logger.info("\n🚀 Would you like to run a quick 2-epoch training test?")
        logger.info("   This will validate the complete training pipeline.")
        
        return True
    else:
        logger.error(f"❌ {total_tests - tests_passed} tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*50)
        print("✅ SYSTEM READY FOR TRAINING!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run quick test: python scripts/run_foundation_pretrain.py --preset debug")
        print("2. Run full training: python scripts/run_foundation_pretrain.py --preset quick_test")
        print("3. Monitor with: tensorboard --logdir ./logs/runs")
    else:
        print("\n" + "="*50)
        print("❌ PLEASE FIX ISSUES BEFORE TRAINING")
        print("="*50) 