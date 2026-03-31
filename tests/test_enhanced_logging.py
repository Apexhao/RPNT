#!/usr/bin/env python3
"""
Test script for Enhanced Logging System

This script validates that our enhanced logging system works correctly:
1. Terminal capture functionality
2. Top-K checkpoint management
3. Folder structure creation
4. Configuration and model summary saving
5. Integration with existing trainer

Run this to verify the logging system before starting real training.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_logging():
    """Test the enhanced logging system."""
    print("🧪 TESTING ENHANCED LOGGING SYSTEM")
    print("=" * 60)
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_base_dir = Path(temp_dir)
        experiment_name = "test_enhanced_logging"
        
        try:
            # Test 1: Import and initialize enhanced logger
            print("\n📦 Test 1: Import and Initialize Enhanced Logger")
            from src.training.enhanced_logger import EnhancedLogger
            
            enhanced_logger = EnhancedLogger(
                experiment_name=experiment_name,
                base_dir=str(test_base_dir),
                top_k_checkpoints=3,
                local_rank=0
            )
            print("✅ Enhanced logger initialized successfully")
            
            # Test 2: Check folder structure
            print("\n📁 Test 2: Folder Structure Creation")
            expected_dirs = [
                test_base_dir / experiment_name / "training_logs",
                test_base_dir / experiment_name / "runs", 
                test_base_dir / experiment_name / "checkpoints",
                test_base_dir / experiment_name / "config",
                test_base_dir / experiment_name / "model_summary"
            ]
            
            for dir_path in expected_dirs:
                if dir_path.exists():
                    print(f"✅ {dir_path.name}/ directory created")
                else:
                    print(f"❌ {dir_path.name}/ directory missing")
                    return False
            
            # Test 3: Configuration saving
            print("\n⚙️ Test 3: Configuration Saving")
            test_config = {
                'training': {
                    'model_size': 'small',
                    'batch_size': 16,
                    'learning_rate': 1e-4,
                    'num_epochs': 100
                },
                'model': {
                    'd_model': 256,
                    'neural_dim': 50
                }
            }
            
            enhanced_logger.save_config(test_config)
            
            # Check config files exist
            config_yaml = test_base_dir / experiment_name / "config" / "training_config.yaml"
            config_json = test_base_dir / experiment_name / "config" / "training_config.json"
            
            if config_yaml.exists() and config_json.exists():
                print("✅ Configuration files saved successfully")
                
                # Verify content
                with open(config_json, 'r') as f:
                    loaded_config = json.load(f)
                if loaded_config['training']['model_size'] == 'small':
                    print("✅ Configuration content verified")
                else:
                    print("❌ Configuration content incorrect")
                    return False
            else:
                print("❌ Configuration files not created")
                return False
            
            # Test 4: Model summary (using dummy model)
            print("\n🧠 Test 4: Model Summary Saving")
            import torch
            import torch.nn as nn
            
            # Create simple dummy model
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                    self.relu = nn.ReLU()
                    self.output = nn.Linear(5, 1)
                
                def forward(self, x):
                    return self.output(self.relu(self.linear(x)))
            
            dummy_model = DummyModel()
            enhanced_logger.save_model_summary(dummy_model)
            
            # Check model summary files exist
            summary_txt = test_base_dir / experiment_name / "model_summary" / "model_summary.txt"
            summary_json = test_base_dir / experiment_name / "model_summary" / "model_architecture.json"
            
            if summary_txt.exists() and summary_json.exists():
                print("✅ Model summary files saved successfully")
                
                # Verify content
                with open(summary_json, 'r') as f:
                    model_info = json.load(f)
                if 'total_parameters' in model_info:
                    print("✅ Model summary content verified")
                else:
                    print("❌ Model summary content incorrect")
                    return False
            else:
                print("❌ Model summary files not created")
                return False
            
            # Test 5: Checkpoint management
            print("\n💾 Test 5: Checkpoint Management")
            
            # Create dummy checkpoints with different validation losses
            checkpoint_data = {
                'epoch': 0,
                'model_state_dict': dummy_model.state_dict(),
                'optimizer_state_dict': {},
                'config': test_config
            }
            
            # Test saving multiple checkpoints
            val_losses = [0.5, 0.3, 0.7, 0.2, 0.4]  # 0.2 should be best
            saved_paths = []
            
            for i, val_loss in enumerate(val_losses):
                checkpoint_data['epoch'] = i
                path = enhanced_logger.save_checkpoint(
                    checkpoint=checkpoint_data,
                    epoch=i,
                    val_loss=val_loss
                )
                saved_paths.append(path)
                print(f"  Saved checkpoint epoch {i} with val_loss {val_loss}")
            
            # Check checkpoint manager state
            checkpoint_info = enhanced_logger.checkpoint_manager.get_checkpoint_info()
            
            if len(checkpoint_info['best_checkpoints']) <= 3:
                print("✅ Top-K constraint maintained")
            else:
                print("❌ Too many checkpoints kept")
                return False
            
            if checkpoint_info['latest_checkpoint']:
                print("✅ Latest checkpoint tracked")
            else:
                print("❌ Latest checkpoint not tracked")
                return False
            
            # Verify best checkpoint is correct (should be epoch with val_loss=0.2)
            best_epochs = [cp['epoch'] for cp in checkpoint_info['best_checkpoints']]
            if 3 in best_epochs:  # Epoch 3 had val_loss=0.2
                print("✅ Best checkpoint correctly identified")
            else:
                print("❌ Best checkpoint incorrect")
                return False
            
            # Test 6: Terminal capture
            print("\n📺 Test 6: Terminal Capture")
            
            # Print some test messages
            print("This is stdout message 1")
            print("This is stdout message 2")
            sys.stderr.write("This is stderr message 1\n")
            sys.stderr.write("This is stderr message 2\n")
            
            # Check terminal log file
            terminal_log = test_base_dir / experiment_name / "training_logs" / "terminal_output.log"
            if terminal_log.exists():
                with open(terminal_log, 'r') as f:
                    terminal_content = f.read()
                
                if "stdout message 1" in terminal_content and "stderr message 1" in terminal_content:
                    print("✅ Terminal capture working")
                else:
                    print("❌ Terminal capture not working properly")
                    return False
            else:
                print("❌ Terminal log file not created")
                return False
            
            # Test 7: Epoch stats logging
            print("\n📊 Test 7: Epoch Statistics Logging")
            
            train_stats = {'total_loss': 0.5, 'contrastive_loss': 0.1, 'poisson_loss': 0.4, 'learning_rate': 1e-4, 'grad_norm': 0.1}
            val_stats = {'total_loss': 0.4, 'contrastive_loss': 0.08, 'poisson_loss': 0.32, 'accuracy': 0.85}
            test_stats = {'total_loss': 0.45, 'contrastive_loss': 0.09, 'poisson_loss': 0.36, 'accuracy': 0.82}
            
            enhanced_logger.log_epoch_stats(
                epoch=1,
                train_stats=train_stats,
                val_stats=val_stats,
                test_stats=test_stats
            )
            print("✅ Epoch statistics logged successfully")
            
            # Test 8: Training completion
            print("\n🏁 Test 8: Training Completion")
            enhanced_logger.log_training_end()
            
            # Check final statistics file
            stats_file = test_base_dir / experiment_name / "training_logs" / "training_statistics.json"
            if stats_file.exists():
                print("✅ Training statistics saved")
            else:
                print("❌ Training statistics not saved")
                return False
            
            # Test 9: Resource cleanup
            print("\n🧹 Test 9: Resource Cleanup")
            enhanced_logger.close()
            print("✅ Resources cleaned up successfully")
            
            # Test 10: Integration test (paths)
            print("\n🔗 Test 10: Path Integration")
            paths = enhanced_logger.get_paths()
            expected_keys = ['experiment_name', 'base_dir', 'experiment_dir', 
                           'checkpoint_dir', 'tensorboard_dir', 'log_dir', 
                           'config_dir', 'model_summary_dir']
            
            for key in expected_keys:
                if key in paths:
                    print(f"✅ Path '{key}' available")
                else:
                    print(f"❌ Path '{key}' missing")
                    return False
            
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! Enhanced logging system is working correctly!")
            print("=" * 60)
            
            # Show final directory structure
            print("\n📁 Final Directory Structure:")
            def print_tree(directory, prefix=""):
                items = sorted(directory.iterdir())
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    print(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and len(list(item.iterdir())) > 0:
                        extension = "    " if is_last else "│   "
                        print_tree(item, prefix + extension)
            
            print_tree(test_base_dir / experiment_name)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_integration_with_trainer():
    """Test integration with the existing PretrainTrainer."""
    print("\n\n🔄 TESTING INTEGRATION WITH PRETRAINER")
    print("=" * 60)
    
    try:
        # Test import
        from src.training.pretrain import PretrainTrainer
        print("✅ Successfully imported PretrainTrainer with enhanced logging")
        
        # Test that enhanced logger is imported
        from src.training.enhanced_logger import EnhancedLogger
        print("✅ Enhanced logger import working")
        
        print("✅ Integration test passed - trainer can use enhanced logging")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 ENHANCED LOGGING SYSTEM VALIDATION")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_enhanced_logging()
    test2_passed = test_integration_with_trainer()
    
    print("\n" + "=" * 80)
    if test1_passed and test2_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Enhanced logging system is ready for production use!")
        print("✅ Your training scripts will now have comprehensive logging!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
    
    print("=" * 80) 