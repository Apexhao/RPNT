#!/usr/bin/env python3
"""
Test Script for Per-Session Implementation
------------------------------------------

This script tests the newly implemented per-session evaluation components
to ensure they work correctly before running full evaluations.

Usage:
    python scripts/test_per_session_implementation.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.public_single_session_dataset import (
    PublicSingleSessionDataset,
    create_single_session_datasets,
    get_session_ids_for_scenario,
    test_public_single_session_dataset
)
from src.training.public_per_session_trainers import (
    PublicPerSessionEvaluationManager,
    test_per_session_trainers
)


def test_session_id_discovery():
    """Test that we can discover session IDs for different scenarios."""
    
    print("🔍 Testing Session ID Discovery")
    print("=" * 50)
    
    scenarios = ['cross_session', 'cross_subject_center', 'cross_subject_random']
    
    for scenario in scenarios:
        try:
            session_ids = get_session_ids_for_scenario(scenario)
            print(f"✅ {scenario}: {len(session_ids)} sessions")
            print(f"   Sessions: {session_ids[:3]}..." if len(session_ids) > 3 else f"   Sessions: {session_ids}")
        except Exception as e:
            print(f"❌ {scenario}: Failed - {str(e)}")
    
    print()


def test_single_session_dataset_loading():
    """Test loading a single session dataset."""
    
    print("📊 Testing Single Session Dataset Loading")
    print("=" * 50)
    
    # Test with a known session (adjust based on your actual data)
    test_session_ids = [
        't_20130819_center_out_reaching',  # Cross-subject center
        't_20130820_random_target_reaching',  # Cross-subject random
        'c_20160909_center_out_reaching'  # Cross-session (if available)
    ]
    
    for session_id in test_session_ids:
        try:
            print(f"\n🔄 Testing session: {session_id}")
            
            dataset = PublicSingleSessionDataset(
                session_id=session_id,
                target_neurons=50,
                neuron_selection_strategy='first_n',
                random_seed=42
            )
            
            # Test basic functionality
            train_data = dataset.get_split_data('train')
            val_data = dataset.get_split_data('val')
            test_data = dataset.get_split_data('test')
            
            print(f"   ✅ Train: {train_data['neural_data'].shape}, {train_data['n_trials']} trials")
            print(f"   ✅ Val: {val_data['neural_data'].shape}, {val_data['n_trials']} trials")
            print(f"   ✅ Test: {test_data['neural_data'].shape}, {test_data['n_trials']} trials")
            
            # Test dataloader creation
            train_loader = dataset.create_dataloader('train', batch_size=4, output_mode='regression')
            batch = next(iter(train_loader))
            print(f"   ✅ Dataloader batch shapes: {[x.shape for x in batch]}")
            
            break  # If one succeeds, that's good enough for testing
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
    
    print()


def test_factory_functions():
    """Test factory functions for creating multiple session datasets."""
    
    print("🏭 Testing Factory Functions")
    print("=" * 50)
    
    scenarios = ['cross_subject_center']  # Test just one scenario for now
    
    for scenario in scenarios:
        try:
            print(f"\n🔄 Testing {scenario} factory function...")
            
            datasets = create_single_session_datasets(
                scenario=scenario,
                target_neurons=50,
                neuron_selection_strategy='first_n',
                random_seed=42
            )
            
            print(f"   ✅ Created {len(datasets)} datasets")
            
            if datasets:
                # Test first dataset
                dataset = datasets[0]
                train_data = dataset.get_split_data('train')
                print(f"   ✅ First dataset: {dataset.get_session_id()}")
                print(f"       Train data: {train_data['neural_data'].shape}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    print()


def test_evaluation_manager_setup():
    """Test evaluation manager setup (without full training)."""
    
    print("🎯 Testing Evaluation Manager Setup")
    print("=" * 50)
    
    # Mock configuration
    test_config = {
        'dataset': {
            'target_neurons': 50,
            'sequence_length': 50,
            'neuron_selection_strategy': 'first_n',
            'random_seed': 42,
            'data_root': "/data/Fang-analysis/causal-nfm/Data/public_data"
        },
        'training': {
            'task_type': 'regression',
            'target_type': 'velocity',
            'training_mode': 'full_finetune',
            'learning_rate': 0.001,
            'batch_size': 4,
            'num_epochs': 2,  # Very few for testing
            'early_stopping_patience': 1
        },
        'paths': {
            'base_dir': './logs_test_per_session',
            'pretrained_path': './logs_public/full_small/checkpoints/best.pth'
        }
    }
    
    try:
        print(f"🔄 Creating evaluation manager...")
        
        # Check if pretrained model exists
        pretrained_path = test_config['paths']['pretrained_path']
        if not Path(pretrained_path).exists():
            print(f"   ⚠️  Pretrained model not found: {pretrained_path}")
            print(f"   🔄 Using mock path for testing setup only")
            pretrained_path = "./mock_pretrained_path.pth"
        
        manager = PublicPerSessionEvaluationManager(
            pretrained_checkpoint_path=pretrained_path,
            evaluation_scenario='cross_subject_center',
            config=test_config
        )
        
        print(f"   ✅ Manager created successfully")
        print(f"   📊 Total sessions: {manager.total_sessions}")
        print(f"   📋 Session datasets: {len(manager.session_datasets)}")
        
        if manager.session_datasets:
            first_session = manager.session_datasets[0]
            print(f"   🔍 First session: {first_session.get_session_id()}")
        
        # Clean up
        manager.enhanced_logger.close()
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    print()


def main():
    """Run all tests."""
    
    print("🧪 Testing Per-Session Implementation")
    print("=" * 80)
    print("This script tests the newly implemented per-session evaluation components.")
    print("=" * 80)
    
    # Run tests
    test_session_id_discovery()
    test_single_session_dataset_loading() 
    test_factory_functions()
    test_evaluation_manager_setup()
    
    print("🎯 Test Summary")
    print("=" * 50)
    print("If all tests above showed ✅, the implementation is working correctly!")
    print("You can now proceed to run actual per-session evaluations.")
    print()
    print("Next steps:")
    print("1. Ensure your pretrained model exists at the specified path")
    print("2. Run per-session evaluation using the new protocol")
    print("3. Compare results with existing multi-session evaluation")
    print("=" * 50)


if __name__ == "__main__":
    main()
