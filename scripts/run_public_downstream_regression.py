#!/usr/bin/env python3
"""
Launcher script for public dataset downstream regression tasks with session-level evaluation.

This script provides easy access to regression training using pretrained temporal-only foundation models
with comprehensive session-level analysis and aggregated metrics.

Usage:
    # Cross-session evaluation (subject c, 2016xxx sessions)
    python scripts/run_public_downstream_regression.py --dataset_type cross_session \
        --pretrained_path ./logs_public/full_small/checkpoints/best.pth

    # Cross-subject center-out evaluation (subject t, center-out tasks)
    python scripts/run_public_downstream_regression.py --dataset_type cross_subject_center \
        --training_mode finetune_encoder --learning_rate 1e-4

    # Cross-subject random-target evaluation (subject t, random-target tasks)
    python scripts/run_public_downstream_regression.py --dataset_type cross_subject_random \
        --target_type velocity --num_epochs 150

    # Full evaluation sweep across all downstream tasks
    python scripts/run_public_downstream_regression.py --eval_mode full_sweep \
        --pretrained_path ./logs_public/full_small/checkpoints/best.pth
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.public_downstream_dataset import (
    PublicCrossSessionDataset, 
    PublicCrossSubjectCenterDataset, 
    PublicCrossSubjectRandomDataset,
    Public_No_T_Subject_Dataset,
    Public_Only_RT_Subject_Dataset,
    Public_Only_CO_Subject_Dataset,
)
from src.training.public_downstream_trainers import create_public_downstream_trainer
from src.utils.helpers import load_config, set_seed


def create_downstream_presets():
    """Create quick configuration presets for downstream tasks."""
    return {
        'debug': {
            'description': 'Quick debug run with minimal epochs',
            'overrides': {
                'dataset_type': 'cross_session',
                'batch_size': 16,
                'num_epochs': 10,
                'learning_rate': 1e-2,
                'training_mode': 'frozen_encoder',
                'experiment_name': 'public_downstream_debug',
            }
        },
        
        'cross_session_frozen': {
            'description': 'Cross-session evaluation with frozen encoder',
            'overrides': {
                'dataset_type': 'cross_session',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 200,
                'experiment_name': 'cross_session_frozen'
            }
        },
        
        'cross_session_finetune': {
            'description': 'Cross-session evaluation with encoder fine-tuning',
            'overrides': {
                'dataset_type': 'cross_session',
                'training_mode': 'full_finetune',
                'learning_rate': 5e-4,
                'num_epochs': 100,
                'experiment_name': 'cross_session_finetune'
            }
        },
        
        'cross_subject_center_frozen': {
            'description': 'Cross-subject center-out with frozen encoder',
            'overrides': {
                'dataset_type': 'cross_subject_center',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'experiment_name': 'cross_subject_center_frozen'
            }
        },
        
        'cross_subject_center_finetune': {
            'description': 'Cross-subject center-out with encoder fine-tuning',
            'overrides': {
                'dataset_type': 'cross_subject_center',
                'training_mode': 'full_finetune',
                'learning_rate': 1e-4,
                'num_epochs': 200,
                'experiment_name': 'cross_subject_center_finetune'
            }
        },
        
        'cross_subject_random_frozen': {
            'description': 'Cross-subject random-target with frozen encoder',
            'overrides': {
                'dataset_type': 'cross_subject_random',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'experiment_name': 'cross_subject_random_frozen'
            }
        },
        
        'cross_subject_random_finetune': {
            'description': 'Cross-subject random-target with encoder fine-tuning',
            'overrides': {
                'dataset_type': 'cross_subject_random',
                'training_mode': 'full_finetune',
                'learning_rate': 1e-4,
                'num_epochs': 200,
                'experiment_name': 'cross_subject_random_finetune'
            }
        }
    }


def print_available_presets():
    """Print available configuration presets."""
    configs = create_downstream_presets()
    
    print("\n" + "="*80)
    print("AVAILABLE PUBLIC DOWNSTREAM REGRESSION PRESETS")
    print("="*80)
    
    for name, config in configs.items():
        print(f"\n📋 {name.upper()}")
        print(f"   Description: {config['description']}")
        print(f"   Key settings:")
        for key, value in config['overrides'].items():
            if key != 'experiment_name':
                print(f"     - {key}: {value}")
        print(f"   Usage: python {__file__} --preset {name}")
    
    print(f"\n💡 You can also override any preset setting:")
    print(f"   python {__file__} --preset cross_session_frozen --learning_rate 5e-4")
    print("="*80)

def create_dataset_by_type(dataset_type: str, config: Dict[str, Any]):
    """Create dataset instance by type."""
    dataset_config = config['dataset']
    
    common_kwargs = {
        'target_neurons': dataset_config.get('target_neurons', 50),
        'random_seed': dataset_config.get('random_seed', 42)
    }
    
    if dataset_type == 'cross_session':
        return PublicCrossSessionDataset(**common_kwargs)
    elif dataset_type == 'cross_subject_center':
        return PublicCrossSubjectCenterDataset(**common_kwargs)
    elif dataset_type == 'cross_subject_random':
        return PublicCrossSubjectRandomDataset(**common_kwargs)
    elif dataset_type == 'no_t_subject':
        return Public_No_T_Subject_Dataset(**common_kwargs)
    elif dataset_type == 'only_rt_task':
        return Public_Only_RT_Subject_Dataset(**common_kwargs)
    elif dataset_type == 'only_co_task':
        return Public_Only_CO_Subject_Dataset(**common_kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def evaluate_single_downstream_task(dataset_type: str, 
                                   pretrained_path: str,
                                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single downstream task with session-level analysis.
    
    Args:
        dataset_type: Type of downstream dataset
        pretrained_path: Path to pretrained model
        config: Training configuration
        
    Returns:
        Dictionary with aggregated and per-session results
    """
    
    print(f"\n🎯 Evaluating {dataset_type} task...")
    print("="*60)
    
    # Create dataset
    dataset = create_dataset_by_type(dataset_type, config)
    
    if len(dataset.session_ids) == 0:
        print(f"⚠️  No sessions found for {dataset_type}")
        return {'error': 'no_sessions_found'}
    
    print(f"📊 Found {len(dataset.session_ids)} sessions: {dataset.session_ids}")
    
    # Create trainer (only regression supported for public dataset)
    trainer = create_public_downstream_trainer(
        pretrained_checkpoint_path=pretrained_path,
        dataset=dataset,
        config=config
    )
    
    print(f"🏋️ Starting training for {config['training']['num_epochs']} epochs...")
    
    # Train the model (trainer handles all evaluation internally)
    trainer.train(config['training']['num_epochs'])
    
    # Return basic results dict for consistency with existing code
    return {
        'dataset_type': dataset_type,
        'num_sessions': len(dataset.session_ids),
        'session_ids': dataset.session_ids,
        'config': config,
        'experiment_dir': trainer.paths_config['experiment_dir']
    }

def main():
    """Main function for downstream regression evaluation."""
    parser = argparse.ArgumentParser(
        description='Public Dataset Downstream Regression Evaluation with Session-Level Analysis',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Preset configurations
    parser.add_argument('--preset', type=str, choices=list(create_downstream_presets().keys()),
                       help='Use a preset configuration')
    parser.add_argument('--list_presets', action='store_true',
                       help='List available preset configurations and exit')
    
    # Evaluation modes
    parser.add_argument('--eval_mode', type=str, choices=['single', 'full_sweep'],
                       default='single', help='Evaluation mode')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/public_downstream_regression.yaml',
                       help='Path to configuration file')
    
    # Dataset configuration
    parser.add_argument('--dataset_type', type=str, 
                       choices=['cross_session', 'cross_subject_center', 'cross_subject_random', 'no_t_subject', 'only_rt_task', 'only_co_task'],
                       help='Type of downstream dataset')
    parser.add_argument('--target_neurons', type=int,
                       help='Target number of neurons')
    parser.add_argument('--neuron_selection_strategy', type=str, choices=['first_n', 'random_n', 'all'],
                       help='Neuron selection strategy')
    
    # NOTE: Model architecture parameters (neural_dim, d_model, etc.) are loaded from 
    # foundation model checkpoint and cannot be overridden via CLI
    
    parser.add_argument('--pretrained_path', type=str,
                       help='Path to pretrained foundation model')
    parser.add_argument('--training_mode', type=str, choices=['frozen_encoder', 'full_finetune', 'partial_finetune'],
                       help='Training mode')
    parser.add_argument('--target_type', type=str, choices=['velocity', 'position'],
                       help='Target type for regression')
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int,
                       help='Early stopping patience')
    
    # Output configuration
    parser.add_argument('--experiment_name', type=str,
                       help='Custom experiment name (auto-generated if not provided)')
    parser.add_argument('--base_dir', type=str,
                       help='Base directory for outputs')
    
    # Other options
    parser.add_argument('--seed', type=int,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_presets:
        print_available_presets()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply preset if specified
    if args.preset:
        presets = create_downstream_presets()
        if args.preset in presets:
            preset_config = presets[args.preset]['overrides']
            print(f"\n🚀 Using preset: {args.preset}")
            print(f"📝 Description: {presets[args.preset]['description']}")
            
            # Apply preset overrides
            for key, value in preset_config.items():
                if key == 'dataset_type':
                    config['dataset'][key] = value
                elif key in ['training_mode', 'learning_rate', 'num_epochs', 'target_type']:
                    config['training'][key] = value
                elif key in ['batch_size']:
                    config['training'][key] = value
                elif key == 'experiment_name':
                    config['paths'][key] = value
    
    # Clean override logic: only override if CLI arg was explicitly provided
    # Config is source of truth, CLI args only override when specified
    
    # Dataset overrides
    if args.dataset_type is not None:
        config['dataset']['dataset_type'] = args.dataset_type
    if args.target_neurons is not None:
        config['dataset']['target_neurons'] = args.target_neurons
    if args.neuron_selection_strategy is not None:
        config['dataset']['neuron_selection_strategy'] = args.neuron_selection_strategy
    
    # Path overrides  
    if args.pretrained_path is not None:
        config['paths']['pretrained_path'] = args.pretrained_path
    
    # Training overrides (only downstream task settings)
    if args.training_mode is not None:
        config['training']['training_mode'] = args.training_mode
    if args.target_type is not None:
        config['training']['target_type'] = args.target_type
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.early_stopping_patience is not None:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    # Output configuration overrides
    if args.experiment_name is not None:
        config['paths']['experiment_name'] = args.experiment_name
    if args.base_dir is not None:
        config['paths']['base_dir'] = args.base_dir
    
    # Seed overrides
    if args.seed is not None:
        config['dataset']['random_seed'] = args.seed
        config['training']['random_seed'] = args.seed
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    print("=" * 80)
    print("PUBLIC DATASET DOWNSTREAM REGRESSION EVALUATION")
    print("=" * 80)
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"Pretrained Model: {config['paths']['pretrained_path']}")
    print(f"Training Mode: {config['training']['training_mode']}")
    print(f"Target Type: {config['training']['target_type']}")
    print("=" * 80)
    
    try: # Run single evaluation
        if not args.dataset_type and not args.preset:
                print("❌ Error: --dataset_type is required for single evaluation mode")
                return
            
        dataset_type = config['dataset']['dataset_type']
        results = evaluate_single_downstream_task(
            dataset_type,
            config['paths']['pretrained_path'],
            config
        )
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise e


if __name__ == '__main__':
    main()
