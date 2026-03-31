#!/usr/bin/env python3
"""
Launch script for public dataset neural foundation model pretraining.

This script provides easy access to the PublicPretrainTrainer with common configurations
and demonstrates proper usage patterns for single-GPU and multi-GPU training.

Usage:
    # Single GPU training
    python scripts/run_public_foundation_pretrain.py --model_size medium --batch_size 32

    # Multi-GPU training (using torchrun)
    torchrun --nproc_per_node=4 scripts/run_public_foundation_pretrain.py --model_size large --batch_size 16

    # Custom experiment with specific subjects
    python scripts/run_public_foundation_pretrain.py --subjects c j m --temporal_layers 8 \
        --experiment_name "public_temporal_deep" --learning_rate 1e-4 --num_epochs 300
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.public_pretrain import main


def create_public_presets():
    """Create quick configuration presets for public dataset pretraining."""
    return {
        'debug': {
            'description': 'Quick debug run with small model, single dataset and few epochs',
            'overrides': {
                'model_size': 'small',
                'batch_size': 16,
                'num_epochs': 5,
                'learning_rate': 1e-3,
                'temporal_mask_ratio': [0.2, 0.2],  # Fixed ratio for debugging
                'neuron_mask_ratio': [0.15, 0.15],
                'subjects': ['j'],  # Single subject for debugging
                'target_trials_per_site': 50,  # Smaller dataset
                'min_val_test_trials': 5,
                'experiment_name': 'debug_run',
            }
        },
        
        'quick_test': {
            'description': 'Quick test run with small model for two subjects',
            'overrides': {
                'model_size': 'small',
                'batch_size': 32,
                'num_epochs': 200,
                'subjects': ['c', 'm'],  # Two subjects for faster testing
                'target_trials_per_site': 2000,
                'experiment_name': 'quick_test'
            }
        },
        
        'full_small': {
            'description': 'Full training run with small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 512,
                'num_epochs': 200,
                'subjects': ['c', 'j', 'm'],
                'experiment_name': 'full_small'
            }
        },

        'full_small_all_replicated_50ms': {
            'description': 'Full training run with small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 128,
                'num_epochs': 500,
                'subjects': ['c', 'j', 'm', 't'],
                'data_root': '/data/Fang-analysis/causal-nfm/Data/processed_replicated_50ms',
                'experiment_name': 'full_small_all_replicated_50ms'
            }
        },

        'full_small_all_replicated_20ms': {
            'description': 'Full training run with small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 128,
                'num_epochs': 500,
                'subjects': ['c', 'j', 'm', 't'],
                'data_root': '/data/Fang-analysis/causal-nfm/Data/processed_replicated_20ms',
                'experiment_name': 'full_small_all_replicated_20ms'
            }
        },

        'full_small_all_normalize_session': {
            'description': 'Full training run with small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 128,
                'num_epochs': 500,
                'subjects': ['c', 'j', 'm', 't'],
                'data_root': '/data/Fang-analysis/causal-nfm/Data/processed_normalize_session',
                'experiment_name': 'full_small_all_normalize_session'
            }
        },

        'ablation_no_temporal_kernels': {
            'description': 'Ablation study: disable adaptive temporal kernels',
            'overrides': {
                'model_size': 'small',
                'batch_size': 512,
                'num_epochs': 200,
                'use_temporal_kernels': False,
                'experiment_name': 'ablation_no_temporal_kernels'
            }
        },
        
        'ablation_large_kernels': {
            'description': 'Ablation study: large kernel size [9,9]',
            'overrides': {
                'model_size': 'small',
                'batch_size': 512,
                'num_epochs': 200,
                'kernel_size': [9, 9],
                'experiment_name': 'ablation_large_kernels'
            }
        },
        
        'ablation_high_masking': {
            'description': 'Ablation study: high masking ratios',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 512,
                'num_epochs': 200,
                'temporal_mask_ratio': [0.6, 0.9],
                'neuron_mask_ratio': [0.6, 0.9],
                'subjects': ['c', 'j', 'm'],
                'experiment_name': 'ablation_high_masking'
            }
        },
        
        'ablation_low_masking': {
            'description': 'Ablation study: low masking ratios',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 512,
                'num_epochs': 200,
                'temporal_mask_ratio': [0.1, 0.3],
                'neuron_mask_ratio': [0.1, 0.3],
                'subjects': ['c', 'j', 'm'],
                'experiment_name': 'ablation_low_masking'
            }
        },
        
        'single_subject_c': {
            'description': 'Single subject (c) pretraining',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 512,
                'num_epochs': 200,
                'subjects': ['c'],
                'target_trials_per_site': 200,  # More trials per session
                'experiment_name': 'single_subject_c'
            }
        },
        
        'full_medium': {
            'description': 'Full training run with medium model',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 512,
                'num_epochs': 200,
                'subjects': ['c', 'j', 'm'],
                'experiment_name': 'full_medium'
            }
        },
        
        'full_large': {
            'description': 'Full training run with large model (multi-GPU recommended)',
            'overrides': {
                'model_size': 'large',
                'batch_size': 16,
                'num_epochs': 200,
                'warmup_epochs': 20,
                'subjects': ['c', 'j', 'm'],
                'experiment_name': 'full_large'
            }
        },
    }
    


def print_available_presets():
    """Print available configuration presets."""
    configs = create_public_presets()
    
    print("\n" + "="*80)
    print("AVAILABLE PUBLIC DATASET PRETRAINING PRESETS")
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
    print(f"   python {__file__} --preset full_medium --batch_size 64 --learning_rate 1e-4")
    print("="*80)


def main_launcher():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description='Public Dataset Neural Foundation Model Pretraining Launcher',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Preset configurations
    parser.add_argument('--preset', type=str, choices=list(create_public_presets().keys()),
                       help='Use a preset configuration')
    parser.add_argument('--list_presets', action='store_true',
                       help='List available preset configurations and exit')
    
    # Common overrides (will be passed to main training script)
    parser.add_argument('--config', type=str, default='config/training/public_pretrain.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, help='Data root')
    parser.add_argument('--target_neurons', type=int, help='Target neurons (overrides config)')
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large', 'custom'],
                       help='Model size preset')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    
    # Public dataset specific options
    parser.add_argument('--subjects', nargs='+', choices=['c', 'j', 'm', 't'],
                       help='Subjects to include in pretraining')
    parser.add_argument('--sample_times', type=int, help='Sample times')
    parser.add_argument('--target_trials_per_site', type=int, help='Target trials per session')
    parser.add_argument('--min_val_test_trials', type=int, help='Minimum trials for val/test')
    
    # Model architecture options
    parser.add_argument('--neural_dim', type=int, help='Neural dimension')  
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--temporal_layers', type=int, help='Temporal encoder layers')
    parser.add_argument('--heads', type=int, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--pos_encoding_type', type=str, choices=['rope_4d', 'standard_rope', 'learnable', 'sinusoidal'],
                       help='Positional encoding type: ["rope_4d", "standard_rope", "learnable", "sinusoidal"]')
    parser.add_argument('--use_temporal_kernels', type=str, default='true', choices=['true', 'false'],
                       help='Enable adaptive causal kernel attention')
    parser.add_argument('--kernel_size', type=int, nargs=2, metavar=('H', 'W'),
                       help='Kernel size for adaptive attention [height width]')
    
    # Training options
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, help='Warmup epochs')
    parser.add_argument('--max_grad_norm', type=float, help='Max gradient norm')
    
    # Masking options
    parser.add_argument('--temporal_mask_ratio', type=float, nargs='+',
                       help='Temporal mask ratio (single value or min max)')
    parser.add_argument('--neuron_mask_ratio', type=float, nargs='+',
                       help='Neuron mask ratio (single value or min max)')
    
    # Paths and logging
    parser.add_argument('--base_dir', type=str, help='Base directory for outputs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Other options
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_presets:
        print_available_presets()
        return
    
    # Build command line arguments for main training script
    cmd_args = []
    
    # Configuration file
    cmd_args.extend(['--config', args.config])
    
    # Apply preset if specified
    if args.preset:
        configs = create_public_presets()
        if args.preset in configs:
            preset_config = configs[args.preset]['overrides']
            print(f"\n🚀 Using preset: {args.preset}")
            print(f"📝 Description: {configs[args.preset]['description']}")
            print(f"⚙️  Applied settings: {preset_config}")
            
            # Add preset overrides to command line
            for key, value in preset_config.items():
                if key == 'temporal_mask_ratio' or key == 'neuron_mask_ratio':
                    if isinstance(value, list):
                        cmd_args.extend([f'--{key}'] + [str(v) for v in value])
                    else:
                        cmd_args.extend([f'--{key}', str(value)])
                elif key == 'subjects':
                    cmd_args.extend([f'--{key}'] + value)
                else:
                    cmd_args.extend([f'--{key}', str(value)])
    
    # Add user overrides (these take precedence over preset)
    if args.data_root:
        cmd_args.extend(['--data_root', args.data_root])
    if args.target_neurons:
        cmd_args.extend(['--target_neurons', str(args.target_neurons)])
    if args.model_size:
        cmd_args.extend(['--model_size', args.model_size])
    if args.batch_size:
        cmd_args.extend(['--batch_size', str(args.batch_size)])
    if args.learning_rate:
        cmd_args.extend(['--learning_rate', str(args.learning_rate)])
    if args.num_epochs:
        cmd_args.extend(['--num_epochs', str(args.num_epochs)])
    if args.experiment_name:
        cmd_args.extend(['--experiment_name', args.experiment_name])
    if args.subjects:
        cmd_args.extend(['--subjects'] + args.subjects)
    if args.sample_times:
        cmd_args.extend(['--sample_times', str(args.sample_times)])
    if args.target_trials_per_site:
        cmd_args.extend(['--target_trials_per_site', str(args.target_trials_per_site)])
    if args.min_val_test_trials:
        cmd_args.extend(['--min_val_test_trials', str(args.min_val_test_trials)])
    if args.pos_encoding_type:
        cmd_args.extend(['--pos_encoding_type', args.pos_encoding_type])
    if args.use_temporal_kernels is not None:
        cmd_args.extend(['--use_temporal_kernels', args.use_temporal_kernels])
    if args.kernel_size:
        cmd_args.extend(['--kernel_size'] + [str(v) for v in args.kernel_size])
    if args.neural_dim:
        cmd_args.extend(['--neural_dim', str(args.neural_dim)])
    if args.d_model:
        cmd_args.extend(['--d_model', str(args.d_model)])
    if args.temporal_layers:
        cmd_args.extend(['--temporal_layers', str(args.temporal_layers)])
    if args.heads:
        cmd_args.extend(['--heads', str(args.heads)])
    if args.dropout is not None:
        cmd_args.extend(['--dropout', str(args.dropout)])
    if args.weight_decay is not None:
        cmd_args.extend(['--weight_decay', str(args.weight_decay)])
    if args.warmup_epochs is not None:
        cmd_args.extend(['--warmup_epochs', str(args.warmup_epochs)])
    if args.max_grad_norm:
        cmd_args.extend(['--max_grad_norm', str(args.max_grad_norm)])
    if args.temporal_mask_ratio:
        cmd_args.extend(['--temporal_mask_ratio'] + [str(v) for v in args.temporal_mask_ratio])
    if args.neuron_mask_ratio:
        cmd_args.extend(['--neuron_mask_ratio'] + [str(v) for v in args.neuron_mask_ratio])
    if args.base_dir:
        cmd_args.extend(['--base_dir', args.base_dir])
    if args.resume:
        cmd_args.extend(['--resume', args.resume])
    if args.seed:
        cmd_args.extend(['--seed', str(args.seed)])
    if args.early_stopping_patience:
        cmd_args.extend(['--early_stopping_patience', str(args.early_stopping_patience)])
    
    print(f"\n🔥 Starting Public Dataset Neural Foundation Model Pretraining")
    print(f"📋 Command line args: {' '.join(cmd_args)}")
    print(f"💻 Working directory: {os.getcwd()}")
    
    # Check for distributed training
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"🌐 Distributed training detected - Rank {rank}/{world_size}")
    else:
        print(f"🖥️  Single GPU training")
    
    print("="*80)
    
    # Override sys.argv and call main training function
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['public_pretrain.py'] + cmd_args
        main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main_launcher()
