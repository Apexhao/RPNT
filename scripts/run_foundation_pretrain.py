#!/usr/bin/env python3
"""
Launch script for professional neural foundation model pretraining.

This script provides easy access to the new PretrainTrainer with common configurations
and demonstrates proper usage patterns for single-GPU and multi-GPU training.

Usage:
    # Single GPU training
    python scripts/run_foundation_pretrain.py --model_size medium --batch_size 32

    # Multi-GPU training (using torchrun)
    torchrun --nproc_per_node=4 scripts/run_foundation_pretrain.py --model_size large --batch_size 16

    # Custom experiment
    python scripts/run_foundation_pretrain.py --config config/training/foundation_pretrain.yaml \
        --experiment_name "custom_experiment" --learning_rate 1e-4 --num_epochs 500
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.pretrain import main


def create_quick_configs():
    """Create quick configuration presets for common use cases."""
    return {
        'debug': {
            'description': 'Quick debug run with small model and few epochs',
            'overrides': {
                'model_size': 'small',
                'batch_size': 16,
                'num_epochs': 5,
                'learning_rate': 1e-3,
                'temporal_mask_ratio': [0.2, 0.2],  # Fixed ratio for debugging
                'neuron_mask_ratio': [0.15, 0.15],
                'exclude_ids': ['13122.0'],  # Train on more data for debug
                'target_trials_per_site': 500,  # Smaller dataset
                'min_val_test_trials': 10, # Smaller dataset
                'experiment_name': 'debug_run',
                'use_site_specific_heads': False,
            }
        },
        
        'quick_test': {
            'description': 'Quick test run with medium model',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 16,
                'num_epochs': 20,
                'learning_rate': 5e-5,
                'target_trials_per_site': 1000,
                'experiment_name': 'quick_test'
            }
        },
        
        'full_small': {
            'description': 'Full training run with small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,  # 32
                'num_epochs': 1000,
                'learning_rate': 5e-5,
                'experiment_name': 'full_small_model'
            }
        },
        
        'full_medium': {
            'description': 'Full training run with medium model',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 32,
                'num_epochs': 1000,
                'learning_rate': 5e-5,
                'experiment_name': 'full_medium_model'
            }
        },
        
        'full_large': {
            'description': 'Full training run with large model (multi-GPU recommended)',
            'overrides': {
                'model_size': 'large',
                'batch_size': 32,
                'num_epochs': 1000,
                'learning_rate': 5e-5,
                'warmup_epochs': 50,
                'experiment_name': 'full_large_model'
            }
        },
        
        'ablation_no_contrastive_small': {
            'description': 'Ablation study: no contrastive loss, small model',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'learning_rate': 5e-5,
                'contrastive_weight': 0.0,
                'experiment_name': 'ablation_no_contrastive_small'
            }
        },
        
        'ablation_no_contrastive_medium': {
            'description': 'Ablation study: no contrastive loss, medium model',
            'overrides': {
                'model_size': 'medium',
                'batch_size': 32,
                'num_epochs': 1000,
                'learning_rate': 5e-5,
                'contrastive_weight': 0.0,
                'experiment_name': 'ablation_no_contrastive_medium'
            }
        },
        
        'ablation_high_masking': {
            'description': 'Ablation study: high masking ratios',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'temporal_mask_ratio': [0.4, 0.6],
                'neuron_mask_ratio': [0.3, 0.5],
                'experiment_name': 'ablation_high_masking'
            }
        },
        
        'ablation_uniform_masking': {
            'description': 'Ablation study: uniform masking ratios',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'temporal_mask_ratio': [0.0, 1.0],
                'neuron_mask_ratio': [0.0, 1.0],
                'experiment_name': 'ablation_uniform_masking'
            }
        },
        
        
        'ablation_high_masking_temporal_only': {
            'description': 'Ablation study: high masking ratios',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'temporal_mask_ratio': [0.4, 0.6],
                'neuron_mask_ratio': [0.1, 0.3],
                'experiment_name': 'ablation_high_masking_temporal_only'
            }
        },
        
        # CRITICAL ARCHITECTURAL ABLATIONS for research comparisons
        'ablation_learnable_pe': {
            'description': 'Ablation study: Learnable vs RoPE3D positional encoding',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'pos_encoding_type': 'learnable',
                'experiment_name': 'ablation_learnable_pe'
            }
        },
        
        'ablation_standard_rope_pe': {
            'description': 'Ablation study: Standard RoPE (1D temporal) vs RoPE3D',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'pos_encoding_type': 'standard_rope',
                'experiment_name': 'ablation_standard_rope_pe'
            }
        },
        
        'ablation_sinusoidal_pe': {
            'description': 'Ablation study: Sinusoidal PE vs RoPE3D positional encoding',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'pos_encoding_type': 'sinusoidal',
                'experiment_name': 'ablation_sinusoidal_pe'
            }
        },
        
        'ablation_no_temporal_kernels': {
            'description': 'Ablation study: disable adaptive temporal kernels',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'no_temporal_kernels': True,
                'experiment_name': 'ablation_no_temporal_kernels'
            }
        },
        
        'ablation_large_kernels': {
            'description': 'Ablation study: large kernel size [9,9]',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'kernel_size': [9, 9],
                'experiment_name': 'ablation_large_kernels'
            }
        },
        
        'ablation_site_specific_decoder': {
            'description': 'Ablation study: shared vs site-specific decoder heads',
            'overrides': {
                'model_size': 'small',
                'batch_size': 64,
                'num_epochs': 1000,
                'use_site_specific_heads': True,
                'experiment_name': 'ablation_site_specific_decoder'
            }
        }
    }


def print_available_presets():
    """Print available configuration presets."""
    configs = create_quick_configs()
    
    print("\n" + "="*80)
    print("AVAILABLE CONFIGURATION PRESETS")
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
        description='Neural Foundation Model Pretraining Launcher',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Preset configurations
    parser.add_argument('--preset', type=str, choices=list(create_quick_configs().keys()),
                       help='Use a preset configuration')
    parser.add_argument('--list_presets', action='store_true',
                       help='List available preset configurations and exit')
    
    # Common overrides (will be passed to main training script)
    parser.add_argument('--config', type=str, default='config/training/foundation_pretrain.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large'],
                       help='Model size preset')
    
    # Individual model parameters (override model_size presets)
    parser.add_argument('--d_model', type=int, help='Model hidden dimension (overrides model_size)')
    parser.add_argument('--temporal_layers', type=int, help='Number of temporal encoder layers (overrides model_size)')
    parser.add_argument('--spatial_layers', type=int, help='Number of spatial encoder layers (overrides model_size)')
    parser.add_argument('--heads', type=int, help='Number of attention heads (overrides model_size)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (overrides model_size)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    
    # Quick dataset options
    parser.add_argument('--exclude_ids', nargs='+', help='Dataset IDs to exclude')
    parser.add_argument('--target_neurons', type=int, help='Target neurons per site')
    parser.add_argument('--target_trials_per_site', type=int, help='Target trials per site')
    parser.add_argument('--min_val_test_trials', type=int, help='Minimum trials for val/test')
    
    # Quick masking options
    parser.add_argument('--temporal_mask_ratio', type=float, nargs='+',
                       help='Temporal mask ratio (single value or min max)')
    parser.add_argument('--neuron_mask_ratio', type=float, nargs='+',
                       help='Neuron mask ratio (single value or min max)')
    parser.add_argument('--contrastive_weight', type=float, help='Contrastive loss weight')
    
    # CRITICAL architectural options for research comparisons
    parser.add_argument('--pos_encoding_type', type=str, choices=['rope_3d', 'standard_rope', 'learnable', 'sinusoidal'],
                       help='Positional encoding type (MAJOR architectural choice)')
    parser.add_argument('--use_temporal_kernels', type=str, default='true', choices=['true', 'false'],
                       help='Enable adaptive causal kernel attention')
    parser.add_argument('--kernel_size', type=int, nargs=2,
                       help='Kernel size for adaptive attention [height width]')
    parser.add_argument('--spatial_scale', type=float, help='Spatial encoding scale factor')
    
    # Other common options
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--base_dir', type=str, help='Base directory for outputs')
    parser.add_argument('--seed', type=int, help='Random seed')
    
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
        configs = create_quick_configs()
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
                elif key == 'exclude_ids':
                    cmd_args.extend([f'--{key}'] + value)
                elif key == 'kernel_size' and isinstance(value, list):
                    cmd_args.extend([f'--{key}'] + [str(v) for v in value])
                elif key in ['use_temporal_kernels']:
                    # Handle boolean flags
                    if value:
                        cmd_args.append(f'--{key}')
                else:
                    cmd_args.extend([f'--{key}', str(value)])
    
    # Add user overrides (these take precedence over preset)
    if args.model_size:
        cmd_args.extend(['--model_size', args.model_size])
    
    # Individual model parameters (override model_size if specified)
    if args.d_model:
        cmd_args.extend(['--d_model', str(args.d_model)])
    if args.temporal_layers:
        cmd_args.extend(['--temporal_layers', str(args.temporal_layers)])
    if args.spatial_layers:
        cmd_args.extend(['--spatial_layers', str(args.spatial_layers)])
    if args.heads:
        cmd_args.extend(['--heads', str(args.heads)])
    if args.dropout is not None:
        cmd_args.extend(['--dropout', str(args.dropout)])
    
    # Training parameters
    if args.batch_size:
        cmd_args.extend(['--batch_size', str(args.batch_size)])
    if args.learning_rate:
        cmd_args.extend(['--learning_rate', str(args.learning_rate)])
    if args.num_epochs:
        cmd_args.extend(['--num_epochs', str(args.num_epochs)])
    if args.experiment_name:
        cmd_args.extend(['--experiment_name', args.experiment_name])
    if args.exclude_ids:
        cmd_args.extend(['--exclude_ids'] + args.exclude_ids)
    if args.target_neurons:
        cmd_args.extend(['--target_neurons', str(args.target_neurons)])
    if args.target_trials_per_site:
        cmd_args.extend(['--target_trials_per_site', str(args.target_trials_per_site)])
    if args.min_val_test_trials:
        cmd_args.extend(['--min_val_test_trials', str(args.min_val_test_trials)])
    if args.temporal_mask_ratio:
        cmd_args.extend(['--temporal_mask_ratio'] + [str(v) for v in args.temporal_mask_ratio])
    if args.neuron_mask_ratio:
        cmd_args.extend(['--neuron_mask_ratio'] + [str(v) for v in args.neuron_mask_ratio])
    if args.contrastive_weight:
        cmd_args.extend(['--contrastive_weight', str(args.contrastive_weight)])
    
    # CRITICAL architectural options for research comparisons
    if args.pos_encoding_type:
        cmd_args.extend(['--pos_encoding_type', args.pos_encoding_type])
    if args.use_temporal_kernels is not None:
        cmd_args.extend(['--use_temporal_kernels', args.use_temporal_kernels])
    if args.kernel_size:
        cmd_args.extend(['--kernel_size'] + [str(v) for v in args.kernel_size])
    if args.spatial_scale:
        cmd_args.extend(['--spatial_scale', str(args.spatial_scale)])
    
    if args.resume:
        cmd_args.extend(['--resume', args.resume])
    if args.base_dir:
        cmd_args.extend(['--base_dir', args.base_dir])
    if args.seed:
        cmd_args.extend(['--seed', str(args.seed)])
    
    print(f"\n🔥 Starting Neural Foundation Model Pretraining")
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
        sys.argv = ['pretrain.py'] + cmd_args
        main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main_launcher() 