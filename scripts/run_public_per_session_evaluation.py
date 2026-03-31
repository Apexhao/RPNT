#!/usr/bin/env python3
"""
Per-Session Evaluation Launcher for Public Dataset (Clean 4-Mode System)
------------------------------------------------------------------------

This script runs the per-session evaluation protocol where each session is trained and evaluated individually, 
then results are aggregated across sessions. Features clean 4-mode initialization system.

4 Clean Initialization Modes:
- 'random': Same architecture as foundation, random weights
- 'foundation_pretrained': Foundation encoder + random prediction head  
- 'downstream_complete': Complete downstream model (encoder + head)
- 'downstream_encoder_only': Downstream encoder + random prediction head

Usage:
    # Quick debug run with foundation pretrained mode
    python scripts/run_public_per_session_evaluation.py --preset debug \
        --pretrained_path ./logs_public/full_small/checkpoints/best.pth
    
    # Cross-subject center evaluation with fine-tuning  
    python scripts/run_public_per_session_evaluation.py --preset cross_subject_center_finetune \
        --pretrained_path ./logs_public/full_small/checkpoints/best.pth
    
    # Test different initialization modes
    python scripts/run_public_per_session_evaluation.py --scenario cross_subject_center \
        --initialization_mode random --pretrained_path ./logs_public/foundation.pth
    
    # Use downstream checkpoint with complete model
    python scripts/run_public_per_session_evaluation.py --scenario cross_session \
        --initialization_mode downstream_complete --pretrained_path ./logs_downstream/model.pth
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.public_per_session_trainers import create_per_session_evaluation_manager
from src.utils.helpers import load_config, set_seed


def create_per_session_presets():
    """Create quick configuration presets for per-session evaluation tasks."""
    return {
        'debug': {
            'description': 'Quick debug run with minimal epochs',
            'overrides': {
                'scenario': 'cross_subject_center',
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 1e-2,
                'training_mode': 'frozen_encoder',
                'experiment_name': 'per_session_debug',
            }
        },
        
        'cross_session_finetune': {
            'description': 'Cross-session evaluation with encoder fine-tuning',
            'overrides': {
                'scenario': 'cross_session',
                'training_mode': 'full_finetune',
                'learning_rate': 5e-4,
                'num_epochs': 200,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_session_finetune'
            }
        },
        
        'cross_subject_center_finetune': {
            'description': 'Cross-subject center-out with encoder fine-tuning',
            'overrides': {
                'scenario': 'cross_subject_center',
                'training_mode': 'full_finetune',
                'learning_rate': 5e-4,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_subject_center_finetune'
            }
        },
        
        'cross_subject_random_finetune': {
            'description': 'Cross-subject random-target with encoder fine-tuning',
            'overrides': {
                'scenario': 'cross_subject_random',
                'training_mode': 'full_finetune',
                'learning_rate': 5e-4,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_subject_random_finetune'
            }
        },
        
        'full_evaluation': {
            'description': 'Comprehensive evaluation across all scenarios with fine-tuning',
            'overrides': {
                'scenario': 'all',
                'training_mode': 'full_finetune',
                'learning_rate': 5e-4,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_full_evaluation'
            }
        },
        
        'cross_session_frozen': {
            'description': 'Cross-session evaluation with frozen encoder',
            'overrides': {
                'scenario': 'cross_session',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 30,
                'early_stopping_patience': 15,
                'experiment_name': 'per_session_cross_session_frozen'
            }
        },
        
        'cross_subject_center_frozen': {
            'description': 'Cross-subject center-out with frozen encoder',
            'overrides': {
                'scenario': 'cross_subject_center',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 40,
                'early_stopping_patience': 15,
                'experiment_name': 'per_session_cross_subject_center_frozen'
            }
        },
        
        
        'cross_subject_random_frozen': {
            'description': 'Cross-subject random-target with frozen encoder',
            'overrides': {
                'scenario': 'cross_subject_random',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 50,
                'early_stopping_patience': 20,
                'experiment_name': 'per_session_cross_subject_random_frozen'
            }
        },
             
        'frozen_comparison': {
            'description': 'Comparison of frozen encoder across all scenarios',
            'overrides': {
                'scenario': 'all',
                'training_mode': 'frozen_encoder',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_frozen_comparison'
            }
        },
        
        # Clean initialization mode presets
        'random_init_debug': {
            'description': 'Quick debug run with random initialization',
            'overrides': {
                'scenario': 'cross_subject_center',
                'initialization_mode': 'random',
                'training_mode': 'full_finetune',
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 1e-2,
                'experiment_name': 'per_session_random_debug',
            }
        },
        
        'cross_session_random': {
            'description': 'Cross-session evaluation with random initialization',
            'overrides': {
                'scenario': 'cross_session',
                'initialization_mode': 'random',
                'training_mode': 'full_finetune',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_session_random'
            }
        },
        
        'cross_subject_center_random': {
            'description': 'Cross-subject center-out with random initialization',
            'overrides': {
                'scenario': 'cross_subject_center',
                'initialization_mode': 'random',
                'training_mode': 'full_finetune',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_subject_center_random'
            }
        },
        
        'cross_subject_random_random': {
            'description': 'Cross-subject random-target with random initialization',
            'overrides': {
                'scenario': 'cross_subject_random',
                'initialization_mode': 'random',
                'training_mode': 'full_finetune',
                'learning_rate': 1e-3,
                'num_epochs': 100,
                'early_stopping_patience': 100,
                'experiment_name': 'per_session_cross_subject_random_random'
            }
        },
        
        'downstream_complete_debug': {
            'description': 'Debug with complete downstream model',
            'overrides': {
                'scenario': 'cross_subject_center',
                'initialization_mode': 'downstream_complete',
                'training_mode': 'full_finetune',
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 1e-3,
                'experiment_name': 'per_session_downstream_complete_debug'
            }
        },
        
        'downstream_encoder_only_debug': {
            'description': 'Debug with downstream encoder only',
            'overrides': {
                'scenario': 'cross_subject_center',
                'initialization_mode': 'downstream_encoder_only',
                'training_mode': 'full_finetune',
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 1e-3,
                'experiment_name': 'per_session_downstream_encoder_only_debug'
            }
        },
    }    


def print_available_presets():
    """Print available configuration presets."""
    configs = create_per_session_presets()
    
    print("\n" + "="*80)
    print("AVAILABLE PER-SESSION EVALUATION PRESETS")
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
    print(f"   python {__file__} --preset cross_subject_center_frozen --learning_rate 5e-4")
    print("="*80)


def evaluate_single_scenario(scenario: str, 
                           checkpoint_path: str,
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single per-session scenario with clean initialization.
    
    Args:
        scenario: Evaluation scenario
        checkpoint_path: Path to checkpoint (foundation or downstream)
        config: Training configuration
        
    Returns:
        Dictionary with aggregated and per-session results
    """
    
    print(f"\n🎯 Evaluating {scenario} scenario...")
    print("="*60)
    
    start_time = time.time()
    
    # Update configuration for this scenario
    config['evaluation']['scenario'] = scenario
    
    # Update experiment name to include initialization mode (only if not custom)
    model_config = config.get('model', {})
    initialization_mode = model_config.get('initialization_mode', 'foundation_pretrained')
    current_experiment_name = config['paths'].get('experiment_name')
    
    # Only auto-generate if no custom name was provided (None, empty string, or default)
    if not current_experiment_name or current_experiment_name == 'null':
        config['paths']['experiment_name'] = f"per_session_{scenario}_{initialization_mode}"
        print(f"🔧 Auto-generated experiment name: {config['paths']['experiment_name']}")
    else:
        print(f"✅ Using custom experiment name: {current_experiment_name}")
    
    try:
        # Create evaluation manager with clean initialization
        evaluator = create_per_session_evaluation_manager(
            checkpoint_path=checkpoint_path,
            evaluation_scenario=scenario,
            config=config
        )
        
        print(f"🏋️ Starting per-session evaluation...")
        
        # Run evaluation (manager handles all evaluation internally)
        results = evaluator.run_per_session_evaluation()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ {scenario.upper()} EVALUATION COMPLETED!")
        print(f"⏱️  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
        # Return basic results dict for consistency
        return {
            'scenario': scenario,
            'elapsed_time': elapsed_time,
            'config': config,
            'experiment_dir': evaluator.paths_config.get('experiment_dir', 'N/A') if hasattr(evaluator, 'paths_config') else 'N/A'
        }
        
    except Exception as e:
        print(f"❌ {scenario.upper()} EVALUATION FAILED!")
        print(f"Error: {str(e)}")
        raise e


def main():
    """Main function for per-session evaluation."""
    parser = argparse.ArgumentParser(
        description='Public Dataset Per-Session Evaluation with Aggregated Session Analysis',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Preset configurations
    parser.add_argument('--dataroot', type=str,
                       help='Data root')
    parser.add_argument('--preset', type=str, choices=list(create_per_session_presets().keys()),
                       help='Use a preset configuration')
    parser.add_argument('--list_presets', action='store_true',
                       help='List available preset configurations and exit')
    
    # NOTE: Only single evaluation mode supported - run scenarios manually
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/public_per_session_regression.yaml',
                       help='Path to configuration file')
    
    # Scenario configuration
    parser.add_argument('--scenario', type=str, 
                       choices=['cross_session', 'cross_subject_center', 'cross_subject_random'],
                       help='Type of per-session scenario')
    
    parser.add_argument('--target_type', type=str, choices=['velocity', 'position'],
                       help='Target type')
    
    # Model configuration (clean - no defaults, config is source of truth)
    parser.add_argument('--pretrained_path', type=str,
                       help='Path to checkpoint (foundation or downstream)')
    parser.add_argument('--initialization_mode', type=str,
                       choices=['random', 'foundation_pretrained', 'downstream_complete', 'downstream_encoder_only'],
                       help='Model initialization mode: random (same arch, random weights), foundation_pretrained (foundation encoder + random head), downstream_complete (complete downstream model), downstream_encoder_only (downstream encoder + random head)')
    parser.add_argument('--training_mode', type=str, choices=['frozen_encoder', 'full_finetune', 'partial_finetune'],
                       help='Training mode')
    
    # NOTE: Model architecture parameters (neural_dim, d_model, etc.) are loaded from 
    # checkpoint and cannot be overridden via CLI
    
    # Dataset configuration
    parser.add_argument('--target_neurons', type=int,
                       help='Target neurons per session')
       
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    
    parser.add_argument('--decay_epochs', type=int,
                       help='Decay epochs')
    
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs per session')
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
        presets = create_per_session_presets()
        if args.preset in presets:
            preset_config = presets[args.preset]['overrides']
            print(f"\n🚀 Using preset: {args.preset}")
            print(f"📝 Description: {presets[args.preset]['description']}")
            
            # Apply preset overrides
            for key, value in preset_config.items():
                if key == 'scenario':
                    config['evaluation'][key] = value
                elif key in ['training_mode', 'learning_rate', 'num_epochs', 'early_stopping_patience']:
                    config['training'][key] = value
                elif key in ['batch_size']:
                    config['training'][key] = value
                elif key == 'initialization_mode':
                    config['model'][key] = value
                elif key in ['compare_pretrained_vs_random']:
                    config['advanced'][key] = value
                elif key == 'experiment_name':
                    config['paths'][key] = value
    
    # Clean override logic: only override if CLI arg was explicitly provided
    # Config is source of truth, CLI args only override when specified
    
    # Scenario override
    if args.dataroot is not None:
        config['dataset']['data_root'] = args.dataroot
    if args.scenario is not None:
        config['evaluation']['scenario'] = args.scenario
    
    # Model overrides
    if args.initialization_mode is not None:
        config['model']['initialization_mode'] = args.initialization_mode
    if args.pretrained_path is not None:
        config['paths']['pretrained_path'] = args.pretrained_path
    
    # Training overrides
    if args.training_mode is not None:
        config['training']['training_mode'] = args.training_mode
    if args.target_type is not None:
        config['training']['target_type'] = args.target_type
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.decay_epochs is not None:
        config['training']['decay_epochs'] = args.decay_epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.early_stopping_patience is not None:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    # Dataset overrides
    if args.target_neurons is not None:
        config['dataset']['target_neurons'] = args.target_neurons
    
    # Output configuration overrides
    if args.experiment_name is not None:
        config['paths']['experiment_name'] = args.experiment_name
        print(f"🎯 Custom experiment name set from CLI: {args.experiment_name}")
    if args.base_dir is not None:
        config['paths']['base_dir'] = args.base_dir
    
    # Seed overrides
    if args.seed is not None:
        config['dataset']['random_seed'] = args.seed
        config['training']['random_seed'] = args.seed
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    print("=" * 80)
    print("PUBLIC DATASET PER-SESSION EVALUATION")
    print("=" * 80)
    
    # Show clean initialization mode
    model_config = config.get('model', {})
    initialization_mode = model_config.get('initialization_mode', 'foundation_pretrained')
    pretrained_path = config['paths'].get('pretrained_path', './logs_public/full_small/checkpoints/best.pth')
    
    print(f"Initialization Mode: {initialization_mode}")
    print(f"Pretrained Path: {pretrained_path}")
    print(f"Training Mode: {config['training']['training_mode']}")
    print("NEW PROTOCOL: Each session trained individually, results aggregated")
    print("=" * 80)
    
    try:
        # Run single evaluation only
        if not args.scenario and not args.preset:
            print("❌ Error: --scenario is required for evaluation")
            return
        
        scenario = config['evaluation']['scenario']
        results = evaluate_single_scenario(
            scenario,
            pretrained_path,
            config
        )
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise e


if __name__ == '__main__':
    main()
