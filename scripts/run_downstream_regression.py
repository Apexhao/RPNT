#!/usr/bin/env python3
"""
Launcher script for downstream regression tasks.

This script provides easy access to regression training using pretrained foundation models.

Usage:
    # Basic regression training
    python scripts/run_downstream_regression.py --dataset_id te14116

    # Custom configuration
    python scripts/run_downstream_regression.py --dataset_id te14116 \
        --target_type velocity --learning_rate 0.001 --num_epochs 100

    # Different pretrained model
    python scripts/run_downstream_regression.py --dataset_id te14116 \
        --pretrained_path ./logs/full_medium_model/checkpoints/best.pth
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downstream_dataset import SingleSiteDownstreamDataset
from src.training.downstream_trainers import create_downstream_trainer
from src.utils.helpers import load_config, set_seed


def main():
    """Main function for downstream regression training."""
    parser = argparse.ArgumentParser(
        description='Downstream Regression Training Launcher',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/downstream_regression.yaml',
                       help='Path to configuration file')
    
    # Dataset configuration
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='Dataset ID (e.g., te14116)')
    parser.add_argument('--target_neurons', type=int,
                       help='Target number of neurons')
    parser.add_argument('--split_ratios', type=float, nargs=3,
                       help='Split ratios for train, val, test')
    parser.add_argument('--neuron_strategy', type=str, choices=['first_n', 'random_n', 'all'],
                       help='Neuron selection strategy')
    
    # Model configuration
    parser.add_argument('--pretrained_path', type=str,
                       help='Path to pretrained foundation model')
    parser.add_argument('--training_mode', type=str, choices=['frozen_encoder', 'finetune_encoder', 'random'],
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
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Clean override logic: only override if CLI arg was explicitly provided
    # Config is source of truth, CLI args only override when specified
    
    # Dataset overrides
    if args.dataset_id is not None:
        config['dataset']['dataset_id'] = args.dataset_id
    if args.target_neurons is not None:
        config['dataset']['target_neurons'] = args.target_neurons
        config['dataset']['selected_neurons'] = args.target_neurons
    if args.split_ratios is not None:
        config['dataset']['split_ratios'] = args.split_ratios
    if args.neuron_strategy is not None:
        config['dataset']['neuron_selection_strategy'] = args.neuron_strategy
    
    # Path overrides
    if args.pretrained_path is not None:
        config['paths']['pretrained_path'] = args.pretrained_path
    
    # Training overrides
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
    else:
        # Auto-generate experiment name if not specified in config or CLI
        if config['paths'].get('experiment_name') is None:
            config['paths']['experiment_name'] = f"regression_{config['dataset']['dataset_id']}_{config['training']['target_type']}_{config['training']['training_mode']}"
    if args.base_dir is not None:
        config['paths']['base_dir'] = args.base_dir
    
    # Seed overrides
    if args.seed is not None:
        config['dataset']['random_seed'] = args.seed
        config['training']['random_seed'] = args.seed
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    print("=" * 80)
    print("DOWNSTREAM REGRESSION TRAINING")
    print("=" * 80)
    print(f"Dataset ID: {config['dataset']['dataset_id']}")
    print(f"Split Ratios: {config['dataset']['split_ratios']}")
    print(f"Target Type: {config['training']['target_type']}")
    print(f"Training Mode: {config['training']['training_mode']}")
    print(f"Pretrained Model: {config['paths']['pretrained_path']}")
    print(f"Experiment: {config['paths']['experiment_name']}")
    print("=" * 80)
    
    try:
        # Initialize dataset
        print("Initializing dataset...")
        dataset = SingleSiteDownstreamDataset(
            dataset_id=config['dataset']['dataset_id'],
            data_root=config['dataset']['data_root'],
            split_ratios=config['dataset']['split_ratios'],
            target_neurons=config['dataset']['target_neurons'],
            width=config['dataset']['width'],
            sequence_length=config['dataset']['sequence_length'],
            neuron_selection_strategy=config['dataset']['neuron_selection_strategy'],
            selected_neurons=config['dataset']['selected_neurons'],
            random_seed=config['dataset']['random_seed']
        )
        
        # Print dataset summary
        dataset.print_summary()
        
        # Create trainer
        print("Creating regression trainer...")
        trainer = create_downstream_trainer(
            task_type='regression',
            pretrained_checkpoint_path=config['paths']['pretrained_path'],
            dataset=dataset,
            config=config
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("Starting training...")
        trainer.train(config['training']['num_epochs'])
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise e


if __name__ == '__main__':
    main() 