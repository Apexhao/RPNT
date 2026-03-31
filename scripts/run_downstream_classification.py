#!/usr/bin/env python3
"""
Launcher script for downstream classification tasks.

This script provides easy access to classification training using pretrained foundation models.

Usage:
    # Basic classification training
    python scripts/run_downstream_classification.py --dataset_id te14116

    # Custom configuration
    python scripts/run_downstream_classification.py --dataset_id te14116 \
        --num_classes 8 --learning_rate 0.001 --num_epochs 100

    # Different pretrained model
    python scripts/run_downstream_classification.py --dataset_id te14116 \
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
    """Main function for downstream classification training."""
    parser = argparse.ArgumentParser(
        description='Downstream Classification Training Launcher',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training/downstream_classification.yaml',
                       help='Path to configuration file')
    
    # Dataset configuration
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='Dataset ID (e.g., te14116)')
    parser.add_argument('--target_neurons', type=int, default=50,
                       help='Target number of neurons')
    parser.add_argument('--neuron_strategy', type=str, choices=['first_n', 'random_n', 'all'],
                       default='first_n', help='Neuron selection strategy')
    
    # Model configuration
    parser.add_argument('--pretrained_path', type=str,
                       default='./logs/full_small_model/checkpoints/best.pth',
                       help='Path to pretrained foundation model')
    parser.add_argument('--training_mode', type=str, choices=['frozen_encoder', 'finetune_encoder'],
                       default='frozen_encoder', help='Training mode')
    parser.add_argument('--num_classes', type=int, default=8,
                       help='Number of classes')
    parser.add_argument('--temporal_pooling', type=str, choices=['mean', 'last', 'attention'],
                       default='mean', help='Temporal pooling strategy')
    
    # Training configuration
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Output configuration
    parser.add_argument('--experiment_name', type=str,
                       help='Custom experiment name (auto-generated if not provided)')
    parser.add_argument('--base_dir', type=str, default='./logs',
                       help='Base directory for outputs')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset_id:
        config['dataset']['dataset_id'] = args.dataset_id
    if args.target_neurons:
        config['dataset']['target_neurons'] = args.target_neurons
        config['dataset']['selected_neurons'] = args.target_neurons
    if args.neuron_strategy:
        config['dataset']['neuron_selection_strategy'] = args.neuron_strategy
    
    if args.pretrained_path:
        config['paths']['pretrained_path'] = args.pretrained_path
    if args.training_mode:
        config['training']['training_mode'] = args.training_mode
    if args.num_classes:
        config['training']['num_classes'] = args.num_classes
    if args.temporal_pooling:
        config['model']['temporal_pooling'] = args.temporal_pooling
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.label_smoothing:
        config['training']['label_smoothing'] = args.label_smoothing
    if args.early_stopping_patience:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    if args.experiment_name:
        config['paths']['experiment_name'] = args.experiment_name
    else:
        # Auto-generate experiment name
        config['paths']['experiment_name'] = f"classification_{args.dataset_id}_{args.num_classes}classes_{args.training_mode}"
    if args.base_dir:
        config['paths']['base_dir'] = args.base_dir
    
    if args.seed:
        config['dataset']['random_seed'] = args.seed
        config['training']['random_seed'] = args.seed
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    print("=" * 80)
    print("DOWNSTREAM CLASSIFICATION TRAINING")
    print("=" * 80)
    print(f"Dataset ID: {config['dataset']['dataset_id']}")
    print(f"Number of Classes: {config['training']['num_classes']}")
    print(f"Training Mode: {config['training']['training_mode']}")
    print(f"Temporal Pooling: {config['model']['temporal_pooling']}")
    print(f"Pretrained Model: {config['paths']['pretrained_path']}")
    print(f"Experiment: {config['paths']['experiment_name']}")
    print("=" * 80)
    
    try:
        # Initialize dataset
        print("Initializing dataset...")
        dataset = SingleSiteDownstreamDataset(
            dataset_id=config['dataset']['dataset_id'],
            data_root=config['dataset']['data_root'],
            split_ratios=tuple(config['dataset']['split_ratios']),
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
        print("Creating classification trainer...")
        trainer = create_downstream_trainer(
            task_type='classification',
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