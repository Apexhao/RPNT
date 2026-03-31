import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
# Import from the new organized structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import KernelTransformerMAE, KernelTransformerRegressor
from src.data.dataset import Monkey_beignet_Dataset_selected_width
from src.utils.helpers import set_seed, load_config
from typing import Dict

class RegressionTrainer:
    def __init__(self, model: KernelTransformerRegressor, dataset, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.target_type = config['finetune']['target_type']  # 'position' or 'velocity'
        
        # Create data loaders
        self.train_loader = dataset.create_dataset('train', 
                                                 batch_size=config['finetune']['batch_size'], 
                                                 shuffle=True)
        self.val_loader = dataset.create_dataset('valid', 
                                               batch_size=config['finetune']['batch_size'], 
                                               shuffle=False)
        self.test_loader = dataset.create_dataset('test', 
                                                batch_size=config['finetune']['batch_size'], 
                                                shuffle=False)
        
        # MSE Loss for training
        self.criterion = nn.MSELoss()
        
        # Optimizer with simple learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['finetune']['learning_rate']),
            weight_decay=0.01
        )
        
        # Simple learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['finetune']['decay_epochs'],
            gamma=float(config['finetune']['decay_rate'])
        )
        
        # Initialize logging and TensorBoard
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        self.log_dir = config['paths']['tensorboard_dir']
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.log_dir, 'finetune.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Log configuration
        self.log_training_config(config)
        
    def log_training_config(self, config: dict):
        """Log training configuration to tensorboard."""
        config_text = f"""
        **Fine-tuning Configuration:**
        - Model Size: {config['finetune']['model_size']}
        - Target Type: {config['finetune']['target_type']}
        - Training Mode: {config['finetune']['training_mode']}
        - Learning Rate: {config['finetune']['learning_rate']}
        - Batch Size: {config['finetune']['batch_size']}
        - Output Dim: {config['finetune']['output_dim']}
        - Decay Epochs: {config['finetune']['decay_epochs']}
        - Decay Rate: {config['finetune']['decay_rate']}
        - Total Epochs: {config['finetune']['num_epochs']}
        
        **Dataset Configuration:**
        - Dataset ID: {config['dataset']['dataset_id']}
        - Width: {config['dataset']['width']}
        - Target Neuron Dim: {config['dataset']['target_neuron_dim']}
        
        **Model Configuration:**
        - Neural Dim: {config['model']['neural_dim']}
        - D Model: {config['model']['d_model']}
        - Heads: {config['model']['heads']}
        - Encoder Layers: {config['model']['encoder_layers']}
        - Decoder Layers: {config['model']['decoder_layers']}
        """
        self.writer.add_text('Training/Configuration', config_text)
    
    def compute_r2_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute R² score between predictions and targets."""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        # Compute R² for each dimension and average
        r2_scores = []
        for i in range(pred.shape[-1]):  # For each output dimension (x,y)
            r2 = r2_score(target[..., i].flatten(), pred[..., i].flatten())
            r2_scores.append(r2)
        
        return np.mean(r2_scores)
    
    def get_target_data(self, batch_data):
        """Extract target data based on target_type."""
        spikes, labels, positions, velocities = batch_data
        if self.target_type == 'position':
            return spikes, positions
        elif self.target_type == 'velocity':
            return spikes, velocities
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_data in self.train_loader:
            # Get appropriate target based on target_type
            spikes, targets = self.get_target_data(batch_data)

            targets = targets[:,:,:2]
            spikes = spikes.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(spikes)

            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Store predictions and targets for R² computation
            all_preds.append(predictions.detach())
            all_targets.append(targets.detach())
            total_loss += loss.item()
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_r2 = self.compute_r2_score(all_preds, all_targets)
        
        return epoch_loss, epoch_r2
    
    @torch.no_grad()
    def evaluate(self, split: str = 'valid') -> Dict[str, float]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        loader = self.val_loader if split == 'valid' else self.test_loader
        
        for batch_data in loader:
            # Get appropriate target based on target_type
            spikes, targets = self.get_target_data(batch_data)

            targets = targets[:,:,:2]

            spikes = spikes.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(spikes)
            loss = self.criterion(predictions, targets)
            
            # Store predictions and targets
            all_preds.append(predictions)
            all_targets.append(targets)
            total_loss += loss.item()
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2 = self.compute_r2_score(all_preds, all_targets)
        
        return {
            'loss': total_loss / len(loader),
            'r2_score': r2
        }

    def train(self, num_epochs: int, save_freq: int = 5):
        """Main training loop."""
        best_val_r2 = float('-inf')
        self.logger.info(f"Starting fine-tuning for {num_epochs} epochs...")
        self.logger.info(f"Target type: {self.target_type}")

        self.logger.info("Initial regressor weight norms:")
        if hasattr(self.model, 'log_weight_norms'):
            self.model.log_weight_norms()
        
        for epoch in range(num_epochs):
            # Training epoch
            train_loss, train_r2 = self.train_epoch()
            if hasattr(self.model, 'log_weight_norms'):
                self.model.log_weight_norms()
            self.scheduler.step()

            # Validation
            val_metrics = self.evaluate('valid')
            test_metrics = self.evaluate('test')
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/test', test_metrics['loss'], epoch)
            self.writer.add_scalar('R2/train', train_r2, epoch)
            self.writer.add_scalar('R2/val', val_metrics['r2_score'], epoch)
            self.writer.add_scalar('R2/test', test_metrics['r2_score'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Test Loss: {test_metrics['loss']:.4f} - "
                f"Train R2: {train_r2:.4f}, Val R2: {val_metrics['r2_score']:.4f}, Test R2: {test_metrics['r2_score']:.4f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if val_metrics['r2_score'] > best_val_r2:
                improvement = val_metrics['r2_score'] - best_val_r2
                best_val_r2 = val_metrics['r2_score']
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"New best model saved! R2 improvement: {improvement:.4f}")
                
                # Log best model metrics
                self.writer.add_scalar('best_model/epoch', epoch, epoch)
                self.writer.add_scalar('best_model/val_r2', best_val_r2, epoch)
            
            # Regular checkpoint saving
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                self.logger.info(f"Regular checkpoint saved at epoch {epoch+1}")
                
        # Final logging
        self.writer.close()
        self.logger.info("Fine-tuning completed!")
        self.logger.info(f"Best validation R2 achieved: {best_val_r2:.4f}")
        self.logger.info(f"Training logs saved to: {self.log_file}")
        self.logger.info(f"TensorBoard logs saved to: {self.log_dir}")

        # Check regressor weights after training
        if hasattr(self.model, 'analyze_regressor_weights'):
            weight_stats = self.model.analyze_regressor_weights()
            self.logger.info("Regressor weight statistics:")
            for layer_name, layer_stats in weight_stats.items():
                self.logger.info(f"{layer_name}:")
                for stat_name, value in layer_stats.items():
                    self.logger.info(f"  {stat_name}: {value}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'target_type': self.target_type,
                'output_dim': self.model.regressor[-1].out_features
            }
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

def generate_experiment_name(config: dict) -> str:
    """Generate experiment name based on configuration."""
    model_size = config['finetune']['model_size']
    dataset_id = config['dataset']['dataset_id']
    target_type = config['finetune']['target_type']
    lr = config['finetune']['learning_rate']
    batch_size = config['finetune']['batch_size']
    training_mode = config['finetune']['training_mode']
    
    # Add split ratios to name
    split_ratios = config['dataset']['split_ratios']
    split_str = f"split{split_ratios[0]:.3f}_{split_ratios[1]:.3f}_{split_ratios[2]:.3f}".replace('.', 'p')
    
    # Add neuron selection info
    neuron_strategy = config['dataset']['neuron_selection_strategy']
    if neuron_strategy == 'all':
        neuron_str = "allneurons"
    else:
        selected_n = config['dataset']['selected_neurons']
        neuron_str = f"{neuron_strategy}{selected_n}neurons"
    
    name = f"finetune_{target_type}_{model_size}_{dataset_id}_bs{batch_size}_lr{lr}_{training_mode}_{split_str}_{neuron_str}"
    return name

def merge_model_config(config: dict) -> dict:
    """Merge model size configuration with base model config."""
    model_size = config['finetune']['model_size']
    
    if model_size not in config['model_sizes']:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(config['model_sizes'].keys())}")
    
    # Merge size-specific config with base model config
    size_config = config['model_sizes'][model_size]
    config['model'].update(size_config)
    
    return config

def setup_paths(config: dict) -> dict:
    """Setup checkpoint and tensorboard paths based on config."""
    if config['paths']['experiment_name'] is None:
        config['paths']['experiment_name'] = generate_experiment_name(config)
    
    base_dir = config['paths']['base_dir']
    exp_name = config['paths']['experiment_name']
    
    config['paths']['checkpoint_dir'] = os.path.join(base_dir, 'checkpoints', exp_name)
    config['paths']['tensorboard_dir'] = os.path.join(base_dir, 'runs', exp_name)
    
    return config

def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """Override config values from command line arguments."""
    if args.model_size:
        config['finetune']['model_size'] = args.model_size
    if args.dataset_id:
        config['dataset']['dataset_id'] = args.dataset_id
    if args.target_type:
        config['finetune']['target_type'] = args.target_type
    if args.batch_size:
        config['finetune']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['finetune']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['finetune']['num_epochs'] = args.num_epochs
    if args.training_mode:
        config['finetune']['training_mode'] = args.training_mode
    if args.experiment_name:
        config['paths']['experiment_name'] = args.experiment_name
    if args.pretrained_path:
        config['paths']['pretrained_path'] = args.pretrained_path
    if args.neuron_selection_strategy:
        config['dataset']['neuron_selection_strategy'] = args.neuron_selection_strategy
    if args.selected_neurons:
        config['dataset']['selected_neurons'] = args.selected_neurons
    if args.split_ratios:
        config['dataset']['split_ratios'] = args.split_ratios
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune Neural Foundation Model for Regression')
    parser.add_argument('--config', type=str, default='config/config_finetune_regression.yaml', 
                       help='Path to config file')
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large'],
                       help='Model size (overrides config)')
    parser.add_argument('--dataset_id', type=str, help='Dataset ID to use (e.g., te14116)')
    parser.add_argument('--target_type', type=str, choices=['position', 'velocity'],
                       help='Target type for regression (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--training_mode', type=str, choices=['from_scratch', 'full_finetune', 'frozen_encoder'],
                       help='Training mode: from_scratch, full_finetune, or frozen_encoder')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained model')
    
    # New arguments for neuron selection and few-shot learning
    parser.add_argument('--neuron_selection_strategy', type=str, choices=['all', 'first_n', 'random_n'],
                       help='Neuron selection strategy (overrides config)')
    parser.add_argument('--selected_neurons', type=int, help='Number of neurons to select (overrides config)')
    parser.add_argument('--split_ratios', nargs=3, type=float, metavar=('TRAIN', 'VALID', 'TEST'),
                       help='Data split ratios for train/valid/test (e.g., 0.02 0.48 0.5)')
    

    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load and process config
    config = load_config(args.config)
    config = override_config_from_args(config, args)
    config = merge_model_config(config)
    config = setup_paths(config)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("NEURAL FOUNDATION MODEL FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model Size: {config['finetune']['model_size']}")
    logger.info(f"Dataset ID: {config['dataset']['dataset_id']}")
    logger.info(f"Target Type: {config['finetune']['target_type']}")
    logger.info(f"Training Mode: {config['finetune']['training_mode']}")
    logger.info(f"Experiment Name: {config['paths']['experiment_name']}")
    logger.info(f"Checkpoint Dir: {config['paths']['checkpoint_dir']}")
    logger.info(f"Tensorboard Dir: {config['paths']['tensorboard_dir']}")
    logger.info(f"Pretrained Path: {config['paths']['pretrained_path']}")
    logger.info("=" * 60)
    
    # Set random seed
    set_seed(config['finetune']['seed'])

    try:
        # Initialize dataset
        logger.info("Initializing dataset...")
        dataset = Monkey_beignet_Dataset_selected_width(
            dataset_id=config['dataset']['dataset_id'], 
            width=config['dataset']['width'],
            target_neuron_dim=config['dataset']['target_neuron_dim'],
            neuron_selection_strategy=config['dataset']['neuron_selection_strategy'],
            selected_neurons=config['dataset']['selected_neurons'],
            split_ratios=tuple(config['dataset']['split_ratios'])
        )

        # Handle three training modes
        training_mode = config['finetune']['training_mode']
        logger.info(f"Training mode: {training_mode}")
        
        if training_mode == "from_scratch":
            # Mode 1: Train from scratch
            logger.info("Initializing model from scratch...")
            pretrained_model = KernelTransformerMAE(**config['model'])  # Create architecture template
            logger.info("Training everything from random initialization.")
            
        else:
            # Mode 2 & 3: Use pretrained weights
            if config['paths']['pretrained_path'] is None:
                raise ValueError(f"pretrained_path is required for training_mode='{training_mode}'")
            
            logger.info("Loading pretrained model...")
            pretrained_model = KernelTransformerMAE(**config['model'])
            
            logger.info("Loading pretrained weights...")
            checkpoint = torch.load(config['paths']['pretrained_path'], map_location='cuda')
            pretrained_model.load_state_dict(checkpoint['model_state_dict'])
            
            if training_mode == "full_finetune":
                logger.info("Full fine-tuning: training all parameters from pretrained weights.")
            elif training_mode == "frozen_encoder":
                logger.info("Frozen encoder fine-tuning: only training regressor head.")

        # Initialize regressor with training mode
        logger.info("Initializing regressor...")
        regressor = KernelTransformerRegressor(
            pretrained_model=pretrained_model,
            output_dim=config['finetune']['output_dim'],
            training_mode=training_mode
        )
        
        # Log parameter status
        logger.info("Parameter status:")
        regressor.log_parameter_status()

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = RegressionTrainer(regressor, dataset, config)
        
        # Train model
        logger.info("Starting fine-tuning...")
        trainer.train(
            num_epochs=config['finetune']['num_epochs'],
            save_freq=config['finetune']['save_freq']
        )
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise e

if __name__ == "__main__":
    main()