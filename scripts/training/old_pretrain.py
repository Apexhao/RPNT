import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix for OpenMP conflict on Windows

import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import numpy as np
import torch.nn as nn
import time
import argparse
from typing import Dict, Optional
from torch.utils.data import DataLoader
# Import from the new organized structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.evaluation.metrics import NeuralMAELoss
from src.models.transformer import KernelTransformerMAE
from src.utils.helpers import LinearWarmUp, plot_training_history, WarmupCosineSchedule, set_seed, load_config
from src.data.dataset import Combined_Monkey_Dataset

class MAETrainer:
    def __init__(self, model: KernelTransformerMAE, dataset, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.batch_size = config['training']['batch_size']
        self.mask_ratio_neurons = config['training']['mask_ratio_neurons']
        self.mask_ratio_time = config['training']['mask_ratio_time']
        self.lambda_reg = config['training']['lambda_reg']
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create data loaders
        self.train_loader = dataset.create_dataset('train', batch_size=config['training']['batch_size'], shuffle=True)
        self.val_loader = dataset.create_dataset('valid', batch_size=config['training']['batch_size'], shuffle=False)
        self.test_loader = dataset.create_dataset('test', batch_size=config['training']['batch_size'], shuffle=False)
        
        # Initialize loss function
        self.criterion = NeuralMAELoss(
            temperature=config['training']['temperature'],
            lambda_rec=config['training']['lambda_rec'],
            lambda_con=config['training']['lambda_con']
        ).to(self.device)
        
        # Initialize optimizer with warm-up
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=0.01)
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=config['training']['warmup_steps'],
            total_steps=config['training']['num_epochs'],
            min_lr_ratio=config['training']['min_lr_ratio']
        )
        
        # Initialize logging with run directory
        self.log_dir = config['paths']['tensorboard_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging to save under runs directory
        self.log_file = os.path.join(self.log_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(config['paths']['tensorboard_dir'])
        
        # Log initial configuration and model info
        self.log_model_info()
        self.log_training_config(config)
        
        # Track additional metrics
        self.step_count = 0
        
    def log_model_info(self):
        """Log detailed model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Log to both file and tensorboard
        self.logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Add model architecture info to tensorboard
        self.writer.add_text('Model/Architecture', f"""
        **Model Configuration:**
        - Total Parameters: {total_params:,}
        - Trainable Parameters: {trainable_params:,}
        - Input Dimension: {self.model.input_proj.in_features}
        - Model Dimension: {self.model.input_proj.out_features}
        - Encoder Layers: {len(self.model.encoder)}
        - Decoder Layers: {len(self.model.decoder)}
        - Threshold Log Rates: {self.model.threshold_log_rates}
        """)
        
    def log_training_config(self, config: dict):
        """Log training configuration to tensorboard."""
        config_text = f"""
        **Training Configuration:**
        - Model Size: {config['training']['model_size']}
        - Learning Rate: {config['training']['learning_rate']}
        - Batch Size: {config['training']['batch_size']}
        - Mask Ratio (Neurons): {config['training']['mask_ratio_neurons']}
        - Mask Ratio (Time): {config['training']['mask_ratio_time']}
        - Temperature: {config['training']['temperature']}
        - Lambda Reconstruction: {config['training']['lambda_rec']}
        - Lambda Contrastive: {config['training']['lambda_con']}
        - Lambda Regularization: {config['training']['lambda_reg']}
        - Warmup Steps: {config['training']['warmup_steps']}
        - Total Epochs: {config['training']['num_epochs']}
        
        **Dataset Configuration:**
        - Excluded IDs: {config['dataset']['exclude_ids']}
        - Width: {config['dataset']['width']}
        - Target Neuron Dim: {config['dataset']['target_neuron_dim']}
        """
        self.writer.add_text('Training/Configuration', config_text)
        
    def compute_l2_regularization(self):
        """Compute L2 regularization loss."""
        l2_loss = 0.0
        for param in self.model.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss
        
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        total_metrics = {}
        num_batches = 0
        
        for batch_idx, (spikes, _, _, _) in enumerate(self.train_loader):
            
            # Create augmented views
            augmented_1, augmented_2 = self.dataset.create_augmented_pairs(spikes)

            augmented_1 = augmented_1.to(self.device)
            augmented_2 = augmented_2.to(self.device)
            
            # Create masked data
            _, binary_mask = self.dataset.create_masked_data(spikes)

            # Move data to device
            spikes = spikes.to(self.device)
            augmented_1 = augmented_1.to(self.device)
            augmented_2 = augmented_2.to(self.device)
            binary_mask = binary_mask.to(self.device)

            # Forward pass for both views
            log_rates_1, encoded_1 = self.model(augmented_1, binary_mask)
            log_rates_2, encoded_2 = self.model(augmented_2, binary_mask)
        
            # Compute main loss
            loss, metrics = self.criterion(
                log_rates_1,
                spikes,
                binary_mask,
                encoded_1,
                encoded_2
            )
            
            # Add L2 regularization
            l2_loss = self.compute_l2_regularization()
            total_loss = loss + self.lambda_reg * l2_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Step the scheduler (now after each batch)
            self.scheduler.step()
            
            # Log current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update metrics
            metrics['l2_reg'] = l2_loss.item()
            metrics['total_loss'] = total_loss.item()
            metrics['learning_rate'] = current_lr
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, split: str = 'valid') -> Dict[str, float]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        loader = self.val_loader if split == 'valid' else self.test_loader
        
        for spikes, _, _, _ in loader:
            
            # Create augmented views
            augmented_1, augmented_2 = self.dataset.create_augmented_pairs(spikes)

            augmented_1 = augmented_1.to(self.device)
            augmented_2 = augmented_2.to(self.device)
            
            # Create masked data
            _, binary_mask = self.dataset.create_masked_data(spikes)
            
            # Forward pass for both views
            log_rates_1, encoded_1 = self.model(augmented_1, binary_mask)
            log_rates_2, encoded_2 = self.model(augmented_2, binary_mask)

            # Move data to device
            spikes = spikes.to(self.device)
            augmented_1 = augmented_1.to(self.device)
            augmented_2 = augmented_2.to(self.device)
            binary_mask = binary_mask.to(self.device)
            
            # Compute loss
            loss, metrics = self.criterion(
                log_rates_1,
                spikes,
                binary_mask,
                encoded_1,
                encoded_2
            )
            
            # Add L2 regularization
            l2_loss = self.compute_l2_regularization()
            metrics['l2_reg'] = l2_loss.item()
            metrics['total_loss'] = loss.item() + self.lambda_reg * l2_loss.item()
            
            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
    
    def train(self, num_epochs: int, save_freq: int = 5):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Log to tensorboard - Training metrics
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            
            # Validation
            val_metrics = self.evaluate('valid')
            for name, value in val_metrics.items():
                self.writer.add_scalar(f'val/{name}', value, epoch)
            
            # Test
            test_metrics = self.evaluate('test')
            for name, value in test_metrics.items():
                self.writer.add_scalar(f'test/{name}', value, epoch)
            
            # Log additional training statistics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate_actual', current_lr, epoch)
            
            # Log optimization statistics every 10 epochs
            if epoch % 10 == 0:
                self.log_gradient_statistics(epoch)
                self.log_model_statistics(epoch)
            
            # Log loss ratios and improvements
            rec_con_ratio = train_metrics.get('weighted_rec_loss', 0) / (train_metrics.get('weighted_con_loss', 1e-8) + 1e-8)
            self.writer.add_scalar('train/rec_con_loss_ratio', rec_con_ratio, epoch)
            
            # Log data statistics
            self.writer.add_scalar('data/mean_spike_rate', train_metrics.get('mean_count', 0), epoch)
            self.writer.add_scalar('data/predicted_rate', train_metrics.get('mean_rate', 0), epoch)
            self.writer.add_scalar('data/masking_efficiency', train_metrics.get('num_masked', 0), epoch)
            
            # Log contrastive learning efficiency
            pos_neg_ratio = train_metrics.get('positive_sim', 0) / (abs(train_metrics.get('negative_sim', 1e-8)) + 1e-8)
            self.writer.add_scalar('contrastive/pos_neg_similarity_ratio', pos_neg_ratio, epoch)
            
            # Enhanced progress logging with more metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Rec: {train_metrics.get('weighted_rec_loss', 0):.4f}, "
                f"Con: {train_metrics.get('weighted_con_loss', 0):.4f}, "
                f"L2: {train_metrics.get('l2_reg', 0):.4f}) - "
                f"Val Loss: {val_metrics['total_loss']:.4f} - "
                f"Test Loss: {test_metrics['total_loss']:.4f} - "
                f"LR: {current_lr:.2e} - "
                f"Mean Rate: {train_metrics.get('mean_rate', 0):.2f}Hz"
            )
            
            # Log significant improvements
            if epoch > 0:
                prev_val_loss = best_val_loss
                if val_metrics['total_loss'] < prev_val_loss * 0.95:  # 5% improvement
                    self.logger.info(f"Significant validation improvement: {prev_val_loss:.4f} -> {val_metrics['total_loss']:.4f}")
            
            # Save checkpoint if best model
            if val_metrics['total_loss'] < best_val_loss:
                improvement = best_val_loss - val_metrics['total_loss']
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(f'best_model.pth')
                self.logger.info(f"New best model saved! Improvement: {improvement:.4f}")
                
                # Log best model metrics
                self.writer.add_scalar('best_model/epoch', epoch, epoch)
                self.writer.add_scalar('best_model/val_loss', best_val_loss, epoch)
            
            # Regular checkpoint saving
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                self.logger.info(f"Regular checkpoint saved at epoch {epoch+1}")
                
        # Final logging
        self.writer.close()
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
        self.logger.info(f"Training logs saved to: {self.log_file}")
        self.logger.info(f"TensorBoard logs saved to: {self.writer.log_dir}")
    
    def log_gradient_statistics(self, epoch: int):
        """Log gradient statistics for monitoring training health."""
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual parameter gradients
                self.writer.add_scalar(f'gradients/{name}/norm', param_norm.item(), epoch)
                self.writer.add_scalar(f'gradients/{name}/mean', param.grad.data.mean().item(), epoch)
                self.writer.add_scalar(f'gradients/{name}/std', param.grad.data.std().item(), epoch)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('gradients/total_norm', total_norm, epoch)
        self.writer.add_scalar('gradients/param_count', param_count, epoch)
        
    def log_model_statistics(self, epoch: int):
        """Log model parameter statistics for monitoring."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Current parameter statistics
                self.writer.add_scalar(f'parameters/{name}/mean', param.data.mean().item(), epoch)
                self.writer.add_scalar(f'parameters/{name}/std', param.data.std().item(), epoch)
                self.writer.add_scalar(f'parameters/{name}/norm', param.data.norm().item(), epoch)
                self.writer.add_scalar(f'parameters/{name}/min', param.data.min().item(), epoch)
                self.writer.add_scalar(f'parameters/{name}/max', param.data.max().item(), epoch)
                
                # Log parameter histograms every 50 epochs
                if epoch % 50 == 0:
                    self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'neural_dim': self.model.input_proj.in_features,
                'd_model': self.model.input_proj.out_features,
                'threshold_log_rates': self.model.threshold_log_rates
            }
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

def generate_experiment_name(config: dict) -> str:
    """Generate experiment name based on configuration."""
    model_size = config['training']['model_size']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    mask_n = config['training']['mask_ratio_neurons']
    mask_t = config['training']['mask_ratio_time']
    lambda_con = config['training']['lambda_con']
    
    # Create exclude_ids string
    exclude_str = "_".join(config['dataset']['exclude_ids']).replace('.0', '')
    
    # Add first_n mode information if applicable
    first_n_str = ""
    if config['dataset'].get('no_chunking_first_n') is not None:
        first_n_str = f"_first{config['dataset']['no_chunking_first_n']}"
    
    name = f"mae_{model_size}_bs{batch_size}_lr{lr}_mn{mask_n}_mt{mask_t}_lc{lambda_con}_ex{exclude_str}{first_n_str}"
    return name

def merge_model_config(config: dict) -> dict:
    """Merge model size configuration with base model config."""
    model_size = config['training']['model_size']
    
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
        config['training']['model_size'] = args.model_size
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.mask_ratio_neurons:
        config['training']['mask_ratio_neurons'] = args.mask_ratio_neurons
    if args.mask_ratio_time:
        config['training']['mask_ratio_time'] = args.mask_ratio_time
    if args.lambda_con:
        config['training']['lambda_con'] = args.lambda_con
    if args.exclude_ids:
        config['dataset']['exclude_ids'] = args.exclude_ids
    if args.experiment_name:
        config['paths']['experiment_name'] = args.experiment_name
    
    return config

def get_model_params_count(config: dict) -> int:
    """Estimate model parameters for logging."""
    neural_dim = config['model']['neural_dim']
    d_model = config['model']['d_model']
    heads = config['model']['heads']
    enc_layers = config['model']['encoder_layers']
    dec_layers = config['model']['decoder_layers']
    
    # Rough estimation (not exact but good for comparison)
    # Input projection + positional encoding
    input_params = neural_dim * d_model
    
    # Encoder layers (attention + FFN)
    enc_attn_params = enc_layers * (3 * d_model * d_model + d_model * d_model)  # QKV + proj
    enc_ffn_params = enc_layers * (d_model * 4 * d_model + 4 * d_model * d_model)  # FFN
    
    # Decoder layers (self-attn + cross-attn + FFN)
    dec_attn_params = dec_layers * (6 * d_model * d_model + 2 * d_model * d_model)  # 2x attn + proj
    dec_ffn_params = dec_layers * (d_model * 4 * d_model + 4 * d_model * d_model)  # FFN
    
    # Output projection
    output_params = d_model * neural_dim
    
    total = input_params + enc_attn_params + enc_ffn_params + dec_attn_params + dec_ffn_params + output_params
    return total

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Neural Foundation Model with MAE')
    parser.add_argument('--config', type=str, default='config/config_pretrain.yaml', 
                       help='Path to config file')
    parser.add_argument('--model_size', type=str, choices=['small', 'medium', 'large'],
                       help='Model size (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--mask_ratio_neurons', type=float, help='Neuron masking ratio (overrides config)')
    parser.add_argument('--mask_ratio_time', type=float, help='Time masking ratio (overrides config)')
    parser.add_argument('--lambda_con', type=float, help='Contrastive loss weight (overrides config)')
    parser.add_argument('--exclude_ids', nargs='+', help='Dataset IDs to exclude (overrides config)')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name (overrides auto-generation)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
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
    logger.info("=" * 50)
    logger.info("NEURAL FOUNDATION MODEL TRAINING")
    logger.info("=" * 50)
    logger.info(f"Model Size: {config['training']['model_size']}")
    logger.info(f"Estimated Parameters: {get_model_params_count(config):,}")
    logger.info(f"Experiment Name: {config['paths']['experiment_name']}")
    logger.info(f"Checkpoint Dir: {config['paths']['checkpoint_dir']}")
    logger.info(f"Tensorboard Dir: {config['paths']['tensorboard_dir']}")
    logger.info("=" * 50)
    
    # Set random seeds
    set_seed(config['training']['seed'])
    
    try:
        # Initialize dataset
        logger.info("Initializing dataset...")
        dataset = Combined_Monkey_Dataset(
            exclude_ids=config['dataset']['exclude_ids'],
            width=config['dataset']['width'],
            target_neuron_dim=config['dataset']['target_neuron_dim'],
            neuron_selection_strategy=config['dataset']['neuron_selection_strategy'],
            selected_neurons=config['dataset']['selected_neurons'],
            split_ratios=tuple(config['dataset']['split_ratios']),
            no_chunking_first_n=config['dataset'].get('no_chunking_first_n', None)
        )
        
        logger.info(f"Dataset loaded - Train: {dataset.train_data.shape}, "
                   f"Valid: {dataset.valid_data.shape}, Test: {dataset.test_data.shape}")

        # Initialize model
        logger.info("Initializing model...")
        model = KernelTransformerMAE(
            neural_dim=config['model']['neural_dim'],
            d_model=config['model']['d_model'],
            heads=config['model']['heads'],
            encoder_layers=config['model']['encoder_layers'],
            decoder_layers=config['model']['decoder_layers'],
            kernel_size=config['model']['kernel_size'],
            dropout=config['model']['dropout'],
            threshold_log_rates=config['model']['threshold_log_rates'],
            activity_threshold=config['model']['activity_threshold']
        )
        
        # Print actual model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = MAETrainer(model, dataset, config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Checkpoint loaded successfully!")
        
        # Train model
        logger.info("Starting training...")
        trainer.train(num_epochs=config['training']['num_epochs'], 
                     save_freq=config['training']['save_freq'])
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 