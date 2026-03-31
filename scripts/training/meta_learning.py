import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
# Import from the new organized structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import KernelTransformerMAE, KernelTransformerRegressor
from src.data.dataset import Session_MAML_Monkey_Dataset, SineWaveDataset
from src.utils.helpers import set_seed, load_config, WarmupCosineSchedule
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn.functional as F
import itertools

class ANILFOMAMLRegressorTrainer:
    def __init__(self, model: KernelTransformerRegressor, dataset: Session_MAML_Monkey_Dataset, config: dict):
        """
        Initialize ANIL trainer.
        
        Args:
            model: Base model with frozen transformer backbone and adaptable readout layer
            dataset: Dataset for meta-learning
            config: Configuration dictionary
        """

        self.dropout = config['model']['dropout']
        self.d_model = config['model']['d_model']
        self.output_dim = config['anil_fomaml']['output_dim']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        
        # Target type
        self.target_type = config['anil_fomaml']['target_type']

        # Meta-learning parameters
        self.k_shot = config['anil_fomaml']['k_shot']
        self.k_query = config['anil_fomaml']['k_query']
        
        # Adjust batch_size to match number of training sessions
        logging.info(f"Number of training sessions: {len(dataset.train_sessions)}")

        self.batch_size = len(dataset.train_sessions)
        if self.batch_size != config['anil_fomaml']['batch_size']:
            logging.info(
                f"Adjusting batch_size from {config['anil_fomaml']['batch_size']} to {self.batch_size} "
                f"to match number of available training sessions"
            )
        
        self.num_batches = config['anil_fomaml']['num_batches']
        self.inner_lr = config['anil_fomaml']['inner_lr']
        self.num_inner_steps = config['anil_fomaml']['num_inner_steps']
        
        # Create meta-learning dataloaders
        self.train_loader = dataset.create_meta_dataloader(
            split='train',
            k_shot=self.k_shot,
            k_query=self.k_query,
            batch_size=self.batch_size,  # Now using adjusted batch_size
            target_type=self.target_type,
            num_batches=self.num_batches
        )
        
        self.val_loader = dataset.create_meta_dataloader(
            split='valid',
            k_shot=self.k_shot,
            k_query=self.k_query,
            batch_size=len(dataset.valid_session),
            target_type=self.target_type,
            num_batches=self.num_batches
        )
        
        self.test_loader = dataset.create_meta_dataloader(
            split='test',
            k_shot=self.k_shot,
            k_query=self.k_query,
            batch_size=len(dataset.test_session),
            target_type=self.target_type,
            num_batches=self.num_batches
        )
        
        # MSE Loss for both inner and outer loops
        self.criterion = nn.MSELoss()

        for param in self.model.regressor[4].parameters():
            param.requires_grad = False
        
        # Outer loop optimizer (only for readout regressor layer)
        first_layer_params = {'params': model.regressor[1].parameters(), 'weight_decay': 0.1}
        #second_layer_params = {'params': model.regressor[4].parameters(), 'weight_decay': 0.5}  # Higher weight decay
        
        self.optimizer = torch.optim.AdamW(
            [first_layer_params], # Only optimize feature transformation layer, not the regressor layer, aka, Linear layer 1 
            lr=float(config['anil_fomaml']['outer_lr']),
        )
        
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=config['anil_fomaml']['warmup_steps'],
            total_steps=config['anil_fomaml']['num_epochs'],
            min_lr_ratio=0.0  # Final learning rate will be 0
        )
        
        # Initialize logging
        self.writer = SummaryWriter(config['paths']['tensorboard_dir'])
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join(config['paths']['checkpoint_dir'], 'anil.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Training history
        self.history = {
            # Losses
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'learning_rates': [],
            
            # Training set metrics
            'train_pre_adapt_support_loss': [], 'train_post_adapt_support_loss': [],
            'train_pre_adapt_query_loss': [], 'train_post_adapt_query_loss': [],
            'train_pre_adapt_support_r2': [], 'train_post_adapt_support_r2': [],
            'train_pre_adapt_query_r2': [], 'train_post_adapt_query_r2': [],
            'train_adaptation_improvement_support_loss': [], 'train_adaptation_improvement_query_loss': [],
            'train_adaptation_improvement_support_r2': [], 'train_adaptation_improvement_query_r2': [],
            
            # Validation set metrics
            'val_pre_adapt_support_loss': [], 'val_post_adapt_support_loss': [],
            'val_pre_adapt_query_loss': [], 'val_post_adapt_query_loss': [],
            'val_pre_adapt_support_r2': [], 'val_post_adapt_support_r2': [],
            'val_pre_adapt_query_r2': [], 'val_post_adapt_query_r2': [],
            'val_adaptation_improvement_support_loss': [], 'val_adaptation_improvement_query_loss': [],
            'val_adaptation_improvement_support_r2': [], 'val_adaptation_improvement_query_r2': [],
            
            # Test set metrics
            'test_pre_adapt_support_loss': [], 'test_post_adapt_support_loss': [],
            'test_pre_adapt_query_loss': [], 'test_post_adapt_query_loss': [],
            'test_pre_adapt_support_r2': [], 'test_post_adapt_support_r2': [],
            'test_pre_adapt_query_r2': [], 'test_post_adapt_query_r2': [],
            'test_adaptation_improvement_support_loss': [], 'test_adaptation_improvement_query_loss': [],
            'test_adaptation_improvement_support_r2': [], 'test_adaptation_improvement_query_r2': [],
            
            # Adaptation trajectories
            'val_adaptation_trajectory_support_loss': [], 'val_adaptation_trajectory_query_loss': [],
            'val_adaptation_trajectory_support_r2': [], 'val_adaptation_trajectory_query_r2': [],
            'test_adaptation_trajectory_support_loss': [], 'test_adaptation_trajectory_query_loss': [],
            'test_adaptation_trajectory_support_r2': [], 'test_adaptation_trajectory_query_r2': []
        }

    def forward_with_params(self, features: torch.Tensor, params: Dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        """
        Forward pass using provided parameters directly with F.* functions to maintain gradient flow.
        """
        x = features
        
        # Layer Norm (using parameters directly)
        x = F.layer_norm(x, 
                        normalized_shape=[self.d_model],
                        weight=None,
                        bias=None,
                        eps=1e-4)
        
        # First Linear
        x = F.linear(x, params['1.weight'], params['1.bias'])
        
        # GELU
        x = F.gelu(x)
        
        # Dropout
        x = F.dropout(x, p=self.dropout, training=training)
        
        # Final Linear
        #x = F.linear(x, params['4.weight'], params['4.bias'])

        # Final Linear (frozen - use directly from model)
        final_linear = self.model.regressor[4]
        x = F.linear(x, final_linear.weight, final_linear.bias)
        
        return x

    def adapt_to_task(self, support_spikes: torch.Tensor, support_target: torch.Tensor, num_steps: int = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Adapt model to a new task (session) using support set (inner loop).
        Args:
            support_spikes: Support set spike data [k_shot, time_steps, num_neurons]
            support_target: Support set targets [k_shot, time_steps, 2]
            num_steps: Optional number of adaptation steps (defaults to self.num_inner_steps)
        """
        num_steps = num_steps if num_steps is not None else self.num_inner_steps
        
        # Get pre-adaptation predictions on support set
        with torch.no_grad():
            support_features = self.model.encode(support_spikes)
            pre_adapt_pred = self.model.regressor(support_features)
            pre_adapt_loss = self.criterion(pre_adapt_pred, support_target).item()
            pre_adapt_support_r2 = self.compute_r2_score(pre_adapt_pred, support_target)
        
        # Clone parameters while maintaining computational graph
        adapted_params = {
            name: param.clone()
            for name, param in self.model.regressor.named_parameters()
            if name.startswith('1.')  # First layer parameters
        }
        
        adaptation_metrics = {
            'pre_adapt_loss': pre_adapt_loss,
            'pre_adapt_r2': pre_adapt_support_r2,
        }
        
        # Inner loop adaptation
        support_features = self.model.encode(support_spikes)
        
        for step in range(num_steps):
            predictions = self.forward_with_params(support_features, adapted_params)
            loss = self.criterion(predictions, support_target)
            
            # Compute gradients with graph tracking
            grads = torch.autograd.grad(loss, adapted_params.values(),
                                      create_graph=True, allow_unused=True)
            
            # Clip gradients to prevent exploding gradients
            clipped_grads = []
            for grad in grads:
                if grad is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad * (1.0 / grad_norm)
                clipped_grads.append(grad)
            
            # Update parameters with clipped gradients
            adapted_params = {
                name: param - self.inner_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(adapted_params.items(), clipped_grads)
                if name.startswith('1.')
            }
        
        # Get post-adaptation performance
        with torch.no_grad():
            post_adapt_pred = self.forward_with_params(support_features, adapted_params, training=False)
            post_adapt_loss = self.criterion(post_adapt_pred, support_target).item()
            post_adapt_support_r2 = self.compute_r2_score(post_adapt_pred, support_target)

        adaptation_metrics['post_adapt_loss'] = post_adapt_loss
        adaptation_metrics['post_adapt_r2'] = post_adapt_support_r2

        return adapted_params, adaptation_metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using ANIL + FOMAML with session-based tasks."""
        self.model.train()
        total_meta_loss = 0
        
        # Track adaptation metrics across all tasks/sessions
        epoch_metrics = {
            'pre_adapt_support_r2': [],
            'pre_adapt_support_loss': [],
            'post_adapt_support_r2': [],
            'post_adapt_support_loss': [],
            'pre_adapt_query_r2': [],
            'pre_adapt_query_loss': [],
            'post_adapt_query_r2': [],
            'post_adapt_query_loss': [],
            'adaptation_improvement_support_r2': [],
            'adaptation_improvement_query_r2': [],
            'adaptation_improvement_support_loss': [],
            'adaptation_improvement_query_loss': []
        }
        
        for batch in self.train_loader:

            # print("Regressor parameters before update:")
            # for name, param in self.model.regressor.named_parameters():
            #      print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}, mean={param.mean().item():.4f}")

            # Clear gradients at the start of each batch
            self.optimizer.zero_grad()
            
            support_spikes = batch['support_spikes'].to(self.device)  # [batch_size, k_shot, time, neurons]
            support_target = batch['support_target'].to(self.device)  # [batch_size, k_shot, time, 2]
            query_spikes = batch['query_spikes'].to(self.device)    # [batch_size, k_query, time, neurons]
            query_target = batch['query_target'].to(self.device)    # [batch_size, k_query, time, 2]
            
            task_losses = []
            
            # Adapt to each task (session) in the batch
            for i in range(support_spikes.size(0)):
                # Get pre-adaptation query performance
                with torch.no_grad():
                    query_features = self.model.encode(query_spikes[i])
                    pre_query_pred = self.model.regressor(query_features)
                    pre_query_loss = self.criterion(pre_query_pred, query_target[i]).item()
                    pre_query_r2 = self.compute_r2_score(pre_query_pred, query_target[i])
                
                # Inner loop: Adapt to task (session) using support set
                adapted_params, adapt_metrics = self.adapt_to_task(
                    support_spikes[i],
                    support_target[i]
                )
                
                # Compute query loss for meta-update
                query_features = self.model.encode(query_spikes[i])
                query_pred = self.forward_with_params(query_features, adapted_params)
                query_loss = self.criterion(query_pred, query_target[i])
                task_losses.append(query_loss)  # Keep as tensor for backward pass
                
                # Store metrics (using no_grad to save memory)
                with torch.no_grad():
                    post_query_r2 = self.compute_r2_score(query_pred, query_target[i])
                    post_query_loss = self.criterion(query_pred, query_target[i]).item()

                    # Store metrics for losses
                    epoch_metrics['pre_adapt_support_loss'].append(adapt_metrics['pre_adapt_loss'])
                    epoch_metrics['post_adapt_support_loss'].append(adapt_metrics['post_adapt_loss'])
                    epoch_metrics['pre_adapt_query_loss'].append(pre_query_loss)
                    epoch_metrics['post_adapt_query_loss'].append(post_query_loss)
                    epoch_metrics['adaptation_improvement_support_loss'].append(adapt_metrics['post_adapt_loss'] - adapt_metrics['pre_adapt_loss'])
                    epoch_metrics['adaptation_improvement_query_loss'].append(post_query_loss - pre_query_loss)

                    # Store metrics for R2 scores
                    epoch_metrics['pre_adapt_support_r2'].append(adapt_metrics['pre_adapt_r2'])
                    epoch_metrics['post_adapt_support_r2'].append(adapt_metrics['post_adapt_r2'])
                    epoch_metrics['pre_adapt_query_r2'].append(pre_query_r2)
                    epoch_metrics['post_adapt_query_r2'].append(post_query_r2)
                    epoch_metrics['adaptation_improvement_support_r2'].append(adapt_metrics['post_adapt_r2'] - adapt_metrics['pre_adapt_r2'])
                    epoch_metrics['adaptation_improvement_query_r2'].append(post_query_r2 - pre_query_r2)

            # Outer loop: Meta-update using query set performance
            meta_loss = torch.stack(task_losses).mean()
            # print(f"meta_loss: {meta_loss}")

            meta_loss.backward()  # Gradients flow through the computational graph
            # Optional: Add gradient norm logging
            grad_norm = torch.norm(torch.stack([
                p.grad.norm() 
                for p in self.model.regressor.parameters() 
                if p.grad is not None
            ]))
            self.writer.add_scalar('Gradients/meta_gradient_norm', grad_norm.item(), self.num_batches * len(self.history['train_loss']))
            
            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(self.model.regressor.parameters(), 1.0)

            # print("\nGradient norms before optimizer step:")
            # for name, param in self.model.regressor.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad norm: {param.grad.norm().item():.4f}")

            self.optimizer.step()
            self.scheduler.step() 
            total_meta_loss += meta_loss.item()

            # Print regressor parameters after optimization step
            # print("\nRegressor parameters after update:")
            # for name, param in self.model.regressor.named_parameters():
            #     print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}, mean={param.mean().item():.4f}")
        
        # Average metrics across all tasks
        avg_metrics = {'loss': total_meta_loss / len(self.train_loader)}
        
        for key in epoch_metrics:
            avg_metrics[key] = np.mean(epoch_metrics[key])
        
        return avg_metrics

    def evaluate(self, split: str = 'valid', num_tasks: int = None, num_steps: int = None) -> Dict[str, float]:
        """
        Evaluate model's fast adaptation ability on validation or test session.
        Each adaptation is independent, starting from the trained initialization.
        
        Args:
            split: 'valid' or 'test'
            num_tasks: Number of tasks to evaluate (if None, evaluate all tasks)
            num_steps: Number of adaptation steps (if None, use self.num_inner_steps)
        """
        self.model.eval()
        total_loss = 0
        
        # Track adaptation metrics
        eval_metrics = {
            'pre_adapt_support_loss': [],
            'post_adapt_support_loss': [],
            'pre_adapt_query_loss': [],
            'post_adapt_query_loss': [],
            'pre_adapt_support_r2': [],
            'post_adapt_support_r2': [],
            'pre_adapt_query_r2': [],
            'post_adapt_query_r2': [],
            'adaptation_improvement_support_loss': [],
            'adaptation_improvement_query_loss': [],
            'adaptation_improvement_support_r2': [],
            'adaptation_improvement_query_r2': [],
            'adaptation_trajectory_support_loss': [],  # List of lists for each task
            'adaptation_trajectory_query_loss': [],    # List of lists for each task
            'adaptation_trajectory_support_r2': [],    # List of lists for each task
            'adaptation_trajectory_query_r2': []       # List of lists for each task
        }
        
        # Select appropriate loader
        loader = self.val_loader if split == 'valid' else self.test_loader
        
        # Use specified number of steps or default
        num_steps = num_steps if num_steps is not None else self.num_inner_steps
        
        # Get iterator for specified number of tasks or all tasks
        task_iterator = itertools.islice(loader, num_tasks) if num_tasks is not None else loader
        
        for batch in task_iterator:
            support_spikes = batch['support_spikes'].to(self.device)
            support_target = batch['support_target'].to(self.device)
            query_spikes = batch['query_spikes'].to(self.device)
            query_target = batch['query_target'].to(self.device)
            
            batch_losses = []
            
            for i in range(support_spikes.size(0)):
                # Initialize trajectories for this task
                task_support_loss_trajectory = []
                task_query_loss_trajectory = []
                task_support_r2_trajectory = []
                task_query_r2_trajectory = []
                
                # Get pre-adaptation performance
                with torch.no_grad():
                    # Support set pre-adaptation
                    support_features = self.model.encode(support_spikes[i])
                    pre_support_pred = self.model.regressor(support_features)
                    pre_support_loss = self.criterion(pre_support_pred, support_target[i]).item()
                    pre_support_r2 = self.compute_r2_score(pre_support_pred, support_target[i])
                    
                    # Query set pre-adaptation
                    query_features = self.model.encode(query_spikes[i])
                    pre_query_pred = self.model.regressor(query_features)
                    pre_query_loss = self.criterion(pre_query_pred, query_target[i]).item()
                    pre_query_r2 = self.compute_r2_score(pre_query_pred, query_target[i])
                
                # Clone initial parameters
                adapted_params = {
                    name: param.clone()
                    for name, param in self.model.regressor.named_parameters()
                    if name.startswith('1.')  # First layer parameters
                }
                
                # Adaptation loop with trajectory tracking
                for step in range(num_steps):
                    # Support set forward pass
                    support_features = self.model.encode(support_spikes[i])
                    support_pred = self.forward_with_params(support_features, adapted_params)
                    support_loss = self.criterion(support_pred, support_target[i])
                    support_r2 = self.compute_r2_score(support_pred, support_target[i])
                    
                    # Query set forward pass
                    query_features = self.model.encode(query_spikes[i])
                    query_pred = self.forward_with_params(query_features, adapted_params)
                    query_loss = self.criterion(query_pred, query_target[i])
                    query_r2 = self.compute_r2_score(query_pred, query_target[i])
                    
                    # Store trajectory
                    task_support_loss_trajectory.append(support_loss.item())
                    task_query_loss_trajectory.append(query_loss.item())
                    task_support_r2_trajectory.append(support_r2)
                    task_query_r2_trajectory.append(query_r2)
                    
                    # Compute gradients and update parameters
                    grads = torch.autograd.grad(support_loss, adapted_params.values(),
                                            create_graph=True)
                    
                    # Update parameters
                    adapted_params = {
                        name: param - self.inner_lr * grad
                        for (name, param), grad in zip(adapted_params.items(), grads)
                        if name.startswith('1.')
                    }
                
                # Get final performance
                with torch.no_grad():
                    # Final support set performance
                    support_features = self.model.encode(support_spikes[i])
                    final_support_pred = self.forward_with_params(support_features, adapted_params, training=False)
                    final_support_loss = self.criterion(final_support_pred, support_target[i]).item()
                    final_support_r2 = self.compute_r2_score(final_support_pred, support_target[i])
                    
                    # Final query set performance
                    query_features = self.model.encode(query_spikes[i])
                    final_query_pred = self.forward_with_params(query_features, adapted_params, training=False)
                    final_query_loss = self.criterion(final_query_pred, query_target[i]).item()
                    final_query_r2 = self.compute_r2_score(final_query_pred, query_target[i])
                    
                    batch_losses.append(final_query_loss)
                    
                    # Store metrics
                    eval_metrics['pre_adapt_support_loss'].append(pre_support_loss)
                    eval_metrics['post_adapt_support_loss'].append(final_support_loss)
                    eval_metrics['pre_adapt_query_loss'].append(pre_query_loss)
                    eval_metrics['post_adapt_query_loss'].append(final_query_loss)
                    
                    eval_metrics['pre_adapt_support_r2'].append(pre_support_r2)
                    eval_metrics['post_adapt_support_r2'].append(final_support_r2)
                    eval_metrics['pre_adapt_query_r2'].append(pre_query_r2)
                    eval_metrics['post_adapt_query_r2'].append(final_query_r2)
                    
                    eval_metrics['adaptation_improvement_support_loss'].append(pre_support_loss - final_support_loss)
                    eval_metrics['adaptation_improvement_query_loss'].append(pre_query_loss - final_query_loss)
                    eval_metrics['adaptation_improvement_support_r2'].append(final_support_r2 - pre_support_r2)
                    eval_metrics['adaptation_improvement_query_r2'].append(final_query_r2 - pre_query_r2)
                    
                    # Store trajectories
                    eval_metrics['adaptation_trajectory_support_loss'].append(task_support_loss_trajectory)
                    eval_metrics['adaptation_trajectory_query_loss'].append(task_query_loss_trajectory)
                    eval_metrics['adaptation_trajectory_support_r2'].append(task_support_r2_trajectory)
                    eval_metrics['adaptation_trajectory_query_r2'].append(task_query_r2_trajectory)
            
            total_loss += np.mean(batch_losses)
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / len(loader)
        }
        
        # Average scalar metrics
        for key in eval_metrics:
            if key.startswith('adaptation_trajectory'):
                # Store full trajectories without averaging
                avg_metrics[key] = eval_metrics[key]
            else:
                avg_metrics[key] = np.mean(eval_metrics[key])
        
        return avg_metrics

    def train(self, num_epochs: int, save_freq: int = 50):
        """Main training loop."""
        best_val_loss = float('inf')  # Track best validation loss

        # Log initial weight norms
        logging.info("Initial weight norms:")
        self.model.log_weight_norms()
        
        # Log training setup
        logging.info(f"Starting ANIL training for {num_epochs} epochs...")
        logging.info(f"Training on {len(self.dataset.train_sessions)} sessions")
        logging.info(f"Training on {self.dataset.train_sessions}")
        logging.info(f"Validation session: {self.dataset.valid_session}")
        logging.info(f"Test session: {self.dataset.test_session}")
        logging.info(f"Meta-learning parameters:")
        logging.info(f"  k_shot: {self.k_shot}")
        logging.info(f"  k_query: {self.k_query}")
        logging.info(f"  batch_size: {self.batch_size}")
        logging.info(f"  num_inner_steps: {self.num_inner_steps}")
        
        for epoch in range(num_epochs):
            # Training epoch
            train_metrics = self.train_epoch()

            self.model.log_weight_norms()
            
            # Validation and Test with enhanced metrics
            val_metrics = self.evaluate(split='valid', num_steps=5)
            test_metrics = self.evaluate(split='test', num_steps=5)
            
            # Update history with all metrics
            # Loss metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Training metrics (excluding trajectories)
            for key in train_metrics:
                if key != 'loss' and not key.startswith('adaptation_trajectory'):
                    self.history[f'train_{key}'].append(train_metrics[key])
            
            # Validation metrics
            for key in val_metrics:
                if key != 'loss':
                    if key.startswith('adaptation_trajectory'):
                        # Store the full trajectory for this epoch
                        self.history[f'val_{key}'].append(val_metrics[key])
                    else:
                        self.history[f'val_{key}'].append(val_metrics[key])
            
            # Test metrics
            for key in test_metrics:
                if key != 'loss':
                    if key.startswith('adaptation_trajectory'):
                        # Store the full trajectory for this epoch
                        self.history[f'test_{key}'].append(test_metrics[key])
                    else:
                        self.history[f'test_{key}'].append(test_metrics[key])
            
            # Log to tensorboard
            # Losses
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/test', test_metrics['loss'], epoch)
            self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log all scalar metrics to tensorboard (excluding trajectories)
            for key in train_metrics:
                if key != 'loss' and not key.startswith('adaptation_trajectory'):
                    self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
            
            for key in val_metrics:
                if key != 'loss' and not key.startswith('adaptation_trajectory'):
                    self.writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)
                    
            for key in test_metrics:
                if key != 'loss' and not key.startswith('adaptation_trajectory'):
                    self.writer.add_scalar(f'Test/{key}', test_metrics[key], epoch)
            
            # Save checkpoint if validation performance improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': self.history
                }
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
                logging.info(f"Saved best model at epoch {epoch} with val loss = {best_val_loss:.4f}")
            
            # Regular checkpoint saving
            if (epoch + 1) % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': self.history
                }
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                logging.info(f"Saved checkpoint at epoch {epoch+1}")
            
            # Log epoch summary
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train loss: {train_metrics['loss']:.4f}, "
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"Test loss: {test_metrics['loss']:.4f}, "
                #f"Train Pre-S R²: {train_metrics['pre_adapt_support_r2']:.4f}, "
                #f"Train Pre-Q R²: {train_metrics['pre_adapt_query_r2']:.4f}, "
                #f"Train Post-S R²: {train_metrics['post_adapt_support_r2']:.4f}, "
                #f"Train Post-Q R²: {train_metrics['post_adapt_query_r2']:.4f}, "
                #f"Val Pre-S R²: {val_metrics['pre_adapt_support_r2']:.4f}, "
                #f"Val Pre-Q R²: {val_metrics['pre_adapt_query_r2']:.4f}, "
                #f"Val Post-S R²: {val_metrics['post_adapt_support_r2']:.4f}, "
                #f"Val Post-Q R²: {val_metrics['post_adapt_query_r2']:.4f}, "
                #f"Test Pre-S R²: {test_metrics['pre_adapt_support_r2']:.4f}, "
                #f"Test Pre-Q R²: {test_metrics['pre_adapt_query_r2']:.4f}, "
                #f"Test Post-S R²: {test_metrics['post_adapt_support_r2']:.4f}, "
                #f"Test Post-Q R²: {test_metrics['post_adapt_query_r2']:.4f}"
            )
        
        # Save and plot training history
        self.save_history()
        self.writer.close()
        logging.info("Training completed!")

    def compute_r2_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute R² score between predictions and targets."""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        r2_scores = []
        for i in range(pred.shape[-1]):  # For each output dimension (x,y)
            r2 = r2_score(target[..., i].flatten(), pred[..., i].flatten())
            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def save_history(self):
        """
        Save training history to CSV.
        Includes all adaptation metrics and trajectories for train/val/test sets.
        """
        # Save main metrics history
        history_df = pd.DataFrame({
            # Losses
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'test_loss': self.history['test_loss'],
            'learning_rate': self.history['learning_rates'],
            
            # Training set metrics
            'train_pre_adapt_support_loss': self.history['train_pre_adapt_support_loss'],
            'train_post_adapt_support_loss': self.history['train_post_adapt_support_loss'],
            'train_pre_adapt_query_loss': self.history['train_pre_adapt_query_loss'],
            'train_post_adapt_query_loss': self.history['train_post_adapt_query_loss'],
            'train_pre_adapt_support_r2': self.history['train_pre_adapt_support_r2'],
            'train_post_adapt_support_r2': self.history['train_post_adapt_support_r2'],
            'train_pre_adapt_query_r2': self.history['train_pre_adapt_query_r2'],
            'train_post_adapt_query_r2': self.history['train_post_adapt_query_r2'],
            'train_adaptation_improvement_support_loss': self.history['train_adaptation_improvement_support_loss'],
            'train_adaptation_improvement_query_loss': self.history['train_adaptation_improvement_query_loss'],
            'train_adaptation_improvement_support_r2': self.history['train_adaptation_improvement_support_r2'],
            'train_adaptation_improvement_query_r2': self.history['train_adaptation_improvement_query_r2'],
            
            # Validation set metrics
            'val_pre_adapt_support_loss': self.history['val_pre_adapt_support_loss'],
            'val_post_adapt_support_loss': self.history['val_post_adapt_support_loss'],
            'val_pre_adapt_query_loss': self.history['val_pre_adapt_query_loss'],
            'val_post_adapt_query_loss': self.history['val_post_adapt_query_loss'],
            'val_pre_adapt_support_r2': self.history['val_pre_adapt_support_r2'],
            'val_post_adapt_support_r2': self.history['val_post_adapt_support_r2'],
            'val_pre_adapt_query_r2': self.history['val_pre_adapt_query_r2'],
            'val_post_adapt_query_r2': self.history['val_post_adapt_query_r2'],
            'val_adaptation_improvement_support_loss': self.history['val_adaptation_improvement_support_loss'],
            'val_adaptation_improvement_query_loss': self.history['val_adaptation_improvement_query_loss'],
            'val_adaptation_improvement_support_r2': self.history['val_adaptation_improvement_support_r2'],
            'val_adaptation_improvement_query_r2': self.history['val_adaptation_improvement_query_r2'],
            
            # Test set metrics
            'test_pre_adapt_support_loss': self.history['test_pre_adapt_support_loss'],
            'test_post_adapt_support_loss': self.history['test_post_adapt_support_loss'],
            'test_pre_adapt_query_loss': self.history['test_pre_adapt_query_loss'],
            'test_post_adapt_query_loss': self.history['test_post_adapt_query_loss'],
            'test_pre_adapt_support_r2': self.history['test_pre_adapt_support_r2'],
            'test_post_adapt_support_r2': self.history['test_post_adapt_support_r2'],
            'test_pre_adapt_query_r2': self.history['test_pre_adapt_query_r2'],
            'test_post_adapt_query_r2': self.history['test_post_adapt_query_r2'],
            'test_adaptation_improvement_support_loss': self.history['test_adaptation_improvement_support_loss'],
            'test_adaptation_improvement_query_loss': self.history['test_adaptation_improvement_query_loss'],
            'test_adaptation_improvement_support_r2': self.history['test_adaptation_improvement_support_r2'],
            'test_adaptation_improvement_query_r2': self.history['test_adaptation_improvement_query_r2']
        })
        history_df.to_csv(os.path.join(self.checkpoint_dir, 'training_history.csv'), index=False)
        
        # Save adaptation trajectories separately
        trajectory_data = {
            'val_adaptation_trajectory_support_loss': self.history['val_adaptation_trajectory_support_loss'],
            'val_adaptation_trajectory_query_loss': self.history['val_adaptation_trajectory_query_loss'],
            'val_adaptation_trajectory_support_r2': self.history['val_adaptation_trajectory_support_r2'],
            'val_adaptation_trajectory_query_r2': self.history['val_adaptation_trajectory_query_r2'],
            'test_adaptation_trajectory_support_loss': self.history['test_adaptation_trajectory_support_loss'],
            'test_adaptation_trajectory_query_loss': self.history['test_adaptation_trajectory_query_loss'],
            'test_adaptation_trajectory_support_r2': self.history['test_adaptation_trajectory_support_r2'],
            'test_adaptation_trajectory_query_r2': self.history['test_adaptation_trajectory_query_r2']
        }
        
        # Save trajectories as numpy arrays for easier loading and analysis
        np.save(os.path.join(self.checkpoint_dir, 'adaptation_trajectories.npy'), trajectory_data)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config('config/config_anil_fomaml.yaml')
    
    # Set random seed
    set_seed(config['anil_fomaml']['seed'])
    
    try:
        # Initialize dataset with session-based meta-learning
        logging.info("Initializing dataset...")

        dataset = Session_MAML_Monkey_Dataset(
            exclude_ids=config['anil_fomaml']['exclude_ids'],
            valid_id=config['anil_fomaml']['valid_id'],
            test_id=config['anil_fomaml']['test_id'],
            width=0.02
        )
        
        # Load pretrained model
        logging.info("Initializing pretrained model...")
        pretrained_model = KernelTransformerMAE(
            **config['model']
        )
        
        # Load pretrained weights
        logging.info("Loading pretrained weights...")
        checkpoint = torch.load(config['paths']['pretrained_path'])
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"We are training ANIL+FOMAML for {config['anil_fomaml']['target_type']} prediction")
        
        # Initialize regressor
        logging.info("Initializing regressor...")
        regressor = KernelTransformerRegressor(
            pretrained_model=pretrained_model,
            output_dim=config['anil_fomaml']['output_dim'],
            freeze_encoder=config['anil_fomaml']['freeze_encoder']
        )
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = ANILFOMAMLRegressorTrainer(regressor, dataset, config)
        
        # Train model
        logging.info("Starting training...")
        trainer.train(
            num_epochs=config['anil_fomaml']['num_epochs'],
            save_freq=config['anil_fomaml']['save_freq']  
        )
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise e


# def main():
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Load configuration
#     config = load_config('config/config_anil_fomaml_sine.yaml')
    
#     # Set random seed
#     set_seed(config['anil_fomaml']['seed'])
    
#     try:
#         # Initialize sine wave dataset
#         logging.info("Initializing sine wave dataset...")
#         dataset = SineWaveDataset(
#             num_neurons=config['model']['neural_dim'],
#             time_steps=50  # match our standard sequence length
#         )
        
#         # Initialize model (without pretrained weights for sine wave)
#         logging.info("Initializing model...")
#         base_model = KernelTransformerMAE(
#             **config['model']
#         )
        
#         # Initialize regressor
#         logging.info("Initializing regressor...")
#         regressor = KernelTransformerRegressor(
#             pretrained_model=base_model,
#             output_dim=config['anil_fomaml']['output_dim'],
#             freeze_encoder=config['anil_fomaml']['freeze_encoder']
#         )
        
#         # Initialize trainer
#         logging.info("Initializing trainer...")
#         trainer = ANILFOMAMLRegressorTrainer(regressor, dataset, config)
        
#         # Train model
#         logging.info("Starting training...")
#         trainer.train(
#             num_epochs=config['anil_fomaml']['num_epochs'],
#             save_freq=config['anil_fomaml']['save_freq']  
#         )
        
#         logging.info("Training completed successfully!")
        
#     except Exception as e:
#         logging.error(f"Error during training: {str(e)}")
#         raise e

if __name__ == "__main__":
    main() 