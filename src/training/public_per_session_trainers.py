"""
Per-Session Evaluation Trainers for Public Dataset Neural Foundation Model
-------------------------------------------------------------------------

This module provides trainers for the NEW per-session evaluation protocol where each
session is trained and evaluated individually, then results are aggregated across sessions.

Key Features:
- PublicPerSessionEvaluationManager: Orchestrates multiple individual training runs
- PublicSingleSessionTrainer: Trains and evaluates on single sessions
- Results aggregation and statistical analysis across sessions
- Professional logging and result storage
- Reuses proven training logic from PublicRegressionTrainer
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics

# Import our modules
from ..models.public_transformer import PublicNeuralFoundationMAE
from ..models.public_downstream import PublicVelocityPredictor
from ..data.public_single_session_dataset import (
    PublicSingleSessionDataset,
    create_single_session_datasets
)
from ..utils.helpers import load_config, set_seed
from .enhanced_logger import EnhancedLogger


# ===========================================================================================
# Clean Model Initialization System
# ===========================================================================================

class PublicModelInitializer:
    """
    Clean model initialization for per-session evaluation with 4 clear modes.
    
    **INITIALIZATION MODES**:
    1. 'random': Same architecture as foundation model, random weights (no load_state_dict)
    2. 'foundation_pretrained': Foundation encoder + random prediction head  
    3. 'downstream_complete': Complete downstream model (encoder + head from downstream)
    4. 'downstream_encoder_only': Downstream encoder + random prediction head
    
    **KEY DESIGN**:
    - Architecture always comes from foundation_config (no duplication)
    - Clean separation of concerns for each initialization strategy
    - Fair comparison by using identical architectures across modes
    """
    
    INITIALIZATION_MODES = {
        'random': 'Same architecture as foundation model, random weights',
        'foundation_pretrained': 'Foundation encoder + random prediction head', 
        'downstream_complete': 'Complete downstream model (encoder + head)',
        'downstream_encoder_only': 'Downstream encoder + random prediction head'
    }
    
    def __init__(self, 
                 initialization_mode: str,
                 checkpoint_path: str,
                 training_mode: str = 'full_finetune',
                 device: torch.device = None):
        """
        Initialize model creator.
        
        Args:
            initialization_mode: One of ['random', 'foundation_pretrained', 
                                       'downstream_complete', 'downstream_encoder_only']
            checkpoint_path: Path to checkpoint (foundation or downstream)
            training_mode: Training mode for PublicVelocityPredictor
            device: Device to load models on
        """
        if initialization_mode not in self.INITIALIZATION_MODES:
            raise ValueError(f"Invalid initialization_mode: {initialization_mode}. "
                           f"Must be one of {list(self.INITIALIZATION_MODES.keys())}")
        
        self.mode = initialization_mode
        self.checkpoint_path = checkpoint_path
        self.training_mode = training_mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """
        Create model using specified initialization mode.
        
        Returns:
            Tuple of (model, foundation_config_used)
        """
        if self.mode == 'random':
            return self._create_random_model()
        elif self.mode == 'foundation_pretrained': 
            return self._create_with_foundation_encoder()
        elif self.mode == 'downstream_complete':
            return self._load_complete_downstream()
        elif self.mode == 'downstream_encoder_only':
            return self._create_with_downstream_encoder()
        else:
            raise ValueError(f"Unknown initialization mode: {self.mode}")
    
    def _load_foundation_config(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load foundation config from any checkpoint type."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to get foundation config
        if 'config' in checkpoint:
            # Legacy format or foundation checkpoint
            return checkpoint['config']
        else:
            raise ValueError(f"No config found in checkpoint: {checkpoint_path}")
    
    def _create_foundation_from_config(self, foundation_config: Dict[str, Any]) -> PublicNeuralFoundationMAE:
        """Create foundation model from config."""
        model_config = foundation_config['model']
        
        model = PublicNeuralFoundationMAE(
            neural_dim=model_config.get('neural_dim', 50),
            d_model=model_config.get('d_model', 512),
            temporal_layers=model_config.get('temporal_layers', 6),
            heads=model_config.get('heads', 8),
            dropout=model_config.get('dropout', 0.1),
            kernel_size=model_config.get('kernel_size', [3, 3]),
            max_seq_length=model_config.get('max_seq_length', 2000),
            pos_encoding_type=model_config.get('pos_encoding_type', 'rope_4d'),
            session_scale=model_config.get('session_scale', 1.0),
            use_temporal_kernels=model_config.get('use_temporal_kernels', True),
            use_mae_decoder=model_config.get('use_mae_decoder', True),
            use_session_specific_heads=model_config.get('use_session_specific_heads', True)
        )
        
        return model.to(self.device)
    
    def _create_random_model(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """Mode 1: Same architecture as foundation model, random weights."""
        # Load foundation config to get architecture
        foundation_config = self._load_foundation_config(self.checkpoint_path)
        
        # Create foundation model with random weights (NO load_state_dict)
        foundation = self._create_foundation_from_config(foundation_config)
        
        # Create PublicVelocityPredictor with random prediction head
        model = PublicVelocityPredictor(
            pretrained_model=foundation,
            training_mode=self.training_mode
        )
        
        # Move entire model to device (ensures prediction head is also on correct device)
        model = model.to(self.device)
        
        logging.info("📦 Created random model with same architecture as foundation")
        return model, foundation_config
    
    def _create_with_foundation_encoder(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """Mode 2: Foundation encoder + random prediction head."""
        # Load foundation config and checkpoint
        foundation_config = self._load_foundation_config(self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create foundation model and load pretrained weights
        foundation = self._create_foundation_from_config(foundation_config)
        foundation.load_state_dict(checkpoint['model_state_dict'])
        
        # Create PublicVelocityPredictor with random prediction head
        model = PublicVelocityPredictor(
            pretrained_model=foundation,
            training_mode=self.training_mode
        )
        
        # Move entire model to device
        model = model.to(self.device)
        
        logging.info("📦 Created model with foundation encoder + random prediction head")
        return model, foundation_config
    
    def _load_complete_downstream(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """Mode 3: Complete downstream model (encoder + head from downstream)."""
        # Load downstream checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get foundation config
        if 'foundation_config' in checkpoint:
            foundation_config = checkpoint['foundation_config']
        elif 'config' in checkpoint:
            foundation_config = checkpoint['config']
        else:
            raise ValueError("No foundation config found in downstream checkpoint")
        
        # Create foundation model architecture
        foundation = self._create_foundation_from_config(foundation_config)
        
        # Create PublicVelocityPredictor
        model = PublicVelocityPredictor(
            pretrained_model=foundation,
            training_mode=self.training_mode
        )
        
        # Load complete downstream state (both encoder and prediction head)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move entire model to device (after loading state dict)
        model = model.to(self.device)
        
        logging.info("📦 Loaded complete downstream model (encoder + prediction head)")
        return model, foundation_config
    
    def _create_with_downstream_encoder(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """Mode 4: Downstream encoder + random prediction head."""
        # Load downstream checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get foundation config
        if 'foundation_config' in checkpoint:
            foundation_config = checkpoint['foundation_config']
        else:
            raise ValueError("No foundation config found in downstream checkpoint")
        
        # Create foundation model architecture
        foundation = self._create_foundation_from_config(foundation_config)
        
        # Load only temporal encoder from downstream
        if 'temporal_encoder_state_dict' in checkpoint:
            # Load from separate encoder state dict
            foundation.temporal_encoder.load_state_dict(checkpoint['temporal_encoder_state_dict'])
        else:
            raise ValueError("No temporal encoder found in downstream checkpoint")
        
        # Create PublicVelocityPredictor with random prediction head
        model = PublicVelocityPredictor(
            pretrained_model=foundation,
            training_mode=self.training_mode
        )
        
        # Move entire model to device
        model = model.to(self.device)
        
        logging.info("📦 Created model with downstream encoder + random prediction head")
        return model, foundation_config


class PublicPerSessionEvaluationManager:
    """
    Orchestrates per-session evaluation across multiple sessions for new evaluation protocol.
    
    **DESIGN GOALS**:
    - Manage multiple individual training runs (one per session)
    - Create fresh model from pretrained checkpoint for each session
    - Aggregate results across sessions with statistical analysis
    - Professional logging and result storage
    - Clean separation from multi-session evaluation approach
    
    **KEY FEATURES**:
    - Session-by-session training and evaluation
    - Statistical aggregation (mean ± std across sessions)
    - Comprehensive result storage and reporting
    - Progress tracking across multiple sessions
    - Reuses proven training components
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 evaluation_scenario: str,  # 'cross_session', 'cross_subject_center', 'cross_subject_random'
                 config: Dict[str, Any]):
        """
        Initialize PublicPerSessionEvaluationManager with clean model initialization.
        
        Args:
            checkpoint_path: Path to checkpoint (foundation or downstream)
            evaluation_scenario: Evaluation scenario identifier
            config: Training configuration dictionary
        """
        
        self.checkpoint_path = checkpoint_path
        self.evaluation_scenario = evaluation_scenario
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration sections
        self.training_config = config.get('training', {})
        self.paths_config = config.get('paths', {})
        self.dataset_config = config.get('dataset', {})
        self.model_config = config.get('model', {})
        
        # Extract clean initialization mode
        self.initialization_mode = self.model_config.get('initialization_mode', 'foundation_pretrained')
        
        # Validate initialization mode
        if self.initialization_mode not in PublicModelInitializer.INITIALIZATION_MODES:
            raise ValueError(f"Invalid initialization_mode: {self.initialization_mode}. "
                           f"Must be one of {list(PublicModelInitializer.INITIALIZATION_MODES.keys())}")
        
        # Setup logging
        self._setup_logging()
        
        # Load session datasets based on scenario
        self.session_datasets = self._create_session_datasets()
        
        # Storage for results
        self.session_results = {}  # {session_id: final_metrics}
        self.aggregated_results = {}  # statistical aggregation
        
        # Progress tracking
        self.total_sessions = len(self.session_datasets)
        self.completed_sessions = 0
        
        # Log initialization info with clean mode description
        mode_description = PublicModelInitializer.INITIALIZATION_MODES[self.initialization_mode]
        
        self.logger.info(f"PublicPerSessionEvaluationManager initialized for {self.evaluation_scenario}")
        self.logger.info(f"Total sessions to evaluate: {self.total_sessions}")
        self.logger.info(f"Model initialization: {self.initialization_mode} ({mode_description})")
        self.logger.info(f"Checkpoint path: {self.checkpoint_path}")
    
    def _setup_logging(self):
        """Setup logging system for per-session evaluation."""
        
        experiment_name = self.paths_config.get('experiment_name', f"per_session_{self.evaluation_scenario}")
        base_dir = self.paths_config.get('base_dir', './logs_public_per_session')
        
        # Initialize enhanced logger
        self.enhanced_logger = EnhancedLogger(
            experiment_name=experiment_name,
            base_dir=base_dir,
            top_k_checkpoints=1,  # Keep only best checkpoint per session
            local_rank=0
        )
        
        # Update paths config
        enhanced_paths = self.enhanced_logger.get_paths()
        self.paths_config.update(enhanced_paths)
        
        # Create logger alias
        self.logger = self.enhanced_logger.logger
    
    def _create_session_datasets(self) -> List[PublicSingleSessionDataset]:
        """Create list of single session datasets for the evaluation scenario."""
        
        dataset_kwargs = {
            'target_neurons': self.dataset_config.get('target_neurons', 50),
            'sequence_length': self.dataset_config.get('sequence_length', 50),
            'neuron_selection_strategy': self.dataset_config.get('neuron_selection_strategy', 'first_n'),
            'random_seed': self.dataset_config.get('random_seed', 42),
            'data_root': self.dataset_config.get('data_root', '/data/Fang-analysis/causal-nfm/Data/public_data')
        }
        
        try:
            session_datasets = create_single_session_datasets(
                scenario=self.evaluation_scenario,
                **dataset_kwargs
            )
            
            if not session_datasets:
                raise ValueError(f"No datasets created for scenario: {self.evaluation_scenario}")
            
            self.logger.info(f"Created {len(session_datasets)} session datasets")
            return session_datasets
            
        except Exception as e:
            self.logger.error(f"Failed to create session datasets: {str(e)}")
            raise e
    
    def run_per_session_evaluation(self) -> Dict[str, Any]:
        """
        Main evaluation loop: train and evaluate each session individually.
        
        Returns:
            Dictionary with session results and aggregated statistics
        """
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STARTING PER-SESSION EVALUATION: {self.evaluation_scenario.upper()}")
        self.logger.info(f"{'='*80}")
        
        # Save config and model summary once at the beginning
        self._save_config_and_model_summary()
        
        start_time = time.time()
        
        # Run evaluation for each session
        for i, session_dataset in enumerate(self.session_datasets):
            session_id = session_dataset.get_session_id()
            
            self.logger.info(f"\n🔄 Session {i+1}/{self.total_sessions}: {session_id}")
            self.logger.info("-" * 60)
            
            session_start_time = time.time()
            
            try:
                # Create fresh model using clean initializer (already on correct device)
                model, foundation_config = self._get_model_and_config()
                
                # Create trainer for this specific session
                session_trainer = PublicSingleSessionTrainer(
                    model=model,
                    session_dataset=session_dataset,
                    config=self.config,
                    foundation_config=foundation_config,
                    session_index=i,
                    total_sessions=self.total_sessions,
                    parent_logger=self.logger,
                    tensorboard_writer=self.enhanced_logger.writer,
                    session_checkpoint_dir=self.paths_config['checkpoint_dir']
                )
                
                # Train and evaluate on this session only
                session_result = session_trainer.train_and_evaluate()
                
                # Store results
                self.session_results[session_id] = session_result
                
                session_time = time.time() - session_start_time
                
                # Log progress
                self.logger.info(f"✅ Session {i+1}/{self.total_sessions} completed in {session_time:.1f}s: "
                               f"{session_id} (R²: {session_result['r2_mean']:.4f})")
                
                self.completed_sessions += 1
                
            except Exception as e:
                self.logger.error(f"❌ Session {i+1}/{self.total_sessions} failed: {session_id}")
                self.logger.error(f"Error: {str(e)}")
                # Continue with next session
                continue
        
        # Aggregate results across sessions
        self.aggregated_results = self._aggregate_session_results()
        
        # Log aggregated results to TensorBoard
        self._log_aggregated_results_to_tensorboard()
        
        total_time = time.time() - start_time
        
        # Save and report final results
        results_dict = self._save_final_results()
        self._print_final_summary(total_time)
        
        # Clean up
        self.enhanced_logger.log_training_end()
        self.enhanced_logger.close()
        
        return results_dict
    
    
    def _get_model_and_config(self) -> Tuple[PublicVelocityPredictor, Dict[str, Any]]:
        """Get model and config using clean initializer."""
        
        training_mode = self.training_config.get('training_mode', 'full_finetune')
        
        self.logger.info(f"🔄 Creating model using {self.initialization_mode} mode...")
        
        # Create clean initializer
        initializer = PublicModelInitializer(
            initialization_mode=self.initialization_mode,
            checkpoint_path=self.checkpoint_path,
            training_mode=training_mode,
            device=self.device
        )
        
        return initializer.create_model()
    
    def _save_config_and_model_summary(self):
        """Save config and model summary once at the beginning of evaluation."""
        
        # Load a temporary model to save the architecture summary
        temp_model, foundation_config = self._get_model_and_config()
        
        # Save config using enhanced logger (include foundation config)
        self.enhanced_logger.save_config(self.config)
        
        # Save model summary using enhanced logger
        self.enhanced_logger.save_model_summary(temp_model)
        
        self.logger.info("📁 Config and model summary saved")
        
        # Clean up temporary model
        del temp_model
    
    def _aggregate_session_results(self) -> Dict[str, float]:
        """
        Compute aggregated statistics across all sessions.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        
        if not self.session_results:
            self.logger.warning("No session results to aggregate")
            return {}
        
        # Get all metric names from first session
        metric_names = list(list(self.session_results.values())[0].keys())
        aggregated = {}
        
        self.logger.info(f"\n📊 Aggregating results across {len(self.session_results)} sessions")
        
        for metric in metric_names:
            values = [session_data[metric] for session_data in self.session_results.values() 
                     if metric in session_data and not math.isnan(session_data[metric])]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_count'] = len(values)
                
                self.logger.info(f"   {metric}: {aggregated[f'{metric}_mean']:.4f} ± {aggregated[f'{metric}_std']:.4f}")
        
        return aggregated
    
    def _log_aggregated_results_to_tensorboard(self):
        """Log aggregated session results to TensorBoard."""
        if not self.enhanced_logger.writer or not self.aggregated_results:
            return
        
        writer = self.enhanced_logger.writer
        
        # Log aggregated metrics
        if 'r2_mean_mean' in self.aggregated_results:
            writer.add_scalar('Aggregated_Results/R2_Mean_Across_Sessions', 
                            self.aggregated_results['r2_mean_mean'], 0)
            writer.add_scalar('Aggregated_Results/R2_Std_Across_Sessions', 
                            self.aggregated_results['r2_mean_std'], 0)
            writer.add_scalar('Aggregated_Results/R2_Min_Across_Sessions', 
                            self.aggregated_results['r2_mean_min'], 0)
            writer.add_scalar('Aggregated_Results/R2_Max_Across_Sessions', 
                            self.aggregated_results['r2_mean_max'], 0)
            
        # Log session count
        writer.add_scalar('Aggregated_Results/Number_of_Sessions', 
                        len(self.session_results), 0)
        
        # Log per-session max R2 scores
        max_r2_scores = []
        for session_id, metrics in self.session_results.items():
            r2_score = metrics.get('r2_mean', 0)
            max_r2_scores.append(r2_score)
            
            # Clean session ID for TensorBoard
            clean_session_id = session_id.replace('/', '_').replace('\\', '_').replace('.', '_')
            writer.add_scalar(f'Final_Session_Results/R2_{clean_session_id}', r2_score, 0)
        
        # Log distribution statistics of max R2 scores
        if max_r2_scores:
            writer.add_histogram('Final_Session_Results/R2_Distribution', 
                               torch.tensor(max_r2_scores), 0)
        
        self.logger.info("📊 Aggregated results logged to TensorBoard")
    
    def _save_final_results(self) -> Dict[str, Any]:
        """Save comprehensive evaluation results to file."""
        
        results_dict = {
            'evaluation_scenario': self.evaluation_scenario,
            'total_sessions': self.total_sessions,
            'completed_sessions': self.completed_sessions,
            'pretrained_model_path': self.checkpoint_path,
            'session_results': self.session_results,
            'aggregated_results': self.aggregated_results,
            'config': self.config,
            'session_ids': list(self.session_results.keys())
        }
        
        # Save to enhanced logger's experiment directory
        results_path = Path(self.paths_config['experiment_dir']) / 'per_session_evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to: {results_path}")
        
        return results_dict
    
    def _print_final_summary(self, total_time: float):
        """Print comprehensive evaluation summary."""
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 PER-SESSION EVALUATION SUMMARY")
        self.logger.info(f"{'='*80}")
        
        # General info
        self.logger.info(f"🎯 Evaluation Scenario: {self.evaluation_scenario}")
        self.logger.info(f"📋 Sessions Completed: {self.completed_sessions}/{self.total_sessions}")
        self.logger.info(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Aggregated metrics
        if self.aggregated_results:
            self.logger.info(f"\n📈 Aggregated Results:")
            
            # Key metrics
            r2_mean = self.aggregated_results.get('r2_mean_mean', 0)
            r2_std = self.aggregated_results.get('r2_mean_std', 0)
            r2_min = self.aggregated_results.get('r2_mean_min', 0)
            r2_max = self.aggregated_results.get('r2_mean_max', 0)
            
            self.logger.info(f"   R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
            self.logger.info(f"   R² Range: [{r2_min:.4f}, {r2_max:.4f}]")
            
        # Per-session breakdown
        self.logger.info(f"\n📋 Per-Session Results:")
        self.logger.info(f"{'Session ID':<30} {'R²':<8}")
        self.logger.info("-" * 70)
        
        for session_id, metrics in self.session_results.items():
            r2 = metrics.get('r2_mean', 0)
            self.logger.info(f"{session_id:<30} {r2:<8.4f}")
        
        self.logger.info(f"\n✅ Per-session evaluation completed successfully!")
        self.logger.info(f"{'='*80}")


class PublicSingleSessionTrainer:
    """
    Trainer for a single session with minimal overhead and focused evaluation.
    
    **DESIGN GOALS**:
    - Train and evaluate on single session only
    - Minimal overhead compared to multi-session trainer
    - Reuse proven training components from PublicRegressionTrainer
    - Clean interface for per-session evaluation
    - Fast training optimized for session-specific adaptation
    
    **KEY FEATURES**:
    - Single session focus with no cross-session complexity
    - Faster training loop with session-specific early stopping
    - Comprehensive metrics computation
    - Professional logging with session context
    - Return final test metrics for aggregation
    """
    
    def __init__(self,
                 model: PublicVelocityPredictor,
                 session_dataset: PublicSingleSessionDataset,
                 config: Dict[str, Any],
                 foundation_config: Dict[str, Any],
                 session_index: int,
                 total_sessions: int,
                 parent_logger: logging.Logger,
                 tensorboard_writer=None,
                 session_checkpoint_dir: Optional[str] = None):
        """
        Initialize PublicSingleSessionTrainer with clean model initialization.
        
        Args:
            model: Ready-to-use PublicVelocityPredictor model (already initialized)
            session_dataset: PublicSingleSessionDataset for this session
            config: Training configuration dictionary
            foundation_config: Foundation model configuration
            session_index: Index of this session (0-based)
            total_sessions: Total number of sessions being evaluated
            parent_logger: Parent logger for consistent logging
            tensorboard_writer: TensorBoard writer for logging training metrics
            session_checkpoint_dir: Directory to save session-specific checkpoints
        """
        
        self.model = model
        self.session_dataset = session_dataset
        self.session_id = session_dataset.get_session_id()
        self.config = config
        self.foundation_config = foundation_config
        self.session_index = session_index
        self.total_sessions = total_sessions
        self.logger = parent_logger
        self.writer = tensorboard_writer
        self.session_checkpoint_dir = session_checkpoint_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration sections
        self.training_config = config.get('training', {})
        
        # Training configuration
        self.target_type = self.training_config.get('target_type', 'velocity')
        self.training_mode = self.training_config.get('training_mode', 'full_finetune')
        self.num_epochs = self.training_config.get('num_epochs', 50)
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 20)
        
        # Initialize training components
        self._setup_training_components()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_r2_score = float('-inf')
        self.epochs_without_improvement = 0
        
        # Setup session-specific checkpoint manager
        self._setup_session_checkpoint_manager()
        
        self.logger.info(f"   Single session trainer initialized for {self.session_id}")
        self.logger.info(f"   Training mode: {self.training_mode}, Target: {self.target_type}")
    
    def _setup_training_components(self):
        """Setup training components (optimizer, scheduler, loss, dataloaders)."""
        
        # Setup dataloaders
        batch_size = self.training_config.get('batch_size', 32)
        
        self.train_loader = self.session_dataset.create_dataloader(
            split='train', batch_size=batch_size, shuffle=True, num_workers=2, output_mode='regression'
        )
        self.val_loader = self.session_dataset.create_dataloader(
            split='val', batch_size=batch_size, shuffle=False, num_workers=2, output_mode='regression'
        )
        self.test_loader = self.session_dataset.create_dataloader(
            split='test', batch_size=batch_size, shuffle=False, num_workers=2, output_mode='regression'
        )
        
        # Setup optimizer and scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.get('learning_rate', 1e-3),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        decay_epochs = self.training_config.get('decay_epochs', 1000)
        decay_rate = self.training_config.get('decay_rate', 0.9)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_epochs, gamma=decay_rate
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup metrics
        self.r2_metric_x = torchmetrics.R2Score()
        self.r2_metric_y = torchmetrics.R2Score()
    
    def _setup_session_checkpoint_manager(self):
        """Setup checkpoint manager for this specific session."""
        if self.session_checkpoint_dir:
            from pathlib import Path
            from .enhanced_logger import SimpleCheckpointManager
            
            # Create session-specific checkpoint directory
            session_ckpt_dir = Path(self.session_checkpoint_dir) / f"{self.session_index:03d}_session_{self.session_id.replace('/', '_')}"
            session_ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint_manager = SimpleCheckpointManager(
                checkpoint_dir=session_ckpt_dir,
                metric_name='val_loss',
                mode='min'
            )
            
            self.logger.info(f"   Session checkpoint manager setup: {session_ckpt_dir}")
        else:
            self.checkpoint_manager = None
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression metrics.
        Args:
            predictions: [B, T, 2] - predicted values
            targets: [B, T, 2] - target values
            
        Returns:
            Dictionary of metrics
        """
        # Update R2 metrics for each dimension separately
            # For each sample in the batch, update metrics across all timepoints
        for b in range(predictions.shape[0]):
                # Get predictions and targets for one sample across all time points
                # [time, 2] for both predictions and targets
                sample_pred = predictions[b]
                sample_target = targets[b]
                
                # Update separate metrics for x and y coordinates
                self.r2_metric_x.update(sample_pred[:, 0], sample_target[:, 0])
                self.r2_metric_y.update(sample_pred[:, 1], sample_target[:, 1])
        
        # Compute final metrics from accumulated state
        r2_x = self.r2_metric_x.compute().item()
        r2_y = self.r2_metric_y.compute().item()
        
        # Average metrics
        epoch_r2 = (r2_x + r2_y) / 2
        
        metrics = {
            'r2_x': r2_x,
            'r2_y': r2_y,
            'r2_mean': epoch_r2
        }   
        
        return metrics
    
    def log_epoch_to_tensorboard(self, epoch: int, train_stats: Dict[str, float], 
                                val_stats: Dict[str, float], test_stats: Dict[str, float]):
        """Log epoch metrics to TensorBoard for this session."""
        if not self.writer:
            return
        
        # Clean session ID for TensorBoard tags
        clean_session_id = self.session_id.replace('/', '_').replace('\\', '_').replace('.', '_')
        
        # Log train metrics
        self.writer.add_scalar(f'Session_{clean_session_id}/Train_Loss', train_stats.get('loss', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Train_R2_Mean', train_stats.get('r2_mean', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Train_R2_X', train_stats.get('r2_x', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Train_R2_Y', train_stats.get('r2_y', 0), epoch)
        
        # Log validation metrics
        self.writer.add_scalar(f'Session_{clean_session_id}/Val_Loss', val_stats.get('loss', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Val_R2_Mean', val_stats.get('r2_mean', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Val_R2_X', val_stats.get('r2_x', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Val_R2_Y', val_stats.get('r2_y', 0), epoch)
        
        # Log test metrics
        self.writer.add_scalar(f'Session_{clean_session_id}/Test_Loss', test_stats.get('loss', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Test_R2_Mean', test_stats.get('r2_mean', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Test_R2_X', test_stats.get('r2_x', 0), epoch)
        self.writer.add_scalar(f'Session_{clean_session_id}/Test_R2_Y', test_stats.get('r2_y', 0), epoch)
        
        # Log learning rate
        if 'learning_rate' in train_stats:
            self.writer.add_scalar(f'Session_{clean_session_id}/Learning_Rate', train_stats['learning_rate'], epoch)
        
        # Track and log best R2 score
        current_r2 = val_stats.get('r2_mean', 0)
        if current_r2 > self.best_r2_score:
            self.best_r2_score = current_r2
            self.writer.add_scalar(f'Session_{clean_session_id}/Best_R2_Score', self.best_r2_score, epoch)
    
    def save_session_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]) -> Dict[str, str]:
        """Save checkpoint for this session."""
        if not self.checkpoint_manager:
            return {}
        
        checkpoint = {
            'epoch': epoch,
            'session_id': self.session_id,
            'session_index': self.session_index,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_r2_score': self.best_r2_score,
            'config': self.config,
            'metrics': metrics,
        }
        
        saved_paths = self.checkpoint_manager.save_checkpoint(checkpoint, epoch, val_loss)
        
        # if saved_paths:
        #     self.logger.info(f"   Session checkpoint saved: {saved_paths}")
        
        return saved_paths
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        epoch_stats = defaultdict(float)
        num_batches = len(self.train_loader)
        
        for batch_data in self.train_loader:
            neural_data = batch_data[0].to(self.device)      # [B, 1, T, N]
            session_coords = batch_data[1].to(self.device)   # [B, 1, 3] - session coordinates
            trajectories = batch_data[2].to(self.device)     # [B, T, 2]
            velocities = batch_data[3].to(self.device)       # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Forward pass
            predictions = self.model(neural_data, session_coords)  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(predictions, targets)
            
            # Update statistics
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        # Add current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_stats['learning_rate'] = current_lr
        
        return dict(epoch_stats)
    
    @torch.no_grad()
    def validate_epoch(self, split: str = 'val') -> Dict[str, float]:
        """Run one validation/test epoch."""
        self.model.eval()
        
        loader = self.val_loader if split == 'val' else self.test_loader
        epoch_stats = defaultdict(float)
        num_batches = len(loader)
        
        for batch_data in loader:
            neural_data = batch_data[0].to(self.device)      # [B, 1, T, N]
            session_coords = batch_data[1].to(self.device)   # [B, 1, 3]
            trajectories = batch_data[2].to(self.device)     # [B, T, 2]
            velocities = batch_data[3].to(self.device)       # [B, T, 2]
            
            # Choose target based on target_type
            targets = velocities if self.target_type == 'velocity' else trajectories
            
            # Forward pass
            predictions = self.model(neural_data, session_coords)  # [B, T, 2]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Compute metrics
            metrics = self.compute_metrics(predictions, targets)
            
            # Update statistics
            epoch_stats['loss'] += loss.item()
            for key, value in metrics.items():
                epoch_stats[key] += value
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return dict(epoch_stats)
    
    def train_and_evaluate(self) -> Dict[str, float]:
        """
        Train and evaluate on this session only.
        
        Returns:
            Final test metrics for this session
        """
        
        # Training loop (focused on single session)
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Training and validation
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch('val')
            test_stats = self.validate_epoch('test')
            
            # Learning rate step
            self.scheduler.step()
            
            # Log to TensorBoard (every epoch)
            self.log_epoch_to_tensorboard(epoch, train_stats, val_stats, test_stats)
            
            # Log progress every 10 epochs or at the end
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                self.logger.info(f"     Epoch {epoch:3d}: Train R²={train_stats['r2_mean']:.4f}, "
                               f"Val R²={val_stats['r2_mean']:.4f}, Test R²={test_stats['r2_mean']:.4f}, Loss={val_stats['loss']:.4f}")
            
            # Early stopping check
            is_best = val_stats['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint for this session
            session_metrics = {
                'train_r2': train_stats['r2_mean'],
                'val_r2': val_stats['r2_mean'],
                'test_r2': test_stats['r2_mean'],
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
                'test_loss': test_stats['loss']
            }
            self.save_session_checkpoint(epoch, val_stats['loss'], session_metrics)
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"     Early stopping at epoch {epoch}")
                break
        
        return test_stats

# ===========================================================================================
# Factory Functions and Utilities
# ===========================================================================================

def create_per_session_evaluation_manager(checkpoint_path: str,
                                         evaluation_scenario: str,
                                         config: Dict[str, Any]) -> PublicPerSessionEvaluationManager:
    """
    Factory function to create per-session evaluation manager with clean initialization.
    
    Args:
        checkpoint_path: Path to checkpoint (foundation or downstream)
        evaluation_scenario: 'cross_session', 'cross_subject_center', or 'cross_subject_random'
        config: Training configuration
        
    Returns:
        Configured PublicPerSessionEvaluationManager instance
    """
    
    # Validate initialization mode
    model_config = config.get('model', {})
    initialization_mode = model_config.get('initialization_mode', 'foundation_pretrained')
    
    if initialization_mode not in PublicModelInitializer.INITIALIZATION_MODES:
        raise ValueError(f"Invalid initialization_mode: {initialization_mode}. "
                       f"Must be one of {list(PublicModelInitializer.INITIALIZATION_MODES.keys())}")
    
    manager = PublicPerSessionEvaluationManager(
        checkpoint_path=checkpoint_path,
        evaluation_scenario=evaluation_scenario,
        config=config
    )
    
    return manager


# ===========================================================================================
# Testing Function
# ===========================================================================================

def test_per_session_trainers():
    """Test per-session trainer implementations."""
    
    print("🧪 Testing Per-Session Trainers")
    print("=" * 60)
    
    # Mock configuration for testing
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
            'batch_size': 4,  # Small for testing
            'num_epochs': 5,  # Few epochs for testing
            'early_stopping_patience': 3
        },
        'paths': {
            'base_dir': './logs_test_per_session',
            'pretrained_path': './logs_public/full_small/checkpoints/best.pth'
        }
    }
    
    try:
        # Test with a small subset (just checking if the system works)
        print(f"\n🔍 Testing evaluation manager creation:")
        
        manager = PublicPerSessionEvaluationManager(
            pretrained_checkpoint_path=test_config['paths']['pretrained_path'],
            evaluation_scenario='cross_subject_center',
            config=test_config
        )
        
        print(f"   Manager created with {manager.total_sessions} sessions")
        
        # Note: Full evaluation test would require actual model training
        # This is just testing the setup and configuration
        
        print(f"\n✅ Per-session trainer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Per-session trainer test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests
    test_per_session_trainers()
