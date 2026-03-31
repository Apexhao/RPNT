"""
Single Session Dataset for Per-Session Evaluation
------------------------------------------------

This module provides dataset classes for individual session loading and evaluation,
supporting the new per-session evaluation protocol where each session is trained
and evaluated independently.

Key Features:
- Single session loading (no cross-session aggregation)
- Direct train/val/test splits from session's provided indices
- Reuses proven data processing logic from PublicDownstreamDatasetBase
- Factory functions for different evaluation scenarios
- Clean interface for per-session training and evaluation
"""

import torch
import numpy as np
import pickle
import logging
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import random
import re
from datetime import datetime

# Import parent class to reuse methods
from .public_downstream_dataset import PublicDownstreamDatasetBase


class PublicSingleSessionDataset(Dataset):
    """
    Dataset for loading and processing a SINGLE session for per-session evaluation.
    
    **DESIGN GOALS**:
    - Load only one session at a time for individual training/evaluation
    - Reuse proven data processing logic from PublicDownstreamDatasetBase
    - Direct access to session's train/val/test splits
    - Compatible with pretrained SparseTemporalEncoder
    - Session coordinates (1,3) for RoPE4D consistency
    
    **KEY DIFFERENCES from PublicDownstreamDatasetBase**:
    - Takes session_id as input, not session_filter_func
    - No _organize_evaluation_data() - already single session
    - Simpler interface focused on single session
    - Direct train/val/test access without concatenation
    """
    
    def __init__(self,
                 session_id: str,
                 data_root: str = "/data/Fang-analysis/causal-nfm/Data/processed_normalize",  #"/data/Fang-analysis/causal-nfm/Data/public_data"
                 target_neurons: int = 50,
                 sequence_length: int = 50,
                 neuron_selection_strategy: str = 'first_n',
                 random_seed: int = 42):
        """
        Initialize PublicSingleSessionDataset for a specific session.
        
        Args:
            session_id: Specific session identifier (e.g., 't_20130819_center_out_reaching')
            data_root: Root directory containing public dataset .pkl files
            target_neurons: Target number of neurons (N dimension)
            sequence_length: Expected sequence length (T=50)
            neuron_selection_strategy: 'first_n', 'random_n', or 'all'
            random_seed: Random seed for reproducible sampling
        """
        
        # Configuration
        self.session_id = session_id
        self.data_root = Path(data_root)
        self.target_neurons = target_neurons
        self.sequence_length = sequence_length
        self.neuron_selection_strategy = neuron_selection_strategy
        self.random_seed = random_seed
        
        # Validate inputs
        valid_strategies = ['first_n', 'random_n', 'all']
        if neuron_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        # Set random seed for reproducible sampling
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Storage for processed data
        self.session_data = None  # {split: {neural_data, labels, trajectories, velocities}}
        self.session_coordinates = None  # (1, 3) tensor
        self.data_stats = {}
        
        # Load and process the specific session
        self._load_and_process_session()
        
        # Compute statistics
        self._compute_statistics()
        
        logging.info(f"PublicSingleSessionDataset initialized for session: {session_id}")
    
    def _load_and_process_session(self):
        """Load and process the specified session."""
        
        print(f"🔄 Loading single session: {self.session_id}")
        
        # Find the session file
        pkl_file = self.data_root / f"{self.session_id}.pkl"
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"Session file not found: {pkl_file}")
        
        # Load session data using reused logic from PublicDownstreamDatasetBase
        session_data = self._load_single_session(self.session_id, pkl_file)
        
        if session_data is None:
            raise ValueError(f"Failed to load session data for: {self.session_id}")
        
        self.session_data = session_data
        self.session_coordinates = self._generate_session_coordinates(self.session_id)
        
        print(f"✅ Successfully loaded session: {self.session_id}")
        
        # Print session info
        for split, data in self.session_data.items():
            print(f"   {split.capitalize()}: {data['n_trials']} trials")
    
    def _load_single_session(self, session_id: str, pkl_file: Path) -> Optional[Dict]:
        """
        Load and process data for the single session.
        **REUSED from PublicDownstreamDatasetBase with identical logic**
        
        Args:
            session_id: Session identifier
            pkl_file: Path to .pkl file
            
        Returns:
            Dictionary containing processed session data or None if failed
        """
        
        try:
            # Load data file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract key components
            spike_data = data['spike_data']  # (n_windows, 50, n_units)
            
            if 'velocity_data' not in data:
                cursor_position = data['cursor_velocity']  # (n_windows, 50, 2)
                cursor_velocity = data['cursor_velocity']  # (n_windows, 50, 2)
            else:
                cursor_position = data['velocity_data']  # (n_windows, 50, 2)
                cursor_velocity = data['velocity_data']  # (n_windows, 50, 2)
            
            train_indices = data['train_indices']
            valid_indices = data['valid_indices']
            test_indices = data['test_indices']
            
            # Validate expected dimensions
            if spike_data.shape[1] != self.sequence_length:
                print(f"⚠️  Unexpected sequence length: {spike_data.shape[1]}, expected {self.sequence_length}")
            
            # Process neural data
            processed_spike_data = self._process_neural_data(spike_data)
            
            # Process behavioral data
            processed_trajectories = cursor_position.astype(np.float32)
            processed_velocities = cursor_velocity.astype(np.float32)
            
            # Generate direction labels for center-out tasks
            direction_labels = self._generate_direction_labels(processed_velocities)
            
            # Apply splits using provided indices
            split_data = self._apply_splits(
                processed_spike_data, direction_labels, processed_trajectories, processed_velocities,
                train_indices, valid_indices, test_indices
            )
            
            return split_data
            
        except Exception as e:
            print(f"💥 Error processing session {session_id}: {str(e)}")
            return None
    
    def _process_neural_data(self, spike_data: np.ndarray) -> np.ndarray:
        """
        Process neural data with neuron selection and standardization.
        **REUSED from PublicDownstreamDatasetBase with identical logic**
        
        Args:
            spike_data: (n_windows, T, n_units)
            
        Returns:
            processed_data: (n_windows, T, target_neurons)
        """
        
        n_windows, time, original_neurons = spike_data.shape
        
        # Apply neuron selection strategy
        if self.neuron_selection_strategy == 'all':
            if original_neurons > self.target_neurons:
                # Randomly sample if too many neurons
                selected_indices = np.random.choice(original_neurons, self.target_neurons, replace=False)
                processed_data = spike_data[:, :, selected_indices]
            elif original_neurons < self.target_neurons:
                # Pad with random sampling with replacement
                additional_needed = self.target_neurons - original_neurons
                additional_indices = np.random.choice(original_neurons, additional_needed, replace=True)
                all_indices = np.concatenate([np.arange(original_neurons), additional_indices])
                processed_data = spike_data[:, :, all_indices]
            else:
                processed_data = spike_data
                
        elif self.neuron_selection_strategy == 'first_n':
            if original_neurons >= self.target_neurons:
                processed_data = spike_data[:, :, :self.target_neurons]
            else:
                # Pad by repeating neurons
                repeats_needed = (self.target_neurons + original_neurons - 1) // original_neurons
                repeated_data = np.tile(spike_data, (1, 1, repeats_needed))
                processed_data = repeated_data[:, :, :self.target_neurons]
                
        elif self.neuron_selection_strategy == 'random_n':
            if original_neurons >= self.target_neurons:
                selected_indices = np.random.choice(original_neurons, self.target_neurons, replace=False)
                processed_data = spike_data[:, :, selected_indices]
            else:
                # Sample with replacement
                selected_indices = np.random.choice(original_neurons, self.target_neurons, replace=True)
                processed_data = spike_data[:, :, selected_indices]
        
        print(f"🧠 Neural data processing: {spike_data.shape} → {processed_data.shape}")
        
        return processed_data.astype(np.float32)
    
    def _generate_direction_labels(self, velocities: np.ndarray) -> np.ndarray:
        """
        Generate 8-direction classification labels from velocity data.
        **REUSED from PublicDownstreamDatasetBase with identical logic**
        
        Args:
            velocities: (n_windows, T, 2) - [vx, vy] velocities
            
        Returns:
            direction_labels: (n_windows,) - 8-direction class labels
        """
        
        # Use mean velocity across time for each window
        mean_velocities = np.mean(velocities, axis=1)  # (n_windows, 2)
        
        # Calculate angles
        angles = np.arctan2(mean_velocities[:, 1], mean_velocities[:, 0])  # (n_windows,)
        
        # Convert to degrees and normalize to [0, 360)
        angles_deg = np.degrees(angles)
        angles_deg[angles_deg < 0] += 360
        
        # Bin into 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
        # Each bin covers 45 degrees
        direction_labels = ((angles_deg + 22.5) / 45).astype(int) % 8
        
        return direction_labels.astype(np.int64)
    
    def _apply_splits(self, neural_data: np.ndarray, labels: np.ndarray, 
                     trajectories: np.ndarray, velocities: np.ndarray,
                     train_indices: np.ndarray, valid_indices: np.ndarray, test_indices: np.ndarray):
        """
        Apply train/validation/test splits using provided indices.
        **REUSED from PublicDownstreamDatasetBase with identical logic**
        """
        
        split_data = {}
        
        for split_name, split_indices in [('train', train_indices), ('val', valid_indices), ('test', test_indices)]:
            split_data[split_name] = {
                'neural_data': neural_data[split_indices],      # (Split_Windows, T, N)
                'labels': labels[split_indices],                # (Split_Windows,)
                'trajectories': trajectories[split_indices],    # (Split_Windows, T, 2)
                'velocities': velocities[split_indices],        # (Split_Windows, T, 2)
                'n_trials': len(split_indices)
            }
            
            print(f"   {split_name.capitalize()} split: {len(split_indices)} trials")
        
        return split_data
    
    def _generate_session_coordinates(self, session_id: str) -> torch.Tensor:
        """
        Generate session coordinates (1,3) from session metadata for RoPE4D.
        **REUSED from PublicDownstreamDatasetBase with identical logic**
        """
        
        # Parse session_id: 'c_20131003_center_out_reaching'
        parts = session_id.split('_')
        
        if len(parts) >= 3:
            subject = parts[0]  # 'c'
            date_str = parts[1]  # '20131003'
            task_parts = parts[2:]  # ['center', 'out', 'reaching'] or ['random', 'target', 'reaching']
            task = '_'.join(task_parts)  # 'center_out_reaching' or 'random_target_reaching'
        else:
            # Fallback for unexpected format
            subject = session_id[0] if len(session_id) > 0 else 'unknown'
            date_str = '20130101'  # Default date
            task = 'unknown'
        
        # Subject embedding (0-based indexing)
        subject_mapping = {'c': 0, 'j': 1, 'm': 2, 't': 3}
        subject_emb = subject_mapping.get(subject, 0)
        
        # Time embedding (normalize date to [0,1] range)
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            # Normalize to range based on dataset span (2013-2016)
            base_date = datetime(2013, 1, 1)
            end_date = datetime(2017, 1, 1)
            time_emb = (date_obj - base_date).days / (end_date - base_date).days
            time_emb = max(0.0, min(1.0, time_emb))  # Clamp to [0,1]
        except:
            time_emb = 0.5  # Default to middle of range
        
        # Task embedding
        task_mapping = {
            'center_out_reaching': 0, 
            'random_target_reaching': 1,
            'unknown': 2
        }
        task_emb = task_mapping.get(task, 2)
        
        # Create coordinate tensor (1, 3)
        coordinates = torch.tensor([[subject_emb, time_emb, task_emb]], dtype=torch.float32)
        
        return coordinates
    
    def _compute_statistics(self):
        """Compute dataset statistics for the single session."""
        
        train_neural = self.session_data['train']['neural_data']
        train_labels = self.session_data['train']['labels']
        train_velocities = self.session_data['train']['velocities']
        
        self.data_stats = {
            'session_id': self.session_id,
            'target_neurons': self.target_neurons,
            'sequence_length': self.sequence_length,
            'split_sizes': {
                split: data['n_trials'] for split, data in self.session_data.items()
            },
            'neural_stats': {
                'mean': float(np.mean(train_neural)),
                'std': float(np.std(train_neural)),
                'min': float(np.min(train_neural)),
                'max': float(np.max(train_neural))
            },
            'label_stats': {
                'num_classes': int(np.max(train_labels) + 1),
                'class_distribution': {int(i): int(np.sum(train_labels == i)) for i in range(int(np.max(train_labels) + 1))}
            },
            'velocity_stats': {
                'mean': [float(np.mean(train_velocities[:, :, 0])), float(np.mean(train_velocities[:, :, 1]))],
                'std': [float(np.std(train_velocities[:, :, 0])), float(np.std(train_velocities[:, :, 1]))]
            }
        }
    
    def get_split_data(self, split: str = 'train') -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Dictionary with neural_data, labels, trajectories, velocities, coordinates
        """
        if split not in self.session_data:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.session_data.keys())}")
        
        split_data = self.session_data[split]
        neural_data = split_data['neural_data']  # (Windows, T, N)
        n_windows = neural_data.shape[0]
        
        # Add site dimension: (Windows, T, N) → (Windows, 1, T, N)
        neural_data_with_site = neural_data[:, np.newaxis, :, :]
        
        # Replicate session coordinates for all windows
        session_coords_batch = self.session_coordinates.unsqueeze(0).repeat(n_windows, 1, 1)  # (Windows, 1, 3)
        
        return {
            'neural_data': neural_data_with_site.astype(np.float32),     # (Windows, 1, T, N)
            'labels': split_data['labels'],                              # (Windows,)
            'trajectories': split_data['trajectories'],                  # (Windows, T, 2)
            'velocities': split_data['velocities'],                      # (Windows, T, 2)
            'coordinates': session_coords_batch,                         # (Windows, 1, 3)
            'n_trials': split_data['n_trials']
        }
    
    def create_dataloader(self,
                         split: str = 'train',
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         output_mode: str = 'regression') -> DataLoader:
        """
        Create a PyTorch DataLoader for the specified split.
        
        Args:
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            output_mode: 'regression', 'classification', or 'both'
            
        Returns:
            DataLoader yielding batches based on output_mode
        """
        
        split_data = self.get_split_data(split)
        
        # Create dataset based on output mode
        if output_mode == 'regression':
            # For regression: neural_data, coordinates, trajectories, velocities
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(split_data['neural_data'], dtype=torch.float32),
                split_data['coordinates'],
                torch.tensor(split_data['trajectories'], dtype=torch.float32),
                torch.tensor(split_data['velocities'], dtype=torch.float32)
            )
        elif output_mode == 'classification':
            # For classification: neural_data, coordinates, labels
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(split_data['neural_data'], dtype=torch.float32),
                split_data['coordinates'],
                torch.tensor(split_data['labels'], dtype=torch.long)
            )
        elif output_mode == 'both':
            # For both: neural_data, coordinates, labels, trajectories, velocities
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(split_data['neural_data'], dtype=torch.float32),
                split_data['coordinates'],
                torch.tensor(split_data['labels'], dtype=torch.long),
                torch.tensor(split_data['trajectories'], dtype=torch.float32),
                torch.tensor(split_data['velocities'], dtype=torch.float32)
            )
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}. Use 'regression', 'classification', or 'both'")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self.data_stats
    
    def get_session_id(self) -> str:
        """Return session ID."""
        return self.session_id
    
    def get_session_coordinates(self) -> torch.Tensor:
        """Return session coordinates (1, 3)."""
        return self.session_coordinates
    
    def print_summary(self):
        """Print a comprehensive dataset summary."""
        
        print(f"\n" + "="*60)
        print(f"📋 PublicSingleSessionDataset Summary")
        print(f"="*60)
        
        print(f"\n🏗️  Configuration:")
        print(f"   • Session ID: {self.session_id}")
        print(f"   • Data root: {self.data_root}")
        print(f"   • Target neurons: {self.target_neurons}")
        print(f"   • Sequence length: {self.sequence_length}")
        print(f"   • Neuron strategy: {self.neuron_selection_strategy}")
        
        print(f"\n📊 Data Shapes:")
        for split, data in self.session_data.items():
            neural_shape = data['neural_data'].shape
            labels_shape = data['labels'].shape
            print(f"   • {split.capitalize()}:")
            print(f"     - Neural: {neural_shape} (Windows, Time, Neurons)")
            print(f"     - Labels: {labels_shape}")
            print(f"     - Trajectories: {data['trajectories'].shape}")
            print(f"     - Velocities: {data['velocities'].shape}")
        
        stats = self.data_stats
        print(f"\n🧠 Statistics:")
        print(f"   • Neural - Mean: {stats['neural_stats']['mean']:.4f}, Std: {stats['neural_stats']['std']:.4f}")
        print(f"   • Classes: {stats['label_stats']['num_classes']}")
        print(f"   • Class distribution: {stats['label_stats']['class_distribution']}")
        
        print(f"\n✅ Ready for per-session evaluation!")
        print(f"="*60)


# ===========================================================================================
# Factory Functions for Different Evaluation Scenarios
# ===========================================================================================

def get_session_ids_for_scenario(scenario: str) -> List[str]:
    """
    Get session IDs for different evaluation scenarios.
    
    Args:
        scenario: 'cross_session', 'cross_subject_center', or 'cross_subject_random'
        
    Returns:
        List of session IDs for the scenario
    """
    
    if scenario == 'cross_session':
        # Subject c, 2016xxx sessions, center-out tasks
        # These session IDs should match what's actually in your data directory
        session_ids = [
            'c_20160909_center_out_reaching',
            'c_20160912_center_out_reaching',
            'c_20160914_center_out_reaching',
            'c_20160915_center_out_reaching',
            'c_20160919_center_out_reaching',
            'c_20160921_center_out_reaching',
            'c_20160923_center_out_reaching',
            'c_20160929_center_out_reaching',
            'c_20161005_center_out_reaching',
            'c_20161006_center_out_reaching',
            'c_20161007_center_out_reaching',
            'c_20161011_center_out_reaching',
            'c_20161013_center_out_reaching',
            'c_20161021_center_out_reaching'
        ]
        
    elif scenario == 'cross_subject_center':
        # Subject t, center-out tasks  
        session_ids = [
            't_20130819_center_out_reaching',
            't_20130821_center_out_reaching',
            't_20130823_center_out_reaching',
            't_20130903_center_out_reaching',
            't_20130905_center_out_reaching',
            't_20130909_center_out_reaching'
        ]
        
    elif scenario == 'cross_subject_random':
        # Subject t, random-target tasks
        session_ids = [
            't_20130820_random_target_reaching',
            't_20130822_random_target_reaching',
            't_20130830_random_target_reaching',
            't_20130904_random_target_reaching',
            't_20130906_random_target_reaching',
            't_20130910_random_target_reaching'
        ]
        
    else:
        raise ValueError(f"Invalid scenario: {scenario}. Must be one of: cross_session, cross_subject_center, cross_subject_random")
    
    return session_ids


def create_single_session_datasets(scenario: str, **kwargs) -> List[PublicSingleSessionDataset]:
    """
    Create list of PublicSingleSessionDataset for different evaluation scenarios.
    
    Args:
        scenario: 'cross_session', 'cross_subject_center', or 'cross_subject_random'
        **kwargs: Additional arguments passed to PublicSingleSessionDataset
        
    Returns:
        List of PublicSingleSessionDataset instances
    """
    
    session_ids = get_session_ids_for_scenario(scenario)
    datasets = []
    
    print(f"\n🏭 Creating single session datasets for scenario: {scenario}")
    print(f"Session IDs: {session_ids}")
    
    for session_id in session_ids:
        try:
            dataset = PublicSingleSessionDataset(session_id=session_id, **kwargs)
            datasets.append(dataset)
            print(f"✅ Created dataset for session: {session_id}")
        except Exception as e:
            print(f"❌ Failed to create dataset for session {session_id}: {str(e)}")
            continue
    
    print(f"📋 Successfully created {len(datasets)}/{len(session_ids)} datasets")
    
    return datasets


def create_cross_session_datasets(**kwargs) -> List[PublicSingleSessionDataset]:
    """Factory function for cross-session evaluation datasets (Subject c, 2016xxx sessions)."""
    return create_single_session_datasets('cross_session', **kwargs)


def create_cross_subject_center_datasets(**kwargs) -> List[PublicSingleSessionDataset]:
    """Factory function for cross-subject center evaluation datasets (Subject t, center-out)."""
    return create_single_session_datasets('cross_subject_center', **kwargs)


def create_cross_subject_random_datasets(**kwargs) -> List[PublicSingleSessionDataset]:
    """Factory function for cross-subject random evaluation datasets (Subject t, random-target)."""
    return create_single_session_datasets('cross_subject_random', **kwargs)


# ===========================================================================================
# Testing Function
# ===========================================================================================

def test_public_single_session_dataset():
    """Test PublicSingleSessionDataset implementation."""
    
    print("🧪 Testing PublicSingleSessionDataset")
    print("=" * 60)
    
    # Test parameters
    test_session_id = 't_20130819_center_out_reaching'  # Known session for testing
    
    try:
        # Test single session dataset
        print(f"\n🔍 Testing single session: {test_session_id}")
        dataset = PublicSingleSessionDataset(
            session_id=test_session_id,
            target_neurons=50,
            neuron_selection_strategy='first_n',
            random_seed=42
        )
        
        # Print summary
        dataset.print_summary()
        
        # Test dataloader creation
        print(f"\n🔄 Testing dataloader creation:")
        train_loader = dataset.create_dataloader('train', batch_size=4, output_mode='regression')
        train_batch = next(iter(train_loader))
        print(f"   Batch shapes: neural={train_batch[0].shape}, coords={train_batch[1].shape}, "
              f"traj={train_batch[2].shape}, vel={train_batch[3].shape}")
        
        # Test factory functions
        print(f"\n🏭 Testing factory functions:")
        cross_subject_datasets = create_cross_subject_center_datasets(
            target_neurons=50,
            random_seed=42
        )
        print(f"   Cross-subject center datasets: {len(cross_subject_datasets)}")
        
        print(f"\n✅ All single session dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Single session dataset test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests
    test_public_single_session_dataset()
