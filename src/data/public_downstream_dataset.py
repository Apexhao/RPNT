"""
Public Dataset for Downstream Tasks Evaluation
---------------------------------------------

This module provides specialized datasets for downstream task evaluation on the Perich-Miller 2018 public dataset.

Key Features:
- 3 evaluation scenarios: cross-session, cross-subject, cross-task
- Single session format [B,S=1,T=50,N=50] for pretrained model compatibility
- Multiple output targets: regression (velocity) + classification (direction)
- Uses provided train/valid/test splits from each session
- Session coordinates (1,3) for RoPE4D consistency
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


class PublicDownstreamDatasetBase(Dataset):
    """
    Base class for public dataset downstream evaluation.
    
    **DESIGN GOALS**:
    - Compatible with pretrained SparseTemporalEncoder
    - Multiple output modalities for comprehensive evaluation
    - Session-specific filtering for different evaluation scenarios
    - Same coordinate system as pretraining dataset
    
    **KEY FEATURES**:
    - Format: [B,S=1,T=50,N=50] for model compatibility
    - Session coordinates: (1,3) matching pretraining
    - Velocity prediction + direction classification
    - Uses provided train/valid/test indices
    """
    
    def __init__(self,
                 data_root: str = "/data/Fang-analysis/causal-nfm/Data/processed_normalize_session",
                 session_filter_func: callable = None,
                 target_neurons: int = 50,
                 sequence_length: int = 50,
                 neuron_selection_strategy: str = 'first_n',
                 random_seed: int = 42):
        """
        Initialize PublicDownstreamDatasetBase.
        
        Args:
            data_root: Root directory containing public dataset .pkl files
            session_filter_func: Function to filter sessions (e.g., lambda x: x.startswith('c_2016'))
            target_neurons: Target number of neurons (N dimension)
            sequence_length: Expected sequence length (T=50)
            neuron_selection_strategy: 'first_n', 'random_n', or 'all'
            random_seed: Random seed for reproducible sampling
        """
        
        # Configuration
        self.data_root = Path(data_root)
        self.session_filter_func = session_filter_func
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
        self.sessions_data = {}  # {session_id: {split: {neural_data, labels, trajectories, velocities}}}
        self.session_coordinates = {}  # {session_id: (1, 3) tensor}
        self.session_ids = []  # List of valid session IDs
        self.data_stats = {}
        
        # Load and process data
        self._discover_and_load_sessions()
        
        # Organize data for evaluation
        self._organize_evaluation_data()
        
        # Compute statistics
        self._compute_statistics()
        
        logging.info(f"PublicDownstreamDatasetBase initialized with {len(self.session_ids)} sessions")
    
    def _discover_and_load_sessions(self):
        """Discover and load sessions based on filter function."""
        
        print("🔄 Discovering downstream evaluation sessions...")
        
        # Find all .pkl files in data_root
        pkl_files = list(self.data_root.glob("*.pkl"))
        print(f"Found {len(pkl_files)} total .pkl files")
        
        # Apply session filter
        filtered_files = []
        for pkl_file in pkl_files:
            session_id = pkl_file.stem  # Remove .pkl extension
            
            if self.session_filter_func is None or self.session_filter_func(session_id):
                filtered_files.append(pkl_file)
        
        print(f"Filtered to {len(filtered_files)} sessions")
        
        successful_loads = 0
        failed_loads = 0
        
        for pkl_file in filtered_files:
            session_id = pkl_file.stem
            
            try:
                # Load session data
                session_data = self._load_single_session(session_id, pkl_file)
                
                if session_data is not None:
                    self.sessions_data[session_id] = session_data
                    self.session_coordinates[session_id] = self._generate_session_coordinates(session_id)
                    self.session_ids.append(session_id)
                    successful_loads += 1
                    print(f"✅ Loaded session {session_id}")
                else:
                    failed_loads += 1
                    print(f"❌ Failed to load session {session_id}")
                    
            except Exception as e:
                failed_loads += 1
                print(f"❌ Error loading session {session_id}: {str(e)}")
        
        print(f"\n📊 Loading Summary:")
        print(f"   ✅ Successfully loaded: {successful_loads} sessions")
        print(f"   ❌ Failed to load: {failed_loads} sessions")
        print(f"   📋 Available sessions: {self.session_ids}")
        
        if successful_loads == 0:
            raise ValueError("No sessions were successfully loaded!")
    
    def _load_single_session(self, session_id: str, pkl_file: Path) -> Optional[Dict]:
        """
        Load and process data for a single session.
        
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
            cursor_position = data['cursor_position']  # (n_windows, 50, 2)
            cursor_velocity = data['cursor_velocity']  # (n_windows, 50, 2)
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
        """Apply train/validation/test splits using provided indices."""
        
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
        Same implementation as pretraining dataset for consistency.
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
    
    def _organize_evaluation_data(self):
        """Organize data for cross-session evaluation."""
        
        self.evaluation_data = {'train': {}, 'val': {}, 'test': {}}
        
        for split in ['train', 'val', 'test']:
            # Collect data from all sessions for this split
            all_neural_data = []
            all_labels = []
            all_trajectories = []
            all_velocities = []
            all_coordinates = []
            
            for session_id in self.session_ids:
                session_data = self.sessions_data[session_id][split]
                session_coords = self.session_coordinates[session_id]
                
                neural_data = session_data['neural_data']  # (Windows, T, N)
                n_windows = neural_data.shape[0]
                
                # Add site dimension: (Windows, T, N) → (Windows, 1, T, N)
                neural_data_with_site = neural_data[:, np.newaxis, :, :]
                
                # Replicate session coordinates for all windows
                session_coords_batch = session_coords.unsqueeze(0).repeat(n_windows, 1, 1)  # (Windows, 1, 3)
                
                all_neural_data.append(neural_data_with_site)
                all_labels.append(session_data['labels'])
                all_trajectories.append(session_data['trajectories'])
                all_velocities.append(session_data['velocities'])
                all_coordinates.append(session_coords_batch)
            
            # Concatenate all sessions
            if all_neural_data:
                self.evaluation_data[split] = {
                    'neural_data': np.concatenate(all_neural_data, axis=0).astype(np.float32),     # (Total_Windows, 1, T, N)
                    'labels': np.concatenate(all_labels, axis=0),                                 # (Total_Windows,)
                    'trajectories': np.concatenate(all_trajectories, axis=0).astype(np.float32), # (Total_Windows, T, 2)
                    'velocities': np.concatenate(all_velocities, axis=0).astype(np.float32),     # (Total_Windows, T, 2)
                    'coordinates': torch.cat(all_coordinates, dim=0),                            # (Total_Windows, 1, 3)
                    'n_trials': sum(len(data) for data in all_neural_data)
                }
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        
        train_neural = self.evaluation_data['train']['neural_data']
        train_labels = self.evaluation_data['train']['labels']
        train_velocities = self.evaluation_data['train']['velocities']
        
        self.data_stats = {
            'session_ids': self.session_ids,
            'n_sessions': len(self.session_ids),
            'target_neurons': self.target_neurons,
            'sequence_length': self.sequence_length,
            'split_sizes': {
                split: data['n_trials'] for split, data in self.evaluation_data.items()
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
        if split not in self.evaluation_data:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.evaluation_data.keys())}")
        
        return self.evaluation_data[split]
    
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
    
    def get_session_ids(self) -> List[str]:
        """Return list of available session IDs."""
        return self.session_ids.copy()
    
    def get_session_data(self, session_id: str, split: str) -> Dict[str, torch.Tensor]:
        """
        Get data for a specific session and split.
        
        Args:
            session_id: Session identifier
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dictionary with session-specific data:
            {
                'neural_data': [Session_Windows, 1, T, N],
                'coordinates': [Session_Windows, 1, 3],
                'trajectories': [Session_Windows, T, 2],
                'velocities': [Session_Windows, T, 2],
                'labels': [Session_Windows],
                'n_trials': int
            }
        """
        if session_id not in self.session_ids:
            raise ValueError(f"Session {session_id} not found in dataset")
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")
        
        # Get session-specific data from stored session data
        session_data = self.sessions_data[session_id][split]
        session_coords = self.session_coordinates[session_id]
        
        neural_data = session_data['neural_data']  # (Windows, T, N)
        n_windows = neural_data.shape[0]
        
        # Add site dimension: (Windows, T, N) → (Windows, 1, T, N)
        neural_data_with_site = neural_data[:, np.newaxis, :, :]
        
        # Replicate session coordinates for all windows
        session_coords_batch = session_coords.unsqueeze(0).repeat(n_windows, 1, 1)  # (Windows, 1, 3)
        
        return {
            'neural_data': torch.tensor(neural_data_with_site, dtype=torch.float32),
            'coordinates': session_coords_batch,
            'trajectories': torch.tensor(session_data['trajectories'], dtype=torch.float32),
            'velocities': torch.tensor(session_data['velocities'], dtype=torch.float32),
            'labels': torch.tensor(session_data['labels'], dtype=torch.long),
            'n_trials': n_windows
        }
    
    def print_summary(self):
        """Print a comprehensive dataset summary."""
        
        print(f"\n" + "="*60)
        print(f"📋 PublicDownstreamDataset Summary")
        print(f"="*60)
        
        print(f"\n🏗️  Configuration:")
        print(f"   • Data root: {self.data_root}")
        print(f"   • Session filter: {self.session_filter_func}")
        print(f"   • Target neurons: {self.target_neurons}")
        print(f"   • Sequence length: {self.sequence_length}")
        print(f"   • Neuron strategy: {self.neuron_selection_strategy}")
        
        print(f"\n📋 Session Information:")
        print(f"   • Total sessions: {len(self.session_ids)}")
        print(f"   • Session IDs: {self.session_ids}")
        
        print(f"\n📊 Data Shapes:")
        for split, data in self.evaluation_data.items():
            neural_shape = data['neural_data'].shape
            labels_shape = data['labels'].shape
            print(f"   • {split.capitalize()}:")
            print(f"     - Neural: {neural_shape} (Windows, Sites=1, Time, Neurons)")
            print(f"     - Labels: {labels_shape}")
            print(f"     - Trajectories: {data['trajectories'].shape}")
            print(f"     - Velocities: {data['velocities'].shape}")
        
        stats = self.data_stats
        print(f"\n🧠 Statistics:")
        print(f"   • Neural - Mean: {stats['neural_stats']['mean']:.4f}, Std: {stats['neural_stats']['std']:.4f}")
        print(f"   • Classes: {stats['label_stats']['num_classes']}")
        print(f"   • Class distribution: {stats['label_stats']['class_distribution']}")
        
        print(f"\n✅ Ready for downstream evaluation!")
        print(f"="*60)


# Specific evaluation dataset classes

class PublicCrossSessionDataset(PublicDownstreamDatasetBase):
    """
    Cross-Session Evaluation: Subject c with 2016xxx center-out sessions.
    Tests same-subject, different-session generalization.
    """
    
    def __init__(self, **kwargs):
        # Filter for subject c, 2016 sessions, center-out tasks
        session_filter = lambda session_id: (
            session_id.startswith('c_2016') and 'center_out' in session_id
        )
        super().__init__(session_filter_func=session_filter, **kwargs)


class PublicCrossSubjectCenterDataset(PublicDownstreamDatasetBase):
    """
    Cross-Subject + Same Task: Subject t with center-out tasks.
    Tests cross-subject generalization with same task type.
    """
    
    def __init__(self, **kwargs):
        # Filter for subject t, center-out tasks
        session_filter = lambda session_id: (
            session_id.startswith('t_') and 'center_out' in session_id
        )
        super().__init__(session_filter_func=session_filter, **kwargs)


class PublicCrossSubjectRandomDataset(PublicDownstreamDatasetBase):
    """
    Cross-Subject + Cross-Task: Subject t with random-target tasks.
    Tests cross-subject + cross-task generalization.
    """
    
    def __init__(self, **kwargs):
        # Filter for subject t, random-target tasks
        session_filter = lambda session_id: (
            session_id.startswith('t_') and 'random_target' in session_id
        )
        super().__init__(session_filter_func=session_filter, **kwargs)

class Public_No_T_Subject_Dataset(PublicDownstreamDatasetBase):
    """
    All Subjects: All subjects with all tasks except t.
    """
    
    def __init__(self, **kwargs):
        # Filter for all subjects with all tasks
        session_filter = lambda session_id: (
            session_id.startswith('c_') or session_id.startswith('m_') or session_id.startswith('j_')
        )
        super().__init__(session_filter_func=session_filter, **kwargs)
        
class Public_Only_RT_Subject_Dataset(PublicDownstreamDatasetBase):
    """
    All Subjects: All subjects with only random-target.
    """
    
    def __init__(self, **kwargs):
        # Filter for all subjects with all tasks
        session_filter = lambda session_id: (
            'random_target' in session_id
        )
        super().__init__(session_filter_func=session_filter, **kwargs)
        
class Public_Only_CO_Subject_Dataset(PublicDownstreamDatasetBase):
    """
    All Subjects: All subjects with only center-out.
    """
    
    def __init__(self, **kwargs):
        # Filter for all subjects with all tasks
        session_filter = lambda session_id: (
            'center_out' in session_id
        )
        super().__init__(session_filter_func=session_filter, **kwargs)


def test_public_downstream_datasets():
    """Test all public downstream dataset implementations."""
    
    print("🧪 Testing Public Downstream Datasets")
    print("=" * 60)
    
    datasets = {
        'Cross-Session': PublicCrossSessionDataset,
        'Cross-Subject (Center)': PublicCrossSubjectCenterDataset,
        'Cross-Subject (Random)': PublicCrossSubjectRandomDataset,
        'All Subjects (No T)': Public_No_T_Subject_Dataset
    }
    
    results = {}
    
    for name, dataset_class in datasets.items():
        print(f"\n{'='*20} {name} {'='*20}")
        
        try:
            # Initialize dataset
            dataset = dataset_class(
                target_neurons=50,
                neuron_selection_strategy='first_n',
                random_seed=42
            )
            
            # Print summary
            dataset.print_summary()
            
            # Test dataloaders
            print(f"\n🔄 Testing dataloaders:")
            
            # Regression dataloader
            if len(dataset.session_ids) > 0:
                regression_loader = dataset.create_dataloader('train', batch_size=4, output_mode='regression')
                regression_batch = next(iter(regression_loader))
                print(f"   Regression batch shapes: neural={regression_batch[0].shape}, coords={regression_batch[1].shape}, "
                      f"traj={regression_batch[2].shape}, vel={regression_batch[3].shape}")
                
                # Classification dataloader
                classification_loader = dataset.create_dataloader('train', batch_size=4, output_mode='classification')
                classification_batch = next(iter(classification_loader))
                print(f"   Classification batch shapes: neural={classification_batch[0].shape}, coords={classification_batch[1].shape}, "
                      f"labels={classification_batch[2].shape}")
            
            results[name] = dataset
            print(f"✅ {name} dataset test passed!")
            
        except Exception as e:
            print(f"❌ {name} dataset test failed: {str(e)}")
            results[name] = None
    
    print(f"\n{'='*60}")
    print(f"📋 Test Summary:")
    for name, result in results.items():
        status = "✅ PASSED" if result is not None else "❌ FAILED"
        n_sessions = len(result.session_ids) if result else 0
        print(f"   • {name}: {status} ({n_sessions} sessions)")
    
    return results


if __name__ == "__main__":
    # Run tests
    datasets = test_public_downstream_datasets() 
