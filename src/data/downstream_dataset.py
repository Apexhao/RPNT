"""
Single-Site Downstream Dataset for Neural Foundation Model Fine-tuning
---------------------------------------------------------------------

This module provides a specialized dataset for downstream tasks (regression/classification)
using single-site neural data with the pretrained foundation model.

Key Features:
- Single-site data in (B,1,T,N) format for temporal encoder input
- Site coordinates for positional encoding
- Multiple output targets: classification labels, cursor trajectories, velocities
- Professional train/val/test splits
- Flexible neuron selection and sampling strategies
"""

import torch
import numpy as np
import pickle
import logging
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import random

# Import helper function
try:
    from ..utils.helpers import load_neuropixel_locations
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.helpers import load_neuropixel_locations


class SingleSiteDownstreamDataset(Dataset):
    """
    Single-Site Dataset for Downstream Tasks (Regression/Classification).
    
    **DESIGN GOALS**:
    - Single-site data in (B,1,T,N) format for temporal encoder
    - Site coordinates for positional encoding compatibility
    - Multi-target outputs: labels, trajectories, velocities
    - Professional train/val/test handling
    
    **KEY FEATURES**:
    - Loads single dataset by ID
    - Standardized neuron counts via sampling
    - Multiple output modalities for different tasks
    - Compatible with pretrained temporal encoder
    """
    
    def __init__(self,
                 dataset_id: str = 'te14116',
                 data_root: str = "/data/Fang-analysis/causal-nfm/Data/Monkey_data_meta",
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 target_neurons: int = 50,
                 width: float = 0.02,
                 sequence_length: int = 50,
                 neuron_selection_strategy: str = 'first_n',
                 selected_neurons: int = 50,
                 random_seed: int = 42):
        """
        Initialize SingleSiteDownstreamDataset.
        
        Args:
            dataset_id: Dataset identifier (e.g., 'te14116')
            data_root: Root directory containing data files
            split_ratios: (train, validation, test) split ratios
            target_neurons: Target number of neurons (N dimension)
            width: Time bin width for spike data
            sequence_length: Expected sequence length (T dimension)
            neuron_selection_strategy: 'first_n', 'random_n', or 'all'
            selected_neurons: Number of neurons to select
            random_seed: Random seed for reproducible sampling
        """
        
        # Configuration
        self.dataset_id = dataset_id
        self.data_root = Path(data_root)
        self.split_ratios = split_ratios
        self.target_neurons = target_neurons
        self.width = width
        self.sequence_length = sequence_length
        self.neuron_selection_strategy = neuron_selection_strategy
        self.selected_neurons = selected_neurons
        self.random_seed = random_seed
    
        # Validate inputs
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        valid_strategies = ['first_n', 'random_n', 'all']
        if neuron_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        # Set random seed for reproducible sampling
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Storage for processed data
        self.split_data = {}  # {split: {neural_data, labels, trajectories, velocities}}
        self.site_coordinates = None
        self.data_stats = {}
        
        # Load and process data
        self._load_and_process_data()
        
        # Compute statistics
        self._compute_statistics()
        
        logging.info(f"SingleSiteDownstreamDataset initialized for {dataset_id}")
    
    def _load_and_process_data(self):
        """Load raw data and process for downstream tasks."""
        
        # Extract numeric ID from dataset_id
        if self.dataset_id.startswith('te'):
            numeric_id = int(self.dataset_id[2:])
        else:
            numeric_id = int(self.dataset_id.split('.')[0])
        
        # Construct filename
        filename = self.data_root / f"beignet_te{numeric_id}_spike_data_{self.width:.2f}.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        print(f"🔄 Loading dataset: {self.dataset_id} from {filename}")
        
        try:
            # Load data file
            with open(filename, 'rb') as f:
                spike_data, labels, cursor_traj, go_cue_times, center_target_on_times, length = pickle.load(f)
            
            print(f"📊 Raw data shapes:")
            print(f"   Spike data: {spike_data.shape}")
            print(f"   Labels: {labels.shape}")
            print(f"   Cursor trajectories: {cursor_traj.shape}")
            
            # Get site coordinates
            self._load_site_coordinates()
            
            # Process neural data
            processed_spike_data = self._process_neural_data(spike_data)
            
            # Process outputs
            processed_labels = self._process_labels(labels)
            processed_trajectories = self._process_trajectories(cursor_traj, go_cue_times, center_target_on_times, length)
            processed_velocities = self._calculate_velocities(processed_trajectories)
            
            # Apply train/val/test splits
            self._apply_splits(processed_spike_data, processed_labels, processed_trajectories, processed_velocities)
            
            print(f"✅ Dataset processing completed for {self.dataset_id}")
            
        except Exception as e:
            print(f"❌ Error loading {self.dataset_id}: {str(e)}")
            raise e
    
    def _load_site_coordinates(self):
        """Load site coordinates for positional encoding."""
        
        try:
            neuropixel_locations = load_neuropixel_locations()
            
            # Extract numeric key
            if self.dataset_id.startswith('te'):
                numeric_key = int(self.dataset_id[2:]) + 0.0  # Convert to float key format
            else:
                numeric_key = float(self.dataset_id)
            
            if numeric_key in neuropixel_locations:
                site_info = neuropixel_locations[numeric_key]
                self.site_coordinates = torch.tensor([site_info['X'], site_info['Y']], dtype=torch.float32)
                print(f"📍 Site coordinates: ({site_info['X']:.2f}, {site_info['Y']:.2f})")
            else:
                print(f"⚠️  Site coordinates not found for {self.dataset_id}, using default (0, 0)")
                self.site_coordinates = torch.tensor([0.0, 0.0], dtype=torch.float32)
                
        except Exception as e:
            print(f"⚠️  Error loading site coordinates: {e}, using default (0, 0)")
            self.site_coordinates = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    def _process_neural_data(self, spike_data: np.ndarray) -> np.ndarray:
        """
        Process neural data with neuron selection and standardization.
        
        Args:
            spike_data: (Trials, Time, Original_Neurons)
            
        Returns:
            processed_data: (Trials, Time, Target_Neurons)
        """
        
        trials, time, original_neurons = spike_data.shape
        
        # Validate sequence length
        if time != self.sequence_length:
            print(f"⚠️  Sequence length mismatch: expected {self.sequence_length}, got {time}")
        
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
            if original_neurons >= self.selected_neurons:
                processed_data = spike_data[:, :, :self.selected_neurons]
            else:
                # Pad by repeating neurons
                repeats_needed = (self.selected_neurons + original_neurons - 1) // original_neurons
                repeated_data = np.tile(spike_data, (1, 1, repeats_needed))
                processed_data = repeated_data[:, :, :self.selected_neurons]
                
        elif self.neuron_selection_strategy == 'random_n':
            if original_neurons >= self.selected_neurons:
                selected_indices = np.random.choice(original_neurons, self.selected_neurons, replace=False)
                processed_data = spike_data[:, :, selected_indices]
            else:
                # Sample with replacement
                selected_indices = np.random.choice(original_neurons, self.selected_neurons, replace=True)
                processed_data = spike_data[:, :, selected_indices]
        
        # Ensure target neuron dimension
        if processed_data.shape[2] != self.target_neurons:
            if processed_data.shape[2] > self.target_neurons:
                processed_data = processed_data[:, :, :self.target_neurons]
            else:
                # Pad to target dimension
                padding_needed = self.target_neurons - processed_data.shape[2]
                padding_indices = np.random.choice(processed_data.shape[2], padding_needed, replace=True)
                padding_data = processed_data[:, :, padding_indices]
                processed_data = np.concatenate([processed_data, padding_data], axis=2)
        
        print(f"🧠 Neural data processing: {spike_data.shape} → {processed_data.shape}")
        
        return processed_data.astype(np.float32)
    
    def _process_labels(self, labels: np.ndarray) -> np.ndarray:
        """Process classification labels."""
        return labels.astype(np.int64)
    
    def _process_trajectories(self, cursor_traj: np.ndarray, go_cue_times: np.ndarray, 
                             center_target_on_times: np.ndarray, length: float) -> np.ndarray:
        """
        Process cursor trajectories to align with movement period.
        
        Args:
            cursor_traj: Raw cursor trajectories
            go_cue_times: Go cue timing
            center_target_on_times: Center target timing
            length: Movement period length
            
        Returns:
            processed_trajectories: (Trials, Time, 2) - aligned trajectories
        """
        
        transformed_trajectories = []
        sampling_rate = 1000  # Original sampling rate in Hz
        
        for traj, go_time, center_time in zip(cursor_traj, go_cue_times, center_target_on_times):
            # Convert times to indices
            start_idx = int((go_time - center_time) * sampling_rate)
            end_idx = start_idx + int(length * sampling_rate)
            
            # Extract movement period
            if start_idx < 0:
                # Pad with initial position if start_idx is negative
                pad_length = abs(start_idx)
                movement_traj = np.vstack([np.tile(traj[0], (pad_length, 1)), traj[:end_idx]])
            else:
                movement_traj = traj[start_idx:end_idx]
            
            # Ensure consistent length
            target_length_samples = int(length * sampling_rate)
            if len(movement_traj) > target_length_samples:
                movement_traj = movement_traj[:target_length_samples]
            elif len(movement_traj) < target_length_samples:
                # Pad with final position if trajectory is too short
                pad_length = target_length_samples - len(movement_traj)
                if len(movement_traj) > 0:
                    movement_traj = np.vstack([movement_traj, np.tile(movement_traj[-1], (pad_length, 1))])
                else:
                    movement_traj = np.tile(traj[0] if len(traj) > 0 else [0, 0], (target_length_samples, 1))
            
            # Downsample to target sequence length
            if len(movement_traj) > 0:
                indices = np.linspace(0, len(movement_traj)-1, self.sequence_length, dtype=int)
                downsampled_traj = movement_traj[indices]
            else:
                downsampled_traj = np.zeros((self.sequence_length, 2))
            
            transformed_trajectories.append(downsampled_traj)
        
        processed_trajectories = np.array(transformed_trajectories, dtype=np.float32)
        print(f"📈 Trajectory processing: {cursor_traj.shape} → {processed_trajectories.shape}")
        
        return processed_trajectories
    
    def _calculate_velocities(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Calculate cursor velocities from trajectories.
        
        Args:
            trajectories: (Trials, Time, 2) - cursor trajectories
            
        Returns:
            velocities: (Trials, Time, 2) - cursor velocities
        """
        
        velocities = np.zeros_like(trajectories)
        
        # Calculate velocities using central difference
        velocities[:, 1:-1] = (trajectories[:, 2:] - trajectories[:, :-2]) / (2 * self.width)
        
        # Handle boundary conditions
        velocities[:, 0] = velocities[:, 1]    # Copy first valid velocity
        velocities[:, -1] = velocities[:, -2]  # Copy last valid velocity
        
        print(f"🚀 Velocity calculation: {trajectories.shape} → {velocities.shape}")
        
        return velocities.astype(np.float32)
    
    def _apply_splits(self, neural_data: np.ndarray, labels: np.ndarray, 
                     trajectories: np.ndarray, velocities: np.ndarray):
        """Apply train/validation/test splits to all data modalities."""
        
        n_trials = neural_data.shape[0]
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        # Calculate split sizes
        train_size = int(train_ratio * n_trials)
        val_size = int(val_ratio * n_trials)
        # test_size is the remainder
        
        # Create shuffled indices
        indices = np.random.permutation(n_trials)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Apply splits to all modalities
        for split_name, split_indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            self.split_data[split_name] = {
                'neural_data': neural_data[split_indices],      # (Split_Trials, Time, Neurons)
                'labels': labels[split_indices],                # (Split_Trials,)
                'trajectories': trajectories[split_indices],    # (Split_Trials, Time, 2)
                'velocities': velocities[split_indices],        # (Split_Trials, Time, 2)
                'n_trials': len(split_indices)
            }
            
            print(f"📋 {split_name.capitalize()} split: {len(split_indices)} trials")
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        
        train_neural = self.split_data['train']['neural_data']
        train_labels = self.split_data['train']['labels']
        train_trajectories = self.split_data['train']['trajectories']
        train_velocities = self.split_data['train']['velocities']
        
        self.data_stats = {
            'dataset_id': self.dataset_id,
            'site_coordinates': self.site_coordinates.tolist(),
            'target_neurons': self.target_neurons,
            'sequence_length': self.sequence_length,
            'split_sizes': {
                split: data['n_trials'] for split, data in self.split_data.items()
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
            'trajectory_stats': {
                'position_mean': [float(np.mean(train_trajectories[:, :, 0])), float(np.mean(train_trajectories[:, :, 1]))],
                'position_std': [float(np.std(train_trajectories[:, :, 0])), float(np.std(train_trajectories[:, :, 1]))],
                'velocity_mean': [float(np.mean(train_velocities[:, :, 0])), float(np.mean(train_velocities[:, :, 1]))],
                'velocity_std': [float(np.std(train_velocities[:, :, 0])), float(np.std(train_velocities[:, :, 1]))]
            }
        }
    
    def get_split_data(self, split: str = 'train') -> Dict[str, np.ndarray]:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Dictionary with neural_data, labels, trajectories, velocities
        """
        if split not in self.split_data:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.split_data.keys())}")
        
        return self.split_data[split]
    
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
        
        # Neural data: (Trials, Time, Neurons) → (Trials, 1, Time, Neurons) for single-site format
        neural_data = split_data['neural_data'][:, np.newaxis, :, :]  # Add site dimension
        
        # Site coordinates: expand to batch format
        # We'll handle this in the dataset __getitem__ method
        
        # Create dataset based on output mode
        if output_mode == 'regression':
            # For regression: neural_data, site_coords, trajectories, velocities
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(neural_data, dtype=torch.float32),
                torch.tensor(split_data['trajectories'], dtype=torch.float32),
                torch.tensor(split_data['velocities'], dtype=torch.float32)
            )
        elif output_mode == 'classification':
            # For classification: neural_data, site_coords, labels
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(neural_data, dtype=torch.float32),
                torch.tensor(split_data['labels'], dtype=torch.long)
            )
        elif output_mode == 'both':
            # For both: neural_data, site_coords, labels, trajectories, velocities
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(neural_data, dtype=torch.float32),
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
    
    def get_site_coordinates_batch(self, batch_size: int) -> torch.Tensor:
        """
        Get site coordinates expanded for batch processing.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            coords: [B, 1, T, 2] - site coordinates expanded for all timesteps
        """
        # Expand coordinates: [2] → [B, 1, T, 2]
        coords = self.site_coordinates.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
        coords = coords.expand(batch_size, 1, self.sequence_length, 2)  # [B, 1, T, 2]
        
        return coords
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self.data_stats
    
    def __len__(self) -> int:
        """Return the total number of training samples."""
        return self.split_data['train']['n_trials']
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single training sample with all outputs.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (neural_data, labels, trajectories, velocities)
            - neural_data: [1, T, N] - single site data
            - labels: [1] - classification label
            - trajectories: [T, 2] - cursor trajectory
            - velocities: [T, 2] - cursor velocity
        """
        train_data = self.split_data['train']
        
        # Neural data: [T, N] → [1, T, N] for single-site format
        neural_data = torch.tensor(train_data['neural_data'][idx], dtype=torch.float32).unsqueeze(0)
        
        # Other outputs
        label = torch.tensor(train_data['labels'][idx], dtype=torch.long)
        trajectory = torch.tensor(train_data['trajectories'][idx], dtype=torch.float32)
        velocity = torch.tensor(train_data['velocities'][idx], dtype=torch.float32)
        
        return neural_data, label, trajectory, velocity
    
    def print_summary(self):
        """Print a comprehensive dataset summary."""
        
        print(f"\n" + "="*60)
        print(f"📋 SingleSiteDownstreamDataset Summary")
        print(f"="*60)
        
        print(f"\n🏗️  Configuration:")
        print(f"   • Dataset ID: {self.dataset_id}")
        print(f"   • Data root: {self.data_root}")
        print(f"   • Split ratios: {self.split_ratios}")
        print(f"   • Target neurons: {self.target_neurons}")
        print(f"   • Sequence length: {self.sequence_length}")
        print(f"   • Neuron strategy: {self.neuron_selection_strategy}")
        print(f"   • Site coordinates: {self.site_coordinates.tolist()}")
        
        print(f"\n📊 Data Shapes:")
        for split, data in self.split_data.items():
            neural_shape = data['neural_data'].shape
            labels_shape = data['labels'].shape
            traj_shape = data['trajectories'].shape
            vel_shape = data['velocities'].shape
            print(f"   • {split.capitalize()}:")
            print(f"     - Neural: {neural_shape} → (B,1,T,N): ({neural_shape[0]},1,{neural_shape[1]},{neural_shape[2]})")
            print(f"     - Labels: {labels_shape}")
            print(f"     - Trajectories: {traj_shape}")
            print(f"     - Velocities: {vel_shape}")
        
        stats = self.data_stats
        print(f"\n🧠 Statistics:")
        print(f"   • Neural - Mean: {stats['neural_stats']['mean']:.4f}, Std: {stats['neural_stats']['std']:.4f}")
        print(f"   • Classes: {stats['label_stats']['num_classes']}")
        print(f"   • Class distribution: {stats['label_stats']['class_distribution']}")
        
        print(f"\n✅ Ready for downstream task training!")
        print(f"="*60)


def test_downstream_dataset():
    """Test the SingleSiteDownstreamDataset implementation."""
    
    print("🧪 Testing SingleSiteDownstreamDataset")
    print("=" * 50)
    
    try:
        # Initialize dataset
        dataset = SingleSiteDownstreamDataset(
            dataset_id='te14116',
            split_ratios=(0.8, 0.1, 0.1),
            target_neurons=50,
            neuron_selection_strategy='first_n',
            selected_neurons=50,
            random_seed=42
        )
        
        # Print summary
        dataset.print_summary()
        
        # Test dataloaders for different modes
        print(f"\n🔄 Testing dataloaders:")
        
        # Regression dataloader
        regression_loader = dataset.create_dataloader('train', batch_size=4, output_mode='regression')
        regression_batch = next(iter(regression_loader))
        print(f"   Regression batch: neural={regression_batch[0].shape}, traj={regression_batch[1].shape}, vel={regression_batch[2].shape}")
        
        # Classification dataloader
        classification_loader = dataset.create_dataloader('train', batch_size=4, output_mode='classification')
        classification_batch = next(iter(classification_loader))
        print(f"   Classification batch: neural={classification_batch[0].shape}, labels={classification_batch[1].shape}")
        
        # Test site coordinates
        coords_batch = dataset.get_site_coordinates_batch(batch_size=4)
        print(f"   Site coordinates batch: {coords_batch.shape}")
        
        print(f"\n✅ All tests passed!")
        return dataset
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests
    dataset = test_downstream_dataset() 