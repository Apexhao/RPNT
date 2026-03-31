"""
Neuropixel Dataset Handler for Neural Spike Data Analysis
-------------------------------------------------------
This module provides a comprehensive dataset handler for Neuropixel recordings from monkey prefrontal, premotor, and motor cortex.
It supports combining multiple recording sessions, data masking for MAE pre-training, and data augmentation for contrastive learning.

Key Features:
- Multi-session data combination
- Train/Valid/Test split handling
- Masking for MAE (Masked Autoencoder) pre-training
- Data augmentation for contrastive learning
- Visualization utilities
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union, Dict
import pandas as pd
# from ..utils.helpers import load_neuropixel_locations
import logging

def load_neuropixel_locations():
    """
    Load and organize neuropixel location data from CSV file.
    Returns a dictionary of meaningful entries (those with IDs).
    """
    # Read CSV file
    filename = "/data/Fang-analysis/causal-nfm/Data/Neuropixel_locations.xlsx"
    df = pd.read_excel(filename)
    
    # Print initial info
    print("Original DataFrame shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Filter out rows with empty IDs
    df_filtered = df.dropna(subset=['ID'])
    
    print(f"\nRows with valid IDs: {len(df_filtered)} (out of {len(df)} total rows)")
    
    # Create dictionary from filtered data
    locations_dict = {}
    for _, row in df_filtered.iterrows():
        locations_dict[row['ID']] = {
            'Site': row['Site'],
            'X': row['X'],
            'Y': row['Y'],
            'PCA': row['PCA'],
            'Neurons': row['Neurons']
        }
    
    print(f"\nTotal number of valid entries: {len(locations_dict)}")
    
    return locations_dict

class SineWaveDataset:
    def __init__(self, num_neurons: int = 100, time_steps: int = 50):
        """
        Initialize Sine Wave Dataset for meta-learning benchmark.
        
        Args:
            num_neurons: Number of neurons to match transformer input
            time_steps: Number of time steps to match transformer input
        """
        self.num_neurons = num_neurons
        self.time_steps = time_steps
        self.amplitude_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        
        # Store train/valid/test sessions for consistent evaluation
        self.train_sessions = np.random.randint(0, 1000, size=100)  # 100 training tasks
        self.valid_session = '1001'
        self.test_session = '1002'
        
    def generate_sine_wave(self, amplitude: float, phase: float, x: np.ndarray) -> np.ndarray:
        """Generate sine wave with given parameters."""
        return amplitude * np.sin(x + phase)
    
    def sample_task(self, session_id: int, k_shot: int, k_query: int) -> Dict[str, torch.Tensor]:
        """
        Sample a single task (support + query) with consistent parameters for a session.
        
        Args:
            session_id: Session identifier to maintain consistent parameters
            k_shot: Number of support points
            k_query: Number of query points
        """
        # Use session_id as seed for consistent parameters
        rng = np.random.RandomState(session_id)
        
        # Sample task parameters
        amplitude = rng.uniform(*self.amplitude_range)
        phase = rng.uniform(*self.phase_range)
        
        # Generate x points
        x_support = rng.uniform(-5, 5, (k_shot, 1))
        x_query = rng.uniform(-5, 5, (k_query, 1))
        
        # Generate y points
        y_support = self.generate_sine_wave(amplitude, phase, x_support)
        y_query = self.generate_sine_wave(amplitude, phase, x_query)
        
        # Create input features (expand x to match transformer input)
        support_spikes = np.tile(x_support, (1, self.time_steps, self.num_neurons))
        query_spikes = np.tile(x_query, (1, self.time_steps, self.num_neurons))
        
        # Create targets (expand y to match transformer output)
        support_target = np.tile(y_support, (1, self.time_steps, 1))
        query_target = np.tile(y_query, (1, self.time_steps, 1))
        
        return {
            'support_spikes': torch.tensor(support_spikes, dtype=torch.float32),
            'support_target': torch.tensor(support_target, dtype=torch.float32),
            'query_spikes': torch.tensor(query_spikes, dtype=torch.float32),
            'query_target': torch.tensor(query_target, dtype=torch.float32)
        }
    
    def create_meta_batch(self, split: str, k_shot: int, k_query: int, 
                         batch_size: int) -> Dict[str, torch.Tensor]:
        """Create a batch of tasks for meta-learning."""
        if split == 'train':
            # Sample tasks from different training sessions
            batch_sessions = np.random.choice(self.train_sessions, size=batch_size, replace=False)
        elif split == 'valid':
            # Use validation session
            batch_sessions = [int(self.valid_session)] * batch_size
        else:  # test
            # Use test session
            batch_sessions = [int(self.test_session)] * batch_size
            
        # Sample tasks from selected sessions
        tasks = [
            self.sample_task(session, k_shot, k_query)
            for session in batch_sessions
        ]
        
        # Collate tasks into a single batch
        return {
            'support_spikes': torch.stack([t['support_spikes'] for t in tasks]),
            'support_target': torch.stack([t['support_target'] for t in tasks]),
            'query_spikes': torch.stack([t['query_spikes'] for t in tasks]),
            'query_target': torch.stack([t['query_target'] for t in tasks])
        }
    
    def create_meta_dataloader(self, split: str, k_shot: int, k_query: int,
                             batch_size: int, target_type: str = 'sine', num_batches: int = 100) -> DataLoader:
        """Create a meta-learning dataloader."""
        class MetaBatchDataset(Dataset):
            def __init__(self, dataset, split, k_shot, k_query, batch_size, num_batches):
                self.dataset = dataset
                self.split = split
                self.k_shot = k_shot
                self.k_query = k_query
                self.batch_size = batch_size
                self.num_batches = num_batches
            
            def __len__(self):
                return self.num_batches
            
            def __getitem__(self, idx):
                return self.dataset.create_meta_batch(
                    self.split,
                    self.k_shot,
                    self.k_query,
                    self.batch_size
                )
        
        dataset = MetaBatchDataset(
            self, split, k_shot, k_query, batch_size, num_batches
        )
        
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=(split == 'train'),
            num_workers=0,
            collate_fn=lambda x: x[0]  # Unpack the batch
        )

class Monkey_beignet_Dataset_selected_width:
    """
    Base dataset class for individual monkey neural recording sessions.
    Handles loading, preprocessing, and splitting of single-session data.
    """
    
    def __init__(self, 
                 dataset_id: str = 'te14116', 
                 width: float = 0.02, 
                 filename: Optional[str] = None, 
                 smoothing: float = 0.0, 
                 target_neuron_dim: int = 100,
                 neuron_selection_strategy: str = 'all',
                 selected_neurons: int = 100,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 no_chunking_first_n: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            dataset_id: Identifier for the dataset
            width: Time bin width for spike data
            filename: Optional custom path to data file
            smoothing: Label smoothing factor
            target_neuron_dim: Target number of neurons per chunk
            neuron_selection_strategy: Strategy for neuron selection ('all', 'first_n', 'random_n')
            selected_neurons: Number of neurons to select (only used for 'first_n' and 'random_n')
            split_ratios: Tuple of (train, valid, test) split ratios
            no_chunking_first_n: If specified, skip chunking and use only first N neurons (overrides other options)
        """
        self.dataset_id = dataset_id
        self.width = width
        self.target_neuron_dim = target_neuron_dim
        self.neuron_selection_strategy = neuron_selection_strategy
        self.selected_neurons = selected_neurons
        self.split_ratios = split_ratios
        self.no_chunking_first_n = no_chunking_first_n
        
        # Validate split ratios
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        # Validate neuron selection strategy
        valid_strategies = ['all', 'first_n', 'random_n']
        if neuron_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid neuron selection strategy. Must be one of: {valid_strategies}")
        
        if filename is None:
            filename = f"/data/Fang-analysis/causal-nfm/Data/Monkey_data_meta/beignet_{dataset_id}_spike_data_{width:.2f}.pkl"
        
        # Load data
        with open(filename, 'rb') as f:
            spike_data, labels, cursor_traj, go_cue_times, center_target_on_times, length = pickle.load(f)
        
        self.data_types = ['data', 'target_direction', 'cursor_traj']
        self.split_types = ['train', 'valid', 'test']
        
        # Transform cursor trajectories
        self.cursor_traj = self.transform_cursor_trajectories(
            cursor_traj, 
            go_cue_times, 
            center_target_on_times, 
            length,
            target_length=int(length/width)  # Downsample to 50 timepoints
        )
        
        # Handle no_chunking_first_n mode
        if no_chunking_first_n is not None:
            print(f"Using no_chunking_first_n mode: selecting first {no_chunking_first_n} neurons")
            # Simply select first N neurons, no chunking or padding
            original_neurons = spike_data.shape[2]
            if no_chunking_first_n > original_neurons:
                raise ValueError(f"Cannot select {no_chunking_first_n} neurons from {original_neurons} available")
            
            self.spike_data = spike_data[:, :, :no_chunking_first_n].astype(np.float32)
            self.labels = self._to_one_hot(labels.astype(np.int64), smoothing=smoothing)
            self.cursor_traj = self.cursor_traj.astype(np.float32)
            
            print(f"Dataset {dataset_id}: Selected first {no_chunking_first_n} neurons from {original_neurons} available")
            print(f"Final data shape: {self.spike_data.shape}")
            
        else:
            # Original chunking logic
            # Apply neuron selection strategy before processing dimension
            spike_data = self.apply_neuron_selection(spike_data)
            
            # Process neuron dimension by chunking or padding
            spike_chunks = self.process_neuron_dimension(spike_data, target_neuron_dim)
            
            # Expand all data to match the number of chunks
            expanded_spike_data = []
            expanded_labels = []
            expanded_cursor_traj = []
            
            for chunk in spike_chunks:
                expanded_spike_data.append(chunk)
                expanded_labels.append(labels)  # Replicate labels for each chunk
                expanded_cursor_traj.append(self.cursor_traj)  # Replicate cursor trajectory for each chunk
            
            # Concatenate all chunks
            self.spike_data = np.concatenate(expanded_spike_data, axis=0).astype(np.float32)
            self.labels = self._to_one_hot(np.concatenate(expanded_labels, axis=0).astype(np.int64), smoothing=smoothing)
            self.cursor_traj = np.concatenate(expanded_cursor_traj, axis=0).astype(np.float32)
            
            # Log chunking information
            print(f"Dataset {dataset_id}: Original neurons={spike_data.shape[2]}, "
                  f"Selected neurons strategy={self.neuron_selection_strategy}, "
                  f"Chunks created={len(spike_chunks)}, "
                  f"Final data shape={self.spike_data.shape}")
        
        # Split the data
        self.split_data()
        
        # Calculate cursor velocities for each split
        self._calculate_cursor_velocities()
    
    def apply_neuron_selection(self, data):
        """
        Apply neuron selection strategy to reduce the number of neurons.
        
        Args:
            data: Input data with shape [trials, time, neurons]
            
        Returns:
            selected_data: Data with selected neurons [trials, time, selected_neurons]
        """
        trials, time, original_neurons = data.shape
        
        if self.neuron_selection_strategy == 'all':
            # Use all neurons (no selection)
            return data
        
        elif self.neuron_selection_strategy == 'first_n':
            # Select first N neurons
            selected_neurons = min(self.selected_neurons, original_neurons)
            selected_data = data[:, :, :selected_neurons]
            print(f"Selected first {selected_neurons} neurons out of {original_neurons}")
            return selected_data
        
        elif self.neuron_selection_strategy == 'random_n':
            # Randomly select N neurons
            selected_neurons = min(self.selected_neurons, original_neurons)
            # Set seed for reproducible random selection
            np.random.seed(42)
            selected_indices = np.random.choice(original_neurons, size=selected_neurons, replace=False)
            selected_indices = np.sort(selected_indices)  # Keep indices sorted for consistency
            selected_data = data[:, :, selected_indices]
            print(f"Randomly selected {selected_neurons} neurons out of {original_neurons}")
            print(f"Selected neuron indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
            return selected_data
        
        else:
            raise ValueError(f"Unknown neuron selection strategy: {self.neuron_selection_strategy}")
    
    def process_neuron_dimension(self, data, target_neuron_dim=100):
        """
        Process neuron dimension by chunking or padding to achieve consistent target dimension.
        
        Args:
            data: Input data with shape [trials, time, neurons]
            target_neuron_dim: Target number of neurons (default: 100)
            
        Returns:
            processed_data: List of data chunks, each with shape [trials, time, target_neuron_dim]
        """
        trials, time, neurons = data.shape
        chunks = []
        
        if neurons <= target_neuron_dim:
            # Pad with zeros if fewer neurons than target
            padded_data = np.zeros((trials, time, target_neuron_dim), dtype=data.dtype)
            padded_data[:, :, :neurons] = data
            chunks.append(padded_data)
        else:
            # Split into multiple chunks if more neurons than target
            num_full_chunks = neurons // target_neuron_dim
            remaining_neurons = neurons % target_neuron_dim
            
            # Process full chunks
            for i in range(num_full_chunks):
                start_idx = i * target_neuron_dim
                end_idx = start_idx + target_neuron_dim
                chunk = data[:, :, start_idx:end_idx]
                chunks.append(chunk)
            
            # Process remaining neurons with padding
            if remaining_neurons > 0:
                padded_chunk = np.zeros((trials, time, target_neuron_dim), dtype=data.dtype)
                start_idx = num_full_chunks * target_neuron_dim
                padded_chunk[:, :, :remaining_neurons] = data[:, :, start_idx:]
                chunks.append(padded_chunk)
        
        return chunks

    def _to_one_hot(self, labels: np.ndarray, num_classes: int = 8, smoothing: float = 0.0) -> np.ndarray:
        """
        Convert integer labels to one-hot vectors with optional smoothing.
        
        Args:
            labels: Integer labels (shape: [batch_size])
            num_classes: Number of classes
            smoothing: Label smoothing factor (0.0 means no smoothing)
        """
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        
        if smoothing > 0:
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes
            
        return one_hot

    def transform_cursor_trajectories(self, cursor_traj, go_cue_times, 
                                   center_target_on_times, length, target_length):
        """Transform cursor trajectories to align with movement period and downsample."""
        transformed_trajectories = []
        sampling_rate = 1000  # Original sampling rate in Hz
        
        for traj, go_time, center_time in zip(cursor_traj, go_cue_times, center_target_on_times):
            # Convert times to indices
            start_idx = int((go_time - center_time) * sampling_rate)
            end_idx = start_idx + int(length * sampling_rate)

            # Extract movement period
            if start_idx < 0:
                # Pad with initial position if start_idx is negative
                print(f"start_idx is negative: {start_idx}")
                pad_length = abs(start_idx)
                movement_traj = np.vstack([np.tile(traj[0], (pad_length, 1)),traj[:end_idx]])
            else:
                movement_traj = traj[start_idx:end_idx]

            # movement_traj = traj[start_idx:end_idx]
            
            # Ensure consistent length
            if len(movement_traj) > int(length * sampling_rate):
                movement_traj = movement_traj[:int(length * sampling_rate)]
            elif len(movement_traj) < int(length * sampling_rate):
                # Pad with final position if trajectory is too short
                pad_length = int(length * sampling_rate) - len(movement_traj)

                #print(f"pad_length: {pad_length}")

                if len(movement_traj) > 0:
                    movement_traj = np.vstack([movement_traj,np.tile(movement_traj[-1], (pad_length, 1))])
                else:
                    movement_traj = np.tile(traj[0], (int(length * sampling_rate), 1))
            
            # Downsample to target length
            indices = np.linspace(0, len(movement_traj)-1, target_length, dtype=int)
            downsampled_traj = movement_traj[indices]
            
            transformed_trajectories.append(downsampled_traj)
        
        return np.array(transformed_trajectories)
    
    def _calculate_cursor_velocities(self):
        """Calculate cursor velocities for all splits."""
        for split in self.split_types:
            traj = getattr(self, f"{split}_cursor_traj")
            vel = np.zeros_like(traj)
            vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2 * self.width)
            vel[:, 0] = vel[:, 1]  # Copy first valid velocity
            vel[:, -1] = vel[:, -2]  # Copy last valid velocity
            setattr(self, f"{split}_cursor_vel", vel)

    def split_data(self):
        """Split data into train, validation, and test sets using configurable ratios."""
        total_samples = len(self.spike_data)
        train_ratio, valid_ratio, test_ratio = self.split_ratios
        
        train_size = int(train_ratio * total_samples)
        valid_size = int(valid_ratio * total_samples)
        # Test size is the remainder to ensure we use all samples
        
        # Set seed for reproducible splits
        np.random.seed(42)
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]
        
        print(f"Data split with ratios {self.split_ratios}: "
              f"Train={len(train_indices)}, Valid={len(valid_indices)}, Test={len(test_indices)}")
        
        for split, idx in zip(self.split_types, [train_indices, valid_indices, test_indices]):
            setattr(self, f"{split}_data", self.spike_data[idx])
            setattr(self, f"{split}_target_direction", self.labels[idx])
            setattr(self, f"{split}_cursor_traj", self.cursor_traj[idx])

    def create_dataset(self, split: str = 'train', 
                      batch_size: int = 32, 
                      shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader for the specified split.
        
        Args:
            split: Data split to use ('train', 'valid', or 'test')
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
        
        Returns:
            DataLoader: PyTorch DataLoader
        """
        if split not in self.split_types:
            raise ValueError(f"Invalid split type. Expected one of: {self.split_types}")
        
        data = getattr(self, f"{split}_data")
        labels = getattr(self, f"{split}_target_direction")
        cursor_traj = getattr(self, f"{split}_cursor_traj")
        cursor_vel = getattr(self, f"{split}_cursor_vel")
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(cursor_traj, dtype=torch.float32),
            torch.tensor(cursor_vel, dtype=torch.float32)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4
        )

    def get_data(self, split: str = 'train', data_type: str = 'data') -> np.ndarray:
        """Get a specific type of data for a given split."""
        if split not in self.split_types:
            raise ValueError(f"Invalid split type. Expected one of: {self.split_types}")
        if data_type not in self.data_types:
            raise ValueError(f"Invalid data type. Expected one of: {self.data_types}")
        
        return getattr(self, f"{split}_{data_type}")

    def get_batch_data(self, split: str = 'train', 
                      batch_size: int = 32, 
                      batch_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a specific batch of data for a given split."""
        if split not in self.split_types:
            raise ValueError(f"Invalid split type. Expected one of: {self.split_types}")

        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size

        data = getattr(self, f"{split}_data")[start_idx:end_idx]
        target_direction = getattr(self, f"{split}_target_direction")[start_idx:end_idx]
        cursor_traj = getattr(self, f"{split}_cursor_traj")[start_idx:end_idx]
        cursor_vel = getattr(self, f"{split}_cursor_vel")[start_idx:end_idx]

        return data, target_direction, cursor_traj, cursor_vel

class Combined_Monkey_Dataset(Monkey_beignet_Dataset_selected_width):
    """
    A dataset class that combines multiple Neuropixel recording sessions from monkey prefrontal, premotor, and motor cortex.
    
    This class handles:
    1. Loading and combining multiple recording sessions
    2. Managing train/valid/test splits
    3. Creating masked data for MAE pre-training
    4. Generating augmented pairs for contrastive learning
    
    Attributes:
        width (float): Time bin width for spike data
        data_types (List[str]): Types of data available ('data', 'target_direction', 'cursor_traj','cursor_vel')
        split_types (List[str]): Available data splits ('train', 'valid', 'test')
        neuropixel_locations (Dict): Mapping of recording locations
        individual_test_datasets (Dict): Individual test datasets for analysis
    """

    def __init__(self, 
                 exclude_ids: List[str] = ['13122.0'], 
                 width: float = 0.02, 
                 smoothing: float = 0.0, 
                 target_neuron_dim: int = 100,
                 neuron_selection_strategy: str = 'all',
                 selected_neurons: int = 100,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 no_chunking_first_n: Optional[int] = None):
        """
        Initialize the combined dataset.
        
        Args:
            exclude_ids: List of dataset IDs to exclude from combination
            width: Time bin width for spike data processing
            smoothing: Label smoothing factor (0.0 means no smoothing)
            target_neuron_dim: Target number of neurons per chunk
            neuron_selection_strategy: Strategy for neuron selection ('all', 'first_n', 'random_n')
            selected_neurons: Number of neurons to select (only used for 'first_n' and 'random_n')
            split_ratios: Tuple of (train, valid, test) split ratios
            no_chunking_first_n: If specified, skip chunking and use only first N neurons for all datasets
        """
        self.width = width
        self.target_neuron_dim = target_neuron_dim
        self.neuron_selection_strategy = neuron_selection_strategy
        self.selected_neurons = selected_neurons
        self.split_ratios = split_ratios
        self.no_chunking_first_n = no_chunking_first_n
        self.data_types = ['data', 'target_direction', 'cursor_traj','cursor_vel']
        self.split_types = ['train', 'valid', 'test']
        
        # Load neuropixel locations
        self.neuropixel_locations = load_neuropixel_locations()
        
        # Dictionary to store individual test datasets for analysis
        self.individual_test_datasets = {}
        
        # Combine datasets
        self.combine_datasets(exclude_ids, smoothing=smoothing)

        # Automatically compute statistics
        self.compute_data_statistics()

    def combine_datasets(self, exclude_ids: List[str], smoothing: float = 0.0):
        """
        Combine multiple datasets for both training and evaluation.
        
        Args:
            exclude_ids: List of dataset IDs to exclude
            smoothing: Label smoothing factor
        """
        # Lists for combined data
        train_spike_data, train_labels, train_cursor_traj, train_cursor_vel = [], [], [], []
        valid_spike_data, valid_labels, valid_cursor_traj, valid_cursor_vel = [], [], [], []
        test_spike_data, test_labels, test_cursor_traj, test_cursor_vel = [], [], [], []
        
        count = 0
        # Iterate through all available datasets
        for key, value in self.neuropixel_locations.items():
            if str(key) in exclude_ids:
                continue

            count += 1
            dataset_id = 'te' + str(int(key))
            print(f"Loading dataset: {dataset_id}")
            
            # Load individual dataset with target neuron dimension
            temp_dataset = Monkey_beignet_Dataset_selected_width(
                dataset_id=dataset_id, 
                width=self.width, 
                smoothing=smoothing,
                target_neuron_dim=self.target_neuron_dim,
                neuron_selection_strategy=self.neuron_selection_strategy,
                selected_neurons=self.selected_neurons,
                split_ratios=self.split_ratios,
                no_chunking_first_n=self.no_chunking_first_n
            )
            
            # Add to combined training data
            train_spike_data.append(temp_dataset.train_data)
            train_labels.append(temp_dataset.train_target_direction)
            train_cursor_traj.append(temp_dataset.train_cursor_traj)
            train_cursor_vel.append(temp_dataset.train_cursor_vel)
            
            # Add to combined validation data
            valid_spike_data.append(temp_dataset.valid_data)
            valid_labels.append(temp_dataset.valid_target_direction)
            valid_cursor_traj.append(temp_dataset.valid_cursor_traj)
            valid_cursor_vel.append(temp_dataset.valid_cursor_vel)
            
            # Add to combined test data
            test_spike_data.append(temp_dataset.test_data)
            test_labels.append(temp_dataset.test_target_direction)
            test_cursor_traj.append(temp_dataset.test_cursor_traj)
            test_cursor_vel.append(temp_dataset.test_cursor_vel)
            
            # Store individual test data for analysis
            self.individual_test_datasets[dataset_id] = {
                'spike_data': temp_dataset.test_data,
                'labels': temp_dataset.test_target_direction,
                'cursor_traj': temp_dataset.test_cursor_traj,
                'cursor_vel': temp_dataset.test_cursor_vel
            }

            print(f"Completed loading dataset: {dataset_id}")
        
        self.num_datasets = count
        print(f"Number of datasets loaded: {count}")
        print(f"Completed loading datasets")

        # Combine training data
        self.train_data = np.concatenate(train_spike_data, axis=0).astype(np.float32)
        self.train_target_direction = np.concatenate(train_labels, axis=0).astype(np.float32)  # Already one-hot
        self.train_cursor_traj = np.concatenate(train_cursor_traj, axis=0).astype(np.float32)
        self.train_cursor_vel = np.concatenate(train_cursor_vel, axis=0).astype(np.float32)
        
        # Combine validation data
        self.valid_data = np.concatenate(valid_spike_data, axis=0).astype(np.float32)
        self.valid_target_direction = np.concatenate(valid_labels, axis=0).astype(np.float32)  # Already one-hot
        self.valid_cursor_traj = np.concatenate(valid_cursor_traj, axis=0).astype(np.float32)
        self.valid_cursor_vel = np.concatenate(valid_cursor_vel, axis=0).astype(np.float32)
        
        # Combine test data
        self.test_data = np.concatenate(test_spike_data, axis=0).astype(np.float32)
        self.test_target_direction = np.concatenate(test_labels, axis=0).astype(np.float32)  # Already one-hot
        self.test_cursor_traj = np.concatenate(test_cursor_traj, axis=0).astype(np.float32)
        self.test_cursor_vel = np.concatenate(test_cursor_vel, axis=0).astype(np.float32)
        
        print(f"Completed combining datasets")
        print(f"Final combined data shapes:")
        print(f"  Train: {self.train_data.shape}")
        print(f"  Valid: {self.valid_data.shape}")
        print(f"  Test: {self.test_data.shape}")
        print(f"  Target neuron dimension: {self.target_neuron_dim}")
        
        # Store individual test data for analysis
        self.individual_test_datasets[dataset_id] = {
            'spike_data': temp_dataset.test_data,
            'labels': temp_dataset.test_target_direction,  # Already one-hot
            'cursor_traj': temp_dataset.test_cursor_traj,
            'cursor_vel': temp_dataset.test_cursor_vel  # Get velocity directly from base class
        }

        print(f"Completed storing individual test data")    
        
        # Add shape validation after combination
        def validate_shapes():
            expected_time_steps = self.train_data.shape[1]
            expected_neurons = self.train_data.shape[2]
            
            assert self.valid_data.shape[1:] == (expected_time_steps, expected_neurons), \
                "Validation data shape mismatch"
            assert self.test_data.shape[1:] == (expected_time_steps, expected_neurons), \
                "Test data shape mismatch"
            assert expected_neurons == self.target_neuron_dim, \
                f"Neuron dimension mismatch: expected {self.target_neuron_dim}, got {expected_neurons}"
            
            print(f"Dataset shapes validated successfully:")
            print(f"Time steps: {expected_time_steps}")
            print(f"Number of neurons: {expected_neurons}")
            print(f"Number of classes: {self.train_target_direction.shape[1]}")
        
        validate_shapes()

    def create_dataset(self, 
                      split: str = 'train', 
                      batch_size: int = 32, 
                      shuffle: bool = True,
                      label_smoothing: float = 0.0,
                      normalize: bool = True) -> DataLoader:
        """Create PyTorch DataLoader with optional label smoothing and normalization."""
        if split not in self.split_types:
            raise ValueError(f"Invalid split type. Expected one of: {self.split_types}")
        
        data = getattr(self, f"{split}_data")
        labels = getattr(self, f"{split}_target_direction")
        cursor_traj = getattr(self, f"{split}_cursor_traj")
        cursor_vel = getattr(self, f"{split}_cursor_vel")
        
        if label_smoothing > 0:
            # Re-smooth the labels if needed
            labels = self._to_one_hot(np.argmax(labels, axis=1), 
                                    num_classes=self.num_classes, 
                                    smoothing=label_smoothing)
        
        if normalize:
            # Comment: Normalization is not applied to the spike data
            #data = (data - self.data_stats['spike_mean']) / self.data_stats['spike_std'] 

            # Apply normalization
            cursor_traj = (cursor_traj - self.data_stats['cursor_traj_mean']) / self.data_stats['cursor_traj_std']
            cursor_vel = (cursor_vel - self.data_stats['cursor_vel_mean']) / self.data_stats['cursor_vel_std']
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(cursor_traj, dtype=torch.float32),
            torch.tensor(cursor_vel, dtype=torch.float32)
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=7
        )

    def get_test_dataset_by_id(self, dataset_id: str, batch_size: int = 32) -> DataLoader:
        """
        Get test dataset for a specific dataset_id for analysis.
        
        Args:
            dataset_id: Identifier for the specific test dataset
            batch_size: Number of samples per batch
        
        Returns:
            DataLoader: PyTorch DataLoader containing:
                - Neural spike data: shape [batch, time_steps, n_neurons]
                - Target directions: shape [batch, num_classes] (one-hot)
                - Cursor trajectories: shape [batch, time_steps, 2]
                - Cursor velocities: shape [batch, time_steps, 2]
        
        Raises:
            ValueError: If dataset_id is not found
        """
        if dataset_id not in self.individual_test_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        test_data = self.individual_test_datasets[dataset_id]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_data['spike_data'], dtype=torch.float32),
            torch.tensor(test_data['labels'], dtype=torch.float32),
            torch.tensor(test_data['cursor_traj'], dtype=torch.float32),
            torch.tensor(test_data['cursor_vel'], dtype=torch.float32)
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test data
            pin_memory=True,
            num_workers=7
        )

    def create_masked_data(self, 
                          spike_data: torch.Tensor,
                          ratio_neurons: float = 0.2, 
                          ratio_time: float = 0.2,
                          future_mask_time: Optional[int] = 45) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked version of input spike data with multiple masking strategies:
        1. Random masking for individual time-neuron pairs
        2. Complete neuron masking (mask all times for selected neurons)
        3. Future masking (mask all data after a certain time point)
        
        Args:
            spike_data: Input spike data [batch, time, neurons]
            ratio_neurons: Proportion of neurons to mask completely (0-1)
            ratio_time: Proportion of time-neuron pairs to mask randomly (0-1)
            future_mask_time: Time step after which all future data is masked
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - masked_data: Masked data with same shape as input
                - mask: Binary mask (1 = keep, 0 = mask)
        """
        B, T, N = spike_data.shape
        
        # Initialize mask (1 = keep, 0 = mask)
        mask = torch.ones_like(spike_data, dtype=torch.bool)
        
        # 1. Random masking for individual time-neuron pairs
        random_mask = torch.rand(B, T, N) > ratio_time
        mask = mask & random_mask
        
        # 2. Complete neuron masking
        # Select neurons to be completely masked
        neurons_to_mask = torch.rand(B, 1, N) < ratio_neurons
        neurons_to_mask = neurons_to_mask.expand(-1, T, -1)  # Expand to match time dimension
        mask = mask & ~neurons_to_mask  # Mask selected neurons for all time points
        
        # 3. Future masking
        if future_mask_time is not None:
            future_mask = torch.ones_like(spike_data, dtype=torch.bool)
            future_mask[:, future_mask_time:, :] = False
            mask = mask & future_mask
        
        # Apply combined mask to data
        masked_data = spike_data * mask
        
        return masked_data, mask

    def create_augmented_pairs(self,
                             spike_data: torch.Tensor,
                             jitter_std: float = 0.2,
                             dropout_rate: float = 0.2,
                             time_shift_max: int = 5,
                             scale_range: Tuple[float, float] = (0.6, 1.4)) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of input spike data for contrastive learning.
        
        Args:
            spike_data: Input spike data [batch, time, neurons]
            jitter_std: Standard deviation for temporal jitter
            dropout_rate: Probability of dropping spikes
            time_shift_max: Maximum temporal shift
            scale_range: Range for random scaling (min, max)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two augmented views of the input
        """
        def augment_single(data):
            augmented = data.clone()
            
            # 1. Add temporal jitter (Gaussian noise)
            noise = torch.randn_like(augmented) * jitter_std
            augmented += noise
            
            # 2. Random temporal shift (circular)
            shift = torch.randint(-time_shift_max, time_shift_max + 1, (1,))
            augmented = torch.roll(augmented, shifts=shift.item(), dims=1)
            
            # 3. Random scaling
            scale = torch.empty(1).uniform_(*scale_range)
            augmented *= scale
            
            # 4. Spike dropout
            dropout_mask = (torch.rand_like(augmented) > dropout_rate)
            augmented *= dropout_mask
            
            # 5. Ensure non-negativity
            augmented = torch.clamp(augmented, min=0)
            
            return augmented
        
        # Create two augmented views
        augmented_1 = augment_single(spike_data)
        augmented_2 = augment_single(spike_data)
        
        return augmented_1, augmented_2

    def visualize_masking(self, batch_idx: int = 0, num_samples: int = 4):
        """
        Visualize the masking pattern for a batch of samples.
        
        Args:
            batch_idx: Index of the batch to visualize
            num_samples: Number of samples to visualize from the batch
        """
        # Get a batch of data
        data_loader = self.create_dataset('train', batch_size=num_samples)
        spikes, _, _, _ = next(iter(data_loader))
        
        # Create masked data
        masked_data, mask = self.create_masked_data(
            spikes,
            ratio_neurons=0.5,
            ratio_time=0.5
        )
        
        # Create subplot for each sample
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        for i in range(num_samples):
            # Plot original data
            im1 = axes[i, 0].imshow(spikes[i].numpy(), aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Sample {i}: Original Data')
            axes[i, 0].set_xlabel('Neurons')
            axes[i, 0].set_ylabel('Time Steps')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot mask pattern
            im2 = axes[i, 1].imshow(mask[i].numpy(), aspect='auto', cmap='binary')
            axes[i, 1].set_title(f'Sample {i}: Mask Pattern\n(Black: masked, White: kept)')
            axes[i, 1].set_xlabel('Neurons')
            axes[i, 1].set_ylabel('Time Steps')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot masked data
            im3 = axes[i, 2].imshow(masked_data[i].numpy(), aspect='auto', cmap='viridis')
            axes[i, 2].set_title(f'Sample {i}: Masked Data')
            axes[i, 2].set_xlabel('Neurons')
            axes[i, 2].set_ylabel('Time Steps')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.show()
        plt.savefig('masking.png')
        
        # Print masking statistics for each sample
        for i in range(num_samples):
            total_elements = mask[i].numel()
            masked_elements = (~mask[i]).sum().item()
            print(f"\nSample {i} Masking Statistics:")
            print(f"Total elements: {total_elements}")
            print(f"Masked elements: {masked_elements}")
            print(f"Masking percentage: {(masked_elements/total_elements)*100:.2f}%")

    def visualize_augmentations(self, batch_idx: int = 0, num_samples: int = 4):
        """
        Visualize different augmentations of the same neural data.
        
        Args:
            batch_idx: Index of the batch to visualize
            num_samples: Number of samples to visualize
        """
        # Get a batch of data
        data_loader = self.create_dataset('train', batch_size=num_samples)
        spikes, _, _, _ = next(iter(data_loader))
        
        # Create augmented pairs
        augmented_1, augmented_2 = self.create_augmented_pairs(spikes)
        
        # Create subplot for each sample
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        for i in range(num_samples):
            # Plot original data
            im1 = axes[i, 0].imshow(spikes[i].numpy(), aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Sample {i}: Original Data')
            axes[i, 0].set_xlabel('Neurons')
            axes[i, 0].set_ylabel('Time Steps')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot first augmentation
            im2 = axes[i, 1].imshow(augmented_1[i].numpy(), aspect='auto', cmap='viridis')
            axes[i, 1].set_title(f'Sample {i}: Augmentation 1')
            axes[i, 1].set_xlabel('Neurons')
            axes[i, 1].set_ylabel('Time Steps')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot second augmentation
            im3 = axes[i, 2].imshow(augmented_2[i].numpy(), aspect='auto', cmap='viridis')
            axes[i, 2].set_title(f'Sample {i}: Augmentation 2')
            axes[i, 2].set_xlabel('Neurons')
            axes[i, 2].set_ylabel('Time Steps')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.show()

    def compute_data_statistics(self):
        """Compute and store dataset statistics."""
        self.data_stats = {
            'spike_max': float(np.max(self.train_data)),
            'spike_min': float(np.min(self.train_data)),
            'spike_mean': float(np.mean(self.train_data)),
            'spike_std': float(np.std(self.train_data)),
            'cursor_traj_mean': np.mean(self.train_cursor_traj, axis=(0, 1)),
            'cursor_traj_std': np.std(self.train_cursor_traj, axis=(0, 1)),
            'cursor_vel_mean': np.mean(self.train_cursor_vel, axis=(0, 1)),
            'cursor_vel_std': np.std(self.train_cursor_vel, axis=(0, 1))
        }
        
        # Class distribution
        class_dist = np.mean(self.train_target_direction, axis=0)
        self.data_stats['class_distribution'] = class_dist
        
        return self.data_stats

    @property
    def num_classes(self) -> int:
        """Number of target direction classes."""
        return self.train_target_direction.shape[1]
    
    @property
    def num_neurons(self) -> int:
        """Number of neurons in recordings (after chunking/padding)."""
        return self.train_data.shape[2]
    
    @property
    def time_steps(self) -> int:
        """Number of time steps in each trial."""
        return self.train_data.shape[1]
    
    @property 
    def target_neuron_dimension(self) -> int:
        """Target neuron dimension used for chunking/padding."""
        return self.target_neuron_dim
    
    ## This code with two functions are not correct for sampling the tasks, i.e., it samples directly from the combined dataset instead of each session.
    # def sample_meta_batch(self,
    #                  data_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #                  k_shot: int = 5,
    #                  k_query: int = 15,
    #                  batch_size: int = 16,
    #                  target_type: str = 'velocity'
    #                  ) -> Dict[str, torch.Tensor]:
    #     """
    #     Reshape a single large batch into support/query sets.
        
    #     Args:
    #         data_batch: Tuple of (spikes, labels, cursor_traj, cursor_vel)
    #         k_shot: Number of support samples per task
    #         k_query: Number of query samples per task
    #         batch_size: Number of tasks
    #         target_type: Type of target variable
    #     """
    #     data, _, cursor_traj, cursor_vel = data_batch
    #     samples_per_task = k_shot + k_query
        
    #     # Reshape data into [batch_size, samples_per_task, ...]
    #     data = data.view(batch_size, samples_per_task, *data.shape[1:])
    #     cursor_traj = cursor_traj.view(batch_size, samples_per_task, *cursor_traj.shape[1:])
    #     cursor_vel = cursor_vel.view(batch_size, samples_per_task, *cursor_vel.shape[1:])
        
    #     # Split into support and query
    #     support_spikes = data[:, :k_shot]
    #     query_spikes = data[:, k_shot:]
        
    #     if target_type == 'velocity':
    #         support_target = cursor_vel[:, :k_shot]
    #         query_target = cursor_vel[:, k_shot:]
    #     else:  # position
    #         support_target = cursor_traj[:, :k_shot]
    #         query_target = cursor_traj[:, k_shot:]
        
    #     return {
    #         'support_spikes': support_spikes,  # [batch_size, k_shot, time, neurons]
    #         'support_target': support_target,  # [batch_size, k_shot, time, 2]
    #         'query_spikes': query_spikes,    # [batch_size, k_query, time, neurons]
    #         'query_target': query_target     # [batch_size, k_query, time, 2]
    #     }
    
    # def create_meta_dataloader(self,
    #                         split: str = 'train',
    #                         k_shot: int = 5,
    #                         k_query: int = 15,
    #                         batch_size: int = 16,
    #                         target_type: str = 'velocity',
    #                         num_batches: int = 100
    #                         ) -> DataLoader:
    #     """
    #     Create a DataLoader for meta-learning with efficient batching.
    #     """
    #     # Calculate total samples needed per batch
    #     samples_per_batch = (k_shot + k_query) * batch_size
        
    #     # Create base dataloader with larger batch size
    #     base_loader = self.create_dataset(
    #         split=split,
    #         batch_size=samples_per_batch,
    #         shuffle=(split == 'train'),
    #     )
        
    #     # Pre-sample and process all batches
    #     meta_batches = []
    #     iterator = iter(base_loader)
        
    #     for _ in range(num_batches):
    #         try:
    #             data_batch = next(iterator)
    #         except StopIteration:
    #             iterator = iter(base_loader)
    #             data_batch = next(iterator)
                
    #         meta_batch = self.sample_meta_batch(
    #             data_batch,
    #             k_shot=k_shot,
    #             k_query=k_query,
    #             batch_size=batch_size,
    #             target_type=target_type
    #         )
    #         meta_batches.append(meta_batch)
        
    #     # Create dataset from pre-processed batches
    #     meta_dataset = torch.utils.data.TensorDataset(
    #         torch.arange(len(meta_batches))  # Simplified index tensor
    #     )
        
    #     # Define collate function properly
    #     def collate_fn(indices):
    #         # indices is a list of tuples, we need to extract the tensor
    #         idx = indices[0][0]  # Get the first element of the first tuple
    #         return meta_batches[idx]  # Return the corresponding meta batch
        
    #     return DataLoader(
    #         meta_dataset,
    #         batch_size=1,
    #         shuffle=(split == 'train'),
    #         num_workers=0,
    #         pin_memory=True,
    #         collate_fn=collate_fn  # Use the properly defined collate function
    #     )

class Session_MAML_Monkey_Dataset(Combined_Monkey_Dataset):
    def __init__(self, 
                 exclude_ids: List[str] = ['13122.0'], 
                 valid_id: List[str] = ['14878'], 
                 test_id: List[str] = ['14116'], 
                 width: float = 0.02,
                 target_neuron_dim: int = 100,
                 neuron_selection_strategy: str = 'all',
                 selected_neurons: int = 100,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Initialize the session-aware meta-learning dataset.
        
        Args:
            exclude_ids: List of session IDs to exclude from training
            valid_id: Session ID for validation
            test_id: Session ID for testing
            width: Time bin width for spike data
            target_neuron_dim: Target number of neurons per chunk
            neuron_selection_strategy: Strategy for neuron selection ('all', 'first_n', 'random_n')
            selected_neurons: Number of neurons to select (only used for 'first_n' and 'random_n')
            split_ratios: Tuple of (train, valid, test) split ratios
        """
        # Initialize parent class
        super().__init__(
            exclude_ids=exclude_ids, 
            width=width, 
            target_neuron_dim=target_neuron_dim,
            neuron_selection_strategy=neuron_selection_strategy,
            selected_neurons=selected_neurons,
            split_ratios=split_ratios
        )

        # Store configuration
        self.exclude_ids = exclude_ids

        self.valid_id = f'te{valid_id}'
        self.test_id = f'te{test_id}'
        
        # Organize data by sessions
        self.session_data = {}
        self._organize_sessions()
        
        # Split sessions for meta-learning
        self.train_sessions = []
        self.valid_session = self.valid_id
        self.test_session = self.test_id
        self._split_sessions()

    def _organize_sessions(self):
        """
        Organize all data by sessions, using complete session data instead of just test data.
        """
        for key, value in self.neuropixel_locations.items():

            if str(key) in self.exclude_ids:
                continue

            session_id = f'te{str(int(key))}'
            
            # Load the complete dataset for this session with target neuron dimension
            temp_dataset = Monkey_beignet_Dataset_selected_width(
                dataset_id=session_id, 
                width=self.width,
                target_neuron_dim=self.target_neuron_dim,
                neuron_selection_strategy=self.neuron_selection_strategy,
                selected_neurons=self.selected_neurons,
                split_ratios=self.split_ratios
            )

            # Combine all data for this session (no train/valid/test split)
            session_spikes = np.concatenate([
                temp_dataset.train_data,
                temp_dataset.valid_data,
                temp_dataset.test_data
            ])
            
            session_velocities = np.concatenate([
                temp_dataset.train_cursor_vel,
                temp_dataset.valid_cursor_vel,
                temp_dataset.test_cursor_vel
            ])
            
            session_positions = np.concatenate([
                temp_dataset.train_cursor_traj,
                temp_dataset.valid_cursor_traj,
                temp_dataset.test_cursor_traj
            ])
            
            # Store complete session data
            self.session_data[session_id] = {
                'spikes': session_spikes,
                'velocities': session_velocities,
                'positions': session_positions
            }

    def _split_sessions(self):
        """
        Split sessions into train/valid/test, considering excluded sessions.
        """
        for session_id in self.session_data.keys():
            if session_id == self.valid_id:
                continue
            elif session_id == self.test_id:
                continue
            else:
                self.train_sessions.append(session_id)

    def sample_task(self, session_id: str, k_shot: int, k_query: int, target_type: str = 'velocity') -> Dict[str, torch.Tensor]:
        """
        Sample a single task (support + query) from a specific session.
        Allow replacement to handle small session sizes.
        
        Args:
            session_id: The session to sample from
            k_shot: Number of support samples
            k_query: Number of query samples
            target_type: 'velocity' or 'position'
            
        Returns:
            Dict containing support and query sets
        """
        session = self.session_data[session_id]
        total_samples = len(session['spikes'])
        
        # Sample with replacement to handle small session sizes
        indices = np.random.choice(
            total_samples, 
            size=k_shot + k_query, 
            replace=True  # Changed to True to allow resampling
        )
        
        # Split into support and query indices
        support_indices = indices[:k_shot]
        query_indices = indices[k_shot:]


        # query_indices = indices[:k_query]
        # support_indices = indices[k_query:]
        
        # Get target data based on type
        target_data = session['velocities'] if target_type == 'velocity' else session['positions']
        
        return {
            'support_spikes': torch.tensor(session['spikes'][support_indices], dtype=torch.float32),
            'support_target': torch.tensor(target_data[support_indices], dtype=torch.float32),
            'query_spikes': torch.tensor(session['spikes'][query_indices], dtype=torch.float32),
            'query_target': torch.tensor(target_data[query_indices], dtype=torch.float32)
        }

    def create_meta_batch(self, split: str, k_shot: int, k_query: int, 
                         batch_size: int, target_type: str = 'velocity') -> Dict[str, torch.Tensor]:
        """
        Create a single meta-batch of tasks.
        
        Args:
            split: 'train', 'valid', or 'test'
            k_shot: Support set size per task
            k_query: Query set size per task
            batch_size: For train split, should match number of available sessions
            target_type: Type of target variable
        """
        if split == 'train':
            # Ensure batch_size doesn't exceed number of available sessions
            if batch_size > len(self.train_sessions):
                logging.warning(
                    f"Batch size ({batch_size}) is larger than number of available "
                    f"training sessions ({len(self.train_sessions)}). "
                    f"Setting batch_size to {len(self.train_sessions)}."
                )
                batch_size = len(self.train_sessions)
            
            # Sample tasks from different sessions without replacement
            batch_sessions = np.random.choice(
                self.train_sessions, 
                size=batch_size, 
                replace=False  # Changed to False to ensure unique sessions
            )
        elif split == 'valid':
            # For validation, sample multiple tasks from validation session
            batch_sessions = [self.valid_session] * batch_size
        else:  # test
            # For testing, sample multiple tasks from test session
            batch_sessions = [self.test_session] * batch_size
            
        # Sample tasks from selected sessions
        tasks = [
            self.sample_task(session, k_shot, k_query, target_type)
            for session in batch_sessions
        ]
        
        # Collate tasks into a single batch
        return {
            'support_spikes': torch.stack([t['support_spikes'] for t in tasks]),
            'support_target': torch.stack([t['support_target'] for t in tasks]),
            'query_spikes': torch.stack([t['query_spikes'] for t in tasks]),
            'query_target': torch.stack([t['query_target'] for t in tasks])
        }

    def create_meta_dataloader(self, split: str, k_shot: int, k_query: int,
                             batch_size: int, target_type: str = 'velocity',
                             num_batches: int = 100) -> DataLoader:
        """
        Create a meta-learning dataloader.
        
        Args:
            split: 'train', 'valid', or 'test'
            k_shot: Number of support samples per task
            k_query: Number of query samples per task
            batch_size: Number of tasks per batch
            target_type: Type of target variable ('velocity' or 'position')
            num_batches: Number of batches to generate
        """
        class MetaBatchDataset(Dataset):
            def __init__(self, dataset, split, k_shot, k_query, batch_size, target_type, num_batches):
                self.dataset = dataset
                self.split = split
                self.k_shot = k_shot
                self.k_query = k_query
                self.batch_size = batch_size
                self.target_type = target_type
                self.num_batches = num_batches
            
            def __len__(self):
                return self.num_batches
            
            def __getitem__(self, idx):
                return self.dataset.create_meta_batch(
                    self.split,
                    self.k_shot,
                    self.k_query,
                    self.batch_size,
                    self.target_type
                )
        
        dataset = MetaBatchDataset(
            self, split, k_shot, k_query, batch_size, target_type, num_batches
        )
        
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=(split == 'train'),
            num_workers=0,
            collate_fn=lambda x: x[0]  # Unpack the batch
        )

def test_dataset():
    """Comprehensive test suite for the Combined_Monkey_Dataset class."""
    
    # Initialize dataset
    print("\nInitializing dataset...")
    dataset = Combined_Monkey_Dataset(exclude_ids=['13122.0'], width=0.02, smoothing=0.05, target_neuron_dim=100)
    
    # Test data statistics
    print("\nComputing dataset statistics...")
    stats = dataset.compute_data_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test data loading
    print("\nTesting data loading...")
    train_loader = dataset.create_dataset('train', batch_size=32)
    spikes, labels, cursor_traj, cursor_vel = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"Spikes: {spikes.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Cursor trajectory: {cursor_traj.shape}")
    print(f"Cursor velocity: {cursor_vel.shape}")

    print(f"\nSpikes:\n", spikes[5,1,:50])
    print("labels: ", labels[0][:10])
    print("cursor_traj: ", cursor_traj[0][:10])
    print("cursor_vel: ", cursor_vel[0][:10])
    
    # Test masking
    print("\nTesting masking functionality...")
    masked_data, mask = dataset.create_masked_data(spikes)
    print(f"Masked data shape: {masked_data.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test augmentation
    print("\nTesting augmentation functionality...")
    aug1, aug2 = dataset.create_augmented_pairs(spikes)
    print(f"Augmentation 1 shape: {aug1.shape}")
    print(f"Augmentation 2 shape: {aug2.shape}")
    
    # Test visualizations
    print("\nGenerating visualizations...")
    dataset.visualize_masking(num_samples=2)
    dataset.visualize_augmentations(num_samples=2)
    
    print("\nAll tests completed successfully!")

def test_session_maml_dataset():
    """Test suite for Session_MAML_Monkey_Dataset class."""
    
    print("\n=== Testing Session MAML Dataset ===")
    
    # Initialize dataset
    print("\nInitializing dataset...")
    dataset = Session_MAML_Monkey_Dataset(
        exclude_ids=['13122.0', '12290.0', '9940.0', '13272.0', '10812.0', '10820.0'],
        valid_id='14878',
        test_id='14116',
        width=0.02,
        target_neuron_dim=100
    )
    
    # Test session organization
    print("\nTesting session organization...")
    print(f"Number of train sessions: {len(dataset.train_sessions)}")
    print(f"Train sessions: {dataset.train_sessions}")
    print(f"Validation session: {dataset.valid_session}")
    print(f"Test session: {dataset.test_session}")
    
    # Test single task sampling
    print("\nTesting single task sampling...")
    task = dataset.sample_task(
        session_id=dataset.train_sessions[0],
        k_shot=5,
        k_query=15,
        target_type='velocity'
    )
    
    print("Task shapes:")
    print(f"Support spikes: {task['support_spikes'].shape}")
    print(f"Support target: {task['support_target'].shape}")
    print(f"Query spikes: {task['query_spikes'].shape}")
    print(f"Query target: {task['query_target'].shape}")
    
    # Test meta-batch creation
    print("\nTesting meta-batch creation...")
    meta_batch = dataset.create_meta_batch(
        split='train',
        k_shot=5,
        k_query=15,
        batch_size=4,
        target_type='velocity'
    )
    
    print("Meta-batch shapes:")
    print(f"Support spikes: {meta_batch['support_spikes'].shape}")
    print(f"Support target: {meta_batch['support_target'].shape}")
    print(f"Query spikes: {meta_batch['query_spikes'].shape}")
    print(f"Query target: {meta_batch['query_target'].shape}")
    

    # Test parameters
    k_shot = 5
    k_query = 15
    batch_size = 4
    num_batches = 3

    # Create dataloaders for all splits
    dataloaders = {
        'train': dataset.create_meta_dataloader(
            split='train',
            k_shot=k_shot,
            k_query=k_query,
            batch_size=batch_size,
            num_batches=num_batches
        ),
        'valid': dataset.create_meta_dataloader(
            split='valid',
            k_shot=k_shot,
            k_query=k_query,
            batch_size=batch_size,
            num_batches=num_batches
        ),
        'test': dataset.create_meta_dataloader(
            split='test',
            k_shot=k_shot,
            k_query=k_query,
            batch_size=batch_size,
            num_batches=num_batches
        )
    }
    
    # Test each dataloader
    for split, loader in dataloaders.items():
        print(f"\nTesting {split} dataloader:")
        print(f"Number of batches: {len(loader)}")
        
        # Test iteration
        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx + 1}:")
            
            # Check batch contents
            expected_shapes = {
                'support_spikes': (batch_size, k_shot, dataset.time_steps, dataset.num_neurons),
                'support_target': (batch_size, k_shot, dataset.time_steps, 2),
                'query_spikes': (batch_size, k_query, dataset.time_steps, dataset.num_neurons),
                'query_target': (batch_size, k_query, dataset.time_steps, 2)
            }
            
            # Verify shapes
            for key, expected_shape in expected_shapes.items():
                actual_shape = batch[key].shape
                print(f"{key}: {actual_shape} {'✓' if actual_shape == expected_shape else '✗'}")
                assert actual_shape == expected_shape, f"Wrong shape for {key}"
            
            # For training split, verify tasks come from different sessions
            if split == 'train':
                print("Batch contains tasks from training sessions")
            else:
                print(f"Batch contains tasks from {split} session: {getattr(dataset, f'{split}_session')}")
    
    print("\nAll dataloader tests completed successfully!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_dataset()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")

# if __name__ == '__main__':

#     dataset = Combined_Monkey_Dataset(exclude_ids=['13122.0'], width=0.02,smoothing=0.05)
#     meta_loader = dataset.create_meta_dataloader(
#         split='train',
#         k_shot=5,
#         k_query=15,
#         batch_size=16,
#         target_type='velocity',
#         num_batches= 30
#     )

#     for batch in meta_loader:
#         print(batch['support_spikes'].shape)
#         print(batch['support_target'].shape)
#         print(batch['query_spikes'].shape)
#         print(batch['query_target'].shape)
#         break


# if __name__ == "__main__":
#     try:
#         test_session_maml_dataset()
#     except Exception as e:
#         print(f"\nTest failed with error: {str(e)}")
