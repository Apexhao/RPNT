"""
Public Dataset for Neural Foundation Model Pretraining
-----------------------------------------------------

This module provides a specialized dataset for pretraining on the Perich-Miller 2018 public dataset.

Key Features:
- Loads subjects c,j,m for comprehensive pretraining (~99 sessions)
- Uses SparseTemporalEncoder format [B,S=1,T=50,N=50]
- Session coordinates (1,3) from (subject, time, task) for RoPE4D
- Uses provided train/valid/test indices from each session
- Same augmentation strategy as cross_site_dataset.py
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


class PublicDatasetForPretraining(Dataset):
    """
    Public Dataset for SparseTemporalEncoder Pretraining.
    
    **DESIGN GOALS**:
    - Temporal-only architecture (S=1, no spatial transformer needed)
    - Session-based RoPE4D coordinates for generalization
    - Same neuron augmentation as existing infrastructure
    - Uses provided splits from public dataset
    
    **KEY FEATURES**:
    - Filter sessions by subject: ['c', 'j', 'm'] only
    - Format: [B,S=1,T=50,N=50] for SparseTemporalEncoder input
    - Session coordinates: (1,3) embeddings from session metadata
    - Neuron standardization via sampling/padding to N=50
    - Cross-session data organization for pretraining
    """
    
    def __init__(self,
                 data_root: str = "/data/Fang-analysis/causal-nfm/Data/processed_normalize_session",
                 subjects: List[str] = ['c', 'j', 'm'],
                 target_neurons: int = 50,
                 sample_times: int = 5,
                 target_trials_per_site: int = 4000,
                 min_val_test_trials: int = 100,
                 sequence_length: int = 50,
                 random_seed: int = 42):
        """
        Initialize PublicDatasetForPretraining.
        
        Args:
            data_root: Root directory containing public dataset .pkl files
            subjects: List of subjects for pretraining (default: ['c', 'j', 'm'])
            target_neurons: Target number of neurons (N dimension)
            sample_times: Number of sampling repetitions for neuron standardization
            target_trials_per_site: Target number of trials for TRAINING split
            min_val_test_trials: Minimum trials for validation/test splits
            sequence_length: Expected sequence length (T=50)
            random_seed: Random seed for reproducible sampling
        """
        
        # Configuration
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.target_neurons = target_neurons
        self.sample_times = sample_times
        self.target_trials_per_site = target_trials_per_site
        self.min_val_test_trials = min_val_test_trials
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        
        # Set random seed for reproducible sampling
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Storage for session data
        self.sessions_data = {}      # {session_id: {train/val/test: data}}
        self.session_coordinates = {}  # {session_id: (1, 3) tensor}
        self.session_ids = []         # List of valid session IDs
        
        # Data statistics
        self.data_stats = {}
        
        # Load all public dataset sessions
        self._discover_and_load_sessions()
        
        # Organize data for pretraining
        self._organize_pretraining_data()
        
        # Compute dataset statistics
        self._compute_statistics()
        
        logging.info(f"PublicDatasetForPretraining initialized with {len(self.session_ids)} sessions")
    
    def _discover_and_load_sessions(self):
        """Discover and load sessions from public dataset."""
        
        print("🔄 Discovering public dataset sessions...")
        
        # Find all .pkl files in data_root

        print(f"Data root: {self.data_root}")
        pkl_files = list(self.data_root.glob("*.pkl"))
        print(f"Found {len(pkl_files)} total .pkl files")
        
        # Filter by subjects
        filtered_files = []
        for pkl_file in pkl_files:
            session_id = pkl_file.stem  # Remove .pkl extension
            subject = session_id.split('_')[0]  # Extract subject from filename
            
            if subject in self.subjects:
                filtered_files.append(pkl_file)
        
        print(f"Filtered to {len(filtered_files)} sessions for subjects {self.subjects}")
        
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
                    print(f"✅ Loaded session {session_id}: {session_data['original_shape']}")
                else:
                    failed_loads += 1
                    print(f"❌ Failed to load session {session_id}")
                    
            except Exception as e:
                failed_loads += 1
                print(f"❌ Error loading session {session_id}: {str(e)}")
        
        print(f"\n📊 Loading Summary:")
        print(f"   ✅ Successfully loaded: {successful_loads} sessions")
        print(f"   ❌ Failed to load: {failed_loads} sessions")
        print(f"   📋 Available sessions: {len(self.session_ids)}")
        
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
            train_indices = data['train_indices']
            valid_indices = data['valid_indices']
            test_indices = data['test_indices']
            
            # Store original shape for reference
            original_shape = spike_data.shape
            
            # Validate expected dimensions
            if spike_data.shape[1] != self.sequence_length:
                print(f"⚠️  Unexpected sequence length: {spike_data.shape[1]}, expected {self.sequence_length}")
            
            # Apply session split using provided indices
            split_data = self._apply_session_split(spike_data, train_indices, valid_indices, test_indices)
            
            # Apply neuron and trial standardization (same as cross_site_dataset.py)
            standardized_data = self._standardize_data(split_data)
            
            return {
                'data': standardized_data,
                'original_shape': original_shape,
                'session_info': {
                    'subject_id': data.get('subject_id', 'unknown'),
                    'session_id': data.get('session_id', session_id),
                    'n_units': data.get('n_units', original_shape[2]),
                    'n_windows': data.get('n_windows', original_shape[0])
                }
            }
            
        except Exception as e:
            print(f"💥 Error processing session {session_id}: {str(e)}")
            return None
    
    def _apply_session_split(self, spike_data: np.ndarray, 
                           train_indices: np.ndarray, 
                           valid_indices: np.ndarray, 
                           test_indices: np.ndarray) -> Dict[str, Dict]:
        """
        Apply train/validation/test split using provided indices.
        
        Args:
            spike_data: (n_windows, T, n_units)
            train_indices: Training window indices
            valid_indices: Validation window indices  
            test_indices: Test window indices
            
        Returns:
            Dictionary with train/val/test splits
        """
        
        return {
            'train': {
                'spike_data': spike_data[train_indices],
                'n_trials': len(train_indices)
            },
            'val': {
                'spike_data': spike_data[valid_indices],
                'n_trials': len(valid_indices)
            },
            'test': {
                'spike_data': spike_data[test_indices],
                'n_trials': len(test_indices)
            }
        }
    
    def _standardize_data(self, split_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Standardize neuron and trial counts using the same approach as cross_site_dataset.py.
        
        1. Data already split by _apply_session_split()
        2. Neuron sampling: Standardize neuron count and create variety through multi-sampling
        3. Trial sampling: Match target trial counts for each split
        
        Args:
            split_data: Dictionary with train/val/test data already split
            
        Returns:
            Dictionary with standardized trial and neuron counts
        """
        
        standardized_data = {}
        
        for split_name, split_info in split_data.items():
            spike_data = split_info['spike_data']  # (Split_Trials, Time, Original_Neurons)
            split_trials, time, original_neurons = spike_data.shape
            
            # Step 1: Neuron sampling with multi-sampling to standardize neuron count
            total_target_neurons = self.target_neurons * self.sample_times
            
            if original_neurons >= total_target_neurons:
                # Sufficient neurons: randomly sample without replacement
                neuron_indices = np.random.choice(
                    original_neurons, 
                    size=total_target_neurons, 
                    replace=False
                )
            else:
                # Insufficient neurons: sample with replacement
                neuron_indices = np.random.choice(
                    original_neurons,
                    size=total_target_neurons,
                    replace=True
                )
            
            # Apply neuron sampling
            neuron_sampled_data = spike_data[:, :, neuron_indices]
            
            # Step 2: Reshape to create multiple samples per trial
            multi_sampled_data = neuron_sampled_data.reshape(
                split_trials * self.sample_times, 
                time, 
                self.target_neurons
            )
            
            # Step 3: Trial sampling to match target trial counts
            if split_name == 'train':
                target_trials = self.target_trials_per_site
            else:
                target_trials = self.min_val_test_trials
            
            available_trials = multi_sampled_data.shape[0]
            
            # Sample/trim to target trial count
            if available_trials >= target_trials:
                trial_indices = np.random.choice(
                    available_trials, 
                    size=target_trials, 
                    replace=False
                )
            else:
                trial_indices = np.random.choice(
                    available_trials,
                    size=target_trials,
                    replace=True
                )
            
            # Apply trial sampling
            final_data = multi_sampled_data[trial_indices]
            
            standardized_data[split_name] = {
                'spike_data': final_data.astype(np.float32),
                'n_trials': final_data.shape[0],
                'sampling_info': {
                    'split_trials': split_trials,
                    'original_neurons': original_neurons,
                    'multi_sampled_trials': available_trials,
                    'target_trials': target_trials,
                    'target_neurons': self.target_neurons,
                    'sample_times': self.sample_times,
                    'final_samples': final_data.shape[0]
                }
            }
            
            print(f"  {split_name}: {final_data.shape} (split: {split_trials} → multi-sampled: {available_trials} → final: {target_trials})")
        
        return standardized_data
    
    def _generate_session_coordinates(self, session_id: str) -> torch.Tensor:
        """
        Generate session coordinates (1,3) from session metadata for RoPE4D.
        
        Args:
            session_id: Session identifier (e.g., 'c_20131003_center_out_reaching')
            
        Returns:
            Session coordinates: (1, 3) tensor with (subject, time, task) embeddings
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
    
    def _organize_pretraining_data(self):
        """Organize data for pretraining format [B,S=1,T,N]."""
        
        self.pretraining_data = {'train': {}, 'val': {}, 'test': {}}
        
        for split in ['train', 'val', 'test']:
            # Collect data from all sessions for this split
            all_session_data = []
            all_session_coords = []
            
            for session_id in self.session_ids:
                session_data = self.sessions_data[session_id]['data'][split]
                session_coords = self.session_coordinates[session_id]
                
                spike_data = session_data['spike_data']  # (Trials, Time, Neurons)
                n_trials = spike_data.shape[0]
                
                # Add site dimension: (Trials, Time, Neurons) → (Trials, 1, Time, Neurons)
                spike_data_with_site = spike_data[:, np.newaxis, :, :]  # (Trials, 1, Time, Neurons)
                
                # Replicate session coordinates for all trials
                session_coords_batch = session_coords.unsqueeze(0).repeat(n_trials, 1, 1)  # (Trials, 1, 3)
                
                all_session_data.append(spike_data_with_site)
                all_session_coords.append(session_coords_batch)
            
            # Concatenate all sessions
            if all_session_data:
                concatenated_data = np.concatenate(all_session_data, axis=0)  # (Total_Trials, 1, Time, Neurons)
                concatenated_coords = torch.cat(all_session_coords, dim=0)     # (Total_Trials, 1, 3)
                
                self.pretraining_data[split] = {
                    'data': concatenated_data.astype(np.float32),       # (Total_Trials, 1, T, N)
                    'coordinates': concatenated_coords,                 # (Total_Trials, 1, 3)
                    'shape': concatenated_data.shape
                }
    
    def _compute_statistics(self):
        """Compute dataset statistics for normalization and monitoring."""
        
        train_data = self.pretraining_data['train']['data']  # (Total_Trials, 1, T, N)
        
        self.data_stats = {
            'shape': {
                'n_sessions': len(self.session_ids),
                'target_neurons': self.target_neurons,
                'target_trials_per_site': self.target_trials_per_site,
                'sequence_length': self.sequence_length,
                'sample_times': self.sample_times
            },
            'splits': {
                split: {
                    'n_trials': self.pretraining_data[split]['data'].shape[0],
                    'data_shape': self.pretraining_data[split]['shape']
                } for split in ['train', 'val', 'test']
            },
            'neural_stats': {
                'mean': np.mean(train_data),
                'std': np.std(train_data),
                'min': np.min(train_data),
                'max': np.max(train_data)
            },
            'subjects': self.subjects,
            'total_sessions': len(self.session_ids)
        }
        
        # Print summary
        print(f"\n📊 Dataset Statistics:")
        print(f"   🏢 Sessions: {self.data_stats['shape']['n_sessions']}")
        print(f"   👥 Subjects: {self.subjects}")
        print(f"   🧠 Neurons per session: {self.target_neurons}")
        print(f"   📋 Target trials per site: {self.target_trials_per_site}")
        print(f"   ⏱️ Sequence length: {self.sequence_length}")
        print(f"   🔄 Sample times: {self.sample_times}")
        print(f"   📈 Data shape: {train_data.shape}")
        print(f"   📊 Neural data: mean={self.data_stats['neural_stats']['mean']:.4f}, "
              f"std={self.data_stats['neural_stats']['std']:.4f}")
    
    def get_session_coordinates(self, session_id: str) -> torch.Tensor:
        """
        Get session coordinates for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            torch.Tensor: [1, 3] - (subject, time, task) coordinates
        """
        return self.session_coordinates.get(session_id, torch.zeros(1, 3))
    
    def get_split_data(self, split: str = 'train') -> Tuple[np.ndarray, torch.Tensor]:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Tuple of (neural_data, session_coordinates)
            - neural_data: (Total_Trials, 1, T, N)
            - session_coordinates: (Total_Trials, 1, 3)
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")
        
        split_data = self.pretraining_data[split]
        return split_data['data'], split_data['coordinates']
    
    def create_dataloader(self, 
                         split: str = 'train',
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4) -> DataLoader:
        """
        Create a PyTorch DataLoader for the specified split.
        
        Args:
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader yielding (neural_data, session_coords) with shapes:
            - neural_data: [B, 1, T, N]
            - session_coords: [B, 1, 3]
        """
        
        data, coordinates = self.get_split_data(split)
        
        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data, dtype=torch.float32),        # [Total_Trials, 1, T, N]
            coordinates                                      # [Total_Trials, 1, 3]
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __len__(self) -> int:
        """Return the total number of training samples."""
        return self.pretraining_data['train']['data'].shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (neural_data, session_coords)
            - neural_data: [1, T, N] - neural activity
            - session_coords: [1, 3] - session coordinates
        """
        data = self.pretraining_data['train']['data'][idx]          # [1, T, N]
        coords = self.pretraining_data['train']['coordinates'][idx]  # [1, 3]
        
        return torch.tensor(data, dtype=torch.float32), coords
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self.data_stats
    
    def print_summary(self):
        """Print a comprehensive dataset summary."""
        
        print(f"\n" + "="*60)
        print(f"📋 PublicDatasetForPretraining Summary")
        print(f"="*60)
        
        print(f"\n🏗️  Configuration:")
        print(f"   • Data root: {self.data_root}")
        print(f"   • Subjects: {self.subjects}")
        print(f"   • Target neurons: {self.target_neurons}")
        print(f"   • Target trials (train): {self.target_trials_per_site}")
        print(f"   • Min trials (val/test): {self.min_val_test_trials}")
        print(f"   • Sample times: {self.sample_times}")
        print(f"   • Random seed: {self.random_seed}")
        
        print(f"\n📋 Session Information:")
        print(f"   • Total sessions loaded: {len(self.session_ids)}")
        by_subject = {}
        for session_id in self.session_ids:
            subject = session_id.split('_')[0]
            by_subject[subject] = by_subject.get(subject, 0) + 1
        for subject, count in by_subject.items():
            print(f"   • Subject {subject}: {count} sessions")
        
        print(f"\n📊 Final Data Shapes:")
        for split in ['train', 'val', 'test']:
            shape = self.pretraining_data[split]['shape']
            print(f"   • {split.capitalize()}: {shape} (Total_Trials, Sites=1, Time, Neurons)")
        
        print(f"\n🧠 Neural Statistics:")
        stats = self.data_stats['neural_stats']
        print(f"   • Mean: {stats['mean']:.4f}")
        print(f"   • Std: {stats['std']:.4f}")
        print(f"   • Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print(f"\n✅ Ready for SparseTemporalEncoder pretraining!")
        print(f"="*60)


def test_public_pretrain_dataset():
    """Test the PublicDatasetForPretraining implementation."""
    
    print("🧪 Testing PublicDatasetForPretraining")
    print("=" * 50)
    
    try:
        # Initialize dataset
        dataset = PublicDatasetForPretraining(
            subjects=['t'],
            target_neurons=50,
            sample_times=1,  # For testing
            target_trials_per_site=1000,  # Smaller for testing
            min_val_test_trials=50,
            random_seed=42
        )
        
        # Print summary
        dataset.print_summary()
        
        # Test data access
        print(f"\n📦 Testing data access:")
        train_data, train_coords = dataset.get_split_data('train')
        print(f"   Train data shape: {train_data.shape}")
        print(f"   Train coordinates shape: {train_coords.shape}")
        
        # Test dataloader
        print(f"\n🔄 Testing dataloader:")
        train_loader = dataset.create_dataloader('train', batch_size=4, shuffle=True)
        batch_data, batch_coords = next(iter(train_loader))
        print(f"   Batch data shape: {batch_data.shape}")
        print(f"   Batch coordinates shape: {batch_coords.shape}")
        
        # Test single sample access
        print(f"\n🎯 Testing single sample:")
        sample_data, sample_coords = dataset[0]
        print(f"   Sample data shape: {sample_data.shape}")
        print(f"   Sample coordinates shape: {sample_coords.shape}")
        print(f"   Sample coordinates values: {sample_coords}")
        
        print(f"   Sample coordinates values: {sample_coords}")
        
        print(f"\n✅ All tests passed!")
        return dataset
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests
    dataset = test_public_pretrain_dataset() 
