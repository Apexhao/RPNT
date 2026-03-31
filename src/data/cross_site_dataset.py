"""
Cross-Site Monkey Dataset for Neural Foundation Model Training
-------------------------------------------------------------

This module provides a specialized dataset for cross-site neural transformer training.

Key Features:
- Loads multiple sites using neuropixel_locations
- Standardizes neuron counts across sites via sampling
- Provides (B,S,T,N) format for transformer input
- Site-specific coordinates for positional encoding
- Configurable split ratios and sampling strategies
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
    # Try relative import first (when used as module)
    from ..utils.helpers import load_neuropixel_locations
except ImportError:
    # Fall back to absolute import (when run directly)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.helpers import load_neuropixel_locations


class CrossSiteMonkeyDataset(Dataset):
    """
    Cross-Site Monkey Dataset for Neural Foundation Model Training.
    
    **DESIGN GOALS**:
    - Standardized neuron counts across all sites
    - (B,S,T,N) format for transformer input
    - Site coordinates for positional encoding
    - Flexible sampling and splitting strategies
    
    **KEY FEATURES**:
    - Multi-site data loading and organization
    - Neuron standardization via sampling
    - Train/validation/test splits per site
    - Batch-first data format
    """
    
    def __init__(self,
                 data_root: str = "/data/Fang-analysis/causal-nfm//Data/Monkey_data_meta",
                 exclude_ids: List[str] = ['13122.0'],
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 target_neurons: int = 50,
                 sample_times: int = 5,
                 target_trials_per_site: int = 4000,
                 min_val_test_trials: int = 100,
                 width: float = 0.02,
                 sequence_length: int = 50,
                 random_seed: int = 42):
        """
        Initialize CrossSiteMonkeyDataset.
        
        Args:
            data_root: Root directory containing monkey data files
            exclude_ids: List of site IDs to exclude from loading
            split_ratios: (train, validation, test) split ratios
            target_neurons: Target number of neurons per site (N dimension)
            sample_times: Number of sampling repetitions for neuron standardization
            target_trials_per_site: Target number of trials for TRAINING split
            min_val_test_trials: Minimum trials for validation/test splits (prevents oversampling)
            width: Time bin width for spike data (should match data files)
            sequence_length: Expected sequence length (T dimension)
            random_seed: Random seed for reproducible sampling
        """
        
        # Configuration
        self.data_root = Path(data_root)
        self.exclude_ids = exclude_ids
        self.split_ratios = split_ratios
        self.target_neurons = target_neurons
        self.sample_times = sample_times
        self.target_trials_per_site = target_trials_per_site
        self.min_val_test_trials = min_val_test_trials
        self.width = width
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        
        # Validate split ratios
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        # Set random seed for reproducible sampling
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Storage for site data
        self.sites_data = {}      # {site_id: {train/val/test: data}}
        self.site_coordinates = {}  # {site_id: (x, y)}
        self.site_ids = []         # List of valid site IDs
        
        # Data statistics
        self.data_stats = {}
        
        # Load neuropixel locations and data
        self._load_all_sites()
        
        # Organize data for cross-site training
        self._organize_cross_site_data()
        
        # Compute dataset statistics
        self._compute_statistics()
        
        logging.info(f"CrossSiteMonkeyDataset initialized with {len(self.site_ids)} sites")
    
    def _load_all_sites(self):
        """Load data from all available sites using neuropixel_locations."""
        
        print("🔄 Loading neuropixel locations...")
        neuropixel_locations = load_neuropixel_locations()
        
        successful_loads = 0
        failed_loads = 0
        
        for site_key, site_info in neuropixel_locations.items():
            site_id = str(site_key)
            
            # Skip excluded sites
            if site_id in self.exclude_ids:
                print(f"⏭️  Skipping excluded site: {site_id}")
                continue
            
            try:
                # Load individual site data
                site_data = self._load_single_site(site_id, site_info)
                
                if site_data is not None:
                    self.sites_data[site_id] = site_data
                    self.site_coordinates[site_id] = (site_info['X'], site_info['Y'])
                    self.site_ids.append(site_id)
                    successful_loads += 1
                    print(f"✅ Loaded site {site_id}: {site_data['original_shape']}")
                else:
                    failed_loads += 1
                    print(f"❌ Failed to load site {site_id}")
                    
            except Exception as e:
                failed_loads += 1
                print(f"❌ Error loading site {site_id}: {str(e)}")
        
        print(f"\n📊 Loading Summary:")
        print(f"   ✅ Successfully loaded: {successful_loads} sites")
        print(f"   ❌ Failed to load: {failed_loads} sites")
        print(f"   📋 Available sites: {self.site_ids}")
        
        if successful_loads == 0:
            raise ValueError("No sites were successfully loaded!")
    
    def _load_single_site(self, site_id: str, site_info: Dict) -> Optional[Dict]:
        """
        Load and process data for a single site.
        
        Args:
            site_id: Site identifier
            site_info: Site information from neuropixel_locations
            
        Returns:
            Dictionary containing processed site data or None if failed
        """
        
        # Construct filename (following existing pattern)
        dataset_id = int(site_id.split('.')[0])
        print(f"dataset_id: {dataset_id}")
        filename = self.data_root / f"beignet_te{dataset_id}_spike_data_{self.width:.2f}.pkl"
        
        if not filename.exists():
            print(f"⚠️  File not found: {filename}")
            return None
        
        try:
            # Load data file
            with open(filename, 'rb') as f:
                spike_data, labels, cursor_traj, go_cue_times, center_target_on_times, length = pickle.load(f)
            
            # Store original shape for reference
            original_shape = spike_data.shape  # (Trials, Time, Neurons)
            
            # Validate expected dimensions
            if spike_data.shape[1] != self.sequence_length:
                print(f"⚠️  Unexpected sequence length: {spike_data.shape[1]}, expected {self.sequence_length}")
            
            # Apply train/validation/test split  
            split_data = self._apply_site_split(spike_data)
            
            # Apply neuron and trial standardization
            standardized_data = self._standardize_data(split_data)
            
            return {
                'data': standardized_data,
                'original_shape': original_shape,
                'site_info': site_info
            }
            
        except Exception as e:
            print(f"💥 Error processing site te{dataset_id}: {str(e)}")
            return None
    
    def _apply_site_split(self, spike_data: np.ndarray) -> Dict[str, Dict]:
        """
        Apply train/validation/test split to site data.
        
        Args:
            spike_data: (Trials, Time, Neurons)
            
        Returns:
            Dictionary with train/val/test splits
        """
        
        n_trials = spike_data.shape[0]
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        # Calculate split indices
        train_size = int(train_ratio * n_trials)
        val_size = int(val_ratio * n_trials)
        # test_size is the remainder
        
        # Create shuffled indices for splitting
        indices = np.random.permutation(n_trials)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return {
            'train': {
                'spike_data': spike_data[train_indices],
                'n_trials': len(train_indices)
            },
            'val': {
                'spike_data': spike_data[val_indices],
                'n_trials': len(val_indices)
            },
            'test': {
                'spike_data': spike_data[test_indices],
                'n_trials': len(test_indices)
            }
        }
    
    def _standardize_data(self, split_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Standardize neuron and trial counts using the clean three-step approach:
        1. Data already split by _apply_site_split()
        2. Neuron sampling: Standardize neuron count and create variety through multi-sampling
        3. Trial sampling: Match target trial counts for each split
        
        **CLEAN APPROACH**: Split → Neuron Sampling → Trial Sampling
        - Neuron sampling is the main way to augment data and standardize dimensions
        - Trial sampling just matches final target counts (not for massive augmentation)
        
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
            
            # Apply neuron sampling: (Split_Trials, Time, Original_Neurons) → (Split_Trials, Time, Total_Target_Neurons)
            neuron_sampled_data = spike_data[:, :, neuron_indices]
            
            # Step 2: Reshape to create multiple samples per trial (neuron multi-sampling for variety)
            # (Split_Trials, Time, Total_Target_Neurons) → (Split_Trials*Sample_Times, Time, Target_Neurons)
            multi_sampled_data = neuron_sampled_data.reshape(
                split_trials * self.sample_times, 
                time, 
                self.target_neurons
            )  # (Split_Trials*Sample_Times, Time, Target_Neurons)
            
            # Step 3: Trial sampling to match target trial counts for each split
            if split_name == 'train':
                target_trials = self.target_trials_per_site
            else:
                target_trials = self.min_val_test_trials
            
            available_trials = multi_sampled_data.shape[0]  # Split_Trials * Sample_Times
            
            # Sample/trim to target trial count
            if available_trials >= target_trials:
                # Sufficient samples: randomly select without replacement
                trial_indices = np.random.choice(
                    available_trials, 
                    size=target_trials, 
                    replace=False
                )
            else:
                # Insufficient samples: sample with replacement
                trial_indices = np.random.choice(
                    available_trials,
                    size=target_trials,
                    replace=True
                )
            
            # Apply trial sampling
            final_data = multi_sampled_data[trial_indices]  # (Target_Trials, Time, Target_Neurons)
            
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
            
            print(f"{split_name}_data: {final_data.shape} (split: {split_trials} → multi-sampled: {available_trials} → final: {target_trials})")
        
        return standardized_data
    
    def _organize_cross_site_data(self):
        """Organize data for cross-site training format (B,S,T,N)."""
        
        self.cross_site_data = {'train': {}, 'val': {}, 'test': {}}
        
        for split in ['train', 'val', 'test']:
            # Collect data from all sites for this split
            all_site_data = []
            
            for site_id in self.site_ids:
                site_data = self.sites_data[site_id]['data'][split]
                all_site_data.append(site_data['spike_data'])  # (Trials, Time, Neurons)
            
            # Stack along site dimension: (Sites, Trials, Time, Neurons)
            # Now all sites have the same number of trials, so stacking will work
            stacked_data = np.stack(all_site_data, axis=0)
            
            # Transpose to (Trials, Sites, Time, Neurons) for batch-first format
            transposed_data = np.transpose(stacked_data, (1, 0, 2, 3))
            
            self.cross_site_data[split] = {
                'data': transposed_data.astype(np.float32),  # (Trials, Sites, Time, Neurons)
                'shape': transposed_data.shape
            }
    
    def _compute_statistics(self):
        """Compute dataset statistics for normalization and monitoring."""
        
        train_data = self.cross_site_data['train']['data']  # (Trials, Sites, Time, Neurons)
        
        self.data_stats = {
            'shape': {
                'n_sites': len(self.site_ids),
                'target_neurons': self.target_neurons,
                'target_trials_per_site': self.target_trials_per_site,
                'sequence_length': self.sequence_length,
                'sample_times': self.sample_times
            },
            'splits': {
                split: {
                    'n_trials': self.cross_site_data[split]['data'].shape[0],
                    'data_shape': self.cross_site_data[split]['shape']
                } for split in ['train', 'val', 'test']
            },
            'neural_stats': {
                'mean': np.mean(train_data),
                'std': np.std(train_data),
                'min': np.min(train_data),
                'max': np.max(train_data)
            },
            'site_coordinates': self.site_coordinates
        }
        
        # Print summary
        print(f"\n📊 Dataset Statistics:")
        print(f"   🏢 Sites: {self.data_stats['shape']['n_sites']}")
        print(f"   🧠 Neurons per site: {self.target_neurons}")
        print(f"   📋 Trials per site: {self.target_trials_per_site}")
        print(f"   ⏱️ Sequence length: {self.sequence_length}")
        print(f"   🔄 Sample times: {self.sample_times}")
        print(f"   📈 Data shape: {train_data.shape}")
        print(f"   📊 Neural data: mean={self.data_stats['neural_stats']['mean']:.4f}, "
              f"std={self.data_stats['neural_stats']['std']:.4f}")
    
    def get_site_coordinates(self) -> torch.Tensor:
        """
        Get site coordinates for positional encoding.
        
        Returns:
            torch.Tensor: [S, 2] - (X, Y) coordinates for each site
        """
        coords = []
        for site_id in self.site_ids:
            x, y = self.site_coordinates[site_id]
            coords.append([x, y])
        
        return torch.tensor(coords, dtype=torch.float32)
    
    def get_split_data(self, split: str = 'train') -> np.ndarray:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Neural data with shape (Trials, Sites, Time, Neurons)
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")
        
        split_data = self.cross_site_data[split]
        return split_data['data']
    
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
            DataLoader yielding neural_data with shape [B, S, T, N]
        """
        
        data = self.get_split_data(split)
        
        # Create TensorDataset with only neural data (no labels for MAE pretraining)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data, dtype=torch.float32)    # [Trials, Sites, Time, Neurons]
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
        return self.cross_site_data['train']['data'].shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Neural data with shape [S, T, N] - neural activity for all sites
        """
        data = self.cross_site_data['train']['data'][idx]  # [Sites, Time, Neurons]
        
        return torch.tensor(data, dtype=torch.float32)
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return self.data_stats
    
    def print_summary(self):
        """Print a comprehensive dataset summary."""
        
        print(f"\n" + "="*60)
        print(f"📋 CrossSiteMonkeyDataset Summary")
        print(f"="*60)
        
        print(f"\n🏗️  Configuration:")
        print(f"   • Data root: {self.data_root}")
        print(f"   • Split ratios: {self.split_ratios}")
        print(f"   • Target neurons: {self.target_neurons}")
        print(f"   • Target trials (train): {self.target_trials_per_site}")
        print(f"   • Min trials (val/test): {self.min_val_test_trials}")
        print(f"   • Sample times: {self.sample_times}")
        print(f"   • Random seed: {self.random_seed}")
        
        print(f"\n📋 Dataset Information Table:")
        self.print_dataset_table()
        
        print(f"\n📊 Final Data Shapes:")
        for split in ['train', 'val', 'test']:
            shape = self.cross_site_data[split]['shape']
            print(f"   • {split.capitalize()}: {shape} (Trials, Sites, Time, Neurons)")
        
        print(f"\n🧠 Neural Statistics:")
        stats = self.data_stats['neural_stats']
        print(f"   • Mean: {stats['mean']:.4f}")
        print(f"   • Std: {stats['std']:.4f}")
        print(f"   • Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print(f"\n✅ Ready for transformer training!")
        print(f"="*60)
    
    def print_dataset_table(self):
        """Print a detailed table of all loaded datasets."""
        
        print(f"\n{'Site ID':<10} {'Coordinates':<12} {'Raw Shape':<15} {'Train Trials':<12} {'Val Trials':<10} {'Test Trials':<11} {'Neurons':<8}")
        print(f"{'-'*10} {'-'*12} {'-'*15} {'-'*12} {'-'*10} {'-'*11} {'-'*8}")
        
        for site_id in self.site_ids:
            # Get coordinates
            x, y = self.site_coordinates[site_id]
            coords_str = f"({x:.1f}, {y:.1f})"
            
            # Get original shape
            original_shape = self.sites_data[site_id]['original_shape']
            shape_str = f"{original_shape}"
            
            # Get split trial counts
            train_info = self.sites_data[site_id]['data']['train']['sampling_info']
            val_info = self.sites_data[site_id]['data']['val']['sampling_info']
            test_info = self.sites_data[site_id]['data']['test']['sampling_info']
            
            train_trials = train_info['target_trials']
            val_trials = val_info['target_trials']
            test_trials = test_info['target_trials']
            target_neurons = train_info['target_neurons']
            
            print(f"{site_id:<10} {coords_str:<12} {shape_str:<15} {train_trials:<12} {val_trials:<10} {test_trials:<11} {target_neurons:<8}")
        
        # Summary row
        total_sites = len(self.site_ids)
        total_train = total_sites * self.target_trials_per_site
        total_val = total_sites * self.min_val_test_trials  
        total_test = total_sites * self.min_val_test_trials
        
        print(f"{'-'*10} {'-'*12} {'-'*15} {'-'*12} {'-'*10} {'-'*11} {'-'*8}")
        print(f"{'TOTAL':<10} {f'{total_sites} sites':<12} {'(varies)':<15} {total_train:<12} {total_val:<10} {total_test:<11} {self.target_neurons:<8}")
    
    def get_dataset_info_dict(self):
        """Get dataset information as a structured dictionary for external use."""
        
        dataset_info = []
        
        for site_id in self.site_ids:
            # Get site information
            x, y = self.site_coordinates[site_id]
            original_shape = self.sites_data[site_id]['original_shape']
            
            # Get sampling information
            train_info = self.sites_data[site_id]['data']['train']['sampling_info']
            val_info = self.sites_data[site_id]['data']['val']['sampling_info']
            test_info = self.sites_data[site_id]['data']['test']['sampling_info']
            
            site_dict = {
                'site_id': site_id,
                'coordinates': (x, y),
                'raw_shape': original_shape,
                'raw_trials': original_shape[0],
                'raw_neurons': original_shape[2],
                'train_trials': train_info['target_trials'],
                'val_trials': val_info['target_trials'],
                'test_trials': test_info['target_trials'],
                'target_neurons': train_info['target_neurons'],
                'sample_times': train_info['sample_times']
            }
            dataset_info.append(site_dict)
        
        return dataset_info


def test_cross_site_dataset():
    """Test the CrossSiteMonkeyDataset implementation."""
    
    print("🧪 Testing CrossSiteMonkeyDataset")
    print("=" * 50)
    
    try:
        # Initialize dataset
        dataset = CrossSiteMonkeyDataset(
            exclude_ids=['13122.0'],
            split_ratios=(0.8, 0.1, 0.1),
            target_neurons=50,
            sample_times=1,
            target_trials_per_site=5000,        # Full sampling for training
            min_val_test_trials=100,             # Minimal sampling for val/test
            random_seed=42
        )
        
        # Print summary
        dataset.print_summary()
        
        # Test site coordinates
        print(f"\n🗺️  Testing site coordinates:")
        site_coords = dataset.get_site_coordinates()
        print(f"   Site coordinates shape: {site_coords.shape}")
        print(f"   First 3 coordinates: {site_coords[:3]}")
        
        # Test data loading
        print(f"\n📦 Testing data access:")
        train_data = dataset.get_split_data('train')
        print(f"   Train data shape: {train_data.shape}")
        
        # Test dataloader
        print(f"\n🔄 Testing dataloader:")
        train_loader = dataset.create_dataloader('train', batch_size=4, shuffle=True)
        batch_data = next(iter(train_loader))
        print(f"   Batch data shape: {batch_data[0].shape}")
        
        # Test single sample access
        print(f"\n🎯 Testing single sample:")
        sample_data = dataset[0]
        print(f"   Sample data shape: {sample_data.shape}")
        
        print(f"\n✅ All tests passed!")
        return dataset
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests
    dataset = test_cross_site_dataset() 