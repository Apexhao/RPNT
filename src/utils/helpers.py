import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import os
import math
import yaml
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Union

def load_config(yaml_path: str) -> dict:
    """Load configuration from yaml file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across multiple platforms.

    This function sets the seed for Python's random module, NumPy, PyTorch,
    and configures CUDA for deterministic behavior.

    Args:
        seed (int): The random seed to set.

    Raises:
        ValueError: If the seed is not a positive integer.
    """
    if not isinstance(seed, int) or seed <= 0:
        raise ValueError("Seed must be a positive integer.")

    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA's random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Make CUDA deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class LinearWarmUp:
    def __init__(self, initial_learning_rate, warmup_steps, min_lr=1e-7):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def get_lr(self):
        if self.current_step >= self.warmup_steps:
            return self.initial_learning_rate
            
        # Linear warm-up
        warmup_percent = self.current_step / self.warmup_steps
        current_lr = self.min_lr + (self.initial_learning_rate - self.min_lr) * warmup_percent
        self.current_step += 1
        return current_lr

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining steps following a cosine curve.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        """
        Args:
            optimizer: Optimizer to wrap
            warmup_steps: Number of steps for linear warmup
            total_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate ratio (final_lr = min_lr_ratio * initial_lr)
        """
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1. + math.cos(math.pi * progress)))
            
        super().__init__(optimizer, lr_lambda)