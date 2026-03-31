"""
Training module for neural foundation models.

This module contains:
- PretrainTrainer: Professional trainer class for foundation model pretraining
- Training utilities and helper functions
- Distributed training support
"""

from .pretrain import PretrainTrainer

__all__ = ['PretrainTrainer'] 