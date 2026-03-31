# 🧠 RPNT: Robust Pre-trained Neural Transformer - A Pathway for Generalized Motor Decoding

A state-of-the-art RPNT model for cross-site neural signal decoding and behavior prediction. This framework implements causal masked autoencoder pretraining with site-specific positional encoding for robust multi-site neural representation learning.

**RPNT** for multi-site brain recordings that:
- 🧠 **Learns causal cross-site neural representations** from Neuropixel recordings across 16+ brain sites within PMd-M1 region in NHP
- 🎭 **Self-supervised causal pretraining** using causal masked autoencoder with Poisson reconstruction
- 🎯 **Transfer learning** to downstream tasks like velocity prediction and direction classification
- ⚡ **Zero-shot generalization** to new recording sites and electrode configurations

### Key Innovation
**Causal site-specific positional encoding** enables the model to understand temporal causality and spatial relationships between recording sites while maintaining the ability to generalize to new, unseen electrode configurations.

## 🚀 Quick Start

### Installation (Linux)
```bash
git clone https://github.com/Apexhao/RPNT.git
cd causal-multisite-neural-foundation-model
```

**📋 Environment Setup:**
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate casual_nfm

# Or install in existing environment
pip install -e .
```

### Run Your First Model
```bash
# Quick debug run (5 epochs, small model)
python scripts/run_foundation_pretrain.py --preset debug

# Use torchrun for proper multi-GPU distributed training (if your machine supports it)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/run_foundation_pretrain.py --preset debug

# Full medium model training (recommended)
python scripts/run_foundation_pretrain.py --preset full_medium

# Monitor with TensorBoard
tensorboard --logdir ./logs --port 6006
```

## 🏗️ Architecture Overview

```
Multi-Site Neural Data → CausalCrossSiteFoundationMAE → Pretrained Representations
     (B,S,T,N)                        ↓                          ↓
16 recording sites              Causal MAE Pretraining      Transfer Learning
50 timesteps                   Poisson + Contrastive        Velocity/Direction
50 neurons per site            Loss Functions               Prediction Tasks
```

**Key Components:**
- **CausalCrossSiteFoundationMAE**: Causal transformer with site-specific positional encoding
- **Enhanced Logging System**: Professional experiment tracking and visualization
- **Multi-GPU Training**: Distributed training support for large models
- **Comprehensive Evaluation**: Built-in metrics and experiment management

## 🎯 Downstream Tasks

Our framework provides comprehensive infrastructure for downstream task evaluation:

### **Professional Training Modes**
```bash
# Frozen encoder (fast, robust baseline)
python scripts/run_downstream_regression.py --dataset_id te14116 --training_mode frozen_encoder

# Full fine-tuning (maximum performance)
python scripts/run_downstream_regression.py --dataset_id te14116 --training_mode full_finetune
```

### **Task Types**
- **🎯 Regression**: Velocity prediction from neural activity


## 📦 Project Structure

```
causal-multisite-neural-foundation-model/
├── src/                    # Core implementation
├── scripts/                # Training and experiment scripts  
├── config/                 # Configuration files
├── docs/                   # Documentation
├── logs/                   # Experiment outputs
├── environment.yml         # Conda environment specification
└── README.md              # This file
```

## 🖥️ System Requirements

- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**: 3.10+
- **CUDA**: 12.4+ (for GPU acceleration)
- **Memory**: 16GB+ RAM recommended

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{fang2026rpnt,
  title={RPNT: Robust Pre-trained Neural Transformer--A Pathway for Generalized Motor Decoding},
  author={Fang, Hao and Canfield, Ryan A and Ouchi, Tomohiro and Macagno, Beatrice and Shlizerman, Eli and Orsborn, Amy L},
  journal={arXiv preprint arXiv:2601.17641},
  year={2026}
}
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
