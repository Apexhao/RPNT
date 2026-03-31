"""
Demonstration: CrossSiteMonkeyDataset + Transformer Integration for MAE Training

This script shows how to:
1. Load the fixed CrossSiteMonkeyDataset
2. Create dataloaders for training
3. Integrate with the transformer model and loss functions
4. Demonstrate a complete training step
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def demo_complete_integration():
    """Demonstrate complete dataset + transformer integration."""
    
    print("🚀 Demo: CrossSiteMonkeyDataset + Transformer Integration")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # ===============================
    # STEP 1: Initialize Dataset
    # ===============================
    print(f"\n📦 STEP 1: Initialize CrossSiteMonkeyDataset")
    
    # Import dataset (direct import to avoid package issues)
    sys.path.append(str(Path(__file__).parent / "src" / "data"))
    sys.path.append(str(Path(__file__).parent / "src" / "utils"))
    from cross_site_dataset import CrossSiteMonkeyDataset
    
    # Create dataset with reasonable parameters for demo
    dataset = CrossSiteMonkeyDataset(
        exclude_ids=['13122.0', '10812.0'],  # Exclude just a few for faster loading
        split_ratios=(0.8, 0.1, 0.1),
        target_neurons=50,                   # Standard neuron count
        sample_times=5,                      # Standard sampling
        target_trials_per_site=2000,        # Full sampling for training
        min_val_test_trials=150,             # Minimal sampling for val/test (efficiency!)
        random_seed=42
    )
    
    print(f"✅ Dataset created:")
    print(f"   Sites: {len(dataset.site_ids)}")
    print(f"   Shape format: (B={dataset.target_trials_per_site * dataset.sample_times}, S={len(dataset.site_ids)}, T=50, N={dataset.target_neurons})")
    
    # Get site coordinates for positional encoding
    site_coords = dataset.get_site_coordinates().to(device)
    print(f"   Site coordinates: {site_coords.shape}")
    
    # ===============================
    # STEP 2: Create DataLoaders
    # ===============================
    print(f"\n🔄 STEP 2: Create DataLoaders")
    
    train_loader = dataset.create_dataloader('train', batch_size=8, shuffle=True)
    val_loader = dataset.create_dataloader('val', batch_size=8, shuffle=False)
    
    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # ===============================
    # STEP 3: Initialize Transformer Model
    # ===============================
    print(f"\n🧠 STEP 3: Initialize Transformer Model")
    
    from models.transformer import CrossSiteFoundationMAE
    
    model = CrossSiteFoundationMAE(
        neural_dim=dataset.target_neurons,   # Match dataset neurons
        d_model=256,                         # Smaller for demo
        n_sites=len(dataset.site_ids),       # Match dataset sites
        temporal_layers=4,                   # Fewer layers for demo
        spatial_layers=2,
        heads=4,
        use_mae_decoder=True                 # Enable MAE reconstruction
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Input format: [B, S, T, N] = [B, {len(dataset.site_ids)}, 50, {dataset.target_neurons}]")
    
    # ===============================
    # STEP 4: Initialize Masking and Loss
    # ===============================
    print(f"\n🎭 STEP 4: Initialize Masking and Loss Functions")
    
    from utils.masking import CausalMaskingEngine
    from evaluation.loss_functions import compute_neural_mae_loss
    
    # Create masking engine with dynamic ratios
    masking_engine = CausalMaskingEngine(
        temporal_mask_ratio=[0.15, 0.35],    # Dynamic temporal masking
        neuron_mask_ratio=[0.10, 0.25],      # Dynamic neuron masking
        min_unmasked_timesteps=5,
        min_unmasked_neurons=10
    )
    
    print(f"✅ Masking and loss functions ready:")
    print(f"   Temporal mask ratio: {masking_engine.temporal_mask_ratio}")
    print(f"   Neuron mask ratio: {masking_engine.neuron_mask_ratio}")
    
    # ===============================
    # STEP 5: Demonstrate Training Step
    # ===============================
    print(f"\n🏋️  STEP 5: Demonstrate Training Step")
    
    # Get a training batch
    batch_data = next(iter(train_loader))
    neural_data = batch_data[0].to(device)  # [B, S, T, N]
    
    print(f"   Batch data shape: {neural_data.shape}")
    
    # Forward pass with loss computation
    model.train()
    
    loss_dict = compute_neural_mae_loss(
        model=model,
        neural_data=neural_data,
        site_coords=site_coords,
        masking_engine=masking_engine,
        contrastive_weight=0.1,
        reconstruction_weight=1.0
    )
    
    total_loss = loss_dict['total_loss']
    poisson_loss = loss_dict['poisson_loss']
    contrastive_loss = loss_dict['contrastive_loss']
    
    print(f"✅ Forward pass completed:")
    print(f"   Total loss: {total_loss.item():.6f}")
    print(f"   Poisson loss: {poisson_loss.item():.6f}")
    print(f"   Contrastive loss: {contrastive_loss.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"   Gradient norm: {grad_norm:.6f}")
    
    # ===============================
    # STEP 6: Optimizer Integration
    # ===============================
    print(f"\n⚡ STEP 6: Demonstrate Optimizer Integration")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01  # Replaces L2 regularization
    )
    
    # Take an optimization step
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"✅ Optimization step completed")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    
    # ===============================
    # STEP 7: Validation Step
    # ===============================
    print(f"\n📊 STEP 7: Demonstrate Validation Step")
    
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if i >= 3:  # Just test a few batches
                break
                
            neural_data = batch_data[0].to(device)
            
            val_loss_dict = compute_neural_mae_loss(
                model=model,
                neural_data=neural_data,
                site_coords=site_coords,
                masking_engine=masking_engine,
                contrastive_weight=0.1,
                reconstruction_weight=1.0
            )
            
            val_losses.append(val_loss_dict['total_loss'].item())
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"✅ Validation completed:")
    print(f"   Average val loss: {avg_val_loss:.6f}")
    print(f"   Batches processed: {len(val_losses)}")
    
    print(f"\n🎉 Complete Integration Demo Successful!")
    print(f"\n💡 Key Integration Points:")
    print(f"   • ✅ Dataset provides consistent (B,S,T,N) format")
    print(f"   • ✅ Site coordinates work with positional encoding") 
    print(f"   • ✅ No labels needed for MAE pretraining")
    print(f"   • ✅ Masking engine works with 4D tensors")
    print(f"   • ✅ Loss functions integrate seamlessly")
    print(f"   • ✅ Training loop ready for full implementation")
    
    return {
        'dataset': dataset,
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'masking_engine': masking_engine,
        'optimizer': optimizer
    }

if __name__ == "__main__":
    components = demo_complete_integration()
    print(f"\n🚀 Ready to build full training loop!")
    print(f"📋 Next steps:")
    print(f"   1. Implement training loop with these components")
    print(f"   2. Add learning rate scheduling")
    print(f"   3. Add checkpointing and logging")
    print(f"   4. Scale up to full dataset") 