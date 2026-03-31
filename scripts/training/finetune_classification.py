import torch
import logging
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# Import from the new organized structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import KernelTransformerMAE, KernelTransformerClassifier
from src.data.dataset import Combined_Monkey_Dataset
from src.utils.helpers import set_seed, load_config
from src.evaluation.metrics import SmoothCrossEntropyLoss
from typing import Dict

class ClassificationTrainer:
    def __init__(self, model: KernelTransformerClassifier, dataset, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        
        # Create data loaders
        self.train_loader = dataset.create_dataset('train', batch_size=config['finetune']['batch_size'], shuffle=True)
        self.val_loader = dataset.create_dataset('valid', batch_size=config['finetune']['batch_size'], shuffle=False)
        self.test_loader = dataset.create_dataset('test', batch_size=config['finetune']['batch_size'], shuffle=False)
        
        # Smooth cross entropy loss for soft labels
        self.criterion = SmoothCrossEntropyLoss()
        
        # Optimizer with simple learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['finetune']['learning_rate']),
            weight_decay=0.01
        )
        
        # Simple learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['finetune']['decay_epochs'],
            gamma=float(config['finetune']['decay_rate'])
        )
        
        # Initialize logging
        self.writer = SummaryWriter(config['paths']['tensorboard_dir'])
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        
        # Initialize logging
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(config['paths']['checkpoint_dir'], 'finetune.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'learning_rates': []
        }
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for spikes, soft_labels, _, _ in self.train_loader:
            spikes = spikes.to(self.device)
            soft_labels = soft_labels.to(self.device)
            
            # Forward pass
            logits = self.model(spikes)
            loss = self.criterion(logits, soft_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Compute accuracy (using argmax for both prediction and soft labels)
            _, predicted = torch.max(logits.data, 1)
            _, true_labels = torch.max(soft_labels.data, 1)
            total += soft_labels.size(0)
            correct += (predicted == true_labels).sum().item()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def evaluate(self, split: str = 'valid') -> Dict[str, float]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        loader = self.val_loader if split == 'valid' else self.test_loader
        
        for spikes, soft_labels, _, _ in loader:
            spikes = spikes.to(self.device)
            soft_labels = soft_labels.to(self.device)
            
            # Forward pass
            logits = self.model(spikes)
            loss = self.criterion(logits, soft_labels)
            
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            _, true_labels = torch.max(soft_labels.data, 1)
            total += soft_labels.size(0)
            correct += (predicted == true_labels).sum().item()
            total_loss += loss.item()
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total
        }
    
    def train(self, num_epochs: int, save_freq: int = 5):
        """Main training loop."""
        best_val_acc = 0.0
        logging.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training epoch
            train_loss, train_acc = self.train_epoch()
            self.scheduler.step()
            
            # Validation
            val_metrics = self.evaluate('valid')
            test_metrics = self.evaluate('test')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['test_acc'].append(test_metrics['accuracy'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/test', test_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/test', test_metrics['accuracy'], epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log progress
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Test Loss: {test_metrics['loss']:.4f} - "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint saving
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                
        # Save and plot training history
        self.save_history()
        self.writer.close()
        logging.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def save_history(self):
        """Save training history to CSV."""
        df = pd.DataFrame({
            'Epoch': range(1, len(self.history['train_loss']) + 1),
            'Train_Loss': self.history['train_loss'],
            'Val_Loss': self.history['val_loss'],
            'Test_Loss': self.history['test_loss'],
            'Train_Acc': self.history['train_acc'],
            'Val_Acc': self.history['val_acc'],
            'Test_Acc': self.history['test_acc'],
            'Learning_Rate': self.history['learning_rates']
        })
        df.to_csv(os.path.join(self.checkpoint_dir, 'training_history.csv'), index=False)

def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Load fine-tuning config
    config = load_config('config/config_finetune_classification.yaml')
    
    # Set random seed
    set_seed(config['finetune']['seed'])
    
    try:
        # Initialize dataset
        logging.info("Initializing dataset...")
        dataset = Combined_Monkey_Dataset(config['finetune']['exclude_ids'], width=0.02)

        # Load pretrained model
        logging.info("Initializing pretrained model...")
        pretrained_model = KernelTransformerMAE(
            **config['model']
        )
        
        # Load pretrained weights
        logging.info("Loading pretrained weights...")
        checkpoint = torch.load(config['paths']['pretrained_path'])
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize classifier
        logging.info("Initializing classifier...")
        classifier = KernelTransformerClassifier(
            pretrained_model=pretrained_model,
            num_classes=config['finetune']['num_classes'],
            freeze_encoder=config['finetune']['freeze_encoder']
        )
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = ClassificationTrainer(classifier, dataset, config)
        
        # Train model
        logging.info("Starting training...")
        trainer.train(
            num_epochs=config['finetune']['num_epochs'],
            save_freq=config['finetune']['save_freq']
        )
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()