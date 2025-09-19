import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

from mcnn_model import MCNN
from improved_dataloader import PelletDataset, create_data_loaders

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class LearningRateScheduler:
    """Custom learning rate scheduler for counting tasks"""
    def __init__(self, optimizer, mode='plateau', factor=0.5, patience=10, min_lr=1e-8):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = None
        self.counter = 0

    def step(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
        elif metric < self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.reduce_lr()
                self.counter = 0

    def reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_metrics(predicted_dmap, gt_dmap):
    """Calculate various evaluation metrics"""
    pred_count = predicted_dmap.sum().item()
    gt_count = gt_dmap.sum().item()
    
    mae = abs(pred_count - gt_count)
    mse = (pred_count - gt_count) ** 2
    
    # Avoid division by zero
    if gt_count > 0:
        mape = abs(pred_count - gt_count) / gt_count * 100
    else:
        mape = 0 if pred_count == 0 else 100
    
    return {
        'mae': mae,
        'mse': mse,
        'mape': mape,
        'pred_count': pred_count,
        'gt_count': gt_count
    }

def train_epoch_memory_efficient(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    """Memory-efficient training for one epoch with gradient accumulation"""
    model.train()
    epoch_loss = 0
    epoch_mae = 0
    num_batches = len(train_loader)
    
    optimizer.zero_grad()
    accumulated_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, gt_dmaps) in enumerate(pbar):
        try:
            # Resize images to reduce memory usage
            if images.shape[-1] > 800 or images.shape[-2] > 600:
                # Resize to max 800x600 while maintaining aspect ratio
                h, w = images.shape[-2], images.shape[-1]
                max_h, max_w = 600, 800
                scale = min(max_h/h, max_w/w, 1.0)
                new_h, new_w = int(h * scale), int(w * scale)
                
                images = torch.nn.functional.interpolate(
                    images, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
                gt_dmaps = torch.nn.functional.interpolate(
                    gt_dmaps, size=(new_h//4, new_w//4), mode='bilinear', align_corners=False
                )
            
            images = images.to(device, non_blocking=True)
            gt_dmaps = gt_dmaps.to(device, non_blocking=True)
            
            # Forward pass
            pred_dmaps = model(images)
            loss = criterion(pred_dmaps, gt_dmaps) / accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += accumulated_loss
                accumulated_loss = 0
            
            # Calculate metrics for display
            with torch.no_grad():
                metrics = calculate_metrics(pred_dmaps, gt_dmaps)
                epoch_mae += metrics['mae']
            
            # Clear intermediate tensors
            del images, gt_dmaps, pred_dmaps, loss
            
            # Update progress bar
            if batch_idx % 10 == 0:  # Update less frequently to save time
                pbar.set_postfix({
                    'Loss': f"{accumulated_loss:.4f}",
                    'MAE': f"{metrics['mae']:.2f}"
                })
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error at batch {batch_idx}, clearing cache and skipping...")
                clear_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    clear_memory()
    return epoch_loss / num_batches, epoch_mae / num_batches

def validate_epoch_memory_efficient(model, val_loader, criterion, device):
    """Memory-efficient validation for one epoch"""
    model.eval()
    epoch_loss = 0
    epoch_mae = 0
    epoch_mse = 0
    epoch_mape = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_idx, (images, gt_dmaps) in enumerate(pbar):
            try:
                # Resize images to reduce memory usage
                if images.shape[-1] > 800 or images.shape[-2] > 600:
                    h, w = images.shape[-2], images.shape[-1]
                    max_h, max_w = 600, 800
                    scale = min(max_h/h, max_w/w, 1.0)
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    images = torch.nn.functional.interpolate(
                        images, size=(new_h, new_w), mode='bilinear', align_corners=False
                    )
                    gt_dmaps = torch.nn.functional.interpolate(
                        gt_dmaps, size=(new_h//4, new_w//4), mode='bilinear', align_corners=False
                    )
                
                images = images.to(device, non_blocking=True)
                gt_dmaps = gt_dmaps.to(device, non_blocking=True)
                
                pred_dmaps = model(images)
                loss = criterion(pred_dmaps, gt_dmaps)
                
                epoch_loss += loss.item()
                
                metrics = calculate_metrics(pred_dmaps, gt_dmaps)
                epoch_mae += metrics['mae']
                epoch_mse += metrics['mse']
                epoch_mape += metrics['mape']
                
                # Clear tensors
                del images, gt_dmaps, pred_dmaps, loss
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f"{epoch_loss/(batch_idx+1):.4f}",
                        'MAE': f"{epoch_mae/(batch_idx+1):.2f}"
                    })
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in validation at batch {batch_idx}, clearing cache and skipping...")
                    clear_memory()
                    continue
                else:
                    raise e
    
    clear_memory()
    return {
        'loss': epoch_loss / num_batches,
        'mae': epoch_mae / num_batches,
        'mse': epoch_mse / num_batches,
        'rmse': np.sqrt(epoch_mse / num_batches),
        'mape': epoch_mape / num_batches
    }

def save_checkpoint(model, optimizer, epoch, best_mae, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mae': best_mae
    }
    torch.save(checkpoint, save_path)

def plot_training_history(train_losses, val_metrics, save_path):
    """Plot and save training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE
    ax2.plot(epochs, [m['mae'] for m in val_metrics], 'g-', label='Validation MAE')
    ax2.set_title('Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    # RMSE
    ax3.plot(epochs, [m['rmse'] for m in val_metrics], 'm-', label='Validation RMSE')
    ax3.set_title('Validation RMSE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE')
    ax3.legend()
    ax3.grid(True)
    
    # MAPE
    ax4.plot(epochs, [m['mape'] for m in val_metrics], 'c-', label='Validation MAPE (%)')
    ax4.set_title('Validation MAPE')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MAPE (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Memory-optimized configuration for 4GB GPU
    config = {
        'batch_size': 1,  # Keep at 1 due to variable image sizes
        'learning_rate': 1e-5,
        'num_epochs': 100,
        'gt_downsample': 4,  # Keep this to reduce memory
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'lr_patience': 8,
        'save_every': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'accumulation_steps': 4,  # Simulate larger batch size
        'max_image_size': (600, 800)  # Limit image size to save memory
    }
    
    print("=== MCNN Pellet Counter Training (Memory Optimized) ===")
    print(f"Device: {config['device']}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
        
    print(f"Configuration: {config}")
    
    # Clear any existing memory
    clear_memory()
    
    # Data paths
    data_root = "./data"
    train_img_root = os.path.join(data_root, "train_data", "images")
    train_gt_root = os.path.join(data_root, "train_data", "densitymaps")
    test_img_root = os.path.join(data_root, "test_data", "images")
    test_gt_root = os.path.join(data_root, "test_data", "densitymaps")
    
    # For now, use test data as validation
    val_img_root = test_img_root
    val_gt_root = test_gt_root
    
    # Create data loaders with memory optimization
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_img_root, train_gt_root,
        val_img_root, val_gt_root,
        test_img_root, test_gt_root,
        batch_size=config['batch_size'],
        gt_downsample=config['gt_downsample'],
        num_workers=0  # Disable multiprocessing to save memory
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    device = torch.device(config['device'])
    model = MCNN().to(device)
    
    # Enable mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Loss function and optimizer
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler and early stopping
    lr_scheduler = LearningRateScheduler(optimizer, patience=config['lr_patience'])
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Create output directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Training history
    train_losses = []
    val_metrics = []
    best_mae = float('inf')
    
    print("\nStarting memory-optimized training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train with memory optimization
        train_loss, train_mae = train_epoch_memory_efficient(
            model, train_loader, criterion, optimizer, device, 
            accumulation_steps=config['accumulation_steps']
        )
        train_losses.append(train_loss)
        
        # Validate with memory optimization
        val_results = validate_epoch_memory_efficient(model, val_loader, criterion, device)
        val_metrics.append(val_results)
        
        # Update learning rate
        lr_scheduler.step(val_results['mae'])
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}")
        print(f"Val Loss: {val_results['loss']:.4f}, Val MAE: {val_results['mae']:.2f}")
        print(f"Val RMSE: {val_results['rmse']:.2f}, Val MAPE: {val_results['mape']:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory Used: {memory_used:.2f} GB")
        
        # Save best model
        if val_results['mae'] < best_mae:
            best_mae = val_results['mae']
            save_checkpoint(model, optimizer, epoch, best_mae, './checkpoints/best_model.pth')
            print(f"New best model saved! MAE: {best_mae:.2f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, best_mae, f'./checkpoints/epoch_{epoch+1}.pth')
        
        # Early stopping
        if early_stopping(val_results['mae'], model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Clear memory after each epoch
        clear_memory()
    
    # Plot training history
    plot_training_history(train_losses, val_metrics, './results/training_history.png')
    
    # Save training log
    training_log = {
        'config': config,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'best_mae': best_mae,
        'training_completed': datetime.now().isoformat()
    }
    
    with open('./results/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation MAE: {best_mae:.2f}")
    print(f"Results saved in ./results/")
    print(f"Best model saved as ./checkpoints/best_model.pth")

if __name__ == "__main__":
    main()