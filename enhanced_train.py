import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FocalMSELoss(nn.Module):
    """Focal MSE Loss - focuses on hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')
        # Calculate focal weight
        pt = torch.exp(-mse)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_mse = focal_weight * mse
        
        if self.reduction == 'sum':
            return focal_mse.sum()
        elif self.reduction == 'mean':
            return focal_mse.mean()
        else:
            return focal_mse

class CountLoss(nn.Module):
    """Count-preserving loss that directly optimizes counting accuracy"""
    def __init__(self, mse_weight=1.0, count_weight=10.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.count_weight = count_weight
    
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = F.mse_loss(pred, target, reduction='sum')
        
        # Count loss - penalize count differences heavily
        pred_count = pred.sum()
        target_count = target.sum()
        count_loss = F.mse_loss(pred_count, target_count)
        
        return self.mse_weight * mse_loss + self.count_weight * count_loss

class SSIM_Loss(nn.Module):
    """SSIM loss for structural similarity"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, pred, target):
        # Simple SSIM approximation
        mu1 = F.avg_pool2d(pred, self.window_size, 1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(target, self.window_size, 1, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, self.window_size, 1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, self.window_size, 1, padding=self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, self.window_size, 1, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()

class CombinedLoss(nn.Module):
    """Combined loss function for better counting accuracy"""
    def __init__(self, mse_weight=1.0, count_weight=20.0, focal_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.count_weight = count_weight
        self.focal_weight = focal_weight
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.focal_loss = FocalMSELoss(alpha=1.0, gamma=2.0, reduction='sum')
        self.count_loss = CountLoss(mse_weight=0, count_weight=1.0)
    
    def forward(self, pred, target):
        # Standard MSE
        mse = self.mse_loss(pred, target)
        
        # Focal loss for hard examples
        focal = self.focal_loss(pred, target)
        
        # Count preservation loss
        pred_count = pred.sum()
        target_count = target.sum()
        count = F.mse_loss(pred_count, target_count)
        
        total_loss = (self.mse_weight * mse + 
                     self.focal_weight * focal + 
                     self.count_weight * count)
        
        return total_loss, {
            'mse': mse.item(),
            'focal': focal.item(),
            'count': count.item(),
            'total': total_loss.item()
        }

class EarlyStopping:
    """Enhanced early stopping with better patience handling"""
    def __init__(self, patience=20, min_delta=0.1, restore_best_weights=True):
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

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_advanced_metrics(predicted_dmap, gt_dmap):
    """Calculate advanced evaluation metrics"""
    pred_count = predicted_dmap.sum().item()
    gt_count = gt_dmap.sum().item()
    
    mae = abs(pred_count - gt_count)
    mse = (pred_count - gt_count) ** 2
    
    # Relative metrics
    if gt_count > 0:
        mape = abs(pred_count - gt_count) / gt_count * 100
        relative_error = (pred_count - gt_count) / gt_count
    else:
        mape = 0 if pred_count == 0 else 100
        relative_error = 0
    
    # Accuracy at different thresholds
    acc_1 = 1 if mae <= 1 else 0
    acc_2 = 1 if mae <= 2 else 0
    acc_5 = 1 if mae <= 5 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'mape': mape,
        'relative_error': relative_error,
        'pred_count': pred_count,
        'gt_count': gt_count,
        'acc_1': acc_1,
        'acc_2': acc_2,
        'acc_5': acc_5
    }

def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    """Enhanced training with better loss tracking"""
    model.train()
    epoch_metrics = {
        'total_loss': 0,
        'mse_loss': 0,
        'focal_loss': 0,
        'count_loss': 0,
        'mae': 0
    }
    num_batches = len(train_loader)
    
    optimizer.zero_grad()
    accumulated_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, gt_dmaps) in enumerate(pbar):
        try:
            # Memory optimization - resize large images
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
            
            # Forward pass
            pred_dmaps = model(images)
            
            # Calculate combined loss
            if isinstance(criterion, CombinedLoss):
                loss, loss_components = criterion(pred_dmaps, gt_dmaps)
                loss = loss / accumulation_steps
                
                # Track loss components
                epoch_metrics['total_loss'] += loss_components['total'] / accumulation_steps
                epoch_metrics['mse_loss'] += loss_components['mse'] / accumulation_steps
                epoch_metrics['focal_loss'] += loss_components['focal'] / accumulation_steps
                epoch_metrics['count_loss'] += loss_components['count'] / accumulation_steps
            else:
                loss = criterion(pred_dmaps, gt_dmaps) / accumulation_steps
                epoch_metrics['total_loss'] += loss.item()
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_advanced_metrics(pred_dmaps, gt_dmaps)
                epoch_metrics['mae'] += metrics['mae']
            
            # Clear intermediate tensors
            del images, gt_dmaps, pred_dmaps, loss
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Loss': f"{epoch_metrics['total_loss']/(batch_idx+1):.4f}",
                    'MAE': f"{epoch_metrics['mae']/(batch_idx+1):.2f}"
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
    
    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_metrics

def validate_epoch_enhanced(model, val_loader, criterion, device):
    """Enhanced validation with detailed metrics"""
    model.eval()
    epoch_metrics = {
        'total_loss': 0,
        'mae': 0,
        'mse': 0,
        'mape': 0,
        'acc_1': 0,
        'acc_2': 0,
        'acc_5': 0
    }
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_idx, (images, gt_dmaps) in enumerate(pbar):
            try:
                # Memory optimization
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
                
                # Calculate loss
                if isinstance(criterion, CombinedLoss):
                    loss, _ = criterion(pred_dmaps, gt_dmaps)
                else:
                    loss = criterion(pred_dmaps, gt_dmaps)
                
                epoch_metrics['total_loss'] += loss.item()
                
                # Calculate detailed metrics
                metrics = calculate_advanced_metrics(pred_dmaps, gt_dmaps)
                epoch_metrics['mae'] += metrics['mae']
                epoch_metrics['mse'] += metrics['mse']
                epoch_metrics['mape'] += metrics['mape']
                epoch_metrics['acc_1'] += metrics['acc_1']
                epoch_metrics['acc_2'] += metrics['acc_2']
                epoch_metrics['acc_5'] += metrics['acc_5']
                
                del images, gt_dmaps, pred_dmaps, loss
                
                if batch_idx % 20 == 0:
                    pbar.set_postfix({
                        'Loss': f"{epoch_metrics['total_loss']/(batch_idx+1):.4f}",
                        'MAE': f"{epoch_metrics['mae']/(batch_idx+1):.2f}"
                    })
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in validation, skipping batch...")
                    clear_memory()
                    continue
                else:
                    raise e
    
    clear_memory()
    
    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    # Calculate accuracy percentages
    epoch_metrics['acc_1'] *= 100
    epoch_metrics['acc_2'] *= 100
    epoch_metrics['acc_5'] *= 100
    epoch_metrics['rmse'] = np.sqrt(epoch_metrics['mse'])
    
    return epoch_metrics

def main():
    """Enhanced training with advanced techniques"""
    config = {
        'batch_size': 1,
        'learning_rate': 5e-6,  # Lower learning rate for stability
        'num_epochs': 150,  # More epochs with better early stopping
        'gt_downsample': 4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 25,  # More patience
        'lr_patience': 12,
        'save_every': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'accumulation_steps': 8,  # Larger effective batch size
        'use_enhanced_densitymaps': True,  # Use enhanced density maps
        'loss_type': 'combined'  # 'mse', 'focal', 'count', 'combined'
    }
    
    print("=== Enhanced MCNN Pellet Counter Training ===")
    print(f"Device: {config['device']}")
    print(f"Configuration: {config}")
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    clear_memory()
    
    # Data paths - use enhanced density maps if available
    data_root = "./data"
    densitymap_suffix = "_enhanced" if config['use_enhanced_densitymaps'] else ""
    
    train_img_root = os.path.join(data_root, "train_data", "images")
    train_gt_root = os.path.join(data_root, "train_data", f"densitymaps{densitymap_suffix}")
    test_img_root = os.path.join(data_root, "test_data", "images")
    test_gt_root = os.path.join(data_root, "test_data", f"densitymaps{densitymap_suffix}")
    
    # Check if enhanced density maps exist
    if config['use_enhanced_densitymaps'] and not os.path.exists(train_gt_root):
        print(f"Enhanced density maps not found at {train_gt_root}")
        print("Using regular density maps instead...")
        train_gt_root = os.path.join(data_root, "train_data", "densitymaps")
        test_gt_root = os.path.join(data_root, "test_data", "densitymaps")
    
    val_img_root = test_img_root
    val_gt_root = test_gt_root
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_img_root, train_gt_root,
        val_img_root, val_gt_root,
        test_img_root, test_gt_root,
        batch_size=config['batch_size'],
        gt_downsample=config['gt_downsample'],
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device(config['device'])
    model = MCNN().to(device)
    
    # Advanced loss function
    if config['loss_type'] == 'combined':
        criterion = CombinedLoss(mse_weight=1.0, count_weight=25.0, focal_weight=0.5)
        print("Using Combined Loss (MSE + Count + Focal)")
    elif config['loss_type'] == 'count':
        criterion = CountLoss(mse_weight=1.0, count_weight=20.0)
        print("Using Count Loss")
    elif config['loss_type'] == 'focal':
        criterion = FocalMSELoss(alpha=1.0, gamma=2.0, reduction='sum')
        print("Using Focal MSE Loss")
    else:
        criterion = nn.MSELoss(reduction='sum')
        print("Using Standard MSE Loss")
    
    # Advanced optimizer with better parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['lr_patience'],
        min_lr=1e-8, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], min_delta=0.1)
    
    # Create output directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Training history
    train_metrics_history = []
    val_metrics_history = []
    best_mae = float('inf')
    
    print("\nStarting enhanced training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch_enhanced(
            model, train_loader, criterion, optimizer, device, 
            accumulation_steps=config['accumulation_steps']
        )
        train_metrics_history.append(train_metrics)
        
        # Validate
        val_metrics = validate_epoch_enhanced(model, val_loader, criterion, device)
        val_metrics_history.append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics['mae'])
        
        # Print detailed results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, Train MAE: {train_metrics['mae']:.2f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}, Val MAE: {val_metrics['mae']:.2f}")
        print(f"Val RMSE: {val_metrics['rmse']:.2f}, Val MAPE: {val_metrics['mape']:.2f}%")
        print(f"Accuracy (MAE≤1): {val_metrics['acc_1']:.1f}%, (MAE≤2): {val_metrics['acc_2']:.1f}%, (MAE≤5): {val_metrics['acc_5']:.1f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory: {memory_used:.2f} GB")
        
        # Save best model
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'config': config
            }, './checkpoints/best_model_enhanced.pth')
            print(f"🎉 New best model saved! MAE: {best_mae:.2f}")
        
        # Early stopping
        if early_stopping(val_metrics['mae'], model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        clear_memory()
    
    # Save final results
    results = {
        'config': config,
        'train_metrics_history': train_metrics_history,
        'val_metrics_history': val_metrics_history,
        'best_mae': best_mae,
        'final_epoch': epoch + 1
    }
    
    with open('./results/enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Enhanced training completed!")
    print(f"Best validation MAE: {best_mae:.2f}")
    print(f"Model saved as: ./checkpoints/best_model_enhanced.pth")

if __name__ == "__main__":
    main()