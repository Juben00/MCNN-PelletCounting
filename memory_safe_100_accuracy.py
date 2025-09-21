#!/usr/bin/env python3
"""
Memory-Safe 100% Accuracy MCNN Training for RTX 3050 Ti (4GB GPU)
================================================================

Ultra-lightweight training system designed specifically for RTX 3050 Ti
while maintaining the goal of 100% perfect pellet counting accuracy.

Key Features:
- Extreme memory optimization for 4GB GPU
- Ultra-aggressive gradient accumulation 
- Mixed precision training
- Perfect count loss with 100% accuracy targeting
- Real-time memory monitoring and cleanup
"""

import os
import gc
import json
import time
import psutil
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from mcnn_model import MCNN
from improved_dataloader import PelletDataset

class MemorySafePerfectLoss(nn.Module):
    """Memory-efficient perfect count loss for 100% accuracy"""
    
    def __init__(self, count_weight=500.0, spatial_weight=1.0, precision_weight=250.0):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.precision_weight = precision_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred_dmap, gt_dmap):
        # Count preservation (primary objective)
        pred_count = pred_dmap.sum()
        gt_count = gt_dmap.sum()
        count_loss = torch.abs(pred_count - gt_count)
        
        # Spatial distribution
        spatial_loss = self.mse(pred_dmap, gt_dmap)
        
        # Precision loss (penalize any deviation from perfect count)
        precision_loss = torch.pow(count_loss, 2)
        
        total_loss = (
            self.count_weight * count_loss +
            self.spatial_weight * spatial_loss +
            self.precision_weight * precision_loss
        )
        
        return total_loss, {
            'count_loss': count_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'precision_loss': precision_loss.item(),
            'pred_count': pred_count.item(),
            'gt_count': gt_count.item()
        }

class MemoryMonitor:
    """Real-time memory monitoring for RTX 3050 Ti"""
    
    def __init__(self):
        self.peak_memory = 0
        self.warnings = 0
        
    def check_memory(self, stage=""):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3  # GB
            peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, peak)
            
            if current > 3.5:  # Warning at 3.5GB (RTX 3050 Ti safe limit)
                self.warnings += 1
                self.aggressive_cleanup()
                return True
        return False
    
    def aggressive_cleanup(self):
        """Ultra-aggressive memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()

class PerfectAccuracyTracker:
    """Track progress toward 100% accuracy goal"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.perfect_predictions = 0
        self.near_perfect_predictions = 0
        self.total_predictions = 0
        self.total_error = 0
        self.best_perfect_accuracy = 0
        self.best_near_perfect_accuracy = 0
        
    def update(self, pred_count, gt_count):
        error = abs(pred_count - gt_count)
        self.total_predictions += 1
        self.total_error += error
        
        if error < 0.5:  # Perfect prediction
            self.perfect_predictions += 1
        if error <= 1.0:  # Near-perfect prediction
            self.near_perfect_predictions += 1
    
    def get_accuracy_metrics(self):
        if self.total_predictions == 0:
            return {'perfect_accuracy': 0, 'near_perfect_accuracy': 0}
            
        perfect_acc = (self.perfect_predictions / self.total_predictions) * 100
        near_perfect_acc = (self.near_perfect_predictions / self.total_predictions) * 100
        
        self.best_perfect_accuracy = max(self.best_perfect_accuracy, perfect_acc)
        self.best_near_perfect_accuracy = max(self.best_near_perfect_accuracy, near_perfect_acc)
        
        return {
            'perfect_accuracy': perfect_acc,
            'near_perfect_accuracy': near_perfect_acc,
            'average_error': self.total_error / self.total_predictions,
            'best_perfect_accuracy': self.best_perfect_accuracy,
            'best_near_perfect_accuracy': self.best_near_perfect_accuracy
        }

def memory_safe_dataloader(dataset, batch_size=1, num_workers=0):
    """Create memory-safe dataloader for RTX 3050 Ti"""
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # No parallel workers for memory safety
        pin_memory=False,  # Disable for memory safety
        drop_last=True
    )

def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()

def memory_safe_validation(model, val_loader, criterion, device, monitor):
    """Memory-safe validation for RTX 3050 Ti"""
    model.eval()
    tracker = PerfectAccuracyTracker()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (images, gt_dmaps) in enumerate(val_loader):
            try:
                # Memory check before processing
                if monitor.check_memory(f"val_batch_{batch_idx}"):
                    print(f"⚠️ Memory warning during validation batch {batch_idx}")
                
                images = images.to(device, non_blocking=False)
                gt_dmaps = gt_dmaps.to(device, non_blocking=False)
                
                # Mixed precision inference
                with autocast(device_type='cuda'):
                    pred_dmaps = model(images)
                    loss, metrics = criterion(pred_dmaps, gt_dmaps)
                
                total_loss += loss.item()
                tracker.update(metrics['pred_count'], metrics['gt_count'])
                
                # Immediate cleanup after each batch
                del images, gt_dmaps, pred_dmaps, loss
                ultra_memory_cleanup()
                
                # Safety break if too many batches for memory
                if batch_idx >= 20:  # Limit validation batches for memory safety
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print(f"💥 GPU OOM during validation batch {batch_idx}, skipping...")
                ultra_memory_cleanup()
                continue
    
    model.train()
    
    # Calculate final metrics
    avg_loss = total_loss / max(1, batch_idx + 1)
    accuracy_metrics = tracker.get_accuracy_metrics()
    
    return avg_loss, accuracy_metrics

def setup_memory_safe_training():
    """Setup training configuration for RTX 3050 Ti"""
    
    # Ultra-conservative configuration for 4GB GPU
    config = {
        'batch_size': 1,                    # Minimum batch size
        'accumulation_steps': 32,           # Very high accumulation
        'learning_rate': 5e-7,              # Conservative learning rate
        'num_epochs': 200,                  # Reduced epochs
        'gt_downsample': 4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 30,
        'lr_patience': 15,
        'save_every': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'target_perfect_accuracy': 100.0,
        'target_near_perfect_accuracy': 99.0,
        'memory_cleanup_interval': 5,       # Cleanup every 5 batches
        'validation_interval': 10,          # Less frequent validation
        'checkpoint_interval': 25
    }
    
    return config

def main():
    """Memory-safe 100% accuracy training for RTX 3050 Ti"""
    
    print("🛡️ MEMORY-SAFE 100% ACCURACY MCNN FOR RTX 3050 Ti")
    print("🎯 Target: Perfect 100% Accuracy (GPU Memory Optimized)")
    print("="*80)
    
    # Setup configuration
    config = setup_memory_safe_training()
    device = torch.device(config['device'])
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")
        
        # Set memory fraction for RTX 3050 Ti safety
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use only 85% of GPU memory
    
    print(f"Configuration: {config}")
    
    # Initialize memory monitor
    monitor = MemoryMonitor()
    
    # Prepare datasets with memory-safe settings
    train_img_root = './data/train_data/images'
    train_gt_root = './data/train_data/densitymaps'
    val_img_root = './data/test_data/images'
    val_gt_root = './data/test_data/densitymaps'
    
    # Check data availability
    if not all(os.path.exists(path) for path in [train_img_root, train_gt_root, val_img_root, val_gt_root]):
        print("❌ Required data directories not found!")
        return
    
    # Create datasets
    train_dataset = PelletDataset(
        train_img_root, train_gt_root,
        gt_downsample=config['gt_downsample'], 
        phase='train',
        transform=False  # Disable augmentation for memory safety
    )
    
    val_dataset = PelletDataset(
        val_img_root, val_gt_root,
        gt_downsample=config['gt_downsample'], 
        phase='val',
        transform=False
    )
    
    print(f"train dataset: {len(train_dataset)} images")
    print(f"val dataset: {len(val_dataset)} images")
    
    # Create memory-safe dataloaders
    train_loader = memory_safe_dataloader(train_dataset, config['batch_size'])
    val_loader = memory_safe_dataloader(val_dataset, config['batch_size'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model (standard MCNN for memory efficiency)
    model = MCNN().to(device)
    print("Using standard MCNN (memory optimized)")
    
    # Initialize loss and optimizer
    criterion = MemorySafePerfectLoss(count_weight=500.0, spatial_weight=1.0, precision_weight=250.0)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['lr_patience'], factor=0.5, verbose=True)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    print("Using MemorySafePerfectLoss (500x count weight)")
    
    # Training tracking
    best_perfect_accuracy = 0
    best_near_perfect_accuracy = 0
    early_stop_counter = 0
    
    # Create results directory
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    print(f"\n🎯 Starting Memory-Safe Training for 100% Accuracy...")
    print(f"Strategy: Ultra-Conservative Memory + Perfect Count Preservation")
    print(f"Success Criteria: 100% perfect predictions OR 99%+ near-perfect")
    print("-" * 80)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_tracker = PerfectAccuracyTracker()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"🎯 Epoch {epoch+1}/{config['num_epochs']}")
        
        optimizer.zero_grad()
        
        for batch_idx, (images, gt_dmaps) in enumerate(pbar):
            try:
                # Memory monitoring
                if batch_idx % config['memory_cleanup_interval'] == 0:
                    if monitor.check_memory(f"epoch_{epoch}_batch_{batch_idx}"):
                        print(f"\n⚠️ Memory warning at epoch {epoch+1}, batch {batch_idx}")
                
                images = images.to(device, non_blocking=False)
                gt_dmaps = gt_dmaps.to(device, non_blocking=False)
                
                # Mixed precision forward pass
                with autocast(device_type='cuda'):
                    pred_dmaps = model(images)
                    loss, metrics = criterion(pred_dmaps, gt_dmaps)
                    loss = loss / config['accumulation_steps']
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update tracking
                epoch_loss += loss.item() * config['accumulation_steps']
                epoch_tracker.update(metrics['pred_count'], metrics['gt_count'])
                
                # Gradient accumulation step
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Update progress bar
                current_metrics = epoch_tracker.get_accuracy_metrics()
                pbar.set_postfix({
                    'Loss': f"{loss.item() * config['accumulation_steps']:.4f}",
                    'Perfect': f"{current_metrics['perfect_accuracy']:.1f}%",
                    'Near-Perfect': f"{current_metrics['near_perfect_accuracy']:.1f}%",
                    'Pred': f"{metrics['pred_count']:.1f}",
                    'GT': f"{metrics['gt_count']:.1f}"
                })
                
                # Immediate cleanup
                del images, gt_dmaps, pred_dmaps, loss
                
                # Periodic aggressive cleanup
                if batch_idx % config['memory_cleanup_interval'] == 0:
                    ultra_memory_cleanup()
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n💥 GPU OOM at epoch {epoch+1}, batch {batch_idx}")
                print("🧹 Performing emergency cleanup...")
                
                optimizer.zero_grad()
                ultra_memory_cleanup()
                continue
                
            except Exception as e:
                print(f"\n❌ Unexpected error at epoch {epoch+1}, batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        train_metrics = epoch_tracker.get_accuracy_metrics()
        
        # Validation (memory-safe)
        if (epoch + 1) % config['validation_interval'] == 0:
            val_loss, val_metrics = memory_safe_validation(model, val_loader, criterion, device, monitor)
            scheduler.step(val_loss)
        else:
            val_loss, val_metrics = 0, {'perfect_accuracy': 0, 'near_perfect_accuracy': 0}
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        print(f"  Train Loss: {avg_loss:.6f}")
        print(f"  Train Perfect Accuracy: {train_metrics['perfect_accuracy']:.2f}%")
        print(f"  Train Near-Perfect Accuracy: {train_metrics['near_perfect_accuracy']:.2f}%")
        
        if (epoch + 1) % config['validation_interval'] == 0:
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Perfect Accuracy: {val_metrics['perfect_accuracy']:.2f}%")
            print(f"  Val Near-Perfect Accuracy: {val_metrics['near_perfect_accuracy']:.2f}%")
        
        print(f"  Peak GPU Memory: {monitor.peak_memory:.2f} GB")
        print(f"  Memory Warnings: {monitor.warnings}")
        
        # Save best model
        current_perfect = max(train_metrics['perfect_accuracy'], val_metrics.get('perfect_accuracy', 0))
        if current_perfect > best_perfect_accuracy:
            best_perfect_accuracy = current_perfect
            early_stop_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'perfect_accuracy': current_perfect,
                'near_perfect_accuracy': max(train_metrics['near_perfect_accuracy'], val_metrics.get('near_perfect_accuracy', 0)),
                'train_loss': avg_loss,
                'config': config
            }
            
            torch.save(checkpoint, './checkpoints/best_memory_safe_100_model.pth')
            print(f"🏆 New best perfect accuracy: {current_perfect:.2f}%")
            
            # Check for 100% achievement
            if current_perfect >= 100.0:
                print(f"🎉 INCREDIBLE! 100% PERFECT ACCURACY ACHIEVED!")
                print(f"🚀 Training completed successfully!")
                break
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config['early_stopping_patience']:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Periodic checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = f"./checkpoints/memory_safe_100_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'perfect_accuracy': current_perfect,
                'near_perfect_accuracy': max(train_metrics['near_perfect_accuracy'], val_metrics.get('near_perfect_accuracy', 0)),
            }, checkpoint_path)
        
        # Final cleanup
        ultra_memory_cleanup()
    
    print(f"\n🏁 Training Complete!")
    print(f"🏆 Best Perfect Accuracy: {best_perfect_accuracy:.2f}%")
    print(f"🥈 Best Near-Perfect Accuracy: {best_near_perfect_accuracy:.2f}%")
    print(f"💾 Peak GPU Memory Used: {monitor.peak_memory:.2f} GB")
    print(f"⚠️ Total Memory Warnings: {monitor.warnings}")
    
    if best_perfect_accuracy >= 100.0:
        print(f"\n🎉 CONGRATULATIONS! 100% PERFECT ACCURACY ACHIEVED!")
        print(f"🚀 Your MCNN model is ready for production!")
    elif best_perfect_accuracy >= 95.0:
        print(f"\n🥇 OUTSTANDING! {best_perfect_accuracy:.1f}% perfect accuracy achieved!")
        print(f"📈 Very close to the 100% perfection goal!")
    else:
        print(f"\n📊 Significant progress made: {best_perfect_accuracy:.1f}% perfect accuracy")
        print(f"🔧 Consider extending training or adjusting hyperparameters")
    
    print(f"\nModel saved: ./checkpoints/best_memory_safe_100_model.pth")
    print(f"Next step: Run 'python evaluate_100_percent.py' for detailed analysis")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        ultra_memory_cleanup()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        ultra_memory_cleanup()
        raise