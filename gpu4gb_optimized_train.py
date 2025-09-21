#!/usr/bin/env python3
"""
GPU-Optimized MCNN Training for RTX 3050 Ti (4GB)
==================================================

Specifically optimized for your RTX 3050 Ti with 4GB memory while still
targeting 90% accuracy through smart optimizations and memory management.
"""

import os
import gc
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from mcnn_model import MCNN  # Use original MCNN to save memory
from improved_dataloader import PelletDataset, create_data_loaders

class MemoryEfficientCountLoss(nn.Module):
    """Memory-efficient count-focused loss for 4GB GPU"""
    def __init__(self, count_weight=100.0, focal_gamma=1.5):
        super(MemoryEfficientCountLoss, self).__init__()
        self.count_weight = count_weight
        self.focal_gamma = focal_gamma
    
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = F.mse_loss(pred, target, reduction='mean')
        
        # Count preservation (most important for accuracy)
        pred_count = pred.sum()
        target_count = target.sum()
        count_loss = F.mse_loss(pred_count, target_count)
        
        # Light focal weighting (memory efficient)
        if self.focal_gamma > 0:
            focal_weight = (1 + mse_loss).pow(self.focal_gamma)
            mse_loss = mse_loss * focal_weight
        
        total_loss = mse_loss + self.count_weight * count_loss
        return total_loss

class GPU4GBAccuracyTracker:
    """Lightweight accuracy tracker for 4GB GPU"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total = 0
        self.acc_1 = 0
        self.acc_2 = 0
        self.acc_5 = 0
        self.mae_sum = 0
        
    def update(self, pred_count, gt_count):
        self.total += 1
        mae = abs(pred_count - gt_count)
        self.mae_sum += mae
        
        if mae <= 1: self.acc_1 += 1
        if mae <= 2: self.acc_2 += 1
        if mae <= 5: self.acc_5 += 1
    
    def get_metrics(self):
        if self.total == 0:
            return {'acc_1': 0, 'acc_2': 0, 'acc_5': 0, 'mae': 0}
        
        return {
            'acc_1': (self.acc_1 / self.total) * 100,
            'acc_2': (self.acc_2 / self.total) * 100,
            'acc_5': (self.acc_5 / self.total) * 100,
            'mae': self.mae_sum / self.total
        }

def aggressive_memory_cleanup():
    """Aggressive memory cleanup for 4GB GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def train_epoch_gpu4gb(model, train_loader, criterion, optimizer, device, accumulation_steps=16):
    """Memory-optimized training epoch for 4GB GPU"""
    model.train()
    epoch_loss = 0
    tracker = GPU4GBAccuracyTracker()
    
    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, gt_dmaps) in enumerate(progress_bar):
        try:
            # Move to GPU with minimal memory footprint
            images = images.to(device, non_blocking=True, dtype=torch.float16)  # Use half precision
            gt_dmaps = gt_dmaps.to(device, non_blocking=True, dtype=torch.float16)
            
            # Forward pass
            with torch.cuda.amp.autocast():  # Mixed precision
                pred_dmaps = model(images.float())  # Model needs float32
                pred_dmaps = pred_dmaps.half()  # Convert back to half
                loss = criterion(pred_dmaps.float(), gt_dmaps.float())
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Calculate metrics (move to CPU to save GPU memory)
            with torch.no_grad():
                pred_count = pred_dmaps.float().sum().item()
                gt_count = gt_dmaps.float().sum().item()
                tracker.update(pred_count, gt_count)
            
            epoch_loss += loss.item() * accumulation_steps
            
            # Gradient accumulation with more frequent cleanup
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                aggressive_memory_cleanup()
            
            # More frequent memory cleanup
            if batch_idx % 4 == 0:
                aggressive_memory_cleanup()
            
            # Update progress with current metrics
            current_metrics = tracker.get_metrics()
            progress_bar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'MAE': f"{current_metrics['mae']:.2f}",
                'Acc2': f"{current_metrics['acc_2']:.1f}%"
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ GPU memory error at batch {batch_idx}, cleaning up...")
                aggressive_memory_cleanup()
                continue
            else:
                raise e
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss /= len(train_loader)
    epoch_metrics = tracker.get_metrics()
    
    return epoch_loss, epoch_metrics

def validate_epoch_gpu4gb(model, val_loader, criterion, device):
    """Memory-optimized validation for 4GB GPU"""
    model.eval()
    val_loss = 0
    tracker = GPU4GBAccuracyTracker()
    
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, gt_dmaps) in enumerate(progress_bar):
            try:
                # Use half precision for memory efficiency
                images = images.to(device, non_blocking=True, dtype=torch.float16)
                gt_dmaps = gt_dmaps.to(device, non_blocking=True, dtype=torch.float16)
                
                with torch.cuda.amp.autocast():
                    pred_dmaps = model(images.float())
                    pred_dmaps = pred_dmaps.half()
                    loss = criterion(pred_dmaps.float(), gt_dmaps.float())
                
                # Calculate metrics
                pred_count = pred_dmaps.float().sum().item()
                gt_count = gt_dmaps.float().sum().item()
                tracker.update(pred_count, gt_count)
                
                val_loss += loss.item()
                
                # Memory cleanup
                if batch_idx % 8 == 0:
                    aggressive_memory_cleanup()
                
                # Update progress
                current_metrics = tracker.get_metrics()
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'MAE': f"{current_metrics['mae']:.2f}",
                    'Acc2': f"{current_metrics['acc_2']:.1f}%"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ GPU memory error during validation, cleaning up...")
                    aggressive_memory_cleanup()
                    continue
                else:
                    raise e
    
    val_loss /= len(val_loader)
    val_metrics = tracker.get_metrics()
    
    return val_loss, val_metrics

def main():
    print("🚀 GPU-Optimized MCNN Training for RTX 3050 Ti (4GB)")
    print("🎯 Target: 90% Accuracy with Memory Optimization")
    print("=" * 70)
    
    # GPU-optimized configuration
    config = {
        'batch_size': 1,
        'learning_rate': 2e-6,  # Lower for stability
        'num_epochs': 150,
        'gt_downsample': 4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 25,
        'lr_patience': 10,
        'save_every': 15,
        'device': 'cuda',
        'accumulation_steps': 16,  # Higher accumulation for 4GB
        'target_accuracy': 90.0
    }
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
        
        # Optimize GPU settings for 4GB
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    print(f"Configuration: {config}")
    
    # Data setup
    train_img_root = './data/train_data/images'
    train_gt_root = './data/train_data/densitymaps'
    val_img_root = './data/test_data/images'
    val_gt_root = './data/test_data/densitymaps'
    test_img_root = './data/test_data/images'
    test_gt_root = './data/test_data/densitymaps'
    
    # Create data loaders with memory optimization
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_img_root, train_gt_root,
        val_img_root, val_gt_root,
        test_img_root, test_gt_root,
        batch_size=config['batch_size'],
        gt_downsample=config['gt_downsample'],
        num_workers=0  # Reduce workers for memory
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Use original MCNN to save memory
    model = MCNN().to(device)
    print(f"Using Original MCNN (memory optimized)")
    
    # Memory-efficient loss
    criterion = MemoryEfficientCountLoss(count_weight=150.0).to(device)
    print(f"Using Memory-Efficient Count Loss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.8,
        patience=config['lr_patience'],
        min_lr=1e-8
    )
    
    # Create directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_2': [], 'val_acc_2': [],
        'train_mae': [], 'val_mae': []
    }
    
    print(f"\n🎯 Starting GPU-Optimized Training for 90% Accuracy...")
    print(f"Memory Strategy: Half precision + Gradient accumulation")
    print("-" * 70)
    
    best_accuracy = 0.0
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Training
        train_loss, train_metrics = train_epoch_gpu4gb(
            model, train_loader, criterion, optimizer, device,
            config['accumulation_steps']
        )
        
        aggressive_memory_cleanup()
        
        # Validation
        val_loss, val_metrics = validate_epoch_gpu4gb(
            model, val_loader, criterion, device
        )
        
        aggressive_memory_cleanup()
        
        # Update scheduler
        scheduler.step(val_metrics['acc_2'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_metrics['mae']:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_metrics['mae']:.2f}")
        print(f"Train Acc (±2): {train_metrics['acc_2']:.1f}%")
        print(f"Val Acc (±2): {val_metrics['acc_2']:.1f}%")
        print(f"Val Acc (±1): {val_metrics['acc_1']:.1f}%, (±5): {val_metrics['acc_5']:.1f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Check for improvement
        if val_metrics['acc_2'] > best_accuracy:
            best_accuracy = val_metrics['acc_2']
            best_mae = val_metrics['mae']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'best_mae': best_mae,
                'config': config
            }, './checkpoints/best_gpu4gb_model.pth')
            
            print(f"🎉 New best model! Accuracy: {best_accuracy:.1f}%, MAE: {best_mae:.2f}")
        else:
            patience_counter += 1
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_2'].append(train_metrics['acc_2'])
        history['val_acc_2'].append(val_metrics['acc_2'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        
        # Check if target reached
        if val_metrics['acc_2'] >= config['target_accuracy']:
            print(f"\n🎯 TARGET ACHIEVED! {val_metrics['acc_2']:.1f}% >= {config['target_accuracy']}%")
            break
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_metrics['acc_2'],
                'mae': val_metrics['mae']
            }, f'./checkpoints/gpu4gb_epoch_{epoch+1}.pth')
    
    # Save final results
    results = {
        'config': config,
        'history': history,
        'best_accuracy': best_accuracy,
        'best_mae': best_mae,
        'final_epoch': epoch + 1,
        'target_achieved': best_accuracy >= config['target_accuracy']
    }
    
    with open('./results/gpu4gb_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 TRAINING COMPLETED!")
    print(f"Best Accuracy: {best_accuracy:.1f}%")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Target Achieved: {'✅ YES' if best_accuracy >= config['target_accuracy'] else '❌ NO'}")
    print(f"Total Epochs: {epoch + 1}")
    
    if best_accuracy >= config['target_accuracy']:
        print(f"\n🎉 SUCCESS! 90% accuracy achieved on RTX 3050 Ti!")
    else:
        print(f"\n📈 Progress: {best_accuracy:.1f}% accuracy achieved")
        print(f"Gap to target: {config['target_accuracy'] - best_accuracy:.1f}%")

if __name__ == "__main__":
    main()