#!/usr/bin/env python3
"""
Ultra-Minimal 4GB GPU Trainer for 100% Pellet Counting Accuracy
===============================================================

Specifically designed for RTX 3050 Ti 4GB GPU with extreme memory constraints.
This trainer sacrifices speed for memory efficiency to achieve 100% accuracy.
"""

import os
import json
import numpy as np
from datetime import datetime
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from mcnn_model import MCNN
from dataloader import CrowdDataset
from config import Config

# Force aggressive memory management
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class UltraMinimalLoss(nn.Module):
    """Ultra-simple loss for minimal memory usage"""
    
    def __init__(self, count_weight=100.0):
        super(UltraMinimalLoss, self).__init__()
        self.count_weight = count_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred_dmap, gt_dmap):
        # Simple MSE + count preservation
        mse_loss = self.mse(pred_dmap, gt_dmap)
        
        # Count loss (minimal computation)
        pred_count = pred_dmap.sum()
        gt_count = gt_dmap.sum()
        count_loss = torch.abs(pred_count - gt_count)
        
        return mse_loss + self.count_weight * count_loss

def extreme_memory_cleanup():
    """Most aggressive memory cleanup possible"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()

def mini_batch_training_loop(model, dataloader, criterion, optimizer, scaler, device, config, accumulation_steps=64):
    """Ultra-memory-efficient training loop"""
    
    model.train()
    total_loss = 0
    perfect_predictions = 0
    total_predictions = 0
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Process in tiny batches with massive accumulation
    pbar = tqdm(dataloader, desc="Ultra-Minimal Training")
    
    for batch_idx, (img, gt_dmap) in enumerate(pbar):
        try:
            # Move to GPU
            img = img.to(device, non_blocking=True)
            gt_dmap = gt_dmap.to(device, non_blocking=True)
            
            # Forward pass with autocast
            with autocast():
                pred_dmap = model(img)
                loss = criterion(pred_dmap, gt_dmap) / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Calculate accuracy (minimal computation)
            with torch.no_grad():
                pred_count = pred_dmap.sum().item()
                gt_count = gt_dmap.sum().item()
                if abs(pred_count - gt_count) < 0.5:
                    perfect_predictions += 1
                total_predictions += 1
            
            total_loss += loss.item() * accumulation_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Apply gradient clipping from config.py
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Aggressive cleanup every step
                extreme_memory_cleanup()
            
            # Update progress
            perfect_acc = (perfect_predictions / total_predictions) * 100
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.2f}',
                'Perfect': f'{perfect_acc:.1f}%',
                'Pred': f'{pred_count:.0f}',
                'GT': f'{gt_count:.0f}'
            })
            
            # Immediate cleanup after each batch
            del img, gt_dmap, pred_dmap, loss
            extreme_memory_cleanup()
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n💥 OOM at batch {batch_idx} - emergency cleanup...")
            extreme_memory_cleanup()
            continue
            
        except Exception as e:
            print(f"\n❌ Error at batch {batch_idx}: {e}")
            extreme_memory_cleanup()
            continue
    
    # Final gradient step if needed
    if len(dataloader) % accumulation_steps != 0:
        # Apply gradient clipping from config.py
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    perfect_accuracy = (perfect_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    return avg_loss, perfect_accuracy

def ultra_minimal_validation(model, dataloader, criterion, device):
    """Ultra-minimal validation to conserve memory"""
    
    model.eval()
    total_loss = 0
    perfect_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (img, gt_dmap) in enumerate(tqdm(dataloader, desc="Validation")):
            try:
                img = img.to(device, non_blocking=True)
                gt_dmap = gt_dmap.to(device, non_blocking=True)
                
                with autocast():
                    pred_dmap = model(img)
                    loss = criterion(pred_dmap, gt_dmap)
                
                # Count accuracy
                pred_count = pred_dmap.sum().item()
                gt_count = gt_dmap.sum().item()
                if abs(pred_count - gt_count) < 0.5:
                    perfect_predictions += 1
                total_predictions += 1
                
                total_loss += loss.item()
                
                # Cleanup
                del img, gt_dmap, pred_dmap, loss
                extreme_memory_cleanup()
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in validation batch {batch_idx}")
                extreme_memory_cleanup()
                continue
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    perfect_accuracy = (perfect_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    return avg_loss, perfect_accuracy

def main():
    """Ultra-minimal main training function"""
    
    print("🔥 ULTRA-MINIMAL 4GB GPU TRAINER FOR 100% ACCURACY")
    print("⚡ Extreme Memory Optimization for RTX 3050 Ti")
    print("=" * 80)
    
    # Load configuration from Config class
    config_obj = Config()
    config = {
        'batch_size': config_obj.batch_size,  # 4 from config.py
        'learning_rate': config_obj.lr,       # 1e-5 from config.py
        'num_epochs': config_obj.epochs,      # 300 from config.py
        'weight_decay': config_obj.weight_decay,  # 1e-4 from config.py
        'gradient_clip_norm': config_obj.gradient_clip_norm,  # 0.5 from config.py
        'early_stopping_patience': config_obj.early_stopping_patience,  # 55 from config.py
        'lr_scheduler_patience': config_obj.lr_scheduler_patience,  # 12 from config.py
        'lr_scheduler_factor': config_obj.lr_scheduler_factor,  # 0.3 from config.py
        'min_lr': config_obj.min_lr,  # 1e-7 from config.py
        'device': config_obj.device,
        'save_every': 10,
        'validation_every': 5,
        'accumulation_steps': config_obj.batch_size * 16  # Adjust based on batch size
    }
    
    device = config_obj.device
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Set memory fraction to be very conservative
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    
    # Load data with minimal batch size
    train_img_root = './data/train_data/images'
    train_gt_root = './data/train_data/densitymaps'
    val_img_root = './data/test_data/images'
    val_gt_root = './data/test_data/densitymaps'
    
    try:
        train_dataset = CrowdDataset(train_img_root, train_gt_root, gt_downsample=4, phase='train')
        val_dataset = CrowdDataset(val_img_root, val_gt_root, gt_downsample=4, phase='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], 
            shuffle=True, num_workers=0, pin_memory=False  # No multiprocessing
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], 
            shuffle=False, num_workers=0, pin_memory=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return
    
    # Initialize model (standard MCNN - most memory efficient)
    model = MCNN().to(device)
    print("Using standard MCNN (ultra-minimal)")
    
    # Ultra-simple loss and optimizer with config.py parameters
    criterion = UltraMinimalLoss(count_weight=50.0)  # Lower weight to avoid huge numbers
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Add learning rate scheduler from config.py
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    scaler = GradScaler()
    
    print("Using UltraMinimalLoss (50x count weight)")
    
    # Training tracking
    best_perfect_accuracy = 0
    training_history = []
    epochs_without_improvement = 0
    
    # Create checkpoints directory
    os.makedirs('./checkpoints', exist_ok=True)
    
    print("\n🔥 Starting Ultra-Minimal Training for 100% Accuracy...")
    print("Strategy: Extreme Memory Conservation + Perfect Count Focus")
    print("-" * 80)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n🎯 Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 70)
        
        # Training
        try:
            train_loss, train_perfect_acc = mini_batch_training_loop(
                model, train_loader, criterion, optimizer, scaler, device, config,
                config['accumulation_steps']
            )
            
            print(f"Training - Loss: {train_loss:.6f}, Perfect Accuracy: {train_perfect_acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            extreme_memory_cleanup()
            continue
        
        # Validation
        if (epoch + 1) % config['validation_every'] == 0:
            try:
                val_loss, val_perfect_acc = ultra_minimal_validation(
                    model, val_loader, criterion, device
                )
                
                print(f"Validation - Loss: {val_loss:.6f}, Perfect Accuracy: {val_perfect_acc:.2f}%")
                
                # Update learning rate scheduler
                lr_scheduler.step(val_loss)
                
                # Save best model
                if val_perfect_acc > best_perfect_accuracy:
                    best_perfect_accuracy = val_perfect_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_perfect_accuracy': best_perfect_accuracy,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': config
                    }
                    
                    torch.save(checkpoint, './checkpoints/best_ultra_minimal_model.pth')
                    print(f"✅ New best model saved! Perfect Accuracy: {best_perfect_accuracy:.2f}%")
                
                # Record history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_perfect_acc': train_perfect_acc,
                    'val_loss': val_loss,
                    'val_perfect_acc': val_perfect_acc
                })
                
            except Exception as e:
                print(f"❌ Validation failed: {e}")
                extreme_memory_cleanup()
        
        # Early stopping check
        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered after {config['early_stopping_patience']} epochs without improvement")
            break
        
        # Periodic save
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = f'./checkpoints/ultra_minimal_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'perfect_accuracy': train_perfect_acc,
                'config': config
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Progress report
        if best_perfect_accuracy >= 100.0:
            print(f"🎉 INCREDIBLE! 100% PERFECT ACCURACY ACHIEVED!")
            break
        elif best_perfect_accuracy >= 95.0:
            print(f"🥇 OUTSTANDING! {best_perfect_accuracy:.1f}% perfect accuracy!")
        elif best_perfect_accuracy >= 90.0:
            print(f"🥈 EXCELLENT! {best_perfect_accuracy:.1f}% perfect accuracy!")
        
        # Cleanup after epoch
        extreme_memory_cleanup()
    
    # Save final results
    results = {
        'best_perfect_accuracy': best_perfect_accuracy,
        'training_history': training_history,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('./ultra_minimal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 ULTRA-MINIMAL TRAINING COMPLETE!")
    print(f"Best Perfect Accuracy: {best_perfect_accuracy:.2f}%")
    print(f"Results saved to: ultra_minimal_results.json")
    
    if best_perfect_accuracy >= 100.0:
        print(f"🎉 CONGRATULATIONS! 100% PERFECT ACCURACY ACHIEVED!")
    elif best_perfect_accuracy >= 95.0:
        print(f"🥇 OUTSTANDING PERFORMANCE! Close to perfection!")
    else:
        print(f"📈 Good progress! Continue training for higher accuracy.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        extreme_memory_cleanup()