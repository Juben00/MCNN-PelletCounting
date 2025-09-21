# enhanced_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

# Import our new model and the existing dataloader
from enhanced_mcnn_model import EnhancedMCNN
from dataloader import PelletDataset

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for img, gt_dmap in pbar:
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        
        # Forward pass
        pred_dmap = model(img)
        
        # Calculate loss
        loss = criterion(pred_dmap, gt_dmap)
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_mae = 0
    total_mse = 0
    perfect_predictions = 0
    
    with torch.no_grad():
        for img, gt_dmap in tqdm(val_loader, desc="Validating"):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            
            pred_dmap = model(img)
            
            # Calculate counts
            pred_count = pred_dmap.sum().item()
            gt_count = gt_dmap.sum().item()
            
            # Calculate metrics
            mae = abs(pred_count - gt_count)
            mse = (pred_count - gt_count)**2
            total_mae += mae
            total_mse += mse
            
            if abs(pred_count - gt_count) < 0.5: # Consider it perfect if error is less than 0.5
                perfect_predictions += 1

    avg_mae = total_mae / len(val_loader)
    avg_mse = np.sqrt(total_mse / len(val_loader))
    perfect_accuracy = (perfect_predictions / len(val_loader)) * 100
    
    return avg_mae, avg_mse, perfect_accuracy

def main():
    # --- Configuration ---
    config = {
        'batch_size': 8,          # Start with 8. If you get memory errors, try 4 or 2.
        'learning_rate': 1e-5,    # A better starting learning rate.
        'num_epochs': 200,        # Train for more epochs.
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 20,
        'gt_downsample': 4        # Match the downsampling of the model.
    }
    
    device = torch.device(config['device'])
    print(f"🚀 Using device: {device}")

    # --- Data Loading ---
    train_img_root = './data/train_data/images'
    train_gt_root = './data/train_data/densitymaps'
    val_img_root = './data/test_data/images'
    val_gt_root = './data/test_data/densitymaps'

    train_dataset = PelletDataset(train_img_root, train_gt_root, gt_downsample=config['gt_downsample'], phase='train')
    val_dataset = PelletDataset(val_img_root, val_gt_root, gt_downsample=config['gt_downsample'], phase='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True # Validate one by one
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model, Loss, Optimizer ---
    model = EnhancedMCNN().to(device)
    criterion = nn.MSELoss() # A standard and effective loss for density maps.
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    # Learning rate scheduler: reduces LR if performance plateaus.
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    os.makedirs('./checkpoints', exist_ok=True)
    best_mae = float('inf')

    # --- Training Loop ---
    print("\n🔥 Starting Enhanced Training...")
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} --- LR: {scheduler.get_last_lr()[0]:.1e}")

        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_mae, val_mse, perfect_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation MAE: {val_mae:.2f}")
        print(f"  Validation RMSE: {val_mse:.2f}")
        print(f"  Perfect Accuracy: {perfect_acc:.2f}%")

        # --- Save Best Model ---
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), './checkpoints/best_model.pth')
            print(f"✅ New best model saved! MAE: {best_mae:.2f}")
            
        # --- Periodic Save ---
        if (epoch + 1) % config['save_every'] == 0:
            torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch+1}.pth')
            print(f"💾 Checkpoint saved for epoch {epoch+1}")

    print("\n🏆 Training Complete!")
    print(f"Best Validation MAE: {best_mae:.2f}")

if __name__ == "__main__":
    main()