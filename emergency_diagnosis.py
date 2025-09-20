import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from mcnn_model import MCNN
from improved_dataloader import PelletDataset

def diagnose_critical_issues():
    """Diagnose the critical issues causing MAE = 436.62"""
    
    print("🚨 CRITICAL ISSUE DIAGNOSIS - MAE 436.62")
    print("=" * 60)
    
    issues_found = []
    
    # 1. Check model loading
    print("\n1. 🔍 Model Loading Check:")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MCNN().to(device)
        
        checkpoint_path = "./checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'best_mae' in checkpoint:
                training_mae = checkpoint['best_mae']
                print(f"   ✅ Model loaded. Training MAE was: {training_mae:.2f}")
                if training_mae > 100:
                    issues_found.append("Training MAE was already very high")
            else:
                print("   ⚠️  No training MAE recorded in checkpoint")
        else:
            print("   ❌ No checkpoint found")
            return
            
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return
    
    # 2. Check data loading
    print("\n2. 📊 Data Loading Check:")
    try:
        test_img_root = "./data/test_data/images"
        test_gt_root = "./data/test_data/densitymaps"
        
        if not os.path.exists(test_img_root):
            print(f"   ❌ Test images not found: {test_img_root}")
            issues_found.append("Test images directory missing")
            return
            
        if not os.path.exists(test_gt_root):
            print(f"   ❌ Test density maps not found: {test_gt_root}")
            issues_found.append("Test density maps directory missing")
            return
            
        dataset = PelletDataset(test_img_root, test_gt_root, gt_downsample=4, phase='test', transform=False)
        print(f"   ✅ Dataset loaded: {len(dataset)} images")
        
        # Check a sample
        if len(dataset) > 0:
            img, dmap = dataset[0]
            gt_count = dmap.sum().item()
            print(f"   Sample ground truth count: {gt_count:.1f}")
            
            if gt_count > 1000:
                issues_found.append("Ground truth counts are abnormally high")
            elif gt_count < 1:
                issues_found.append("Ground truth counts are abnormally low")
                
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        issues_found.append("Data loading error")
        return
    
    # 3. Check density map quality
    print("\n3. 🗺️  Density Map Quality Check:")
    try:
        # Load a few density maps directly
        dmap_files = [f for f in os.listdir(test_gt_root) if f.endswith('.npy')][:5]
        
        dmap_stats = []
        for dmap_file in dmap_files:
            dmap_path = os.path.join(test_gt_root, dmap_file)
            dmap = np.load(dmap_path)
            count = dmap.sum()
            dmap_stats.append(count)
            print(f"   {dmap_file}: sum={count:.1f}, shape={dmap.shape}, max={dmap.max():.3f}")
        
        avg_count = np.mean(dmap_stats)
        if avg_count > 500:
            issues_found.append("Density maps have abnormally high counts")
        elif avg_count < 1:
            issues_found.append("Density maps have abnormally low counts")
            
    except Exception as e:
        print(f"   ❌ Density map check failed: {e}")
    
    # 4. Test model prediction on one image
    print("\n4. 🔮 Model Prediction Test:")
    try:
        model.eval()
        with torch.no_grad():
            img_tensor, gt_dmap = dataset[0]
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            pred_dmap = model(img_batch)
            pred_count = pred_dmap.sum().item()
            gt_count = gt_dmap.sum().item()
            
            print(f"   Ground Truth: {gt_count:.1f}")
            print(f"   Predicted: {pred_count:.1f}")
            print(f"   Error: {abs(pred_count - gt_count):.1f}")
            
            if pred_count > gt_count * 5:
                issues_found.append("Model severely overestimating counts")
            elif pred_count < gt_count * 0.2:
                issues_found.append("Model severely underestimating counts")
                
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
    
    # 5. Check training configuration
    print("\n5. ⚙️  Training Configuration Check:")
    results_path = "./results/training_log.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                training_log = json.load(f)
            
            config = training_log.get('config', {})
            print(f"   Learning Rate: {config.get('learning_rate', 'Unknown')}")
            print(f"   Downsample Factor: {config.get('gt_downsample', 'Unknown')}")
            print(f"   Epochs: {len(training_log.get('val_metrics', []))}")
            
            val_metrics = training_log.get('val_metrics', [])
            if val_metrics:
                final_mae = val_metrics[-1].get('mae', 0) if isinstance(val_metrics[-1], dict) else val_metrics[-1]
                print(f"   Final Training MAE: {final_mae:.2f}")
                
                if final_mae > 100:
                    issues_found.append("Training never converged properly")
                    
        except Exception as e:
            print(f"   ⚠️  Could not read training log: {e}")
    else:
        print("   ⚠️  No training log found")
    
    return issues_found

def create_emergency_fixes(issues):
    """Create emergency fixes for the critical issues"""
    
    print(f"\n🚨 EMERGENCY FIXES NEEDED")
    print("=" * 60)
    
    if not issues:
        print("No specific issues detected, but MAE 436.62 indicates major problems.")
        issues = ["General model failure"]
    
    print("🔧 Critical Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n🎯 IMMEDIATE ACTION PLAN:")
    
    print("\n1. 🔴 CRITICAL: Check Your Density Maps")
    print("   Your model predicts 662.5 pellets on average vs GT 225.9")
    print("   This suggests density map generation issues.")
    print("   Commands to run:")
    print("   → cd data_preparation")
    print("   → python enhanced_dmap_generator.py")
    
    print("\n2. 🔴 CRITICAL: Retrain with Better Parameters")
    print("   Current model is completely broken. Need fresh training:")
    print("   → python enhanced_train.py")
    print("   (Uses better loss functions and parameters)")
    
    print("\n3. 🟡 HIGH: Check Annotation Quality")
    print("   GT counts range 5-494, which is very wide.")
    print("   Verify your annotations are correct.")
    
    print("\n4. 🟡 HIGH: Use Count-Focused Training")
    print("   Standard MSE loss is failing. Need count-specific loss.")
    
    print(f"\n📋 STEP-BY-STEP RECOVERY PLAN:")
    print("   Step 1: Generate enhanced density maps")
    print("   Step 2: Train with enhanced script (count-focused loss)")
    print("   Step 3: Test again - target MAE < 20")
    print("   Step 4: If still high, check annotation quality")

def quick_fix_config():
    """Generate a quick fix configuration"""
    
    print(f"\n⚡ QUICK FIX CONFIGURATION")
    print("=" * 40)
    
    quick_fix_code = '''
# Quick Fix Training Configuration
# Add this to your training script:

config = {
    'batch_size': 1,
    'learning_rate': 1e-6,  # MUCH lower learning rate
    'num_epochs': 200,      # More epochs
    'gt_downsample': 4,
    'weight_decay': 1e-3,   # Higher regularization
    'early_stopping_patience': 50,
    'count_loss_weight': 100.0,  # HEAVY count loss weighting
}

# Use this loss function:
class EmergencyCountLoss(nn.Module):
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction='sum')
        count_loss = F.mse_loss(pred.sum(), target.sum()) * 100  # VERY heavy
        return mse_loss + count_loss
'''
    
    print(quick_fix_code)

def main():
    """Main diagnosis function"""
    
    # Diagnose issues
    issues = diagnose_critical_issues()
    
    # Create emergency fixes
    create_emergency_fixes(issues)
    
    # Quick fix configuration
    quick_fix_config()
    
    print(f"\n💡 SUMMARY:")
    print("   Your MAE of 436.62 indicates complete model failure.")
    print("   This is likely due to:")
    print("   1. Poor density map generation")
    print("   2. Inappropriate loss function (standard MSE)")
    print("   3. Training hyperparameters not optimized for counting")
    
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Run: python data_preparation/enhanced_dmap_generator.py")
    print("   2. Run: python enhanced_train.py")
    print("   3. Target: Get MAE below 20 first, then optimize further")

if __name__ == "__main__":
    main()