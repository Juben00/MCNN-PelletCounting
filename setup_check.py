#!/usr/bin/env python3
"""
Quick test and setup script for ultra-optimized MCNN training
"""

import os
import sys
import torch

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    dependencies = []
    
    # Check PyTorch
    try:
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        dependencies.append("pytorch")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Check other dependencies
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        dependencies.append("numpy")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        dependencies.append("matplotlib")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        dependencies.append("opencv")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        from PIL import Image
        print(f"✅ PIL/Pillow")
        dependencies.append("pillow")
    except ImportError:
        print("❌ PIL/Pillow not found")
        return False
    
    try:
        from tqdm import tqdm
        print(f"✅ tqdm")
        dependencies.append("tqdm")
    except ImportError:
        print("❌ tqdm not found")
        return False
    
    return True

def check_data_structure():
    """Check if data is properly organized"""
    print("\n📁 Checking data structure...")
    
    required_dirs = [
        './data/train_data/images',
        './data/train_data/densitymaps',
        './data/test_data/images', 
        './data/test_data/densitymaps'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_path}: {file_count} files")
        else:
            print(f"❌ {dir_path}: Not found")
            return False
    
    return True

def check_models():
    """Check if model files are importable"""
    print("\n🧠 Checking model imports...")
    
    try:
        from mcnn_model import MCNN
        print("✅ Original MCNN model")
    except ImportError as e:
        print(f"❌ MCNN model import failed: {e}")
        return False
    
    try:
        from enhanced_mcnn_model import PelletMCNN
        print("✅ Enhanced PelletMCNN model")
    except ImportError as e:
        print(f"❌ Enhanced MCNN model import failed: {e}")
        return False
    
    try:
        from dataloader import PelletDataset, create_data_loaders
        print("✅ Data loader")
    except ImportError as e:
        print(f"❌ Data loader import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    print("\n🔧 Testing model creation...")
    
    try:
        from enhanced_mcnn_model import PelletMCNN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = PelletMCNN().to(device)
        print(f"✅ PelletMCNN created successfully on {device}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Model forward pass successful: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📂 Creating directories...")
    
    dirs_to_create = [
        './checkpoints',
        './results',
        './results/evaluation'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def main():
    """Run all checks and setup"""
    print("🚀 Ultra-Optimized MCNN Setup and Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check dependencies
    if not check_dependencies():
        all_good = False
    
    # Check data structure
    if not check_data_structure():
        all_good = False
    
    # Check model imports
    if not check_models():
        all_good = False
    
    # Test model creation
    if not test_model_creation():
        all_good = False
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("🚀 Ready to start ultra-optimized MCNN training!")
        print("\nTo start training, run:")
        print("   python ultra_optimized_mcnn_train.py")
        print("\nTo evaluate trained model, run:")
        print("   python evaluate_90_percent.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before starting training.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()