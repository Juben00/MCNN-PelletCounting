import torch
import numpy as np
import os
from improved_dataloader import PelletDataset

def check_gpu_memory():
    """Check GPU memory and capabilities"""
    print("=== GPU Memory Check ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Will use CPU training.")
        return False
    
    device = torch.device('cuda')
    gpu_props = torch.cuda.get_device_properties(0)
    
    print(f"GPU: {gpu_props.name}")
    print(f"Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Clear any existing memory
    torch.cuda.empty_cache()
    
    # Check available memory
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = (gpu_props.total_memory - torch.cuda.memory_reserved()) / 1024**3
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB") 
    print(f"Free: {free:.2f} GB")
    
    return True

def test_model_memory():
    """Test memory usage with different configurations"""
    print("\n=== Model Memory Test ===")
    
    from mcnn_model import MCNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test basic model loading
        model = MCNN().to(device)
        print("✓ Model loaded successfully")
        
        # Test different input sizes
        test_sizes = [
            (1, 3, 400, 600),   # Small
            (1, 3, 600, 800),   # Medium  
            (1, 3, 800, 1200),  # Large
            (1, 3, 1200, 1600), # Very Large
        ]
        
        max_working_size = None
        
        for size in test_sizes:
            try:
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
                # Create dummy input
                dummy_input = torch.randn(size, device=device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                
                print(f"✓ Size {size[2]}x{size[3]} works - Output: {output.shape}")
                max_working_size = size
                
                del dummy_input, output
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"✗ Size {size[2]}x{size[3]} - Out of memory")
                    break
                else:
                    print(f"✗ Size {size[2]}x{size[3]} - Error: {e}")
                    break
        
        if max_working_size:
            print(f"\nRecommended max image size: {max_working_size[2]}x{max_working_size[3]}")
        
        del model
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False
    
    return True

def test_dataloader_memory():
    """Test dataloader memory usage"""
    print("\n=== Dataloader Memory Test ===")
    
    # Check if data exists
    train_img_root = "./data/train_data/images"
    train_gt_root = "./data/train_data/densitymaps"
    
    if not os.path.exists(train_img_root):
        print(f"Training data not found at {train_img_root}")
        return False
    
    try:
        # Create small dataset for testing
        dataset = PelletDataset(
            train_img_root, train_gt_root, 
            gt_downsample=4, phase='train', transform=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading one sample
            img, dmap = dataset[0]
            print(f"Sample image shape: {img.shape}")
            print(f"Sample density map shape: {dmap.shape}")
            print(f"Sample pellet count: {dmap.sum():.1f}")
            
            del img, dmap
            print("✓ Dataloader test successful")
            return True
        else:
            print("No samples in dataset")
            return False
            
    except Exception as e:
        print(f"Dataloader test failed: {e}")
        return False

def get_memory_recommendations():
    """Provide memory optimization recommendations"""
    print("\n=== Memory Optimization Recommendations ===")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory < 6:
            print("🔧 Your GPU has limited memory. Recommendations:")
            print("   - Use memory_optimized_train.py instead of improved_train.py")
            print("   - Keep batch_size = 1")
            print("   - Use gt_downsample = 4 or 8")
            print("   - Limit image size to 600x800 or smaller")
            print("   - Consider training on CPU if GPU OOM persists")
            
        elif gpu_memory < 8:
            print("⚡ Moderate GPU memory. Recommendations:")
            print("   - Use batch_size = 1-2")
            print("   - Use gt_downsample = 4")
            print("   - Limit image size to 800x1200")
            
        else:
            print("🚀 Good GPU memory. You can use:")
            print("   - Use improved_train.py")
            print("   - Consider batch_size = 2-4")
            print("   - Use gt_downsample = 2-4")
    else:
        print("💻 Using CPU training:")
        print("   - Training will be slower but should work")
        print("   - Consider using smaller images")
        print("   - Use gt_downsample = 8 for faster training")

def main():
    """Run all memory tests"""
    print("🧪 MCNN Memory Diagnostic Tool")
    print("="*50)
    
    # Check GPU
    has_gpu = check_gpu_memory()
    
    # Test model
    model_ok = test_model_memory()
    
    # Test dataloader
    data_ok = test_dataloader_memory()
    
    # Provide recommendations
    get_memory_recommendations()
    
    print("\n" + "="*50)
    if has_gpu and model_ok and data_ok:
        print("✅ System ready for training with memory_optimized_train.py")
    elif model_ok and data_ok:
        print("⚠️  System ready for CPU training")
    else:
        print("❌ Issues detected. Check the errors above.")
    
    print("\n🚀 Next steps:")
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
        print("   python memory_optimized_train.py")
    else:
        print("   python improved_train.py")

if __name__ == "__main__":
    main()