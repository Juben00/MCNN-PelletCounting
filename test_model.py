import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from mcnn_model import MCNN
from improved_dataloader import PelletDataset

def load_trained_model(checkpoint_path="./checkpoints/best_model.pth"):
    """Load your trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found at: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth'):
                    print(f"   - {os.path.join(checkpoint_dir, file)}")
        return None, None
    
    # Load model
    model = MCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Device: {device}")
    if 'best_mae' in checkpoint:
        print(f"   Best MAE: {checkpoint['best_mae']:.2f}")
    
    return model, device

def test_single_image(model, device, image_path, show_result=True):
    """Test model on a single image"""
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if too large (memory optimization)
    h, w = image_rgb.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        print(f"   Resized from {w}x{h} to {new_w}x{new_h}")
    
    # Prepare tensor
    image_tensor = torch.FloatTensor(image_rgb.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        density_map = model(image_tensor)
        predicted_count = density_map.sum().item()
    
    # Convert back for visualization
    density_np = density_map.squeeze().cpu().numpy()
    
    print(f"📊 Prediction Results:")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Predicted Count: {predicted_count:.1f} pellets")
    
    if show_result:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title(f'Original Image\n{os.path.basename(image_path)}')
        ax1.axis('off')
        
        # Density map
        im = ax2.imshow(density_np, cmap='hot')
        ax2.set_title(f'Density Map\nPredicted: {predicted_count:.1f} pellets')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_count, density_np

def test_on_dataset(model, device, dataset_type="test"):
    """Test model on your test dataset"""
    
    # Dataset paths
    if dataset_type == "test":
        img_root = "./data/test_data/images"
        gt_root = "./data/test_data/densitymaps"
    elif dataset_type == "train":
        img_root = "./data/train_data/images"
        gt_root = "./data/train_data/densitymaps"
    else:
        print(f"❌ Unknown dataset type: {dataset_type}")
        return
    
    # Check if enhanced density maps exist
    enhanced_root = gt_root + "_enhanced"
    if os.path.exists(enhanced_root):
        gt_root = enhanced_root
        print(f"✅ Using enhanced density maps")
    
    # Create dataset
    try:
        dataset = PelletDataset(img_root, gt_root, gt_downsample=4, phase='test', transform=False)
        print(f"📊 Testing on {len(dataset)} {dataset_type} images")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Test on subset (first 10 images)
    results = []
    n_test = min(10, len(dataset))
    
    for i in range(n_test):
        try:
            img_tensor, gt_dmap_tensor = dataset[i]
            img_name = dataset.img_names[i]
            
            # Add batch dimension and move to device
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                pred_dmap = model(img_batch)
            
            # Calculate counts
            pred_count = pred_dmap.sum().item()
            gt_count = gt_dmap_tensor.sum().item()
            mae = abs(pred_count - gt_count)
            
            results.append({
                'image': img_name,
                'predicted': pred_count,
                'ground_truth': gt_count,
                'mae': mae
            })
            
            print(f"   {img_name}: Predicted={pred_count:.1f}, GT={gt_count:.1f}, MAE={mae:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error processing {dataset.img_names[i]}: {e}")
            continue
    
    # Calculate overall metrics
    if results:
        total_mae = sum(r['mae'] for r in results)
        avg_mae = total_mae / len(results)
        
        print(f"\n📈 Overall Results ({len(results)} images):")
        print(f"   Average MAE: {avg_mae:.2f}")
        print(f"   Best MAE: {min(r['mae'] for r in results):.2f}")
        print(f"   Worst MAE: {max(r['mae'] for r in results):.2f}")
    
    return results

def quick_test():
    """Quick test function - just run this!"""
    
    print("🧪 MCNN Pellet Counter - Quick Test")
    print("=" * 50)
    
    # Load model
    model, device = load_trained_model()
    if model is None:
        return
    
    # Test options
    print("\n🎯 Testing Options:")
    print("1. Test on dataset (automatic)")
    print("2. Test single image (manual)")
    
    # Auto test on dataset
    print("\n📊 Testing on dataset...")
    try:
        results = test_on_dataset(model, device, "test")
        if results:
            print("✅ Dataset testing completed!")
        else:
            print("⚠️  No results from dataset testing")
    except Exception as e:
        print(f"❌ Dataset testing failed: {e}")
    
    # Check for sample images to test
    sample_paths = [
        "./data/test_data/images",
        "./data/train_data/images"
    ]
    
    sample_image = None
    for path in sample_paths:
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_image = os.path.join(path, images[0])
                break
    
    if sample_image:
        print(f"\n🖼️  Testing single image: {os.path.basename(sample_image)}")
        try:
            count, density = test_single_image(model, device, sample_image, show_result=False)
            if count is not None:
                print("✅ Single image testing completed!")
        except Exception as e:
            print(f"❌ Single image testing failed: {e}")

def main():
    """Main testing interface"""
    
    print("🧪 MCNN Pellet Counter - Model Testing")
    print("=" * 50)
    
    # Load model
    model, device = load_trained_model()
    if model is None:
        print("\n💡 To test your model:")
        print("   1. Train a model first: python enhanced_train.py")
        print("   2. Or use existing checkpoint: python test_model.py")
        return
    
    while True:
        print("\n🎯 Testing Options:")
        print("1. Quick test (automatic)")
        print("2. Test on full dataset")
        print("3. Test single image")
        print("4. Run comprehensive evaluation")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            quick_test()
        elif choice == '2':
            dataset_type = input("Dataset (train/test): ").strip().lower()
            if dataset_type in ['train', 'test']:
                test_on_dataset(model, device, dataset_type)
            else:
                print("Invalid dataset type")
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                test_single_image(model, device, image_path)
            else:
                print("Image not found")
        elif choice == '4':
            print("Running comprehensive evaluation...")
            os.system("python evaluate_model.py")
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    # For quick testing, just run this
    quick_test()
    
    # For interactive testing, uncomment this:
    # main()