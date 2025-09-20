import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import cv2
from datetime import datetime

from mcnn_model import MCNN
from improved_dataloader import PelletDataset

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = MCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, dataset, device, save_predictions=True, save_dir='./results/predictions'):
    """Comprehensive model evaluation"""
    
    if save_predictions:
        os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    results = []
    total_mae = 0
    total_mse = 0
    total_mape = 0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            img_tensor, gt_dmap_tensor = dataset[idx]
            img_name = dataset.img_names[idx]
            
            # Add batch dimension
            img_batch = img_tensor.unsqueeze(0).to(device)
            gt_dmap_batch = gt_dmap_tensor.unsqueeze(0).to(device)
            
            # Predict
            pred_dmap_batch = model(img_batch)
            
            # Remove batch dimension
            pred_dmap = pred_dmap_batch.squeeze(0).cpu().numpy()
            gt_dmap = gt_dmap_batch.squeeze(0).cpu().numpy()
            
            # Calculate counts
            pred_count = pred_dmap.sum()
            gt_count = gt_dmap.sum()
            
            # Calculate metrics
            mae = abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            mape = abs(pred_count - gt_count) / max(gt_count, 1) * 100
            
            total_mae += mae
            total_mse += mse
            total_mape += mape
            
            result = {
                'image_name': img_name,
                'predicted_count': float(pred_count),
                'ground_truth_count': float(gt_count),
                'mae': float(mae),
                'mse': float(mse),
                'mape': float(mape)
            }
            results.append(result)
            
            # Save prediction visualization
            if save_predictions and idx < 20:  # Save first 20 predictions
                save_prediction_visualization(
                    img_tensor, gt_dmap, pred_dmap, img_name, 
                    gt_count, pred_count, save_dir
                )
    
    # Calculate overall metrics
    n_samples = len(dataset)
    overall_metrics = {
        'mae': total_mae / n_samples,
        'mse': total_mse / n_samples,
        'rmse': np.sqrt(total_mse / n_samples),
        'mape': total_mape / n_samples,
        'num_samples': n_samples
    }
    
    return results, overall_metrics

def save_prediction_visualization(img_tensor, gt_dmap, pred_dmap, img_name, gt_count, pred_count, save_dir):
    """Save visualization of prediction vs ground truth"""
    
    # Convert image tensor to numpy (denormalize if needed)
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # If image was normalized, denormalize it
    if img.min() < 0:  # Likely normalized
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    
    img = np.clip(img, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f'Original Image\n{img_name}')
    axes[0].axis('off')
    
    # Ground truth density map
    gt_display = gt_dmap.squeeze() if len(gt_dmap.shape) > 2 else gt_dmap
    im1 = axes[1].imshow(gt_display, cmap='hot')
    axes[1].set_title(f'Ground Truth\nCount: {gt_count:.1f}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Predicted density map
    pred_display = pred_dmap.squeeze() if len(pred_dmap.shape) > 2 else pred_dmap
    im2 = axes[2].imshow(pred_display, cmap='hot')
    axes[2].set_title(f'Prediction\nCount: {pred_count:.1f}\nMAE: {abs(pred_count-gt_count):.1f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save
    save_name = os.path.splitext(img_name)[0] + '_prediction.png'
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_results(results, save_path):
    """Plot evaluation results and statistics"""
    
    pred_counts = [r['predicted_count'] for r in results]
    gt_counts = [r['ground_truth_count'] for r in results]
    maes = [r['mae'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot: Predicted vs Ground Truth
    ax1.scatter(gt_counts, pred_counts, alpha=0.6)
    min_count = min(min(gt_counts), min(pred_counts))
    max_count = max(max(gt_counts), max(pred_counts))
    ax1.plot([min_count, max_count], [min_count, max_count], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Ground Truth Count')
    ax1.set_ylabel('Predicted Count')
    ax1.set_title('Predicted vs Ground Truth Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE distribution
    ax2.hist(maes, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of MAE')
    ax2.axvline(np.mean(maes), color='red', linestyle='--', label=f'Mean MAE: {np.mean(maes):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error vs Ground Truth Count
    ax3.scatter(gt_counts, maes, alpha=0.6)
    ax3.set_xlabel('Ground Truth Count')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error vs Ground Truth Count')
    ax3.grid(True, alpha=0.3)
    
    # Cumulative error distribution
    sorted_maes = np.sort(maes)
    cumulative = np.arange(1, len(sorted_maes) + 1) / len(sorted_maes)
    ax4.plot(sorted_maes, cumulative)
    ax4.set_xlabel('Mean Absolute Error')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Error Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_stats(results, overall_metrics):
    """Print detailed evaluation statistics"""
    
    print("\n" + "="*60)
    print("DETAILED EVALUATION RESULTS")
    print("="*60)
    
    print(f"Number of test samples: {overall_metrics['num_samples']}")
    print(f"Overall MAE: {overall_metrics['mae']:.2f}")
    print(f"Overall RMSE: {overall_metrics['rmse']:.2f}")
    print(f"Overall MAPE: {overall_metrics['mape']:.2f}%")
    
    # Error distribution analysis
    maes = [r['mae'] for r in results]
    print(f"\nError Distribution:")
    print(f"  Min MAE: {min(maes):.2f}")
    print(f"  Max MAE: {max(maes):.2f}")
    print(f"  Median MAE: {np.median(maes):.2f}")
    print(f"  Std MAE: {np.std(maes):.2f}")
    
    # Accuracy at different thresholds
    thresholds = [1, 2, 5, 10]
    print(f"\nAccuracy at different error thresholds:")
    for threshold in thresholds:
        accuracy = np.mean([mae <= threshold for mae in maes]) * 100
        print(f"  MAE ≤ {threshold}: {accuracy:.1f}%")
    
    # Count range analysis
    gt_counts = [r['ground_truth_count'] for r in results]
    pred_counts = [r['predicted_count'] for r in results]
    
    print(f"\nCount Statistics:")
    print(f"  GT Count Range: {min(gt_counts):.1f} - {max(gt_counts):.1f}")
    print(f"  Predicted Count Range: {min(pred_counts):.1f} - {max(pred_counts):.1f}")
    print(f"  Average GT Count: {np.mean(gt_counts):.1f}")
    print(f"  Average Predicted Count: {np.mean(pred_counts):.1f}")

def main():
    """Main evaluation function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = './checkpoints/best_model.pth'
    
    # Data paths
    test_img_root = "./data/test_data/images"
    test_gt_root = "./data/test_data/densitymaps"
    
    print("=== MCNN Pellet Counter Evaluation ===")
    print(f"Device: {device}")
    print(f"Loading model from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using improved_train.py")
        return
    
    # Load model
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully!")
    
    # Create test dataset
    test_dataset = PelletDataset(
        test_img_root, test_gt_root, 
        gt_downsample=4, phase='test', transform=False
    )
    
    print(f"Test dataset: {len(test_dataset)} images")
    
    # Evaluate model
    results, overall_metrics = evaluate_model(model, test_dataset, device)
    
    # Print detailed statistics
    print_detailed_stats(results, overall_metrics)
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Save results to JSON
    evaluation_results = {
        'overall_metrics': overall_metrics,
        'individual_results': results,
        'evaluation_date': datetime.now().isoformat()
    }
    
    with open('./results/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Plot results
    plot_evaluation_results(results, './results/evaluation_plots.png')
    
    print(f"\n✅ Evaluation completed!")
    print(f"Results saved in ./results/")
    print(f"Prediction visualizations saved in ./results/predictions/")

if __name__ == "__main__":
    main()