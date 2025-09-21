#!/usr/bin/env python3
"""
Ultra-Optimized MCNN Evaluation for 90% Accuracy Target
======================================================

Comprehensive evaluation system specifically designed to track progress
toward 90% pellet counting accuracy with detailed metrics and visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
import cv2

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from enhanced_mcnn_model import PelletMCNN
from improved_dataloader import PelletDataset

class AccuracyEvaluator:
    """Comprehensive accuracy evaluator for pellet counting"""
    
    def __init__(self, target_accuracy=0.90):
        self.target_accuracy = target_accuracy
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.absolute_errors = []
        self.relative_errors = []
        self.image_names = []
    
    def add_prediction(self, pred_count, gt_count, image_name=None):
        """Add a prediction for evaluation"""
        self.predictions.append(pred_count)
        self.ground_truths.append(gt_count)
        
        abs_error = abs(pred_count - gt_count)
        self.absolute_errors.append(abs_error)
        
        if gt_count > 0:
            rel_error = abs_error / gt_count
        else:
            rel_error = 0.0 if pred_count == 0 else 1.0
        self.relative_errors.append(rel_error)
        
        if image_name:
            self.image_names.append(image_name)
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        ground_truths = np.array(self.ground_truths)
        abs_errors = np.array(self.absolute_errors)
        rel_errors = np.array(self.relative_errors)
        
        # Basic metrics
        mae = np.mean(abs_errors)
        mse = np.mean((predictions - ground_truths) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(rel_errors) * 100
        
        # Accuracy at different thresholds
        acc_1 = np.mean(abs_errors <= 1) * 100  # ±1 pellet
        acc_2 = np.mean(abs_errors <= 2) * 100  # ±2 pellets
        acc_3 = np.mean(abs_errors <= 3) * 100  # ±3 pellets
        acc_5 = np.mean(abs_errors <= 5) * 100  # ±5 pellets
        acc_10 = np.mean(abs_errors <= 10) * 100  # ±10 pellets
        
        # Relative accuracy (within percentage of true count)
        rel_acc_5 = np.mean(rel_errors <= 0.05) * 100   # Within 5%
        rel_acc_10 = np.mean(rel_errors <= 0.10) * 100  # Within 10%
        rel_acc_20 = np.mean(rel_errors <= 0.20) * 100  # Within 20%
        
        # Counting statistics
        total_pred = np.sum(predictions)
        total_gt = np.sum(ground_truths)
        count_bias = (total_pred - total_gt) / total_gt * 100 if total_gt > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truths)[0, 1]
        
        # Target achievement
        target_achieved = acc_2 >= (self.target_accuracy * 100)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'count_bias': count_bias,
            'total_predictions': len(predictions),
            'total_predicted_count': total_pred,
            'total_ground_truth_count': total_gt,
            'accuracy': {
                'acc_1': acc_1,
                'acc_2': acc_2,
                'acc_3': acc_3,
                'acc_5': acc_5,
                'acc_10': acc_10
            },
            'relative_accuracy': {
                'rel_acc_5': rel_acc_5,
                'rel_acc_10': rel_acc_10,
                'rel_acc_20': rel_acc_20
            },
            'target_accuracy': self.target_accuracy * 100,
            'target_achieved': target_achieved,
            'target_metric': acc_2  # Primary metric for 90% target
        }
    
    def get_error_distribution(self):
        """Get error distribution for analysis"""
        if not self.absolute_errors:
            return {}
        
        abs_errors = np.array(self.absolute_errors)
        
        return {
            'min_error': np.min(abs_errors),
            'max_error': np.max(abs_errors),
            'median_error': np.median(abs_errors),
            'q25_error': np.percentile(abs_errors, 25),
            'q75_error': np.percentile(abs_errors, 75),
            'std_error': np.std(abs_errors),
            'perfect_predictions': np.sum(abs_errors == 0),
            'error_distribution': {
                'error_0': np.sum(abs_errors == 0),
                'error_1': np.sum(abs_errors == 1),
                'error_2': np.sum(abs_errors == 2),
                'error_3_5': np.sum((abs_errors >= 3) & (abs_errors <= 5)),
                'error_6_10': np.sum((abs_errors >= 6) & (abs_errors <= 10)),
                'error_gt_10': np.sum(abs_errors > 10)
            }
        }

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = PelletMCNN().to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    model.eval()
    return model

def evaluate_model_comprehensive(model, dataset, device, save_dir='./results/evaluation'):
    """Comprehensive model evaluation with detailed analysis"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = AccuracyEvaluator(target_accuracy=0.90)
    model.eval()
    
    print("🔍 Performing comprehensive evaluation...")
    
    detailed_results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            # Get data
            image, gt_dmap = dataset[idx]
            image_name = f"image_{idx:04d}"
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            gt_dmap = gt_dmap.unsqueeze(0).to(device)
            
            # Predict
            pred_dmap = model(image)
            
            # Calculate counts
            pred_count = pred_dmap.sum().item()
            gt_count = gt_dmap.sum().item()
            
            # Add to evaluator
            evaluator.add_prediction(pred_count, gt_count, image_name)
            
            # Store detailed result
            abs_error = abs(pred_count - gt_count)
            rel_error = abs_error / gt_count if gt_count > 0 else (0.0 if pred_count == 0 else 1.0)
            
            detailed_results.append({
                'image_name': image_name,
                'predicted_count': pred_count,
                'ground_truth_count': gt_count,
                'absolute_error': abs_error,
                'relative_error': rel_error,
                'accuracy_1': abs_error <= 1,
                'accuracy_2': abs_error <= 2,
                'accuracy_5': abs_error <= 5
            })
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    error_dist = evaluator.get_error_distribution()
    
    # Generate report
    generate_evaluation_report(metrics, error_dist, detailed_results, save_dir)
    
    return metrics, detailed_results

def generate_evaluation_report(metrics, error_dist, detailed_results, save_dir):
    """Generate comprehensive evaluation report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print console report
    print_evaluation_summary(metrics, error_dist)
    
    # Save detailed JSON report
    report = {
        'timestamp': timestamp,
        'metrics': metrics,
        'error_distribution': error_dist,
        'detailed_results': detailed_results
    }
    
    with open(f"{save_dir}/evaluation_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate visualizations
    create_evaluation_plots(metrics, error_dist, detailed_results, save_dir, timestamp)
    
    # Generate text summary
    create_text_summary(metrics, error_dist, save_dir, timestamp)

def print_evaluation_summary(metrics, error_dist):
    """Print evaluation summary to console"""
    
    print("\n" + "="*70)
    print("🎯 MCNN PELLET COUNTING EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 CORE METRICS:")
    print(f"   MAE (Mean Absolute Error): {metrics['mae']:.2f} pellets")
    print(f"   RMSE (Root Mean Square Error): {metrics['rmse']:.2f} pellets")
    print(f"   MAPE (Mean Absolute Percentage Error): {metrics['mape']:.1f}%")
    print(f"   Correlation Coefficient: {metrics['correlation']:.3f}")
    print(f"   Count Bias: {metrics['count_bias']:+.1f}%")
    
    print(f"\n🎯 ACCURACY METRICS (90% Target):")
    acc = metrics['accuracy']
    print(f"   ±1 pellet accuracy: {acc['acc_1']:.1f}%")
    print(f"   ±2 pellet accuracy: {acc['acc_2']:.1f}% {'✅' if acc['acc_2'] >= 90 else '❌'}")
    print(f"   ±3 pellet accuracy: {acc['acc_3']:.1f}%")
    print(f"   ±5 pellet accuracy: {acc['acc_5']:.1f}%")
    print(f"   ±10 pellet accuracy: {acc['acc_10']:.1f}%")
    
    print(f"\n📈 RELATIVE ACCURACY:")
    rel_acc = metrics['relative_accuracy']
    print(f"   Within 5% of true count: {rel_acc['rel_acc_5']:.1f}%")
    print(f"   Within 10% of true count: {rel_acc['rel_acc_10']:.1f}%")
    print(f"   Within 20% of true count: {rel_acc['rel_acc_20']:.1f}%")
    
    print(f"\n📋 SUMMARY STATISTICS:")
    print(f"   Total Images Evaluated: {metrics['total_predictions']}")
    print(f"   Total Predicted Pellets: {metrics['total_predicted_count']:.0f}")
    print(f"   Total Ground Truth Pellets: {metrics['total_ground_truth_count']:.0f}")
    print(f"   Perfect Predictions (0 error): {error_dist['perfect_predictions']}")
    
    print(f"\n🏆 TARGET ACHIEVEMENT:")
    if metrics['target_achieved']:
        print(f"   🎉 TARGET ACHIEVED! 90% accuracy reached: {acc['acc_2']:.1f}%")
        print(f"   🚀 Your MCNN model can accurately count feed pellets!")
    else:
        print(f"   📈 Progress toward 90% target: {acc['acc_2']:.1f}%")
        print(f"   🔧 Need {90 - acc['acc_2']:.1f}% more accuracy to reach target")
    
    print("="*70)

def create_evaluation_plots(metrics, error_dist, detailed_results, save_dir, timestamp):
    """Create comprehensive evaluation plots"""
    
    # Set up matplotlib
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Extract data for plotting
    pred_counts = [r['predicted_count'] for r in detailed_results]
    gt_counts = [r['ground_truth_count'] for r in detailed_results]
    abs_errors = [r['absolute_error'] for r in detailed_results]
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MCNN Pellet Counting Evaluation - {timestamp}', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Predicted vs Ground Truth
    axes[0, 0].scatter(gt_counts, pred_counts, alpha=0.6, s=30)
    max_count = max(max(pred_counts), max(gt_counts))
    axes[0, 0].plot([0, max_count], [0, max_count], 'r--', alpha=0.8, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Ground Truth Count')
    axes[0, 0].set_ylabel('Predicted Count')
    axes[0, 0].set_title(f'Predicted vs Ground Truth\nr = {metrics["correlation"]:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution histogram
    axes[0, 1].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(metrics['mae'], color='red', linestyle='--', label=f'MAE = {metrics["mae"]:.2f}')
    axes[0, 1].set_xlabel('Absolute Error (pellets)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy at different thresholds
    thresholds = [1, 2, 3, 5, 10]
    accuracies = [metrics['accuracy'][f'acc_{t}'] for t in thresholds]
    bars = axes[0, 2].bar([str(t) for t in thresholds], accuracies, alpha=0.7)
    axes[0, 2].axhline(90, color='red', linestyle='--', label='90% Target')
    axes[0, 2].set_xlabel('Error Threshold (±pellets)')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Accuracy at Different Thresholds')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Highlight the target metric
    for i, bar in enumerate(bars):
        if thresholds[i] == 2:  # Highlight ±2 pellet accuracy
            bar.set_color('gold')
            bar.set_edgecolor('red')
            bar.set_linewidth(2)
    
    # 4. Error vs Ground Truth Count
    axes[1, 0].scatter(gt_counts, abs_errors, alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Ground Truth Count')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error vs Ground Truth Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative accuracy
    sorted_errors = np.sort(abs_errors)
    cum_accuracy = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1, 1].plot(sorted_errors, cum_accuracy)
    axes[1, 1].axhline(90, color='red', linestyle='--', label='90% Target')
    axes[1, 1].axvline(2, color='orange', linestyle='--', label='±2 pellet threshold')
    axes[1, 1].set_xlabel('Error Threshold')
    axes[1, 1].set_ylabel('Cumulative Accuracy (%)')
    axes[1, 1].set_title('Cumulative Accuracy Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Error distribution by bins
    error_bins = ['0', '1', '2', '3-5', '6-10', '>10']
    error_counts = [
        error_dist['error_distribution']['error_0'],
        error_dist['error_distribution']['error_1'],
        error_dist['error_distribution']['error_2'],
        error_dist['error_distribution']['error_3_5'],
        error_dist['error_distribution']['error_6_10'],
        error_dist['error_distribution']['error_gt_10']
    ]
    
    bars = axes[1, 2].bar(error_bins, error_counts, alpha=0.7)
    axes[1, 2].set_xlabel('Error Range (pellets)')
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].set_title('Error Distribution by Bins')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Color code the bars (green for good, red for bad)
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation plots saved to: {save_dir}/evaluation_plots_{timestamp}.png")

def create_text_summary(metrics, error_dist, save_dir, timestamp):
    """Create a detailed text summary report"""
    
    summary_text = f"""
MCNN PELLET COUNTING EVALUATION REPORT
=====================================
Generated: {timestamp}

EXECUTIVE SUMMARY
-----------------
Target: 90% accuracy (±2 pellets tolerance)
Achieved: {metrics['accuracy']['acc_2']:.1f}%
Status: {'✅ TARGET ACHIEVED' if metrics['target_achieved'] else '❌ TARGET NOT REACHED'}

DETAILED METRICS
----------------
Core Performance:
• Mean Absolute Error (MAE): {metrics['mae']:.2f} pellets
• Root Mean Square Error (RMSE): {metrics['rmse']:.2f} pellets
• Mean Absolute Percentage Error (MAPE): {metrics['mape']:.1f}%
• Correlation Coefficient: {metrics['correlation']:.3f}
• Count Bias: {metrics['count_bias']:+.1f}%

Accuracy Breakdown:
• ±1 pellet accuracy: {metrics['accuracy']['acc_1']:.1f}%
• ±2 pellet accuracy: {metrics['accuracy']['acc_2']:.1f}% (TARGET METRIC)
• ±3 pellet accuracy: {metrics['accuracy']['acc_3']:.1f}%
• ±5 pellet accuracy: {metrics['accuracy']['acc_5']:.1f}%
• ±10 pellet accuracy: {metrics['accuracy']['acc_10']:.1f}%

Relative Accuracy:
• Within 5% of true count: {metrics['relative_accuracy']['rel_acc_5']:.1f}%
• Within 10% of true count: {metrics['relative_accuracy']['rel_acc_10']:.1f}%
• Within 20% of true count: {metrics['relative_accuracy']['rel_acc_20']:.1f}%

ERROR ANALYSIS
--------------
Total Images: {metrics['total_predictions']}
Perfect Predictions (0 error): {error_dist['perfect_predictions']}
Error Statistics:
• Minimum Error: {error_dist['min_error']:.1f} pellets
• Maximum Error: {error_dist['max_error']:.1f} pellets
• Median Error: {error_dist['median_error']:.1f} pellets
• Standard Deviation: {error_dist['std_error']:.2f} pellets

Error Distribution:
• 0 pellets error: {error_dist['error_distribution']['error_0']} images
• 1 pellet error: {error_dist['error_distribution']['error_1']} images
• 2 pellets error: {error_dist['error_distribution']['error_2']} images
• 3-5 pellets error: {error_dist['error_distribution']['error_3_5']} images
• 6-10 pellets error: {error_dist['error_distribution']['error_6_10']} images
• >10 pellets error: {error_dist['error_distribution']['error_gt_10']} images

COUNTING STATISTICS
-------------------
Total Predicted Pellets: {metrics['total_predicted_count']:.0f}
Total Ground Truth Pellets: {metrics['total_ground_truth_count']:.0f}
Counting Bias: {metrics['count_bias']:+.1f}%

RECOMMENDATIONS
---------------"""

    if metrics['target_achieved']:
        summary_text += """
🎉 CONGRATULATIONS! Your MCNN model has achieved the 90% accuracy target!

The model is ready for deployment in feed pellet counting applications.
Consider the following for optimization:
• Monitor performance on new data
• Fine-tune thresholds if needed
• Document the model configuration for reproducibility
"""
    else:
        gap = 90 - metrics['accuracy']['acc_2']
        summary_text += f"""
📈 Your model achieved {metrics['accuracy']['acc_2']:.1f}% accuracy.
Gap to target: {gap:.1f}%

Improvement Strategies:
• Increase training data if accuracy < 80%
• Adjust loss function weights for better count preservation
• Implement data augmentation for better generalization
• Consider ensemble methods if accuracy > 85%
• Fine-tune hyperparameters (learning rate, batch size)
"""

    summary_text += f"""

TECHNICAL DETAILS
-----------------
Model: Enhanced MCNN with Attention Mechanisms
Evaluation Date: {timestamp}
Target Threshold: ±2 pellets (reasonable for pellet counting)
Evaluation Dataset: {metrics['total_predictions']} images

This report provides a comprehensive analysis of your MCNN model's
performance for feed pellet counting applications.
"""
    
    # Save summary
    with open(f"{save_dir}/evaluation_summary_{timestamp}.txt", 'w') as f:
        f.write(summary_text)
    
    print(f"📝 Detailed summary saved to: {save_dir}/evaluation_summary_{timestamp}.txt")

def main():
    """Main evaluation function"""
    print("🔍 Ultra-Optimized MCNN Evaluation for 90% Accuracy")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model paths to try (in order of preference)
    model_paths = [
        './checkpoints/best_ultra_optimized_model.pth',
        './checkpoints/best_model.pth',
        './checkpoints/enhanced_model.pth'
    ]
    
    # Find available model
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found!")
        print("Please train a model first using ultra_optimized_mcnn_train.py")
        return
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return
    
    # Load test dataset
    test_img_root = './data/test_data/images'
    test_gt_root = './data/test_data/densitymaps'
    
    if not os.path.exists(test_img_root) or not os.path.exists(test_gt_root):
        print("❌ Test data not found!")
        print("Please ensure test data is available in ./data/test_data/")
        return
    
    # Create test dataset
    test_dataset = PelletDataset(
        test_img_root, test_gt_root,
        gt_downsample=4, phase='test', transform=False
    )
    
    print(f"Test dataset: {len(test_dataset)} images")
    
    # Evaluate model
    metrics, detailed_results = evaluate_model_comprehensive(
        model, test_dataset, device
    )
    
    # Print final status
    if metrics['target_achieved']:
        print(f"\n🎉 SUCCESS: 90% accuracy target achieved!")
        print(f"Your MCNN model is ready for pellet counting applications!")
    else:
        print(f"\n📈 Progress: {metrics['accuracy']['acc_2']:.1f}% accuracy achieved")
        print(f"Continue training to reach the 90% target")

if __name__ == "__main__":
    main()