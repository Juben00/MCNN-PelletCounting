#!/usr/bin/env python3
"""
100% Accuracy Evaluation System for Perfect Pellet Counting
==========================================================

Ultra-precise evaluation system designed to validate and verify
100% accuracy achievement in feed pellet counting.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch

from mcnn_model import MCNN
from improved_dataloader import PelletDataset

class PerfectionAnalyzer:
    """Comprehensive analyzer for 100% accuracy validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking metrics"""
        self.results = []
        self.perfect_predictions = 0
        self.near_perfect_1 = 0
        self.acceptable_2 = 0
        self.total_predictions = 0
        self.total_error = 0
        self.max_error = 0
        self.error_distribution = {}
        
    def add_prediction(self, pred_count, gt_count, image_name=None):
        """Add a prediction for detailed analysis"""
        error = abs(pred_count - gt_count)
        
        # Track detailed result
        result = {
            'image_name': image_name or f"image_{self.total_predictions}",
            'predicted_count': pred_count,
            'ground_truth_count': gt_count,
            'absolute_error': error,
            'relative_error': error / gt_count if gt_count > 0 else (0 if pred_count == 0 else 1),
            'is_perfect': error == 0,
            'is_near_perfect': error <= 1,
            'is_acceptable': error <= 2
        }
        
        self.results.append(result)
        self.total_predictions += 1
        self.total_error += error
        self.max_error = max(self.max_error, error)
        
        # Update accuracy counters
        if error == 0:
            self.perfect_predictions += 1
        if error <= 1:
            self.near_perfect_1 += 1
        if error <= 2:
            self.acceptable_2 += 1
            
        # Track error distribution
        error_key = str(int(error))
        self.error_distribution[error_key] = self.error_distribution.get(error_key, 0) + 1
    
    def calculate_perfection_metrics(self):
        """Calculate comprehensive perfection metrics"""
        if self.total_predictions == 0:
            return {}
        
        # Core perfection metrics
        perfect_accuracy = (self.perfect_predictions / self.total_predictions) * 100
        near_perfect_accuracy = (self.near_perfect_1 / self.total_predictions) * 100
        acceptable_accuracy = (self.acceptable_2 / self.total_predictions) * 100
        
        # Error statistics
        average_error = self.total_error / self.total_predictions
        
        # Advanced metrics
        predictions = [r['predicted_count'] for r in self.results]
        ground_truths = [r['ground_truth_count'] for r in self.results]
        
        total_predicted = sum(predictions)
        total_ground_truth = sum(ground_truths)
        count_bias = ((total_predicted - total_ground_truth) / total_ground_truth * 100) if total_ground_truth > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truths)[0, 1] if len(predictions) > 1 else 0
        
        # Error analysis
        errors = [r['absolute_error'] for r in self.results]
        error_std = np.std(errors)
        error_median = np.median(errors)
        
        return {
            'perfect_accuracy': perfect_accuracy,
            'near_perfect_accuracy': near_perfect_accuracy,
            'acceptable_accuracy': acceptable_accuracy,
            'average_error': average_error,
            'max_error': self.max_error,
            'error_std': error_std,
            'error_median': error_median,
            'correlation': correlation,
            'count_bias': count_bias,
            'perfect_count': self.perfect_predictions,
            'total_predictions': self.total_predictions,
            'total_predicted_pellets': total_predicted,
            'total_ground_truth_pellets': total_ground_truth,
            'error_distribution': self.error_distribution
        }
    
    def get_worst_predictions(self, top_n=10):
        """Get the worst predictions for analysis"""
        sorted_results = sorted(self.results, key=lambda x: x['absolute_error'], reverse=True)
        return sorted_results[:top_n]
    
    def get_perfect_predictions(self):
        """Get all perfect predictions"""
        return [r for r in self.results if r['is_perfect']]

def load_best_model(device):
    """Load the best trained model"""
    model_paths = [
        './checkpoints/best_100percent_model.pth',
        './checkpoints/best_ultra_optimized_model.pth',
        './checkpoints/best_gpu4gb_model.pth',
        './checkpoints/best_model.pth'
    ]
    
    model = MCNN().to(device)
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"📂 Loading model from: {path}")
            checkpoint = torch.load(path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Print model info if available
                if 'perfect_accuracy' in checkpoint:
                    print(f"   Previous Perfect Accuracy: {checkpoint['perfect_accuracy']:.1f}%")
                if 'near_perfect_accuracy' in checkpoint:
                    print(f"   Previous Near-Perfect Accuracy: {checkpoint['near_perfect_accuracy']:.1f}%")
                    
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            return model, path
    
    print("❌ No trained model found!")
    return None, None

def evaluate_for_100_percent(model, dataset, device):
    """Comprehensive evaluation for 100% accuracy validation"""
    
    analyzer = PerfectionAnalyzer()
    model.eval()
    
    print("🔍 Performing 100% Accuracy Evaluation...")
    print(f"📊 Dataset: {len(dataset)} images")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating for 100% accuracy"):
            # Get data
            image, gt_dmap = dataset[idx]
            image_name = f"test_image_{idx:04d}"
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            gt_dmap = gt_dmap.unsqueeze(0).to(device)
            
            # Predict
            pred_dmap = model(image)
            
            # Calculate counts
            pred_count = pred_dmap.sum().item()
            gt_count = gt_dmap.sum().item()
            
            # Add to analyzer
            analyzer.add_prediction(pred_count, gt_count, image_name)
    
    return analyzer

def generate_100_percent_report(analyzer, model_path, save_dir='./results/100_percent_evaluation'):
    """Generate comprehensive 100% accuracy report"""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate metrics
    metrics = analyzer.calculate_perfection_metrics()
    
    # Print detailed console report
    print_100_percent_summary(metrics)
    
    # Create visualizations
    create_100_percent_plots(analyzer, save_dir, timestamp)
    
    # Generate detailed analysis
    create_detailed_analysis(analyzer, metrics, save_dir, timestamp)
    
    # Save comprehensive JSON report
    report = {
        'timestamp': timestamp,
        'model_path': model_path,
        'metrics': metrics,
        'detailed_results': analyzer.results,
        'worst_predictions': analyzer.get_worst_predictions(20),
        'perfect_predictions': analyzer.get_perfect_predictions()
    }
    
    with open(f"{save_dir}/100_percent_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    return metrics

def print_100_percent_summary(metrics):
    """Print comprehensive 100% accuracy summary"""
    
    print("\n" + "="*80)
    print("🏆 100% ACCURACY EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🎯 PERFECTION METRICS:")
    print(f"   Perfect Accuracy (0 errors): {metrics['perfect_accuracy']:.2f}%")
    print(f"   Near-Perfect (±1 pellet): {metrics['near_perfect_accuracy']:.2f}%")
    print(f"   Acceptable (±2 pellets): {metrics['acceptable_accuracy']:.2f}%")
    
    print(f"\n📊 ERROR ANALYSIS:")
    print(f"   Average Error: {metrics['average_error']:.3f} pellets")
    print(f"   Maximum Error: {metrics['max_error']:.0f} pellets")
    print(f"   Error Standard Deviation: {metrics['error_std']:.3f}")
    print(f"   Error Median: {metrics['error_median']:.3f}")
    
    print(f"\n🔢 COUNTING STATISTICS:")
    print(f"   Total Images: {metrics['total_predictions']}")
    print(f"   Perfect Predictions: {metrics['perfect_count']}")
    print(f"   Total Predicted Pellets: {metrics['total_predicted_pellets']:.0f}")
    print(f"   Total Ground Truth Pellets: {metrics['total_ground_truth_pellets']:.0f}")
    print(f"   Count Bias: {metrics['count_bias']:+.2f}%")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    
    print(f"\n📈 ERROR DISTRIBUTION:")
    error_dist = metrics['error_distribution']
    for error, count in sorted(error_dist.items(), key=lambda x: int(x[0])):
        percentage = (count / metrics['total_predictions']) * 100
        print(f"   {error} pellet error: {count} images ({percentage:.1f}%)")
    
    print(f"\n🏆 ACHIEVEMENT STATUS:")
    if metrics['perfect_accuracy'] >= 100.0:
        print(f"   🎉 INCREDIBLE! 100% PERFECT ACCURACY ACHIEVED!")
        print(f"   🚀 Your model has ZERO errors - every prediction is perfect!")
    elif metrics['perfect_accuracy'] >= 95.0:
        print(f"   🥇 OUTSTANDING! {metrics['perfect_accuracy']:.1f}% perfect accuracy!")
        print(f"   📈 Only {100 - metrics['perfect_accuracy']:.1f}% away from perfection!")
    elif metrics['near_perfect_accuracy'] >= 99.0:
        print(f"   🥈 EXCELLENT! {metrics['near_perfect_accuracy']:.1f}% near-perfect accuracy!")
        print(f"   🎯 Almost all predictions within ±1 pellet!")
    elif metrics['near_perfect_accuracy'] >= 95.0:
        print(f"   🥉 VERY GOOD! {metrics['near_perfect_accuracy']:.1f}% near-perfect accuracy!")
        print(f"   📊 Strong performance with room for improvement!")
    else:
        print(f"   📈 PROGRESS MADE: {metrics['near_perfect_accuracy']:.1f}% near-perfect accuracy")
        print(f"   🔧 Further training recommended for perfection!")
    
    print("="*80)

def create_100_percent_plots(analyzer, save_dir, timestamp):
    """Create specialized plots for 100% accuracy analysis"""
    
    metrics = analyzer.calculate_perfection_metrics()
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'100% Accuracy Analysis - {timestamp}', fontsize=16, fontweight='bold')
    
    # Extract data
    predictions = [r['predicted_count'] for r in analyzer.results]
    ground_truths = [r['ground_truth_count'] for r in analyzer.results]
    errors = [r['absolute_error'] for r in analyzer.results]
    
    # 1. Perfect vs Ground Truth scatter
    axes[0, 0].scatter(ground_truths, predictions, alpha=0.6, s=40)
    max_count = max(max(predictions), max(ground_truths))
    axes[0, 0].plot([0, max_count], [0, max_count], 'r--', alpha=0.8, label='Perfect Line')
    axes[0, 0].set_xlabel('Ground Truth Count')
    axes[0, 0].set_ylabel('Predicted Count')
    axes[0, 0].set_title(f'Prediction Accuracy\nr = {metrics["correlation"]:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error histogram with perfection focus
    axes[0, 1].hist(errors, bins=range(int(max(errors)) + 2), alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='green', linestyle='-', linewidth=3, label='Perfect (0 error)')
    axes[0, 1].axvline(1, color='orange', linestyle='--', linewidth=2, label='Near-Perfect (±1)')
    axes[0, 1].set_xlabel('Absolute Error (pellets)')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].set_title('Error Distribution (Focus on Perfection)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Perfection metrics bar chart
    categories = ['Perfect\n(0 error)', 'Near-Perfect\n(±1)', 'Acceptable\n(±2)']
    accuracies = [metrics['perfect_accuracy'], metrics['near_perfect_accuracy'], metrics['acceptable_accuracy']]
    colors = ['gold', 'lightgreen', 'lightblue']
    
    bars = axes[0, 2].bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 2].axhline(100, color='red', linestyle='--', linewidth=2, label='100% Target')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Perfection Metrics')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Error vs Count analysis
    axes[1, 0].scatter(ground_truths, errors, alpha=0.6, s=40)
    axes[1, 0].axhline(0, color='green', linestyle='-', linewidth=2, label='Perfect Line')
    axes[1, 0].axhline(1, color='orange', linestyle='--', linewidth=1, label='±1 Tolerance')
    axes[1, 0].set_xlabel('Ground Truth Count')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error vs Pellet Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative perfection curve
    sorted_errors = np.sort(errors)
    cumulative_perfect = np.cumsum(sorted_errors == 0) / len(errors) * 100
    cumulative_near_perfect = np.cumsum(sorted_errors <= 1) / len(errors) * 100
    
    x_range = np.arange(len(errors))
    axes[1, 1].plot(x_range, cumulative_perfect, 'g-', linewidth=2, label='Perfect Accuracy')
    axes[1, 1].plot(x_range, cumulative_near_perfect, 'orange', linewidth=2, label='Near-Perfect Accuracy')
    axes[1, 1].axhline(100, color='red', linestyle='--', alpha=0.7, label='100% Target')
    axes[1, 1].set_xlabel('Number of Images')
    axes[1, 1].set_ylabel('Cumulative Accuracy (%)')
    axes[1, 1].set_title('Cumulative Perfection Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Detailed error breakdown
    error_counts = {}
    for error in errors:
        error_int = int(error)
        error_counts[error_int] = error_counts.get(error_int, 0) + 1
    
    error_labels = sorted(error_counts.keys())
    error_values = [error_counts[e] for e in error_labels]
    
    colors_map = {0: 'gold', 1: 'lightgreen', 2: 'lightblue'}
    bar_colors = [colors_map.get(e, 'lightcoral') for e in error_labels]
    
    axes[1, 2].bar([str(e) for e in error_labels], error_values, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1, 2].set_xlabel('Error (pellets)')
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].set_title('Detailed Error Breakdown')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add percentage labels
    total_images = len(errors)
    for i, (label, value) in enumerate(zip(error_labels, error_values)):
        percentage = (value / total_images) * 100
        axes[1, 2].text(i, value + max(error_values) * 0.01, 
                       f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/100_percent_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 100% accuracy plots saved: {save_dir}/100_percent_analysis_{timestamp}.png")

def create_detailed_analysis(analyzer, metrics, save_dir, timestamp):
    """Create detailed text analysis report"""
    
    worst_predictions = analyzer.get_worst_predictions(10)
    perfect_predictions = analyzer.get_perfect_predictions()
    
    analysis_text = f"""
100% ACCURACY DETAILED ANALYSIS REPORT
======================================
Generated: {timestamp}

EXECUTIVE SUMMARY
-----------------
Perfect Accuracy Achieved: {metrics['perfect_accuracy']:.2f}%
Near-Perfect Accuracy: {metrics['near_perfect_accuracy']:.2f}%
Target Status: {'✅ 100% ACHIEVED' if metrics['perfect_accuracy'] >= 100.0 else '📈 PROGRESS TOWARD 100%'}

PERFECTION BREAKDOWN
--------------------
Total Images Evaluated: {metrics['total_predictions']}
Perfect Predictions (0 error): {metrics['perfect_count']} ({metrics['perfect_accuracy']:.2f}%)
Near-Perfect Predictions (±1): {metrics['total_predictions'] - metrics['perfect_count'] + metrics['perfect_count']} 
Imperfect Predictions: {metrics['total_predictions'] - int(metrics['near_perfect_accuracy'] * metrics['total_predictions'] / 100)}

ERROR ANALYSIS
--------------
Average Error: {metrics['average_error']:.4f} pellets
Maximum Error: {metrics['max_error']:.0f} pellets
Error Standard Deviation: {metrics['error_std']:.4f}
Error Median: {metrics['error_median']:.4f}

Count Statistics:
• Total Predicted Pellets: {metrics['total_predicted_pellets']:.0f}
• Total Ground Truth Pellets: {metrics['total_ground_truth_pellets']:.0f}
• Count Bias: {metrics['count_bias']:+.2f}%
• Correlation Coefficient: {metrics['correlation']:.6f}

ERROR DISTRIBUTION ANALYSIS
---------------------------"""

    error_dist = metrics['error_distribution']
    for error, count in sorted(error_dist.items(), key=lambda x: int(x[0])):
        percentage = (count / metrics['total_predictions']) * 100
        analysis_text += f"""
{error} pellet error: {count} images ({percentage:.2f}%)"""

    analysis_text += f"""

WORST PERFORMING IMAGES
-----------------------
Top 10 images with highest errors:"""

    for i, result in enumerate(worst_predictions[:10], 1):
        analysis_text += f"""
{i}. {result['image_name']}:
   Predicted: {result['predicted_count']:.1f} | Ground Truth: {result['ground_truth_count']:.1f}
   Error: {result['absolute_error']:.1f} pellets ({result['relative_error']*100:.1f}% relative error)"""

    if perfect_predictions:
        analysis_text += f"""

PERFECT PREDICTIONS SAMPLE
--------------------------
Sample of perfect predictions (showing first 20):"""
        
        for i, result in enumerate(perfect_predictions[:20], 1):
            analysis_text += f"""
{i}. {result['image_name']}: {result['ground_truth_count']:.0f} pellets (PERFECT!)"""
    
    analysis_text += f"""

RECOMMENDATIONS FOR 100% ACCURACY
---------------------------------"""

    if metrics['perfect_accuracy'] >= 100.0:
        analysis_text += f"""
🎉 CONGRATULATIONS! You have achieved the ultimate goal of 100% perfect accuracy!

Your MCNN model demonstrates exceptional performance:
• Every single prediction is perfectly accurate
• Zero counting errors across all test images
• Model is ready for production deployment
• Consider validating on additional datasets to confirm robustness

This level of accuracy is extremely rare and represents cutting-edge performance
in computer vision and object counting tasks.
"""
    elif metrics['perfect_accuracy'] >= 95.0:
        analysis_text += f"""
🥇 OUTSTANDING PERFORMANCE! You're extremely close to 100% perfection!

Current Status: {metrics['perfect_accuracy']:.2f}% perfect accuracy
Gap to 100%: {100 - metrics['perfect_accuracy']:.2f}%

Strategies to reach 100%:
• Fine-tune the model with focus on the {metrics['total_predictions'] - metrics['perfect_count']} imperfect cases
• Increase training data for difficult scenarios
• Apply ensemble methods with multiple models
• Implement post-processing count refinement
• Consider data augmentation for edge cases
"""
    elif metrics['near_perfect_accuracy'] >= 99.0:
        analysis_text += f"""
🥈 EXCELLENT PERFORMANCE! Near-perfect accuracy achieved!

Current Status: {metrics['near_perfect_accuracy']:.2f}% near-perfect (±1 pellet)
Perfect Accuracy: {metrics['perfect_accuracy']:.2f}%

Your model shows exceptional reliability with minimal errors. Most practical
applications would consider this performance level as "perfect" for real-world use.

To push toward 100% perfect accuracy:
• Analyze the {100 - metrics['perfect_accuracy']:.1f}% imperfect cases
• Implement advanced loss functions with higher count weights
• Use curriculum learning starting with easier images
• Consider architectural improvements or ensemble methods
"""
    else:
        analysis_text += f"""
📈 SIGNIFICANT PROGRESS MADE!

Current Performance:
• Perfect Accuracy: {metrics['perfect_accuracy']:.2f}%
• Near-Perfect Accuracy: {metrics['near_perfect_accuracy']:.2f}%

Recommendations for improvement:
• Extend training with more epochs
• Implement advanced loss functions focusing on count preservation
• Increase model capacity or use attention mechanisms
• Improve data quality and augmentation strategies
• Consider transfer learning or pre-trained features
• Analyze error patterns to identify systematic issues
"""

    analysis_text += f"""

TECHNICAL DETAILS
-----------------
Evaluation completed at: {timestamp}
Model Performance: {'PRODUCTION READY' if metrics['perfect_accuracy'] >= 95.0 else 'DEVELOPMENT STAGE'}
Recommended Use: {'Deploy immediately' if metrics['perfect_accuracy'] >= 100.0 else 'Further optimization recommended'}

This analysis provides comprehensive insights into your model's journey toward
100% perfect pellet counting accuracy. The metrics and recommendations can guide
future improvements and deployment decisions.
"""

    # Save analysis
    with open(f"{save_dir}/detailed_analysis_{timestamp}.txt", 'w') as f:
        f.write(analysis_text)
    
    print(f"📋 Detailed analysis saved: {save_dir}/detailed_analysis_{timestamp}.txt")

def main():
    """Main evaluation function for 100% accuracy assessment"""
    print("🔍 100% ACCURACY EVALUATION SYSTEM")
    print("🎯 Ultimate Precision Assessment for Feed Pellet Counting")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load best available model
    model, model_path = load_best_model(device)
    if model is None:
        print("❌ No trained model available for evaluation!")
        print("Please train a model first using perfect_100_accuracy_train.py")
        return
    
    # Load test dataset
    test_img_root = './data/test_data/images'
    test_gt_root = './data/test_data/densitymaps'
    
    if not os.path.exists(test_img_root) or not os.path.exists(test_gt_root):
        print("❌ Test data not found!")
        print("Please ensure test data is available in ./data/test_data/")
        return
    
    test_dataset = PelletDataset(
        test_img_root, test_gt_root,
        gt_downsample=4, phase='test', transform=False
    )
    
    print(f"📊 Test dataset: {len(test_dataset)} images")
    
    # Perform evaluation
    analyzer = evaluate_for_100_percent(model, test_dataset, device)
    
    # Generate comprehensive report
    metrics = generate_100_percent_report(analyzer, model_path)
    
    # Final status
    print(f"\n🏆 EVALUATION COMPLETE!")
    if metrics['perfect_accuracy'] >= 100.0:
        print(f"🎉 INCREDIBLE! 100% PERFECT ACCURACY ACHIEVED!")
        print(f"🚀 Your MCNN model has reached the ultimate goal!")
    elif metrics['perfect_accuracy'] >= 95.0:
        print(f"🥇 OUTSTANDING! {metrics['perfect_accuracy']:.1f}% perfect accuracy!")
        print(f"📈 So close to 100% perfection!")
    else:
        print(f"📊 Current Performance: {metrics['perfect_accuracy']:.1f}% perfect accuracy")
        print(f"🔧 Continue training to reach 100% perfection!")

if __name__ == "__main__":
    main()