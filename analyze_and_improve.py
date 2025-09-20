import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_results():
    """Analyze current training results and identify improvement areas"""
    
    print("=== MCNN Training Results Analysis ===")
    
    # Check for existing results
    results_files = [
        './results/training_log.json',
        './results/enhanced_training_results.json'
    ]
    
    results_data = None
    for file_path in results_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results_data = json.load(f)
            print(f"Found results: {file_path}")
            break
    
    if results_data is None:
        print("No training results found. Please run training first.")
        return
    
    # Analyze validation metrics
    if 'val_metrics' in results_data:
        val_metrics = results_data['val_metrics']
    elif 'val_metrics_history' in results_data:
        val_metrics = results_data['val_metrics_history']
    else:
        print("No validation metrics found in results.")
        return
    
    # Extract MAE values
    mae_values = []
    for metric in val_metrics:
        if isinstance(metric, dict) and 'mae' in metric:
            mae_values.append(metric['mae'])
        elif isinstance(metric, (int, float)):
            mae_values.append(metric)
    
    if not mae_values:
        print("No MAE values found in results.")
        return
    
    # Calculate statistics
    best_mae = min(mae_values)
    worst_mae = max(mae_values)
    avg_mae = np.mean(mae_values)
    final_mae = mae_values[-1] if mae_values else None
    
    print(f"\n📊 MAE Analysis:")
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Worst MAE: {worst_mae:.2f}")
    print(f"   Average MAE: {avg_mae:.2f}")
    print(f"   Final MAE: {final_mae:.2f}")
    print(f"   Improvement needed: {best_mae - 3.0:.2f} (target: ≤3.0)")
    
    # Analyze training progression
    epochs = len(mae_values)
    convergence_epoch = None
    for i, mae in enumerate(mae_values):
        if mae == best_mae:
            convergence_epoch = i + 1
            break
    
    print(f"\n📈 Training Analysis:")
    print(f"   Total epochs: {epochs}")
    print(f"   Best result at epoch: {convergence_epoch}")
    if convergence_epoch and convergence_epoch < epochs * 0.5:
        print("   ⚠️  Model converged early - might benefit from longer training")
    
    return best_mae, mae_values

def check_data_quality():
    """Analyze data quality issues that might affect accuracy"""
    
    print("\n=== Data Quality Analysis ===")
    
    data_issues = []
    
    # Check for enhanced density maps
    enhanced_train_path = "./data/train_data/densitymaps_enhanced"
    enhanced_test_path = "./data/test_data/densitymaps_enhanced"
    
    if not os.path.exists(enhanced_train_path):
        data_issues.append("Enhanced density maps not generated")
        print("❌ Enhanced density maps not found")
        print("   Run: python data_preparation/enhanced_dmap_generator.py")
    else:
        print("✅ Enhanced density maps available")
    
    # Check annotation file
    annotation_files = [
        "./data/via_export_json.json",
        "./data/original_796.json"
    ]
    
    annotation_found = False
    for ann_file in annotation_files:
        if os.path.exists(ann_file):
            annotation_found = True
            print(f"✅ Annotation file found: {ann_file}")
            break
    
    if not annotation_found:
        data_issues.append("No annotation file found")
        print("❌ No annotation file found")
    
    # Check data distribution
    train_images = "./data/train_data/images"
    test_images = "./data/test_data/images"
    
    if os.path.exists(train_images) and os.path.exists(test_images):
        train_count = len([f for f in os.listdir(train_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(test_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"📊 Data Distribution:")
        print(f"   Training images: {train_count}")
        print(f"   Test images: {test_count}")
        
        if train_count < 400:
            data_issues.append("Insufficient training data")
            print("   ⚠️  Consider adding more training data")
        
        if test_count < 50:
            data_issues.append("Insufficient test data")
            print("   ⚠️  Consider adding more test data")
    
    return data_issues

def generate_improvement_plan(current_mae, data_issues):
    """Generate a specific improvement plan based on current performance"""
    
    print(f"\n=== Improvement Plan for MAE {current_mae:.2f} ===")
    
    improvements = []
    
    # Critical improvements (MAE > 10)
    if current_mae > 10:
        print("🚨 Critical Issues (High Priority):")
        improvements.extend([
            "1. Check annotation quality - ensure all pellets are marked",
            "2. Verify density map generation is working correctly",
            "3. Use enhanced density map generator with adaptive sigma",
            "4. Check for data preprocessing issues",
            "5. Validate ground truth counts manually for sample images"
        ])
    
    # High priority improvements (MAE 5-10)
    elif current_mae > 5:
        print("⚠️  High Priority Improvements:")
        improvements.extend([
            "1. Use enhanced training script with combined loss function",
            "2. Generate enhanced density maps with multi-scale approach",
            "3. Increase training epochs to 150+ with better early stopping",
            "4. Use lower learning rate (5e-6) for better convergence",
            "5. Implement gradient accumulation for larger effective batch size"
        ])
    
    # Medium priority improvements (MAE 3-5)
    elif current_mae > 3:
        print("📈 Medium Priority Improvements:")
        improvements.extend([
            "1. Fine-tune hyperparameters (learning rate, loss weights)",
            "2. Add more data augmentation techniques",
            "3. Use ensemble methods (train multiple models)",
            "4. Implement test-time augmentation",
            "5. Consider architectural modifications to MCNN"
        ])
    
    # Fine-tuning improvements (MAE < 3)
    else:
        print("🎯 Fine-tuning Improvements:")
        improvements.extend([
            "1. Use advanced optimization techniques",
            "2. Implement sophisticated loss functions",
            "3. Add regularization techniques",
            "4. Use curriculum learning",
            "5. Consider model ensemble or distillation"
        ])
    
    # Data-specific improvements
    if data_issues:
        print("\n📋 Data Quality Improvements:")
        for issue in data_issues:
            improvements.append(f"- Fix: {issue}")
    
    # Print improvement list
    for i, improvement in enumerate(improvements[:8], 1):  # Top 8 improvements
        print(f"   {improvement}")
    
    return improvements

def create_quick_fixes():
    """Create immediate actionable fixes"""
    
    print(f"\n=== Quick Fixes to Try Now ===")
    
    fixes = [
        {
            'name': 'Generate Enhanced Density Maps',
            'command': 'cd data_preparation && python enhanced_dmap_generator.py',
            'description': 'Better density maps with adaptive sigma and multi-scale generation'
        },
        {
            'name': 'Use Enhanced Training',
            'command': 'python enhanced_train.py',
            'description': 'Advanced loss functions and optimization techniques'
        },
        {
            'name': 'Lower Learning Rate',
            'description': 'Change learning_rate to 2e-6 in training config',
            'benefit': 'Better convergence and stability'
        },
        {
            'name': 'Increase Count Loss Weight',
            'description': 'In enhanced_train.py, increase count_weight to 50.0',
            'benefit': 'Direct optimization of counting accuracy'
        },
        {
            'name': 'More Training Epochs',
            'description': 'Increase num_epochs to 200 with patience=30',
            'benefit': 'Allow model more time to converge'
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['name']}")
        if 'command' in fix:
            print(f"   Command: {fix['command']}")
        print(f"   {fix['description']}")
        if 'benefit' in fix:
            print(f"   Benefit: {fix['benefit']}")

def main():
    """Main analysis function"""
    
    # Analyze current results
    try:
        current_mae, mae_history = analyze_training_results()
    except:
        current_mae = 17.18  # User's reported MAE
        mae_history = []
        print(f"Using reported MAE: {current_mae}")
    
    # Check data quality
    data_issues = check_data_quality()
    
    # Generate improvement plan
    improvements = generate_improvement_plan(current_mae, data_issues)
    
    # Provide quick fixes
    create_quick_fixes()
    
    # Final recommendations
    print(f"\n=== Priority Actions ===")
    if current_mae > 10:
        print("🎯 Focus on: Data quality and basic training setup")
        print("   1. Run enhanced_dmap_generator.py")
        print("   2. Verify annotations are correct")
        print("   3. Use enhanced_train.py with combined loss")
    elif current_mae > 5:
        print("🎯 Focus on: Advanced training techniques")
        print("   1. Use enhanced training with combined loss")
        print("   2. Lower learning rate to 2e-6")
        print("   3. Increase training epochs to 150+")
    else:
        print("🎯 Focus on: Fine-tuning and advanced techniques")
        print("   1. Hyperparameter optimization")
        print("   2. Ensemble methods")
        print("   3. Test-time augmentation")
    
    print(f"\n📈 Expected improvement: MAE {current_mae:.2f} → 3-8 (with enhanced methods)")
    print("🎉 Good pellet counting typically achieves MAE < 3.0")

if __name__ == "__main__":
    main()