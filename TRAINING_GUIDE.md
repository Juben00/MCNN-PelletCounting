# MCNN Pellet Counter Training Guide

## Overview
This guide provides comprehensive instructions for training an MCNN (Multi-Column Convolutional Neural Network) model for feed pellet counting using your 700-image dataset.

## Quick Start

### 1. Generate Improved Density Maps
```bash
cd data_preparation
python improved_dmap_generator.py
```

### 2. Train the Model
```bash
python improved_train.py
```

### 3. Evaluate the Model
```bash
python evaluate_model.py
```

## Detailed Training Process

### Step 1: Data Preparation

#### Density Map Generation
The improved density map generator includes:
- **Adaptive Gaussian Kernels**: Automatically adjusts sigma based on pellet density
- **Better Validation**: Ensures all annotations are within image boundaries
- **Error Handling**: Robust handling of missing or corrupted files

**Key Improvements for Pellets:**
- Smaller sigma values (0.5-3.0) suitable for pellet size
- Adaptive sigma calculation based on nearest neighbor distances
- Exact count preservation in density maps

#### Data Organization
Ensure your data follows this structure:
```
data/
├── train_data/
│   ├── images/          # Training images
│   └── densitymaps/     # Generated density maps
├── test_data/
│   ├── images/          # Test images
│   └── densitymaps/     # Generated density maps
└── via_export_json.json # VIA annotations
```

### Step 2: Training Configuration

#### Recommended Settings for 700 Images:
- **Batch Size**: 1 (due to variable image sizes)
- **Learning Rate**: 1e-5 (conservative for stability)
- **Epochs**: 100 (with early stopping)
- **Downsample Factor**: 4 (reduces computational load)
- **Weight Decay**: 1e-4 (prevents overfitting)

#### Data Augmentation:
- Horizontal flipping (50% probability)
- Random brightness/contrast (30% probability)
- Small rotations (-10° to +10°, 30% probability)
- Random scaling (0.9x to 1.1x, 30% probability)
- Gaussian noise (20% probability)

### Step 3: Training Process

#### Key Features:
- **Early Stopping**: Prevents overfitting (patience: 15 epochs)
- **Learning Rate Scheduling**: Reduces LR when validation plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Comprehensive Metrics**: MAE, RMSE, MAPE tracking
- **Visualization**: Real-time training progress plots

#### Monitoring Training:
- Watch validation MAE (primary metric)
- Ensure training/validation curves don't diverge
- Monitor learning rate adjustments
- Check prediction visualizations

### Step 4: Model Evaluation

#### Metrics:
- **MAE (Mean Absolute Error)**: Primary metric for counting
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Relative accuracy

#### Evaluation Features:
- Detailed per-image results
- Error distribution analysis
- Prediction visualizations
- Accuracy at different thresholds

## Improving Model Accuracy

### 1. Data Quality Improvements

#### Annotation Quality:
- **Consistency**: Mark pellet centers consistently
- **Completeness**: Ensure all visible pellets are annotated
- **Precision**: Use zoom for accurate point placement
- **Quality Control**: Review and correct annotations

#### Data Diversity:
- **Lighting Conditions**: Include various lighting scenarios
- **Backgrounds**: Different surface types and colors
- **Pellet Densities**: Range from sparse to dense distributions
- **Camera Angles**: Slight variations in viewing angle
- **Image Quality**: Mix of high and medium resolution images

### 2. Model Architecture Improvements

#### MCNN Enhancements:
```python
# Add batch normalization
nn.BatchNorm2d(num_features)

# Add dropout for regularization
nn.Dropout2d(p=0.2)

# Use different activation functions
nn.LeakyReLU(negative_slope=0.1)
```

#### Multi-Scale Features:
- The MCNN already uses multi-column architecture
- Consider adding attention mechanisms
- Experiment with different kernel sizes

### 3. Training Optimizations

#### Advanced Optimizers:
```python
# AdamW optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-5, 
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-7
)
```

#### Loss Function Variants:
```python
# Focal Loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')
        pt = torch.exp(-mse)
        focal_loss = self.alpha * (1-pt)**self.gamma * mse
        return focal_loss.mean()
```

### 4. Data Augmentation Strategies

#### Advanced Augmentations:
- **Gaussian Blur**: Simulate focus variations
- **Color Jittering**: Adjust hue, saturation
- **Elastic Deformation**: Subtle geometric changes
- **Cutout/Mixup**: Advanced regularization techniques

#### Domain-Specific Augmentations:
- **Pellet Occlusion Simulation**: Partially hide pellets
- **Lighting Variations**: Simulate different illumination
- **Surface Texture Changes**: Various background materials

### 5. Hyperparameter Tuning

#### Systematic Tuning:
```python
# Grid search parameters
learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
batch_sizes = [1, 2, 4]
weight_decays = [1e-5, 1e-4, 1e-3]
downsample_factors = [2, 4, 8]
```

#### Bayesian Optimization:
- Use libraries like Optuna or Hyperopt
- Optimize multiple objectives simultaneously
- Consider computational budget constraints

### 6. Ensemble Methods

#### Model Averaging:
```python
# Train multiple models with different initializations
models = [MCNN() for _ in range(5)]
# Average predictions
ensemble_pred = torch.mean(torch.stack([m(x) for m in models]), dim=0)
```

#### Test-Time Augmentation:
```python
# Apply augmentations during inference
def tta_predict(model, image, n_augmentations=8):
    predictions = []
    for _ in range(n_augmentations):
        aug_image = apply_random_augmentation(image)
        pred = model(aug_image)
        predictions.append(pred)
    return torch.mean(torch.stack(predictions), dim=0)
```

## Troubleshooting Common Issues

### 1. Overfitting
**Symptoms**: Training loss decreases, validation loss increases
**Solutions**:
- Increase data augmentation
- Add dropout/batch normalization
- Reduce model complexity
- Use early stopping

### 2. Underfitting
**Symptoms**: Both training and validation loss remain high
**Solutions**:
- Increase model capacity
- Reduce regularization
- Increase learning rate
- Train for more epochs

### 3. Unstable Training
**Symptoms**: Loss oscillates or explodes
**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Check data preprocessing
- Use batch normalization

### 4. Poor Counting Accuracy
**Symptoms**: High MAE despite low MSE loss
**Solutions**:
- Improve density map generation
- Use count-specific loss functions
- Ensure exact count preservation
- Check annotation quality

## Performance Benchmarks

### Expected Results for Pellet Counting:

#### Good Performance:
- **MAE**: < 3.0 pellets
- **MAPE**: < 10%
- **Accuracy (MAE ≤ 2)**: > 80%

#### Excellent Performance:
- **MAE**: < 1.5 pellets
- **MAPE**: < 5%
- **Accuracy (MAE ≤ 1)**: > 70%

### Factors Affecting Performance:
- **Image Quality**: Higher resolution → better accuracy
- **Pellet Size Consistency**: Uniform pellets → easier counting
- **Background Contrast**: Good contrast → better detection
- **Annotation Quality**: Precise annotations → better learning

## Best Practices Summary

1. **Start Simple**: Use baseline configuration first
2. **Monitor Carefully**: Watch validation metrics closely
3. **Iterate Systematically**: Change one thing at a time
4. **Validate Changes**: Always test on held-out data
5. **Document Results**: Keep track of experiments
6. **Use Version Control**: Save successful configurations

## File Structure After Setup

```
GITHUB_MCNN/
├── data_preparation/
│   ├── dmap_generator.py           # Original generator
│   └── improved_dmap_generator.py  # Enhanced generator
├── checkpoints/                    # Saved models
├── results/                        # Training results and plots
├── mcnn_model.py                   # Model architecture
├── my_dataloader.py               # Original dataloader
├── improved_dataloader.py         # Enhanced dataloader
├── train.py                       # Original training script
├── improved_train.py              # Enhanced training script
├── evaluate_model.py              # Model evaluation
└── README.md                      # This guide
```

## Next Steps

1. **Generate Density Maps**: Run the improved generator
2. **Start Training**: Use the enhanced training script
3. **Monitor Progress**: Watch training metrics and visualizations
4. **Evaluate Results**: Use the evaluation script for detailed analysis
5. **Iterate and Improve**: Apply accuracy improvement techniques

Remember: Deep learning is iterative. Start with the baseline, understand your data, and gradually apply improvements while monitoring performance metrics.