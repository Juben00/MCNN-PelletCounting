#!/usr/bin/env python3
"""
Enhanced MCNN Model for Pellet Counting
=======================================

This module contains an improved MCNN architecture specifically optimized
for feed pellet counting with attention mechanisms and feature enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention(nn.Module):
    """Channel Attention Module for focusing on important features"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on pellet regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class AttentionBlock(nn.Module):
    """Combined Channel and Spatial Attention"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class EnhancedMCNNBranch(nn.Module):
    """Enhanced MCNN branch with attention and better feature extraction"""
    def __init__(self, input_channels, filters, kernels, branch_name):
        super(EnhancedMCNNBranch, self).__init__()
        self.branch_name = branch_name
        
        layers = []
        in_ch = input_channels
        
        for i, (filt, kern) in enumerate(zip(filters, kernels)):
            # Convolutional layer
            layers.append(nn.Conv2d(in_ch, filt, kern, padding=kern//2))
            layers.append(nn.BatchNorm2d(filt))  # Add batch normalization
            layers.append(nn.ReLU(inplace=True))
            
            # Add pooling after first two layers for spatial reduction
            if i < 2:
                layers.append(nn.MaxPool2d(2))
            
            # Add attention after certain layers
            if i == 1:  # Add attention after second layer
                layers.append(AttentionBlock(filt))
            
            in_ch = filt
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PelletMCNN(nn.Module):
    """
    Enhanced Multi-column CNN specifically optimized for pellet counting
    with attention mechanisms and improved feature extraction
    """
    def __init__(self, load_weights=False):
        super(PelletMCNN, self).__init__()
        
        # Enhanced Branch 1: Fine details (small receptive field)
        self.branch1 = EnhancedMCNNBranch(
            input_channels=3,
            filters=[16, 32, 24, 12],
            kernels=[9, 7, 5, 3],
            branch_name="fine"
        )
        
        # Enhanced Branch 2: Medium details (medium receptive field)  
        self.branch2 = EnhancedMCNNBranch(
            input_channels=3,
            filters=[20, 40, 30, 15],
            kernels=[7, 5, 3, 3],
            branch_name="medium"
        )
        
        # Enhanced Branch 3: Coarse details (large receptive field)
        self.branch3 = EnhancedMCNNBranch(
            input_channels=3,
            filters=[24, 48, 36, 18],
            kernels=[5, 3, 3, 3],
            branch_name="coarse"
        )
        
        # Total channels from all branches: 12 + 15 + 18 = 45
        total_channels = 12 + 15 + 18
        
        # Enhanced fusion with attention
        self.fusion_attention = AttentionBlock(total_channels)
        
        # Multi-scale fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels//2, 3, padding=1),
            nn.BatchNorm2d(total_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels//2, total_channels//4, 3, padding=1),
            nn.BatchNorm2d(total_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels//4, 1, 1, padding=0)
        )
        
        # Density refinement layer
        self.density_refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.ReLU(inplace=True)  # Ensure non-negative density
        )
        
        if not load_weights:
            self._initialize_weights()
    
    def forward(self, img_tensor):
        # Extract features from all three branches
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        
        # Concatenate features from all branches
        x = torch.cat((x1, x2, x3), 1)
        
        # Apply attention to fused features
        x = self.fusion_attention(x)
        
        # Fuse features to generate initial density map
        x = self.fusion(x)
        
        # Refine density map
        x = self.density_refine(x)
        
        return x
    
    def _initialize_weights(self):
        """Enhanced weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Backward compatibility - use enhanced version by default
class MCNN(PelletMCNN):
    """Enhanced MCNN - backward compatible with original interface"""
    def __init__(self, load_weights=False):
        super(MCNN, self).__init__(load_weights)

# For comparison, keep original MCNN available
class OriginalMCNN(nn.Module):
    """Original MCNN implementation for comparison"""
    def __init__(self, load_weights=False):
        super(OriginalMCNN, self).__init__()

        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0))

        if not load_weights:
            self._initialize_weights()

    def forward(self, img_tensor):
        x1=self.branch1(img_tensor)
        x2=self.branch2(img_tensor)
        x3=self.branch3(img_tensor)
        x=torch.cat((x1,x2,x3),1)
        x=self.fuse(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Test code
if __name__ == "__main__":
    # Test enhanced MCNN
    print("Testing Enhanced MCNN...")
    img = torch.rand((1, 3, 800, 1200), dtype=torch.float)
    
    # Enhanced model
    enhanced_mcnn = PelletMCNN()
    enhanced_out = enhanced_mcnn(img)
    print(f"Enhanced MCNN output shape: {enhanced_out.shape}")
    
    # Original model for comparison
    original_mcnn = OriginalMCNN()
    original_out = original_mcnn(img)
    print(f"Original MCNN output shape: {original_out.shape}")
    
    # Count parameters
    enhanced_params = sum(p.numel() for p in enhanced_mcnn.parameters())
    original_params = sum(p.numel() for p in original_mcnn.parameters())
    
    print(f"Enhanced MCNN parameters: {enhanced_params:,}")
    print(f"Original MCNN parameters: {original_params:,}")
    print(f"Parameter increase: {((enhanced_params - original_params) / original_params * 100):.1f}%")