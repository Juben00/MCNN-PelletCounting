# enhanced_mcnn_model.py
import torch
import torch.nn as nn

class EnhancedMCNN(nn.Module):
    '''
    An Enhanced MCNN architecture with Batch Normalization and Dropout for improved stability and performance.
    '''
    def __init__(self, load_weights=False):
        super(EnhancedMCNN, self).__init__()

        # Branch 1: Large filters
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2), # Add dropout for regularization
            nn.Conv2d(16, 8, 7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Medium filters
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Small filters
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fuse = nn.Sequential(nn.Conv2d(30, 1, 1, padding=0))

        if not load_weights:
            self._initialize_weights()

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)