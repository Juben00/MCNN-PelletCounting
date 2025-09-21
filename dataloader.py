from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

class PelletDataset(Dataset):
    '''
    Enhanced Dataset for Pellet Counting with Data Augmentation
    '''
    def __init__(self, img_root, gt_dmap_root, gt_downsample=1, 
                 phase='train', transform=True, normalize=True):
        '''
        img_root: the root path of images
        gt_dmap_root: the root path of ground-truth density maps
        gt_downsample: downsample factor for the model output
        phase: 'train', 'val', or 'test'
        transform: whether to apply data augmentation
        normalize: whether to normalize images
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample
        self.phase = phase
        self.transform = transform and (phase == 'train')
        self.normalize = normalize

        # Get list of images
        self.img_names = [filename for filename in os.listdir(img_root) 
                         if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.n_samples = len(self.img_names)
        
        print(f"{phase} dataset: {self.n_samples} images")

        # Image normalization (ImageNet stats work well for most cases)
        if self.normalize:
            self.norm_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        
        img_name = self.img_names[index]
        
        # Load image
        img_path = os.path.join(self.img_root, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Load density map
        dmap_name = img_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')
        gt_dmap = np.load(os.path.join(self.gt_dmap_root, dmap_name))

        # Apply data augmentation if in training phase
        if self.transform:
            img, gt_dmap = self.apply_augmentations(img, gt_dmap)

        # Handle downsampling
        if self.gt_downsample > 1:
            # Resize image to be divisible by downsample factor
            ds_rows = int(img.shape[0] // self.gt_downsample) * self.gt_downsample
            ds_cols = int(img.shape[1] // self.gt_downsample) * self.gt_downsample
            
            img = cv2.resize(img, (ds_cols, ds_rows))
            
            # Downsample density map
            target_rows = ds_rows // self.gt_downsample
            target_cols = ds_cols // self.gt_downsample
            gt_dmap = cv2.resize(gt_dmap, (target_cols, target_rows))
            
            # Adjust density values for downsampling
            gt_dmap = gt_dmap * (self.gt_downsample ** 2)
        
        # Convert to torch tensors
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0,1]
        
        # Apply normalization if enabled
        if self.normalize:
            img_tensor = self.norm_transform(img_tensor)
        
        gt_dmap_tensor = torch.tensor(gt_dmap[np.newaxis, :, :], dtype=torch.float32)

        return img_tensor, gt_dmap_tensor

    def apply_augmentations(self, img, gt_dmap):
        """Apply data augmentation techniques suitable for counting tasks"""
        
        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            gt_dmap = cv2.flip(gt_dmap, 1)
        
        # Random brightness and contrast adjustment
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random rotation (small angles only to preserve pellet positions)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            img, gt_dmap = self.rotate_image_and_dmap(img, gt_dmap, angle)
        
        # Random scaling (small factors only)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            img, gt_dmap = self.scale_image_and_dmap(img, gt_dmap, scale_factor)
        
        # Random Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        return img, gt_dmap

    def rotate_image_and_dmap(self, img, gt_dmap, angle):
        """Rotate image and density map by given angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Rotate density map
        dmap_rotated = cv2.warpAffine(gt_dmap, M, (w, h), flags=cv2.INTER_LINEAR)
        
        return img_rotated, dmap_rotated

    def scale_image_and_dmap(self, img, gt_dmap, scale_factor):
        """Scale image and density map by given factor"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Scale image
        img_scaled = cv2.resize(img, (new_w, new_h))
        
        # Scale density map
        dmap_scaled = cv2.resize(gt_dmap, (new_w, new_h))
        
        # Crop or pad to original size
        if scale_factor > 1:  # Crop
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img_scaled = img_scaled[start_y:start_y+h, start_x:start_x+w]
            dmap_scaled = dmap_scaled[start_y:start_y+h, start_x:start_x+w]
        else:  # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img_scaled = cv2.copyMakeBorder(img_scaled, pad_y, h-new_h-pad_y, 
                                          pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
            dmap_scaled = cv2.copyMakeBorder(dmap_scaled, pad_y, h-new_h-pad_y, 
                                           pad_x, w-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
        
        return img_scaled, dmap_scaled

def create_data_loaders(train_img_root, train_gt_root, 
                       val_img_root, val_gt_root,
                       test_img_root, test_gt_root,
                       batch_size=1, gt_downsample=4, num_workers=0):
    """Create data loaders for training, validation, and testing"""
    
    # Create datasets
    train_dataset = PelletDataset(train_img_root, train_gt_root, 
                                 gt_downsample=gt_downsample, 
                                 phase='train', transform=True)
    
    val_dataset = PelletDataset(val_img_root, val_gt_root, 
                               gt_downsample=gt_downsample, 
                               phase='val', transform=False)
    
    test_dataset = PelletDataset(test_img_root, test_gt_root, 
                                gt_downsample=gt_downsample, 
                                phase='test', transform=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# Test code
if __name__ == "__main__":
    img_root = "../data/train_data/images"
    gt_dmap_root = "../data/train_data/densitymaps"
    
    dataset = PelletDataset(img_root, gt_dmap_root, gt_downsample=4, phase='train')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        img, gt_dmap = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Density map shape: {gt_dmap.shape}")
        print(f"Density map sum (pellet count): {gt_dmap.sum():.2f}")