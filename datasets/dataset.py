"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Dataset Implementation
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# Add letterbox utility
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.letterbox import letterbox_tensor


class LowLightDataset(Dataset):
    """
    Custom Dataset for Low-Light Images
    
    Features:
    - Loads images from a folder
    - Converts to RGB
    - Resizes/crops images
    - Normalizes to [0, 1]
    - No labels/ground truth needed (unsupervised learning)
    """
    def __init__(self, 
                 image_dir, 
                 image_size=640, 
                 random_crop=True,
                 augment=True):
        """
        Args:
            image_dir (str): Path to directory containing low-light images
            image_size (int): Target image size (for training crops)
            random_crop (bool): If True, use random crops; else center crop
            augment (bool): Apply data augmentation (flip, rotation)
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.random_crop = random_crop
        self.augment = augment
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {image_dir}")
        
    def _get_image_files(self):
        """
        Get list of image files from directory
        Supports: .jpg, .jpeg, .png, .bmp
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get image at index idx
        
        Returns:
            img (torch.Tensor): Image tensor [3, H, W], range [0, 1]
        """
        # Load image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Convert PIL to tensor
        img_tensor = transforms.ToTensor()(img)
        
        # Apply letterbox preprocessing to maintain aspect ratio
        img_tensor, _, _ = letterbox_tensor(
            img_tensor, 
            new_shape=self.image_size, 
            auto=True, 
            scaleup=True
        )
        
        # Data augmentation (only applied to training data)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
            
            # Random vertical flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([1, 2, 3])  # 1=90°, 2=180°, 3=270°
                img_tensor = torch.rot90(img_tensor, k=angle, dims=[1, 2])
        
        return img_tensor


class LowLightTestDataset(Dataset):
    """
    Test Dataset for Low-Light Images (No augmentation, full resolution)
    """
    def __init__(self, image_dir, max_size=None):
        """
        Args:
            image_dir (str): Path to directory containing test images
            max_size (int): Maximum image dimension (optional, for memory constraints)
        """
        self.image_dir = image_dir
        self.max_size = max_size
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Loaded {len(self.image_files)} test images from {image_dir}")
        
    def _get_image_files(self):
        """Get list of image files from directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get image at index idx
        
        Returns:
            img (torch.Tensor): Image tensor [3, H, W], range [0, 1]
            img_name (str): Image filename
        """
        # Load image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Convert PIL to tensor
        img_tensor = transforms.ToTensor()(img)
        
        # Apply letterbox preprocessing if max_size is specified
        if self.max_size is not None:
            img_tensor, _, _ = letterbox_tensor(
                img_tensor, 
                new_shape=self.max_size, 
                auto=True, 
                scaleup=False
            )
        else:
            # If no max_size specified, still apply letterbox to maintain aspect ratio
            # but without scaling up
            img_tensor, _, _ = letterbox_tensor(
                img_tensor, 
                new_shape=img_tensor.shape[1:],  # Keep original size
                auto=True, 
                scaleup=False
            )
        
        # Get image filename
        img_name = os.path.basename(img_path)
        
        return img_tensor, img_name


def get_train_dataloader(image_dir, 
                         batch_size=8, 
                         image_size=640,
                         num_workers=4,
                         shuffle=True):
    """
    Create training dataloader
    
    Args:
        image_dir (str): Path to training images
        batch_size (int): Batch size
        image_size (int): Image size for training crops
        num_workers (int): Number of worker threads
        shuffle (bool): Shuffle data
        
    Returns:
        dataloader (DataLoader): Training dataloader
    """
    dataset = LowLightDataset(
        image_dir=image_dir,
        image_size=image_size,
        random_crop=True,
        augment=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_test_dataloader(image_dir,
                        batch_size=1,
                        max_size=None,
                        num_workers=2):
    """
    Create test dataloader
    
    Args:
        image_dir (str): Path to test images
        batch_size (int): Batch size (usually 1 for testing)
        max_size (int): Maximum image dimension
        num_workers (int): Number of worker threads
        
    Returns:
        dataloader (DataLoader): Test dataloader
    """
    dataset = LowLightTestDataset(
        image_dir=image_dir,
        max_size=max_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <image_directory>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        sys.exit(1)
    
    # Test training dataset
    print("=" * 60)
    print("Testing Training Dataset")
    print("=" * 60)
    
    train_loader = get_train_dataloader(
        image_dir=image_dir,
        batch_size=4,
        image_size=640,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"Number of batches: {len(train_loader)}")
    
    # Get one batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Value range: [{batch.min():.4f}, {batch.max():.4f}]")
        break
    
    print("=" * 60)
    print("Dataset test passed!")

