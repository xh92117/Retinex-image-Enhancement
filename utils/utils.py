"""
UP-Retinex: Utility Functions
"""

import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def visualize_results(img_low, img_enhanced, illu_map, save_path=None):
    """
    Visualize input, enhanced image, and illumination map
    
    Args:
        img_low (torch.Tensor): Input low-light image [1, 3, H, W] or [3, H, W]
        img_enhanced (torch.Tensor): Enhanced image [1, 3, H, W] or [3, H, W]
        illu_map (torch.Tensor): Illumination map [1, 3, H, W] or [3, H, W]
        save_path (str): Path to save visualization (optional)
    """
    # Remove batch dimension if present
    if img_low.dim() == 4:
        img_low = img_low.squeeze(0)
    if img_enhanced.dim() == 4:
        img_enhanced = img_enhanced.squeeze(0)
    if illu_map.dim() == 4:
        illu_map = illu_map.squeeze(0)
    
    # Convert to numpy and transpose to [H, W, C]
    img_low = img_low.cpu().detach().numpy().transpose(1, 2, 0)
    img_enhanced = img_enhanced.cpu().detach().numpy().transpose(1, 2, 0)
    illu_map = illu_map.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Convert illumination to grayscale
    illu_gray = np.mean(illu_map, axis=2)
    
    # Clip to [0, 1]
    img_low = np.clip(img_low, 0, 1)
    img_enhanced = np.clip(img_enhanced, 0, 1)
    illu_gray = np.clip(illu_gray, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot input
    axes[0].imshow(img_low)
    axes[0].set_title('Input (Low-light)', fontsize=14)
    axes[0].axis('off')
    
    # Plot enhanced
    axes[1].imshow(img_enhanced)
    axes[1].set_title('Enhanced', fontsize=14)
    axes[1].axis('off')
    
    # Plot illumination
    axes[2].imshow(illu_gray, cmap='gray')
    axes[2].set_title('Illumination Map', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file
    
    Args:
        log_file (str): Path to training log file
        save_path (str): Path to save plot (optional)
    """
    # This is a placeholder - implement based on your logging format
    # You would typically parse the log file and extract loss values
    
    # Example implementation:
    # losses = parse_log_file(log_file)
    # plt.figure(figsize=(10, 6))
    # plt.plot(losses['total'], label='Total Loss')
    # plt.plot(losses['exposure'], label='Exposure Loss')
    # ...
    # plt.legend()
    # plt.savefig(save_path)
    
    print("Training curve plotting not implemented yet")


def calculate_metrics(img_enhanced, img_reference=None):
    """
    Calculate image quality metrics
    
    Args:
        img_enhanced (torch.Tensor): Enhanced image
        img_reference (torch.Tensor): Reference image (optional, for PSNR/SSIM)
        
    Returns:
        metrics (dict): Dictionary of metrics
    """
    metrics = {}
    
    # Convert to numpy
    if isinstance(img_enhanced, torch.Tensor):
        img_enhanced = img_enhanced.cpu().detach().numpy()
    
    # Remove batch dimension if present
    if img_enhanced.ndim == 4:
        img_enhanced = img_enhanced.squeeze(0)
    
    # Transpose to [H, W, C] if needed
    if img_enhanced.shape[0] == 3:
        img_enhanced = img_enhanced.transpose(1, 2, 0)
    
    # Calculate mean brightness
    mean_brightness = np.mean(img_enhanced)
    metrics['mean_brightness'] = float(mean_brightness)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(img_enhanced)
    metrics['contrast'] = float(contrast)
    
    # Calculate entropy (information content)
    hist, _ = np.histogram(img_enhanced.flatten(), bins=256, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    metrics['entropy'] = float(entropy)
    
    return metrics


def create_gif(image_paths, output_path, duration=500):
    """
    Create GIF from a list of images
    
    Args:
        image_paths (list): List of image file paths
        output_path (str): Path to save GIF
        duration (int): Duration per frame in milliseconds
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Created GIF: {output_path}")


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def count_parameters(model):
    """
    Count total and trainable parameters in a model
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        total_params (int): Total number of parameters
        trainable_params (int): Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model):
    """
    Print model summary
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    
    # Test visualization
    import torch
    
    # Create dummy data
    img_low = torch.rand(1, 3, 256, 256) * 0.3
    img_enhanced = torch.rand(1, 3, 256, 256) * 0.8
    illu_map = torch.rand(1, 3, 256, 256) * 0.5 + 0.3
    
    print("Testing visualization...")
    visualize_results(img_low, img_enhanced, illu_map, save_path="test_visualization.png")
    
    print("Testing metrics calculation...")
    metrics = calculate_metrics(img_enhanced)
    print("Metrics:", metrics)
    
    print("\nAll tests passed!")

