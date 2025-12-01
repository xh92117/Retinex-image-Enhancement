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
    Calculate comprehensive image quality metrics
    
    Args:
        img_enhanced (torch.Tensor or np.ndarray): Enhanced image
        img_reference (torch.Tensor or np.ndarray): Reference image (optional, for PSNR/SSIM)
        
    Returns:
        metrics (dict): Dictionary of metrics
    """
    metrics = {}
    
    # Convert to numpy if needed
    if isinstance(img_enhanced, torch.Tensor):
        img_enhanced_np = img_enhanced.cpu().detach().numpy()
    else:
        img_enhanced_np = img_enhanced.copy()
    
    # Remove batch dimension if present
    if img_enhanced_np.ndim == 4:
        img_enhanced_np = img_enhanced_np.squeeze(0)
    
    # Transpose to [H, W, C] if needed
    if img_enhanced_np.shape[0] == 3:
        img_enhanced_np = img_enhanced_np.transpose(1, 2, 0)
    
    # === 基础指标 ===
    
    # Mean brightness
    mean_brightness = np.mean(img_enhanced_np)
    metrics['mean_brightness'] = float(mean_brightness)
    
    # Contrast (standard deviation)
    contrast = np.std(img_enhanced_np)
    metrics['contrast'] = float(contrast)
    
    # Entropy (information content)
    hist, _ = np.histogram(img_enhanced_np.flatten(), bins=256, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    metrics['entropy'] = float(entropy)
    
    # === 无参考质量指标 ===
    
    # NIQE (Natural Image Quality Evaluator)
    try:
        metrics['niqe'] = calculate_niqe(img_enhanced_np)
    except Exception as e:
        metrics['niqe'] = None
        print(f"NIQE calculation failed: {e}")
    
    # === 有参考质量指标 ===
    
    if img_reference is not None:
        # Convert reference to numpy
        if isinstance(img_reference, torch.Tensor):
            img_reference_np = img_reference.cpu().detach().numpy()
        else:
            img_reference_np = img_reference.copy()
        
        if img_reference_np.ndim == 4:
            img_reference_np = img_reference_np.squeeze(0)
        
        if img_reference_np.shape[0] == 3:
            img_reference_np = img_reference_np.transpose(1, 2, 0)
        
        # PSNR
        metrics['psnr'] = calculate_psnr(img_enhanced_np, img_reference_np)
        
        # SSIM
        metrics['ssim'] = calculate_ssim(img_enhanced_np, img_reference_np)
        
        # MSE
        metrics['mse'] = float(np.mean((img_enhanced_np - img_reference_np) ** 2))
    
    # === 低光增强专用指标 ===
    
    # LOE (Lightness Order Error) - 需要原始低光图像
    # 这里暂不实现，因为需要原始图像
    
    # 色彩饱和度
    metrics['saturation'] = calculate_saturation(img_enhanced_np)
    
    # 自然度评分（基于色彩分布）
    metrics['naturalness'] = calculate_naturalness(img_enhanced_np)
    
    return metrics


def calculate_psnr(img1, img2):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1 (np.ndarray): 第一张图像 [H, W, C]
        img2 (np.ndarray): 第二张图像 [H, W, C]
        
    Returns:
        float: PSNR值（dB）
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1, img2):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1 (np.ndarray): 第一张图像 [H, W, C]
        img2 (np.ndarray): 第二张图像 [H, W, C]
        
    Returns:
        float: SSIM值 [0, 1]
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = np.ones((11, 11)) / 121
    
    def convolve2d(img, kernel):
        from scipy.ndimage import convolve
        return convolve(img, kernel, mode='constant')
    
    # 对每个通道计算SSIM
    ssim_channels = []
    for i in range(img1.shape[2]):
        mu1 = convolve2d(img1[:, :, i], kernel)
        mu2 = convolve2d(img2[:, :, i], kernel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve2d(img1[:, :, i] ** 2, kernel) - mu1_sq
        sigma2_sq = convolve2d(img2[:, :, i] ** 2, kernel) - mu2_sq
        sigma12 = convolve2d(img1[:, :, i] * img2[:, :, i], kernel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_channels.append(np.mean(ssim_map))
    
    return float(np.mean(ssim_channels))


def calculate_niqe(img):
    """
    计算NIQE (Natural Image Quality Evaluator) - 简化版本
    
    Args:
        img (np.ndarray): 图像 [H, W, C]
        
    Returns:
        float: NIQE分数（越低越好）
    """
    # 简化的NIQE实现，基于图像统计特性
    # 真实的NIQE需要预训练的模型
    
    # 转换为灰度
    if len(img.shape) == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img
    
    # 计算局部统计特性
    from scipy.ndimage import uniform_filter
    
    mu = uniform_filter(gray, size=7)
    sigma = np.sqrt(uniform_filter(gray**2, size=7) - mu**2)
    
    # 简化的质量评分
    score = np.mean(sigma) / (np.std(mu) + 1e-8)
    
    return float(score)


def calculate_saturation(img):
    """
    计算图像色彩饱和度
    
    Args:
        img (np.ndarray): 图像 [H, W, C]
        
    Returns:
        float: 平均饱和度 [0, 1]
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        return 0.0
    
    # RGB to HSV
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    
    # 饱和度 = (max - min) / max
    saturation = np.zeros_like(max_val)
    mask = max_val > 1e-8
    saturation[mask] = (max_val[mask] - min_val[mask]) / max_val[mask]
    
    return float(np.mean(saturation))


def calculate_naturalness(img):
    """
    计算图像自然度评分
    基于色彩分布和对比度
    
    Args:
        img (np.ndarray): 图像 [H, W, C]
        
    Returns:
        float: 自然度评分 [0, 1]，越高越自然
    """
    # 1. 检查色彩分布的均衡性
    color_balance = 1.0 - np.std([np.mean(img[:, :, i]) for i in range(3)])
    
    # 2. 检查对比度是否合理
    contrast = np.std(img)
    contrast_score = 1.0 - abs(contrast - 0.15) / 0.15  # 理想对比度约为0.15
    contrast_score = max(0, min(1, contrast_score))
    
    # 3. 检查亮度是否合理
    brightness = np.mean(img)
    brightness_score = 1.0 - abs(brightness - 0.5) / 0.5  # 理想亮度约为0.5
    brightness_score = max(0, min(1, brightness_score))
    
    # 综合评分
    naturalness = 0.3 * color_balance + 0.4 * contrast_score + 0.3 * brightness_score
    
    return float(naturalness)


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

