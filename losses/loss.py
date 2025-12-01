"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Loss Functions Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdaptiveExposureLoss(nn.Module):
    """
    Adaptive Exposure Control Loss (L_exp)
    
    Dynamically adjusts the target exposure level based on the input image brightness.
    
    Formula: L_exp = (1/M) * Σ |Y_k - E_adaptive|
    where:
    - Y_k: Average intensity value of a local region k in R
    - E_adaptive: Adaptive target exposure based on global image statistics
    - M: Number of local regions (patches)
    """
    def __init__(self, patch_size=16, base_target_exposure=0.6):
        super(AdaptiveExposureLoss, self).__init__()
        self.patch_size = patch_size
        self.base_target_exposure = base_target_exposure
        
    def forward(self, img_enhanced, img_low):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        B, C, H, W = img_enhanced.shape
        
        # Convert to grayscale (average across channels)
        gray_enhanced = torch.mean(img_enhanced, dim=1, keepdim=True)  # [B, 1, H, W]
        gray_low = torch.mean(img_low, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Calculate global mean brightness of input image
        global_mean = torch.mean(gray_low)
        
        # Adaptively adjust target exposure based on input brightness
        # Darker images need higher target exposure
        adaptive_target = self.base_target_exposure + (0.8 - self.base_target_exposure) * (1 - global_mean)
        
        # Divide image into non-overlapping patches
        # Use average pooling to get mean intensity per patch
        mean_intensity = F.avg_pool2d(gray_enhanced, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Calculate loss: average absolute difference from adaptive target exposure
        loss = torch.mean(torch.abs(mean_intensity - adaptive_target))
        
        return loss


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge-Aware Smoothness Loss (L_smooth)
    
    Enhances the original smoothness loss by incorporating edge detection for better
    preservation of important structural details while maintaining smoothness in flat regions.
    
    Formula: L_smooth = Σ ||∇I ⊙ exp(-λ|∇S|) ⊙ (1 + α|∇E|)||_1
    where:
    - ∇: Gradient operator (horizontal and vertical)
    - λ: Sensitivity factor (default: 10)
    - ⊙: Element-wise multiplication
    - E: Edge map computed from Sobel operators
    - α: Edge sensitivity factor
    """
    def __init__(self, lambda_val=10.0, alpha=1.0):
        super(EdgeAwareSmoothnessLoss, self).__init__()
        self.lambda_val = lambda_val
        self.alpha = alpha
        
        # Sobel filters for edge detection
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        self.sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        
    def gradient(self, img):
        """
        Compute image gradients (horizontal and vertical)
        
        Args:
            img (torch.Tensor): Input image [B, C, H, W]
            
        Returns:
            grad_h (torch.Tensor): Horizontal gradient
            grad_v (torch.Tensor): Vertical gradient
        """
        # Horizontal gradient (difference along width)
        grad_h = img[:, :, :, :-1] - img[:, :, :, 1:]
        
        # Vertical gradient (difference along height)
        grad_v = img[:, :, :-1, :] - img[:, :, 1:, :]
        
        return grad_h, grad_v
    
    def compute_edge_map(self, img):
        """
        Compute edge map using Sobel operators
        
        Args:
            img (torch.Tensor): Input image [B, C, H, W]
            
        Returns:
            edge_map (torch.Tensor): Edge map [B, 1, H, W]
        """
        # Convert to grayscale if needed
        if img.shape[1] > 1:
            gray = torch.mean(img, dim=1, keepdim=True)
        else:
            gray = img
            
        # Pad to maintain size
        padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients using Sobel operators
        grad_x = F.conv2d(padded, self.sobel_x, padding=0)
        grad_y = F.conv2d(padded, self.sobel_y, padding=0)
        
        # Compute edge magnitude
        edge_map = torch.sqrt(grad_x**2 + grad_y**2)
        
        return edge_map
    
    def forward(self, illu_map, img_low):
        """
        Args:
            illu_map (torch.Tensor): Predicted illumination map I [B, C, H, W]
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # Compute gradients of illumination map
        illu_grad_h, illu_grad_v = self.gradient(illu_map)
        
        # Compute gradients of input image
        img_grad_h, img_grad_v = self.gradient(img_low)
        
        # Compute gradient magnitudes of input image
        img_grad_h_mag = torch.abs(img_grad_h)
        img_grad_v_mag = torch.abs(img_grad_v)
        
        # Compute edge map
        edge_map = self.compute_edge_map(img_low)
        
        # Compute weights: exp(-λ|∇S|)
        # Edges in S have large gradients -> small weights (preserve edges)
        # Smooth areas in S have small gradients -> large weights (enforce smoothness)
        weight_h = torch.exp(-self.lambda_val * torch.mean(img_grad_h_mag, dim=1, keepdim=True))
        weight_v = torch.exp(-self.lambda_val * torch.mean(img_grad_v_mag, dim=1, keepdim=True))
        
        # Edge enhancement factor: (1 + α|∇E|)
        edge_factor_h = 1 + self.alpha * F.avg_pool2d(edge_map, kernel_size=(1, weight_h.shape[3]), stride=1)[:, :, :, :-1]
        edge_factor_v = 1 + self.alpha * F.avg_pool2d(edge_map, kernel_size=(weight_v.shape[2], 1), stride=1)[:, :, :-1, :]
        
        # Compute weighted smoothness loss with edge awareness
        loss_h = torch.mean(weight_h * edge_factor_h * torch.abs(illu_grad_h))
        loss_v = torch.mean(weight_v * edge_factor_v * torch.abs(illu_grad_v))
        
        loss = loss_h + loss_v
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual Quality Loss (L_perceptual)
    
    Measures the perceptual difference between enhanced and input images using VGG features.
    This loss helps to preserve high-level visual quality and natural appearance.
    
    Formula: L_perceptual = Σ ||φ(R) - φ(S)||_2^2
    where:
    - φ: VGG feature extractor
    - R: Enhanced image
    - S: Input low-light image
    """
    def __init__(self, device='cpu'):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        
        # Define feature extraction layers (using relu activations)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # Split VGG into slices
        for i, layer in enumerate(vgg.children()):
            if i <= 4:  # conv1_2
                self.slice1.add_module(str(i), layer)
            elif i <= 9:  # conv2_2
                self.slice2.add_module(str(i), layer)
            elif i <= 18:  # conv4_2
                self.slice3.add_module(str(i), layer)
            if i >= 18:
                break
                
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization factors for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize(self, x):
        """Normalize image for VGG input"""
        return (x - self.mean) / self.std
        
    def forward(self, img_enhanced, img_low):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # Normalize images for VGG
        norm_enhanced = self.normalize(img_enhanced)
        norm_low = self.normalize(img_low)
        
        # Extract features
        feat_enhanced_1 = self.slice1(norm_enhanced)
        feat_enhanced_2 = self.slice2(feat_enhanced_1)
        feat_enhanced_3 = self.slice3(feat_enhanced_2)
        
        feat_low_1 = self.slice1(norm_low)
        feat_low_2 = self.slice2(feat_low_1)
        feat_low_3 = self.slice3(feat_low_2)
        
        # Compute perceptual loss at multiple levels
        loss_1 = F.mse_loss(feat_enhanced_1, feat_low_1)
        loss_2 = F.mse_loss(feat_enhanced_2, feat_low_2)
        loss_3 = F.mse_loss(feat_enhanced_3, feat_low_3)
        
        # Weighted combination
        loss = loss_1 + loss_2 + loss_3
        
        return loss


class IlluminationReflectanceDecouplingLoss(nn.Module):
    """
    Illumination-Reflectance Decoupling Loss (L_decouple)
    
    Ensures proper decoupling between illumination and reflectance maps by enforcing
    independence constraints based on Retinex theory.
    
    Formula: L_decouple = ||cov(I, R)||_F^2 + λ * (||I_mean - R_mean||_2^2)
    where:
    - cov(I, R): Covariance between illumination and reflectance maps
    - I_mean, R_mean: Global mean values
    - λ: Weight factor
    """
    def __init__(self, lambda_val=0.1):
        super(IlluminationReflectanceDecouplingLoss, self).__init__()
        self.lambda_val = lambda_val
        
    def forward(self, illu_map, reflectance):
        """
        Args:
            illu_map (torch.Tensor): Illumination map I [B, C, H, W]
            reflectance (torch.Tensor): Reflectance map R [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        B, C_illu, H, W = illu_map.shape
        _, C_refl, _, _ = reflectance.shape
        
        # Reshape to [B, C, H*W]
        # Handle different channel dimensions between illumination (1 channel) and reflectance (3 channels)
        illu_flat = illu_map.view(B, C_illu, -1)
        refl_flat = reflectance.view(B, C_refl, -1)
        
        # Compute global means
        illu_mean = torch.mean(illu_flat, dim=2, keepdim=True)  # [B, C_illu, 1]
        refl_mean = torch.mean(refl_flat, dim=2, keepdim=True)  # [B, C_refl, 1]
        
        # Center the data
        illu_centered = illu_flat - illu_mean
        refl_centered = refl_flat - refl_mean
        
        # For covariance computation, we need to handle different channel counts
        # We compute the covariance between each illumination channel and each reflectance channel
        if C_illu == C_refl:
            # Same number of channels - direct covariance computation
            covariance = torch.bmm(illu_centered, refl_centered.transpose(1, 2)) / (H * W - 1)
        else:
            # Different number of channels - compute cross-covariance
            # Expand illumination to match reflectance channels for meaningful comparison
            if C_illu == 1 and C_refl == 3:
                # Common case: single-channel illumination vs 3-channel reflectance
                # Replicate illumination channel to match reflectance
                illu_replicated = illu_flat.expand(B, C_refl, -1)
                covariance = torch.bmm(illu_replicated, refl_centered.transpose(1, 2)) / (H * W - 1)
            else:
                # General case: compute covariance for compatible dimensions
                # Take the first channel of illumination and all channels of reflectance
                illu_first_channel = illu_flat[:, :1, :]  # [B, 1, H*W]
                covariance = torch.bmm(illu_first_channel, refl_centered.transpose(1, 2)) / (H * W - 1)
        
        # Frobenius norm of covariance (independence constraint)
        cov_loss = torch.norm(covariance, p='fro') ** 2
        
        # Mean difference constraint - only compare compatible channels
        if C_illu == C_refl:
            mean_diff_loss = F.mse_loss(illu_mean, refl_mean)
        else:
            # For different channel counts, compare illumination with mean of reflectance
            refl_mean_avg = torch.mean(refl_mean, dim=1, keepdim=True)  # [B, 1, 1]
            illu_mean_avg = torch.mean(illu_mean, dim=1, keepdim=True)   # [B, 1, 1]
            mean_diff_loss = F.mse_loss(illu_mean_avg, refl_mean_avg)
        
        # Total loss
        loss = cov_loss + self.lambda_val * mean_diff_loss
        
        return loss


class ColorLoss(nn.Module):
    """
    Color Constancy Loss (L_col)
    
    Based on the Gray-World assumption to correct color deviations.
    Assumes that the average of each color channel should be similar.
    
    Formula: L_col = Σ (J^p - J^q)^2 for all (p,q) ∈ (R,G,B), p ≠ q
    where:
    - J^p: Average intensity of channel p in the enhanced image R
    """
    def __init__(self):
        super(ColorLoss, self).__init__()
        
    def forward(self, img_enhanced):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image R [B, 3, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # Compute mean intensity for each channel
        mean_r = torch.mean(img_enhanced[:, 0, :, :])
        mean_g = torch.mean(img_enhanced[:, 1, :, :])
        mean_b = torch.mean(img_enhanced[:, 2, :, :])
        
        # Compute pairwise differences
        loss_rg = (mean_r - mean_g) ** 2
        loss_rb = (mean_r - mean_b) ** 2
        loss_gb = (mean_g - mean_b) ** 2
        
        loss = loss_rg + loss_rb + loss_gb
        
        return loss


class SpatialConsistencyLoss(nn.Module):
    """
    Spatial Consistency Loss (L_spa)
    
    Preserves the texture structure of the input S in the output R.
    Ensures that the gradient patterns are similar between input and output.
    
    Formula: L_spa = (1/K) * Σ ||∇R_i - ∇S_i||^2
    where:
    - K: Number of regions
    - Operates on small local regions (patches)
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        
    def gradient(self, img):
        """
        Compute image gradients (horizontal and vertical)
        
        Args:
            img (torch.Tensor): Input image [B, C, H, W]
            
        Returns:
            grad_h (torch.Tensor): Horizontal gradient
            grad_v (torch.Tensor): Vertical gradient
        """
        # Horizontal gradient
        grad_h = img[:, :, :, :-1] - img[:, :, :, 1:]
        
        # Vertical gradient
        grad_v = img[:, :, :-1, :] - img[:, :, 1:, :]
        
        return grad_h, grad_v
    
    def forward(self, img_enhanced, img_low):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # Compute gradients
        enh_grad_h, enh_grad_v = self.gradient(img_enhanced)
        low_grad_h, low_grad_v = self.gradient(img_low)
        
        # Compute L2 loss between gradients
        loss_h = torch.mean((enh_grad_h - low_grad_h) ** 2)
        loss_v = torch.mean((enh_grad_v - low_grad_v) ** 2)
        
        loss = loss_h + loss_v
        
        return loss


class FrequencyLoss(nn.Module):
    """
    Frequency Domain Loss (L_freq)
    
    在频域中保持图像的高频细节，确保增强后的图像保留原始纹理信息。
    使用FFT将图像转换到频域，比较幅度谱的差异。
    
    特点：
    - 保持图像的频率特性
    - 重点关注高频细节（纹理）
    - 使用傅里叶变换分析
    """
    def __init__(self, weight_high=1.0, weight_low=0.5):
        super(FrequencyLoss, self).__init__()
        self.weight_high = weight_high  # 高频权重
        self.weight_low = weight_low    # 低频权重
    
    def forward(self, img_enhanced, img_low):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image [B, C, H, W]
            img_low (torch.Tensor): Input low-light image [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # 转换到频域 (使用2D FFT)
        fft_enhanced = torch.fft.fft2(img_enhanced, dim=(-2, -1))
        fft_low = torch.fft.fft2(img_low, dim=(-2, -1))
        
        # 计算幅度谱
        mag_enhanced = torch.abs(fft_enhanced)
        mag_low = torch.abs(fft_low)
        
        # 创建高频和低频掩码
        B, C, H, W = img_enhanced.shape
        high_freq_mask, low_freq_mask = self._create_frequency_masks(H, W, img_enhanced.device)
        
        # 扩展掩码以匹配批次和通道维度
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # 计算高频损失（保持细节）
        high_freq_loss = F.mse_loss(
            mag_enhanced * high_freq_mask,
            mag_low * high_freq_mask
        )
        
        # 计算低频损失（整体亮度）
        low_freq_loss = F.mse_loss(
            mag_enhanced * low_freq_mask,
            mag_low * low_freq_mask
        )
        
        # 加权组合
        loss = self.weight_high * high_freq_loss + self.weight_low * low_freq_loss
        
        return loss
    
    def _create_frequency_masks(self, H, W, device):
        """
        创建高频和低频掩码
        
        Args:
            H (int): 图像高度
            W (int): 图像宽度
            device: 设备
            
        Returns:
            high_freq_mask: 高频掩码
            low_freq_mask: 低频掩码
        """
        # 创建坐标网格
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 计算到中心的距离
        dist = torch.sqrt((x - center_w).float() ** 2 + (y - center_h).float() ** 2)
        
        # 定义高频和低频的阈值（低频半径为图像对角线的1/4）
        radius = min(H, W) // 4
        
        # 创建掩码
        low_freq_mask = (dist <= radius).float()
        high_freq_mask = (dist > radius).float()
        
        return high_freq_mask, low_freq_mask


def calculate_texture_complexity(img, method='tv'):
    """
    计算图像的纹理复杂度
    
    Args:
        img (torch.Tensor): 输入图像 [B, C, H, W]
        method (str): 纹理复杂度计算方法，可选值：'tv' (全变分) 或 'edge_density' (边缘密度)
        
    Returns:
        complexity (torch.Tensor): 纹理复杂度值 [B]
    """
    B, C, H, W = img.shape
    
    if method == 'tv':
        # 全变分（Total Variation）：计算图像梯度的L1范数
        # 水平梯度
        grad_h = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        # 垂直梯度
        grad_v = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        
        # 计算每个通道的TV值，然后求平均
        tv_h = torch.mean(grad_h, dim=(1, 2, 3))
        tv_v = torch.mean(grad_v, dim=(1, 2, 3))
        
        # 总TV值
        complexity = tv_h + tv_v
        
    elif method == 'edge_density':
        # 边缘密度：使用Sobel边缘检测计算边缘像素比例
        # 转换为灰度图
        if C > 1:
            gray = torch.mean(img, dim=1, keepdim=True)
        else:
            gray = img
        
        # Sobel滤波器
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=img.device).view(1, 1, 3, 3)
        
        # 扩展滤波器以匹配批次和通道
        sobel_x = sobel_x.expand(1, gray.shape[1], 3, 3)
        sobel_y = sobel_y.expand(1, gray.shape[1], 3, 3)
        
        # 计算边缘
        padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(padded, sobel_x, padding=0)
        grad_y = F.conv2d(padded, sobel_y, padding=0)
        
        # 计算边缘幅度
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 二值化边缘（使用自适应阈值）
        threshold = torch.mean(edge_mag, dim=(1, 2, 3), keepdim=True) * 1.5
        edge_mask = (edge_mag > threshold).float()
        
        # 计算边缘密度
        complexity = torch.mean(edge_mask, dim=(1, 2, 3))
    else:
        raise ValueError(f"不支持的纹理复杂度计算方法: {method}")
    
    return complexity


class TotalLoss(nn.Module):
    """
    Total Loss Function for UP-Retinex (Improved Version)
    
    L_total = L_exp + ω_smooth * L_smooth + ω_col * L_col + ω_spa * L_spa + 
              ω_decouple * L_decouple + ω_perceptual * L_perceptual + ω_freq * L_freq
    
    Default weights (as per specification):
    - L_exp: 10.0
    - L_smooth: 1.0 (动态调整)
    - L_col: 0.5
    - L_spa: 1.0
    - L_decouple: 0.1
    - L_perceptual: 1.0
    - L_freq: 0.5 (新增)
    
    特点：
    - 支持频域损失
    - 可选的自适应权重调整
    - 根据纹理复杂度动态调整平滑损失权重
    """
    def __init__(self, 
                 weight_exp=10.0,
                 weight_smooth=1.0,
                 weight_col=0.5,
                 weight_spa=1.0,
                 weight_decouple=0.1,
                 weight_perceptual=1.0,
                 weight_freq=0.5,
                 use_freq_loss=True,
                 adaptive_weights=False,
                 use_dynamic_smooth_weight=True,
                 texture_method='tv'):
        super(TotalLoss, self).__init__()
        
        # Initialize individual loss functions
        self.exposure_loss = AdaptiveExposureLoss()
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
        self.color_loss = ColorLoss()
        self.spatial_loss = SpatialConsistencyLoss()
        self.decouple_loss = IlluminationReflectanceDecouplingLoss()
        self.perceptual_loss = PerceptualLoss()
        self.frequency_loss = FrequencyLoss() if use_freq_loss else None
        
        # Loss weights
        self.weight_exp = weight_exp
        self.weight_smooth = weight_smooth
        self.weight_col = weight_col
        self.weight_spa = weight_spa
        self.weight_decouple = weight_decouple
        self.weight_perceptual = weight_perceptual
        self.weight_freq = weight_freq
        
        self.use_freq_loss = use_freq_loss
        self.adaptive_weights = adaptive_weights
        self.use_dynamic_smooth_weight = use_dynamic_smooth_weight
        self.texture_method = texture_method
        
        # 用于自适应权重调整的历史记录
        if adaptive_weights:
            self.loss_history = {
                'exposure': [],
                'smoothness': [],
                'color': [],
                'spatial': [],
                'decouple': [],
                'perceptual': [],
                'frequency': []
            }
        
    def forward(self, img_low, img_enhanced, illu_map, reflectance=None, epoch=0):
        """
        Compute total loss with optional frequency loss and adaptive weights
        
        Args:
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            illu_map (torch.Tensor): Illumination map I [B, C, H, W]
            reflectance (torch.Tensor, optional): Reflectance map R [B, C, H, W]
            epoch (int): Current training epoch (for adaptive weights)
            
        Returns:
            total_loss (torch.Tensor): Scalar total loss
            loss_dict (dict): Dictionary containing individual loss values
        """
        # Compute individual losses
        loss_exp = self.exposure_loss(img_enhanced, img_low)
        loss_smooth = self.smoothness_loss(illu_map, img_low)
        loss_col = self.color_loss(img_enhanced)
        loss_spa = self.spatial_loss(img_enhanced, img_low)
        loss_perceptual = self.perceptual_loss(img_enhanced, img_low)
        
        # Compute decoupling loss if reflectance is provided
        if reflectance is not None:
            loss_decouple = self.decouple_loss(illu_map, reflectance)
        else:
            loss_decouple = torch.tensor(0.0, device=img_low.device)
        
        # Compute frequency loss if enabled
        if self.use_freq_loss and self.frequency_loss is not None:
            loss_freq = self.frequency_loss(img_enhanced, img_low)
        else:
            loss_freq = torch.tensor(0.0, device=img_low.device)
        
        # 获取当前权重（可能是自适应的）
        if self.adaptive_weights and epoch > 1:
            weights = self._compute_adaptive_weights()
        else:
            weights = {
                'exposure': self.weight_exp,
                'smoothness': self.weight_smooth,
                'color': self.weight_col,
                'spatial': self.weight_spa,
                'decouple': self.weight_decouple,
                'perceptual': self.weight_perceptual,
                'frequency': self.weight_freq
            }
        
        # 根据纹理复杂度动态调整平滑损失权重
        if self.use_dynamic_smooth_weight:
            # 计算输入图像的纹理复杂度
            texture_complexity = calculate_texture_complexity(img_low, method=self.texture_method)
            
            # 计算批次的平均纹理复杂度
            avg_complexity = torch.mean(texture_complexity)
            
            # 动态调整平滑损失权重：
            # - 纹理复杂度高的图像：降低平滑权重，保留更多细节
            # - 纹理复杂度低的图像：提高平滑权重，增强平滑效果
            # 权重范围：0.1 - 5.0
            dynamic_smooth_weight = self.weight_smooth * (1.0 - avg_complexity * 0.8)
            dynamic_smooth_weight = torch.clamp(dynamic_smooth_weight, 0.1, 5.0)
            
            # 更新权重
            weights['smoothness'] = dynamic_smooth_weight
        
        # Compute weighted total loss
        total_loss = (weights['exposure'] * loss_exp +
                     weights['smoothness'] * loss_smooth +
                     weights['color'] * loss_col +
                     weights['spatial'] * loss_spa +
                     weights['decouple'] * loss_decouple +
                     weights['perceptual'] * loss_perceptual +
                     weights['frequency'] * loss_freq)
        
        # 更新损失历史（用于自适应权重）
        if self.adaptive_weights:
            self.loss_history['exposure'].append(loss_exp.item())
            self.loss_history['smoothness'].append(loss_smooth.item())
            self.loss_history['color'].append(loss_col.item())
            self.loss_history['spatial'].append(loss_spa.item())
            self.loss_history['decouple'].append(loss_decouple.item() if isinstance(loss_decouple, torch.Tensor) else loss_decouple)
            self.loss_history['perceptual'].append(loss_perceptual.item())
            self.loss_history['frequency'].append(loss_freq.item() if isinstance(loss_freq, torch.Tensor) else loss_freq)
        
        # Create dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'exposure': loss_exp.item(),
            'smoothness': loss_smooth.item(),
            'color': loss_col.item(),
            'spatial': loss_spa.item(),
            'decouple': loss_decouple.item() if isinstance(loss_decouple, torch.Tensor) else loss_decouple,
            'perceptual': loss_perceptual.item(),
            'frequency': loss_freq.item() if isinstance(loss_freq, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict
    
    def _compute_adaptive_weights(self):
        """
        使用Dynamic Weight Average (DWA)算法计算自适应权重
        根据损失的相对变化率动态调整权重
        """
        weights = {}
        temperature = 2.0
        
        for key in self.loss_history:
            if len(self.loss_history[key]) >= 2:
                # 计算损失变化率
                current = self.loss_history[key][-1]
                previous = self.loss_history[key][-2]
                
                # 避免除零
                if previous > 1e-8:
                    ratio = current / previous
                else:
                    ratio = 1.0
                
                # 使用指数加权
                weights[key] = ratio / temperature
            else:
                # 使用默认权重
                weight_map = {
                    'exposure': self.weight_exp,
                    'smoothness': self.weight_smooth,
                    'color': self.weight_col,
                    'spatial': self.weight_spa,
                    'decouple': self.weight_decouple,
                    'perceptual': self.weight_perceptual,
                    'frequency': self.weight_freq
                }
                weights[key] = weight_map.get(key, 1.0)
        
        # 归一化权重
        if weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                num_losses = len(weights)
                for key in weights:
                    weights[key] = num_losses * weights[key] / total_weight
        
        return weights


if __name__ == "__main__":
    # Test all loss functions
    import torch
    
    # Create sample images
    batch_size, channels, height, width = 2, 3, 64, 64
    img_low = torch.rand(batch_size, channels, height, width)
    img_enhanced = torch.rand(batch_size, channels, height, width)
    illu_map = torch.rand(batch_size, channels, height, width)
    reflectance = torch.rand(batch_size, channels, height, width)
    
    # Test AdaptiveExposureLoss
    exp_loss_fn = AdaptiveExposureLoss()
    exp_loss = exp_loss_fn(img_enhanced, img_low)
    print(f"Adaptive Exposure Loss: {exp_loss.item():.4f}")
    
    # Test EdgeAwareSmoothnessLoss
    smooth_loss_fn = EdgeAwareSmoothnessLoss()
    smooth_loss = smooth_loss_fn(illu_map, img_low)
    print(f"Edge-Aware Smoothness Loss: {smooth_loss.item():.4f}")
    
    # Test PerceptualLoss
    perceptual_loss_fn = PerceptualLoss()
    perceptual_loss = perceptual_loss_fn(img_enhanced, img_low)
    print(f"Perceptual Loss: {perceptual_loss.item():.4f}")
    
    # Test ColorLoss
    color_loss_fn = ColorLoss()
    color_loss = color_loss_fn(img_enhanced)
    print(f"Color Loss: {color_loss.item():.4f}")
    
    # Test SpatialConsistencyLoss
    spatial_loss_fn = SpatialConsistencyLoss()
    spatial_loss = spatial_loss_fn(img_enhanced, img_low)
    print(f"Spatial Consistency Loss: {spatial_loss.item():.4f}")
    
    # Test IlluminationReflectanceDecouplingLoss
    decouple_loss_fn = IlluminationReflectanceDecouplingLoss()
    decouple_loss = decouple_loss_fn(illu_map, reflectance)
    print(f"Illumination-Reflectance Decoupling Loss: {decouple_loss.item():.4f}")
    
    # Test TotalLoss
    total_loss_fn = TotalLoss()
    total_loss, loss_dict = total_loss_fn(img_low, img_enhanced, illu_map, reflectance)
    print(f"Total Loss: {total_loss.item():.4f}")
    print("Individual Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")