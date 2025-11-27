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
        B, C, H, W = illu_map.shape
        
        # Reshape to [B, C, H*W]
        illu_flat = illu_map.view(B, C, -1)
        refl_flat = reflectance.view(B, C, -1)
        
        # Compute global means
        illu_mean = torch.mean(illu_flat, dim=2, keepdim=True)  # [B, C, 1]
        refl_mean = torch.mean(refl_flat, dim=2, keepdim=True)  # [B, C, 1]
        
        # Center the data
        illu_centered = illu_flat - illu_mean
        refl_centered = refl_flat - refl_mean
        
        # Compute covariance matrix
        covariance = torch.bmm(illu_centered, refl_centered.transpose(1, 2)) / (H * W - 1)
        
        # Frobenius norm of covariance (independence constraint)
        cov_loss = torch.norm(covariance, p='fro') ** 2
        
        # Mean difference constraint
        mean_diff_loss = F.mse_loss(illu_mean, refl_mean)
        
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


class TotalLoss(nn.Module):
    """
    Total Loss Function for UP-Retinex
    
    L_total = L_exp + ω_smooth * L_smooth + ω_col * L_col + ω_spa * L_spa + ω_decouple * L_decouple + ω_perceptual * L_perceptual
    
    Default weights (as per specification):
    - L_exp: 10.0
    - L_smooth: 1.0
    - L_col: 0.5
    - L_spa: 1.0
    - L_decouple: 0.1
    - L_perceptual: 1.0
    """
    def __init__(self, 
                 weight_exp=10.0,
                 weight_smooth=1.0,
                 weight_col=0.5,
                 weight_spa=1.0,
                 weight_decouple=0.1,
                 weight_perceptual=1.0):
        super(TotalLoss, self).__init__()
        
        # Initialize individual loss functions
        self.exposure_loss = AdaptiveExposureLoss()
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
        self.color_loss = ColorLoss()
        self.spatial_loss = SpatialConsistencyLoss()
        self.decouple_loss = IlluminationReflectanceDecouplingLoss()
        self.perceptual_loss = PerceptualLoss()
        
        # Loss weights
        self.weight_exp = weight_exp
        self.weight_smooth = weight_smooth
        self.weight_col = weight_col
        self.weight_spa = weight_spa
        self.weight_decouple = weight_decouple
        self.weight_perceptual = weight_perceptual
        
    def forward(self, img_low, img_enhanced, illu_map, reflectance=None):
        """
        Compute total loss
        
        Args:
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            illu_map (torch.Tensor): Illumination map I [B, C, H, W]
            reflectance (torch.Tensor, optional): Reflectance map R [B, C, H, W]
            
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
            loss_decouple = 0.0
        
        # Compute weighted total loss
        total_loss = (self.weight_exp * loss_exp +
                     self.weight_smooth * loss_smooth +
                     self.weight_col * loss_col +
                     self.weight_spa * loss_spa +
                     self.weight_decouple * loss_decouple +
                     self.weight_perceptual * loss_perceptual)
        
        # Create dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'exposure': loss_exp.item(),
            'smoothness': loss_smooth.item(),
            'color': loss_col.item(),
            'spatial': loss_spa.item(),
            'decouple': loss_decouple.item() if reflectance is not None else 0.0,
            'perceptual': loss_perceptual.item()
        }
        
        return total_loss, loss_dict


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