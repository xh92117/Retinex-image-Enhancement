"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Loss Functions Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExposureLoss(nn.Module):
    """
    Exposure Control Loss (L_exp)
    
    Constrains the average intensity of the enhanced image R to a target level E.
    
    Formula: L_exp = (1/M) * Σ |Y_k - E|
    where:
    - Y_k: Average intensity value of a local region k in R
    - E: Target exposure value (default: 0.6)
    - M: Number of local regions (patches)
    """
    def __init__(self, patch_size=16, target_exposure=0.6):
        super(ExposureLoss, self).__init__()
        self.patch_size = patch_size
        self.target_exposure = target_exposure
        
    def forward(self, img_enhanced):
        """
        Args:
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        B, C, H, W = img_enhanced.shape
        
        # Convert to grayscale (average across channels)
        gray = torch.mean(img_enhanced, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Divide image into non-overlapping patches
        # Use average pooling to get mean intensity per patch
        mean_intensity = F.avg_pool2d(gray, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Calculate loss: average absolute difference from target exposure
        loss = torch.mean(torch.abs(mean_intensity - self.target_exposure))
        
        return loss


class SmoothnessLoss(nn.Module):
    """
    Structure-Aware Smoothness Loss (L_smooth)
    
    Ensures illumination map I is smooth where S is smooth, but retains edges
    where S has gradients (to prevent halo effects).
    
    Formula: L_smooth = Σ ||∇I ⊙ exp(-λ|∇S|)||_1
    where:
    - ∇: Gradient operator (horizontal and vertical)
    - λ: Sensitivity factor (default: 10)
    - ⊙: Element-wise multiplication
    """
    def __init__(self, lambda_val=10.0):
        super(SmoothnessLoss, self).__init__()
        self.lambda_val = lambda_val
        
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
        
        # Compute weights: exp(-λ|∇S|)
        # Edges in S have large gradients -> small weights (preserve edges)
        # Smooth areas in S have small gradients -> large weights (enforce smoothness)
        weight_h = torch.exp(-self.lambda_val * torch.mean(img_grad_h_mag, dim=1, keepdim=True))
        weight_v = torch.exp(-self.lambda_val * torch.mean(img_grad_v_mag, dim=1, keepdim=True))
        
        # Compute weighted smoothness loss
        loss_h = torch.mean(weight_h * torch.abs(illu_grad_h))
        loss_v = torch.mean(weight_v * torch.abs(illu_grad_v))
        
        loss = loss_h + loss_v
        
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
    
    L_total = L_exp + ω_smooth * L_smooth + ω_col * L_col + ω_spa * L_spa
    
    Default weights (as per specification):
    - L_exp: 10.0
    - L_smooth: 1.0
    - L_col: 0.5
    - L_spa: 1.0
    """
    def __init__(self, 
                 weight_exp=10.0,
                 weight_smooth=1.0,
                 weight_col=0.5,
                 weight_spa=1.0):
        super(TotalLoss, self).__init__()
        
        # Initialize individual loss functions
        self.exposure_loss = ExposureLoss()
        self.smoothness_loss = SmoothnessLoss()
        self.color_loss = ColorLoss()
        self.spatial_loss = SpatialConsistencyLoss()
        
        # Loss weights
        self.weight_exp = weight_exp
        self.weight_smooth = weight_smooth
        self.weight_col = weight_col
        self.weight_spa = weight_spa
        
    def forward(self, img_low, img_enhanced, illu_map):
        """
        Compute total loss
        
        Args:
            img_low (torch.Tensor): Input low-light image S [B, C, H, W]
            img_enhanced (torch.Tensor): Enhanced image R [B, C, H, W]
            illu_map (torch.Tensor): Illumination map I [B, C, H, W]
            
        Returns:
            total_loss (torch.Tensor): Scalar total loss
            loss_dict (dict): Dictionary containing individual loss values
        """
        # Compute individual losses
        loss_exp = self.exposure_loss(img_enhanced)
        loss_smooth = self.smoothness_loss(illu_map, img_low)
        loss_col = self.color_loss(img_enhanced)
        loss_spa = self.spatial_loss(img_enhanced, img_low)
        
        # Compute weighted total loss
        total_loss = (self.weight_exp * loss_exp +
                     self.weight_smooth * loss_smooth +
                     self.weight_col * loss_col +
                     self.weight_spa * loss_spa)
        
        # Create dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'exposure': loss_exp.item(),
            'smoothness': loss_smooth.item(),
            'color': loss_col.item(),
            'spatial': loss_spa.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    batch_size = 2
    img_size = 256
    img_low = torch.rand(batch_size, 3, img_size, img_size).to(device) * 0.3  # Dark image
    img_enhanced = torch.rand(batch_size, 3, img_size, img_size).to(device) * 0.8  # Bright image
    illu_map = torch.rand(batch_size, 3, img_size, img_size).to(device) * 0.5 + 0.3  # Illumination
    
    # Initialize total loss
    criterion = TotalLoss().to(device)
    
    # Compute loss
    total_loss, loss_dict = criterion(img_low, img_enhanced, illu_map)
    
    print("=" * 60)
    print("Loss Functions Test")
    print("=" * 60)
    print(f"Total Loss: {loss_dict['total']:.6f}")
    print(f"  - Exposure Loss: {loss_dict['exposure']:.6f}")
    print(f"  - Smoothness Loss: {loss_dict['smoothness']:.6f}")
    print(f"  - Color Loss: {loss_dict['color']:.6f}")
    print(f"  - Spatial Loss: {loss_dict['spatial']:.6f}")
    print("=" * 60)
    print("Loss functions test passed!")

