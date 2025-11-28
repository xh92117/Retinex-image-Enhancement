"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Model Architecture Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedFAM(nn.Module):
    """
    Enhanced Feature Aggregation Module (FAM) with Channel and Spatial Attention
    
    Architecture:
    - Branch 1: 1x1 Conv
    - Branch 2: 3x3 MaxPool -> 1x1 Conv
    - Branch 3: 3x3 Conv -> 3x3 Conv (Cascaded)
    - Branch 4: 3x3 Conv -> 3x3 Conv (Cascaded with dilation)
    - Fusion: Concatenate all branches -> 1x1 Conv
    - Channel Attention: Adaptive feature recalibration
    - Spatial Attention: Spatial feature enhancement
    """
    def __init__(self, in_channels, out_channels):
        super(EnhancedFAM, self).__init__()
        
        # Original FAM branches
        # Branch 1: 1x1 Conv
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Branch 2: 3x3 MaxPool -> 1x1 Conv
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Branch 3: 3x3 Conv -> 3x3 Conv (Cascaded)
        self.branch3_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Branch 4: 3x3 Conv -> 3x3 Dilated Conv (Cascaded)
        self.branch4_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch4_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        
        # Fusion layer: Reduce concatenated channels back to out_channels
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, padding=0)
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Branch 1
        b1 = self.branch1(x)
        
        # Branch 2
        b2 = self.branch2_pool(x)
        b2 = self.branch2_conv(b2)
        
        # Branch 3
        b3 = self.relu(self.branch3_conv1(x))
        b3 = self.branch3_conv2(b3)
        
        # Branch 4
        b4 = self.relu(self.branch4_conv1(x))
        b4 = self.branch4_conv2(b4)
        
        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Fusion
        out = self.fusion(out)
        out = self.relu(out)
        
        # Channel Attention
        ca = self.channel_attention(out)
        out = out * ca
        
        # Spatial Attention
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        out = out * sa
        
        return out


class ResBlock(nn.Module):
    """
    Residual Block with downsampling capability
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出通道数不同，或者需要下采样，则添加shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加shortcut连接
        identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        
        return out


class UpBlock(nn.Module):
    """
    Upsampling Block for decoder path
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ResidualIENet(nn.Module):
    """
    Residual Illumination Estimation Network (IENet) with Improved Decomposition
    
    Architecture:
    - Input Layer: 3->32 channels
    - Encoder: 3 ResBlocks (32->64->128->256 channels)
    - Bottleneck: 2 ResBlocks (256 channels)
    - Decoder: 3 Upsampling Blocks (256->128->64->32 channels)
    - Output Head: 32->1 channel (sigmoid activation)
    - Residual Learning: Direct estimation of illumination residual
    """
    def __init__(self):
        super(ResidualIENet, self).__init__()
        
        # Input layer
        self.input_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ResBlock(32, 64, stride=2)
        self.enc2 = ResBlock(64, 128, stride=2)
        self.enc3 = ResBlock(128, 256, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        
        # Decoder
        self.dec3 = UpBlock(256, 128)
        self.dec2 = UpBlock(128, 64)
        self.dec1 = UpBlock(64, 32)
        
        # Output head for residual
        self.residual_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Final sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input layer
        x1 = F.relu(self.input_layer(x))
        
        # Encoder
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        d3 = self.dec3(x5) + x3  # Skip connection from enc3
        d2 = self.dec2(d3) + x2   # Skip connection from enc2
        d1 = self.dec1(d2) + x1   # Skip connection from enc1
        
        # Residual estimation
        residual = self.residual_head(d1)
        
        # Residual learning: illumination = mean(input) + residual
        mean_illumination = torch.mean(x, dim=1, keepdim=True)  # Global mean as base
        illumination = mean_illumination + residual
        
        # Apply sigmoid for normalized output
        out = self.sigmoid(illumination)
        
        return out


class MultiScaleUP_Retinex(nn.Module):
    """
    Multi-Scale UP-Retinex: Unsupervised Physics-Guided Retinex Network with Multi-Scale Enhancement
    
    Architecture:
    - Input: Low-light image X
    - IENet: Predicts illumination map I
    - Multi-Scale Feature Extraction: Extract features at multiple scales
    - Retinex Decomposition: S = X / I (with regularization)
    - Multi-Scale Enhancement: Combine enhancements from multiple scales
    - Output: Enhanced image Y
    """
    def __init__(self):
        super(MultiScaleUP_Retinex, self).__init__()
        self.ie_net = ResidualIENet()
        
        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedFAM(32, 32)
        )
        
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedFAM(32, 32)
        )
        
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            EnhancedFAM(32, 32)
        )
        
        # Scale fusion
        self.fusion = nn.Conv2d(96, 32, kernel_size=1)
        self.output_layer = nn.Conv2d(32, 3, kernel_size=1)
        
    def retinex_decompose(self, x, illu):
        """
        Retinex decomposition: S = X / I
        With regularization to prevent division by zero.
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        reflectance = x / (illu + epsilon)
        return reflectance
    
    def multi_scale_enhance(self, x, reflectance, illu):
        """
        Multi-scale enhancement function
        """
        # Resize inputs to different scales
        x1 = x
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        # Extract multi-scale features
        f1 = self.scale1(x1)
        f2 = self.scale2(x2)
        f3 = self.scale3(x3)
        
        # Resize features to the same size
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=False)
        
        # Fuse features
        fused = torch.cat([f1, f2_up, f3_up], dim=1)
        fused = self.fusion(fused)
        
        # Generate enhancement map
        enhancement_map = self.output_layer(fused)
        enhancement_map = torch.sigmoid(enhancement_map)
        
        # Apply enhancement
        enhanced = reflectance * enhancement_map + (1 - reflectance) * (enhancement_map ** 2)
        return enhanced
    
    def forward(self, x):
        # Predict illumination map
        illu = self.ie_net(x)
        
        # Retinex decomposition
        reflectance = self.retinex_decompose(x, illu)
        
        # Multi-scale enhancement
        enhanced = self.multi_scale_enhance(x, reflectance, illu)
        
        return enhanced, reflectance, illu


# 为保持向后兼容性，添加别名
UP_Retinex = MultiScaleUP_Retinex


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleUP_Retinex().to(device)
    
    # Print model info
    print("=" * 60)
    print("UP-Retinex Model")
    print("=" * 60)
    print(f"Number of parameters: {count_parameters(model):,}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    img_size = 640
    test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    with torch.no_grad():
        enhanced, reflectance, illumination = model(test_input)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"Enhanced output shape: {enhanced.shape}")
    print(f"Reflectance map shape: {reflectance.shape}")
    print(f"Illumination map shape: {illumination.shape}")
    print("=" * 60)
    print("Model test passed!")