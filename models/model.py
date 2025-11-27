"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Model Architecture Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAM(nn.Module):
    """
    Feature Aggregation Module (FAM)
    Derived from HIDO-Net to capture multi-scale features.
    
    Architecture:
    - Branch 1: 1x1 Conv
    - Branch 2: 3x3 MaxPool -> 1x1 Conv
    - Branch 3: 3x3 Conv -> 3x3 Conv (Cascaded)
    - Branch 4: 3x3 Conv -> 3x3 Conv (Cascaded with dilation)
    - Fusion: Concatenate all branches -> 1x1 Conv
    """
    def __init__(self, in_channels, out_channels):
        super(FAM, self).__init__()
        
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
        
        return out


class IENet(nn.Module):
    """
    Illumination Estimation Network (IE-Net)
    A Fully Convolutional Network (FCN) to predict illumination map I.
    
    Architecture:
    - Input Layer: Conv (3 -> 32 channels)
    - Encoder: 2 layers of [Conv + ReLU + FAM]
    - Bottleneck: FAM block
    - Decoder: 2 layers of [Conv + ReLU]
    - Output Head: 1x1 Conv (reduce to 3 channels) -> Sigmoid Activation
    """
    def __init__(self):
        super(IENet, self).__init__()
        
        # Input Layer
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.input_relu = nn.ReLU(inplace=True)
        
        # Encoder Layer 1
        self.enc1_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc1_relu = nn.ReLU(inplace=True)
        self.enc1_fam = FAM(64, 64)
        
        # Encoder Layer 2
        self.enc2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_relu = nn.ReLU(inplace=True)
        self.enc2_fam = FAM(128, 128)
        
        # Bottleneck
        self.bottleneck = FAM(128, 128)
        
        # Decoder Layer 1
        self.dec1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1_relu = nn.ReLU(inplace=True)
        
        # Decoder Layer 2
        self.dec2_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec2_relu = nn.ReLU(inplace=True)
        
        # Output Head: Reduce to 3 channels (RGB illumination map)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.output_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input
        x = self.input_conv(x)
        x = self.input_relu(x)
        
        # Encoder
        x = self.enc1_conv(x)
        x = self.enc1_relu(x)
        x = self.enc1_fam(x)
        
        x = self.enc2_conv(x)
        x = self.enc2_relu(x)
        x = self.enc2_fam(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec1_conv(x)
        x = self.dec1_relu(x)
        
        x = self.dec2_conv(x)
        x = self.dec2_relu(x)
        
        # Output
        illu_map = self.output_conv(x)
        illu_map = self.output_sigmoid(illu_map)
        
        return illu_map


class UP_Retinex(nn.Module):
    """
    UP-Retinex: Unsupervised Physics-Guided Retinex Network
    
    Main model that combines IE-Net and physical Retinex decomposition.
    
    Theory: S = R ⊙ I
    where:
    - S: Input low-light image
    - I: Illumination map (estimated by IE-Net)
    - R: Reflectance (enhanced result)
    
    Reconstruction: R = S / (I + ε)
    """
    def __init__(self):
        super(UP_Retinex, self).__init__()
        self.ie_net = IENet()
        self.epsilon = 1e-4  # Small constant to prevent division by zero
        
    def forward(self, img_low):
        """
        Forward pass
        
        Args:
            img_low (torch.Tensor): Input low-light image [B, 3, H, W], range [0, 1]
            
        Returns:
            img_high (torch.Tensor): Enhanced image [B, 3, H, W], range [0, 1]
            illu_map (torch.Tensor): Illumination map [B, 3, H, W], range [0, 1]
        """
        # 1. Estimate Illumination
        illu_map = self.ie_net(img_low)
        
        # 2. Physical Reconstruction (Retinex decomposition)
        # R = S / (I + ε)
        img_high = img_low / (illu_map + self.epsilon)
        
        # 3. Clip results to [0, 1]
        img_high = torch.clamp(img_high, 0, 1)
        
        return img_high, illu_map
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UP_Retinex().to(device)
    
    # Print model info
    print("=" * 60)
    print("UP-Retinex Model")
    print("=" * 60)
    print(f"Number of parameters: {model.get_num_params():,}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    img_size = 256
    test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    with torch.no_grad():
        enhanced, illumination = model(test_input)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"Enhanced output shape: {enhanced.shape}")
    print(f"Illumination map shape: {illumination.shape}")
    print("=" * 60)
    print("Model test passed!")

