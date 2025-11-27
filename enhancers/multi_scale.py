"""
UP-Retinex 多尺度增强模块
"""

import torch
import torch.nn.functional as F
import numpy as np


class MultiScaleEnhancer:
    """多尺度增强器，通过多尺度特征提取进行图像增强"""
    
    def __init__(self):
        """初始化多尺度增强器"""
        pass
    
    def extract_multi_scale_features(self, image_tensor):
        """
        提取多尺度特征
        
        Args:
            image_tensor (torch.Tensor): 图像张量 [1, 3, H, W]
            
        Returns:
            features (list): 多尺度特征列表
        """
        # 确保图像张量是正确的形状
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 不同尺度的特征提取
        scales = [1.0, 0.5, 0.25]
        features = []
        
        for scale in scales:
            if scale == 1.0:
                # 原始尺度
                scaled_img = image_tensor
            else:
                # 缩放图像
                h, w = image_tensor.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # 提取简单特征（亮度、边缘等）
            # 亮度特征
            luminance = 0.299 * scaled_img[:, 0:1, :, :] + \
                        0.587 * scaled_img[:, 1:2, :, :] + \
                        0.114 * scaled_img[:, 2:3, :, :]
            
            # 边缘特征（使用Sobel算子）
            grad_x = torch.gradient(scaled_img, dim=3)[0]
            grad_y = torch.gradient(scaled_img, dim=2)[0]
            edges = torch.sqrt(grad_x**2 + grad_y**2)
            
            # 组合特征
            combined_features = torch.cat([scaled_img, luminance, edges], dim=1)
            features.append(combined_features)
        
        return features
    
    def apply_multi_scale_enhancement(self, model, image_tensor, device):
        """
        应用多尺度增强到模型推理
        
        Args:
            model: UP-Retinex模型实例
            image_tensor (torch.Tensor): 输入图像张量
            device: 设备 (cuda/cpu)
            
        Returns:
            enhanced_img (torch.Tensor): 增强后的图像
            illu_map (torch.Tensor): 光照图
        """
        # 将图像移到设备上
        image_tensor = image_tensor.to(device)
        
        # 提取多尺度特征
        multi_scale_features = self.extract_multi_scale_features(image_tensor)
        
        # 模型推理
        with torch.no_grad():
            enhanced_img, illu_map = model(image_tensor)
        
        # 根据多尺度特征调整增强效果
        # 这里是一个简化的实现，实际应用中可以根据不同尺度的特征进行更复杂的调整
        scale_weights = [0.5, 0.3, 0.2]  # 不同尺度的权重
        
        # 基于多尺度特征调整增强强度
        adjustment_factor = 1.0
        for i, features in enumerate(multi_scale_features):
            # 计算特征的重要性（简化实现）
            feature_importance = torch.mean(features)
            adjustment_factor += scale_weights[i] * feature_importance.item() * 0.1
        
        # 应用调整因子
        enhanced_img_adjusted = enhanced_img * adjustment_factor
        enhanced_img_adjusted = torch.clamp(enhanced_img_adjusted, 0, 1)
        
        return enhanced_img_adjusted, illu_map
    
    def enhance_with_pyramid(self, model, image_tensor, device):
        """
        使用金字塔结构进行多尺度增强（兼容simple_enhance.py中的调用）
        
        Args:
            model: UP-Retinex模型实例
            image_tensor (torch.Tensor): 输入图像张量
            device: 设备 (cuda/cpu)
            
        Returns:
            enhanced_img (torch.Tensor): 增强后的图像
            illu_map (torch.Tensor): 光照图
        """
        return self.apply_multi_scale_enhancement(model, image_tensor, device)


# 示例用法
if __name__ == "__main__":
    # 创建多尺度增强器实例
    enhancer = MultiScaleEnhancer()
    
    # 创建一个模拟的图像张量
    test_image = torch.rand(1, 3, 256, 256)
    
    # 提取多尺度特征
    features = enhancer.extract_multi_scale_features(test_image)
    print(f"提取到 {len(features)} 个尺度的特征")
    for i, feat in enumerate(features):
        print(f"  尺度 {i+1} 特征形状: {feat.shape}")