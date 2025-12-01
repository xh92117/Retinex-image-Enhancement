"""
UP-Retinex 自适应参数调整模块
"""

import torch
import numpy as np
from PIL import Image
import cv2


class AdaptiveParameterAdjuster:
    """自适应参数调整器，根据图像特征自动调整增强参数"""
    
    def __init__(self):
        """初始化参数调整器"""
        # 默认参数值
        self.default_params = {
            'enhance_strength': 1.0,  # 增强强度
            'color_balance': 1.0,     # 色彩平衡
            'brightness_boost': 1.0,  # 亮度提升
            'contrast_adjust': 1.0    # 对比度调整
        }
    
    def calculate_brightness_features(self, image_tensor):
        """
        计算图像亮度特征
        
        Args:
            image_tensor (torch.Tensor): 图像张量 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            features (dict): 亮度特征字典
        """
        # 确保图像张量是正确的形状
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 转换为numpy数组
        img_np = image_tensor.squeeze(0).cpu().detach().numpy()
        
        # 转换为 [H, W, C] 格式
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # 转换为OpenCV格式 (BGR)
        img_cv = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 计算亮度特征
        features = {}
        
        # 平均亮度
        features['mean_brightness'] = np.mean(gray) / 255.0
        
        # 亮度标准差 (对比度)
        features['brightness_std'] = np.std(gray) / 255.0
        
        # 低亮度像素比例 (< 50)
        features['dark_pixel_ratio'] = np.sum(gray < 50) / gray.size
        
        # 中等亮度像素比例 (50-200)
        features['mid_pixel_ratio'] = np.sum((gray >= 50) & (gray <= 200)) / gray.size
        
        # 高亮度像素比例 (> 200)
        features['bright_pixel_ratio'] = np.sum(gray > 200) / gray.size
        
        return features
    
    def adjust_parameters(self, image_tensor):
        """
        根据图像特征自适应调整参数
        
        Args:
            image_tensor (torch.Tensor): 图像张量 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            params (dict): 调整后的参数字典
        """
        # 计算图像亮度特征
        features = self.calculate_brightness_features(image_tensor)
        
        # 获取默认参数
        params = self.default_params.copy()
        
        # 根据平均亮度调整增强强度
        mean_brightness = features['mean_brightness']
        if mean_brightness < 0.2:  # 很暗的图像
            params['enhance_strength'] = 1.5
            params['brightness_boost'] = 1.3
        elif mean_brightness < 0.4:  # 暗的图像
            params['enhance_strength'] = 1.3
            params['brightness_boost'] = 1.2
        elif mean_brightness > 0.7:  # 明亮的图像
            params['enhance_strength'] = 0.8
            params['brightness_boost'] = 0.9
        else:  # 中等亮度图像
            params['enhance_strength'] = 1.0
            params['brightness_boost'] = 1.0
        
        # 根据对比度调整对比度参数
        brightness_std = features['brightness_std']
        if brightness_std < 0.1:  # 低对比度
            params['contrast_adjust'] = 1.3
        elif brightness_std < 0.2:  # 中等对比度
            params['contrast_adjust'] = 1.1
        else:  # 高对比度
            params['contrast_adjust'] = 0.9
        
        # 根据暗像素比例调整色彩平衡
        dark_ratio = features['dark_pixel_ratio']
        if dark_ratio > 0.6:  # 大量暗像素
            params['color_balance'] = 1.2
        elif dark_ratio > 0.3:  # 中等暗像素
            params['color_balance'] = 1.1
        else:  # 较少暗像素
            params['color_balance'] = 1.0
        
        return params
    
    def apply_clahe_enhancement(self, image_tensor):
        """
        应用CLAHE增强到图像张量
        
        Args:
            image_tensor (torch.Tensor): 输入图像张量 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            clahe_enhanced (torch.Tensor): CLAHE增强后的图像张量
        """
        # 确保图像张量是正确的形状
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 转换为numpy数组
        img_np = image_tensor.squeeze(0).cpu().detach().numpy()
        
        # 转换为 [H, W, C] 格式
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # 转换为OpenCV格式 (BGR) 并转换为8位
        img_cv = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 转换为Lab色彩空间，只对L通道进行CLAHE增强
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 应用CLAHE到L通道
        l_clahe = clahe.apply(l)
        
        # 合并通道
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # 转换回BGR格式
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # 转换回RGB格式
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        
        # 转换为张量并归一化到[0, 1]
        clahe_tensor = torch.from_numpy(img_clahe.astype(np.float32) / 255.0)
        
        # 转换为 [1, 3, H, W] 格式
        clahe_tensor = clahe_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return clahe_tensor
    
    def apply_adaptive_enhancement(self, model, image_tensor, device):
        """
        应用自适应增强参数到模型推理
        
        Args:
            model: UP-Retinex模型实例
            image_tensor (torch.Tensor): 输入图像张量
            device: 设备 (cuda/cpu)
            
        Returns:
            enhanced_img (torch.Tensor): 增强后的图像
            illu_map (torch.Tensor): 光照图
        """
        # 调整参数
        params = self.adjust_parameters(image_tensor)
        
        # 将图像移到设备上
        image_tensor = image_tensor.to(device)
        
        # 模型推理
        with torch.no_grad():
            enhanced_img, reflectance, illu_map = model(image_tensor)
        
        # 应用CLAHE增强到模型输出
        enhanced_img = self.apply_clahe_enhancement(enhanced_img)
        
        # 将增强后的图像移回设备
        enhanced_img = enhanced_img.to(device)
        
        return enhanced_img, illu_map


# 示例用法
if __name__ == "__main__":
    # 创建参数调整器实例
    adjuster = AdaptiveParameterAdjuster()
    
    # 创建一个模拟的暗图像张量
    dark_image = torch.rand(1, 3, 256, 256) * 0.3  # 模拟暗图像
    
    # 计算亮度特征
    features = adjuster.calculate_brightness_features(dark_image)
    print("图像亮度特征:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # 调整参数
    params = adjuster.adjust_parameters(dark_image)
    print("\n自适应调整后的参数:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")