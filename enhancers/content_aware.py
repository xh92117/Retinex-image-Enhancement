"""
UP-Retinex 内容感知增强模块
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage


class ContentAwareEnhancer:
    """内容感知增强器，根据图像内容区域的重要性进行自适应增强"""
    
    def __init__(self):
        """初始化内容感知增强器"""
        pass
    
    def compute_saliency_map(self, image_tensor):
        """
        计算图像的显著性图（简化版）
        
        Args:
            image_tensor (torch.Tensor): 图像张量 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            saliency_map (torch.Tensor): 显著性图 [1, 1, H, W]
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
        
        # 计算显著性图 - 使用拉普拉斯算子检测边缘作为显著性特征
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        saliency = np.abs(laplacian)
        
        # 应用高斯模糊平滑显著性图
        saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
        
        # 归一化到[0, 1]
        saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
        
        # 转换回张量格式
        saliency_tensor = torch.from_numpy(saliency).float()
        saliency_tensor = saliency_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return saliency_tensor
    
    def compute_attention_map(self, image_tensor):
        """
        计算注意力图，结合亮度和显著性信息
        
        Args:
            image_tensor (torch.Tensor): 图像张量 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            attention_map (torch.Tensor): 注意力图 [1, 1, H, W]
        """
        # 确保图像张量是正确的形状
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 计算亮度图
        luminance = 0.299 * image_tensor[:, 0:1, :, :] + \
                    0.587 * image_tensor[:, 1:2, :, :] + \
                    0.114 * image_tensor[:, 2:3, :, :]
        
        # 计算显著性图
        saliency_map = self.compute_saliency_map(image_tensor)
        
        # 结合亮度和显著性信息
        # 对于较暗的区域，如果具有高显著性，则给予更高的注意力权重
        attention_map = saliency_map * (1.0 / (luminance + 0.1))
        
        # 归一化
        attention_map = (attention_map - torch.min(attention_map)) / \
                       (torch.max(attention_map) - torch.min(attention_map) + 1e-8)
        
        return attention_map
    
    def apply_content_aware_enhancement(self, model, image_tensor, device):
        """
        应用内容感知增强到模型推理
        
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
        
        # 计算注意力图
        attention_map = self.compute_attention_map(image_tensor)
        attention_map = attention_map.to(device)
        
        # 模型推理
        with torch.no_grad():
            enhanced_img, reflectance, illu_map = model(image_tensor)
        
        # 根据注意力图调整增强强度
        # 在高注意力区域应用更强的增强
        enhanced_img_adjusted = enhanced_img * (1.0 + 0.2 * attention_map)
        enhanced_img_adjusted = torch.clamp(enhanced_img_adjusted, 0, 1)
        
        return enhanced_img_adjusted, illu_map


# 示例用法
if __name__ == "__main__":
    # 创建内容感知增强器实例
    enhancer = ContentAwareEnhancer()
    
    # 创建一个模拟的图像张量
    test_image = torch.rand(1, 3, 256, 256)
    
    # 计算显著性图
    saliency = enhancer.compute_saliency_map(test_image)
    print(f"显著性图形状: {saliency.shape}")
    
    # 计算注意力图
    attention = enhancer.compute_attention_map(test_image)
    print(f"注意力图形状: {attention.shape}")