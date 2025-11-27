#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版UP-Retinex图像增强功能实现
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from pathlib import Path

from models.model import UP_Retinex
from enhancers.adaptive_params import AdaptiveParameterAdjuster
from enhancers.multi_scale import MultiScaleEnhancer
from enhancers.content_aware import ContentAwareEnhancer


def load_image(image_path, max_size=None):
    """
    加载并预处理图像
    
    Args:
        image_path (str): 图像路径
        max_size (int): 最大图像尺寸（可选）
        
    Returns:
        img_tensor (torch.Tensor): 图像张量 [1, 3, H, W]
        original_size (tuple): 原始图像尺寸 (W, H)
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # 如果指定了最大尺寸，则调整图像大小
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
    
    # 转换为张量并归一化到[0, 1]
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
    
    return img_tensor, original_size


def save_image(tensor, save_path):
    """
    保存张量为图像
    
    Args:
        tensor (torch.Tensor): 图像张量 [1, 3, H, W] 或 [3, H, W]
        save_path (str): 保存图像的路径
    """
    # 如果有批次维度则移除
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 转换为numpy数组
    img_np = tensor.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
    
    # 裁剪到[0, 1]并转换为[0, 255]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    # 保存图像
    img = Image.fromarray(img_np)
    img.save(save_path)
    print(f"已保存: {save_path}")


def create_comparison(img_low, img_enhanced, save_path):
    """
    创建原始图像与增强图像的对比图
    
    Args:
        img_low (torch.Tensor): 输入的低光图像 [1, 3, H, W]
        img_enhanced (torch.Tensor): 增强后的图像 [1, 3, H, W]
        save_path (str): 保存对比图像的路径
    """
    # 移除批次维度
    img_low = img_low.squeeze(0).cpu().detach().numpy()
    img_enhanced = img_enhanced.squeeze(0).cpu().detach().numpy()
    
    # 转置为 [H, W, C]
    img_low = np.transpose(img_low, (1, 2, 0))
    img_enhanced = np.transpose(img_enhanced, (1, 2, 0))
    
    # 裁剪到[0, 1]并转换为[0, 255]
    img_low = np.clip(img_low, 0, 1)
    img_enhanced = np.clip(img_enhanced, 0, 1)
    
    img_low = (img_low * 255).astype(np.uint8)
    img_enhanced = (img_enhanced * 255).astype(np.uint8)
    
    # 水平拼接原始图像和增强图像
    comparison = np.concatenate([img_low, img_enhanced], axis=1)
    
    # 保存对比图像
    img = Image.fromarray(comparison)
    img.save(save_path)
    print(f"已保存对比图像: {save_path}")


def enhance_single_image(model, image_path, output_dir, device, max_size=None, enable_multi_scale=False, enable_content_aware=False):
    """
    增强单张图像
    
    Args:
        model: UP_Retinex模型实例
        image_path (str): 输入图像路径
        output_dir (str): 结果保存目录
        device: 设备 (cuda/cpu)
        max_size (int): 最大图像尺寸
        enable_multi_scale (bool): 是否启用多尺度增强
        enable_content_aware (bool): 是否启用内容感知增强
    """
    print(f"正在处理: {os.path.basename(image_path)}")
    print("正在加载图像...")
    
    # 加载图像
    img_low, original_size = load_image(image_path, max_size)
    
    print("正在进行图像增强...")
    # 初始化自适应参数调整器
    adjuster = AdaptiveParameterAdjuster()
    
    # 初始化多尺度增强器
    multi_scale_enhancer = MultiScaleEnhancer()
    
    # 初始化内容感知增强器
    content_aware_enhancer = ContentAwareEnhancer()
    
    # 推理（带自适应参数调整）
    start_time = time.time()
    
    if enable_content_aware:
        # 使用内容感知增强
        img_enhanced, illu_map = content_aware_enhancer.apply_content_aware_enhancement(model, img_low, device)
    elif enable_multi_scale:
        # 使用多尺度增强
        img_enhanced, illu_map = multi_scale_enhancer.enhance_with_pyramid(model, img_low, device)
    else:
        # 使用基础自适应增强
        img_enhanced, illu_map = adjuster.apply_adaptive_enhancement(model, img_low, device)
    
    inference_time = time.time() - start_time
    
    print(f"增强耗时: {inference_time:.4f}s")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像名称
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存增强图像
    enhanced_path = os.path.join(output_dir, f"{img_name}_enhanced.png")
    save_image(img_enhanced, enhanced_path)
    
    # 保存光照图
    illu_path = os.path.join(output_dir, f"{img_name}_illumination.png")
    save_image(illu_map, illu_path)
    
    # 生成并保存对比图像
    comparison_path = os.path.join(output_dir, f"{img_name}_comparison.png")
    create_comparison(img_low, img_enhanced, comparison_path)
    
    print("图像增强完成！")


def enhance_batch_images(input_dir, output_dir, device, max_size=None):
    """
    批量增强图像
    
    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 结果保存目录
        device: 设备 (cuda/cpu)
        max_size (int): 最大图像尺寸
    """
    print("正在加载模型...")
    # 初始化模型
    model = UP_Retinex()
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 获取图像文件列表
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_files.append(os.path.join(input_dir, file))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"在目录 '{input_dir}' 中未找到有效图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print("=" * 50)
    
    # 处理每个图像
    total_time = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}]")
        start_time = time.time()
        enhance_single_image(model, image_path, output_dir, device, max_size)
        image_time = time.time() - start_time
        total_time += image_time
        print("-" * 50)
    
    # 打印统计信息
    print("=" * 50)
    print(f"总共处理了 {len(image_files)} 张图像")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每张图像耗时: {total_time/len(image_files):.4f}s")
    print("=" * 50)