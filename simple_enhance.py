#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版UP-Retinex图像增强工具
可直接对图像进行增强，无需显式训练步骤
"""

import os
import argparse
import torch
from pathlib import Path

from enhancers.simple_enhance import enhance_single_image, enhance_batch_images
from models.model import UP_Retinex


def main():
    parser = argparse.ArgumentParser(description='简化版UP-Retinex图像增强工具')
    
    # 输入/输出参数
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='./results',
                        help='结果保存目录')
    
    # 处理参数
    parser.add_argument('--max_size', type=int, default=None,
                        help='最大图像尺寸(用于内存限制)')
    
    # 设备
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    
    # 多尺度增强
    parser.add_argument('--multi_scale', action='store_true',
                        help='启用多尺度增强')
    
    # 内容感知增强
    parser.add_argument('--content_aware', action='store_true',
                        help='启用内容感知增强')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {args.device}")
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入路径 '{args.input}'")
        return
    
    # 增强图像
    print("=" * 50)
    print("开始进行图像增强")
    print("=" * 50)
    
    # 判断输入是文件还是目录
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像处理
        print("处理单张图像...")
        # 初始化模型
        model = UP_Retinex()
        model = model.to(args.device)
        model.eval()  # 设置为评估模式
        
        enhance_single_image(
            model=model,
            image_path=str(input_path),
            output_dir=args.output,
            device=args.device,
            max_size=args.max_size,
            enable_multi_scale=args.multi_scale
        )
    elif input_path.is_dir():
        # 批量图像处理
        print("批量处理图像目录...")
        # 注意：当前版本的批量处理暂不支持多尺度增强
        enhance_batch_images(
            input_dir=str(input_path),
            output_dir=args.output,
            device=args.device,
            max_size=args.max_size
        )
    else:
        print(f"错误: 输入路径 '{args.input}' 不是有效的文件或目录")
        return
    
    print("=" * 50)
    print(f"结果已保存到: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()