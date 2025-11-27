#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Main Program Entry Point
"""

import os
import argparse
import torch
from pathlib import Path

from models.model import UP_Retinex
from trainers.train import train
from predictors.predict import predict_single_image, predict_batch
from enhancers.simple_enhance import enhance_single_image, enhance_batch_images
from enhancers.adaptive_params import AdaptiveParameterAdjuster


def main():
    """
    主程序入口
    提供统一接口来训练模型或进行推理
    """
    parser = argparse.ArgumentParser(description='UP-Retinex: 基于Retinex理论的低光照图像增强')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'enhance'], 
                        default='predict', help='运行模式: train(训练) 或 predict(推理) 或 enhance(简化增强)')
    
    # 路径参数
    parser.add_argument('--train_dir', type=str, default='./data/train',
                        help='训练图像目录路径')
    parser.add_argument('--test_dir', type=str, default='./data/test',
                        help='测试图像目录路径')
    parser.add_argument('--input_path', type=str, default='./data/test',
                        help='输入图像路径(单张图像或目录)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='保存检查点的目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小')
    parser.add_argument('--image_size', type=int, default=256,
                        help='训练时裁剪的图像尺寸')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 损失函数权重
    parser.add_argument('--weight_exp', type=float, default=10.0,
                        help='曝光控制损失权重')
    parser.add_argument('--weight_smooth', type=float, default=1.0,
                        help='平滑性损失权重')
    parser.add_argument('--weight_col', type=float, default=0.5,
                        help='颜色恒常性损失权重')
    parser.add_argument('--weight_spa', type=float, default=1.0,
                        help='空间一致性损失权重')
    
    # 推理参数
    parser.add_argument('--max_size', type=int, default=None,
                        help='推理时最大图像尺寸(用于内存限制)')
    parser.add_argument('--no_comparison', action='store_true',
                        help='不保存对比图像')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    
    # 简化增强模式参数
    parser.add_argument('--multi_scale', action='store_true',
                        help='启用多尺度增强')
    parser.add_argument('--content_aware', action='store_true',
                        help='启用内容感知增强')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作进程数')
    parser.add_argument('--lr_decay_step', type=int, default=30,
                        help='学习率衰减步长')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='学习率衰减因子')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {args.device}")
    print(f"运行模式: {args.mode}")
    
    if args.mode == 'train':
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 开始训练
        print("=" * 60)
        print("开始训练 UP-Retinex 模型")
        print("=" * 60)
        
        train(args)
        
    elif args.mode == 'predict':
        # 检查模型文件是否存在
        if not os.path.exists(args.checkpoint):
            print(f"错误: 找不到模型检查点文件 '{args.checkpoint}'")
            print("请先训练模型或提供有效的检查点文件")
            return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载模型
        print("正在加载模型...")
        model = UP_Retinex()
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(args.device)
        model.eval()
        print("模型加载完成")
        
        # 进行推理
        print("=" * 60)
        print("开始进行图像增强推理")
        print("=" * 60)
        
        # 判断输入是单个文件还是目录
        input_path = Path(args.input_path)
        if input_path.is_file():
            # 单张图像推理
            print(f"处理单张图像: {input_path}")
            predict_single_image(
                model=model,
                image_path=str(input_path),
                output_dir=args.output_dir,
                device=args.device,
                max_size=args.max_size,
                save_comparison=not args.no_comparison
            )
        elif input_path.is_dir():
            # 批量图像推理
            print(f"批量处理目录中的图像: {input_path}")
            predict_batch(
                model=model,
                input_dir=str(input_path),
                output_dir=args.output_dir,
                device=args.device,
                max_size=args.max_size,
                save_comparison=not args.no_comparison
            )
        else:
            print(f"错误: 输入路径 '{args.input_path}' 不存在")
            return
        
        print("=" * 60)
        print("推理完成，结果已保存到:", args.output_dir)
        print("=" * 60)
        
    elif args.mode == 'enhance':
        # 简化版图像增强模式
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 进行图像增强
        print("=" * 60)
        print("开始进行简化版图像增强")
        print("=" * 60)
        
        # 判断输入是单个文件还是目录
        input_path = Path(args.input_path)
        if input_path.is_file():
            # 单张图像增强
            print(f"处理单张图像: {input_path}")
            
            # 初始化模型
            model = UP_Retinex()
            model = model.to(args.device)
            model.eval()
            
            # 初始化自适应参数调整器
            adjuster = AdaptiveParameterAdjuster()
            
            # 检查是否启用多尺度增强或内容感知增强
            enable_multi_scale = hasattr(args, 'multi_scale') and args.multi_scale
            enable_content_aware = hasattr(args, 'content_aware') and args.content_aware
            
            enhance_single_image(
                model=model,
                image_path=str(input_path),
                output_dir=args.output_dir,
                device=args.device,
                max_size=args.max_size,
                adjuster=adjuster,
                enable_multi_scale=enable_multi_scale,
                enable_content_aware=enable_content_aware
            )
        elif input_path.is_dir():
            # 批量图像增强
            print(f"批量处理目录中的图像: {input_path}")
            enhance_batch_images(
                input_dir=str(input_path),
                output_dir=args.output_dir,
                device=args.device,
                max_size=args.max_size
            )
        else:
            print(f"错误: 输入路径 '{args.input_path}' 不存在")
            return
        
        print("=" * 60)
        print("图像增强完成，结果已保存到:", args.output_dir)
        print("=" * 60)


if __name__ == '__main__':
    main()