"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Inference Script
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

from models.model import UP_Retinex


def load_image(image_path, max_size=None):
    """
    Load and preprocess image
    
    Args:
        image_path (str): Path to image
        max_size (int): Maximum image dimension (optional)
        
    Returns:
        img_tensor (torch.Tensor): Image tensor [1, 3, H, W]
        original_size (tuple): Original image size (W, H)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize if max_size is specified
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Convert to tensor and normalize to [0, 1]
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor, original_size


def save_image(tensor, save_path):
    """
    Save tensor as image
    
    Args:
        tensor (torch.Tensor): Image tensor [1, 3, H, W] or [3, H, W]
        save_path (str): Path to save image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy array
    img_np = tensor.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
    
    # Clip to [0, 1] and convert to [0, 255]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    # Save image
    img = Image.fromarray(img_np)
    img.save(save_path)
    print(f"Saved: {save_path}")


def create_comparison(img_low, img_enhanced, illu_map, save_path):
    """
    Create side-by-side comparison image
    
    Args:
        img_low (torch.Tensor): Input low-light image [1, 3, H, W]
        img_enhanced (torch.Tensor): Enhanced image [1, 3, H, W]
        illu_map (torch.Tensor): Illumination map [1, 3, H, W]
        save_path (str): Path to save comparison image
    """
    # Remove batch dimension
    img_low = img_low.squeeze(0).cpu().detach().numpy()
    img_enhanced = img_enhanced.squeeze(0).cpu().detach().numpy()
    illu_map = illu_map.squeeze(0).cpu().detach().numpy()
    
    # Transpose to [H, W, C]
    img_low = np.transpose(img_low, (1, 2, 0))
    img_enhanced = np.transpose(img_enhanced, (1, 2, 0))
    illu_map = np.transpose(illu_map, (1, 2, 0))
    
    # Convert illumination map to grayscale for visualization
    illu_map_gray = np.mean(illu_map, axis=2, keepdims=True)
    illu_map_gray = np.repeat(illu_map_gray, 3, axis=2)
    
    # Clip and convert to [0, 255]
    img_low = np.clip(img_low, 0, 1)
    img_enhanced = np.clip(img_enhanced, 0, 1)
    illu_map_gray = np.clip(illu_map_gray, 0, 1)
    
    img_low = (img_low * 255).astype(np.uint8)
    img_enhanced = (img_enhanced * 255).astype(np.uint8)
    illu_map_gray = (illu_map_gray * 255).astype(np.uint8)
    
    # Concatenate horizontally
    comparison = np.concatenate([img_low, img_enhanced, illu_map_gray], axis=1)
    
    # Save
    img = Image.fromarray(comparison)
    img.save(save_path)
    print(f"Saved comparison: {save_path}")


def predict_single_image(model, image_path, output_dir, device, max_size=None, save_comparison=True):
    """
    Predict on a single image
    
    Args:
        model: Trained UP_Retinex model
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        device: Device (cuda/cpu)
        max_size (int): Maximum image dimension
        save_comparison (bool): Save comparison image
    """
    # Load image
    img_low, original_size = load_image(image_path, max_size)
    img_low = img_low.to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        img_enhanced, illu_map = model(img_low)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.4f}s")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image name
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save enhanced image
    enhanced_path = os.path.join(output_dir, f"{img_name}_enhanced.png")
    save_image(img_enhanced, enhanced_path)
    
    # Save illumination map
    illu_path = os.path.join(output_dir, f"{img_name}_illumination.png")
    save_image(illu_map, illu_path)
    
    # Save comparison image
    if save_comparison:
        comparison_path = os.path.join(output_dir, f"{img_name}_comparison.png")
        create_comparison(img_low, img_enhanced, illu_map, comparison_path)


def predict_batch(model, input_dir, output_dir, device, max_size=None, save_comparison=True):
    """
    Predict on a batch of images
    
    Args:
        model: Trained UP_Retinex model
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save results
        device: Device (cuda/cpu)
        max_size (int): Maximum image dimension
        save_comparison (bool): Save comparison images
    """
    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_files.append(os.path.join(input_dir, file))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("=" * 60)
    
    # Process each image
    total_time = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing [{i}/{len(image_files)}]: {os.path.basename(image_path)}")
        
        start_time = time.time()
        predict_single_image(model, image_path, output_dir, device, max_size, save_comparison)
        image_time = time.time() - start_time
        total_time += image_time
        
        print(f"Time: {image_time:.4f}s")
        print("-" * 60)
    
    # Print summary
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time / len(image_files):.4f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='UP-Retinex Inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save results')
    
    # Processing arguments
    parser.add_argument('--max_size', type=int, default=None,
                        help='Maximum image dimension (for memory constraints)')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Do not save comparison images')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = UP_Retinex().to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Number of parameters: {model.get_num_params():,}")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single image
        print("=" * 60)
        print(f"Processing single image: {args.input}")
        print("=" * 60)
        
        predict_single_image(
            model, args.input, args.output, device,
            args.max_size, not args.no_comparison
        )
    
    elif os.path.isdir(args.input):
        # Directory of images
        print("=" * 60)
        print(f"Processing directory: {args.input}")
        print("=" * 60)
        
        predict_batch(
            model, args.input, args.output, device,
            args.max_size, not args.no_comparison
        )
    
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print("\nInference completed!")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()

