"""
Letterbox utility for YOLO-style image preprocessing
"""

import cv2
import numpy as np


def letterbox(img, new_shape=640, color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
    """
    YOLO-style letterbox image preprocessing
    
    Args:
        img (np.ndarray): Input image in BGR format
        new_shape (int or tuple): Target size (height, width)
        color (tuple): Padding color (B, G, R)
        auto (bool): Automatically calculate padding to make dimensions divisible by 32
        scale_fill (bool): Stretch image to fill new_shape (disables aspect ratio preservation)
        scaleup (bool): Allow scaling up (small images will be scaled up)
        
    Returns:
        img (np.ndarray): Letterboxed image
        ratio (tuple): Width and height ratios
        (dw, dh) (tuple): Padding widths
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Convert new_shape to tuple if it's an integer
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Only scale down, do not scale up (for better test mAP)
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)


def letterbox_tensor(img_tensor, new_shape=640, color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
    """
    Apply letterbox to a PyTorch tensor image
    
    Args:
        img_tensor (torch.Tensor): Input image tensor [C, H, W] in range [0, 1]
        new_shape (int or tuple): Target size (height, width)
        color (tuple): Padding color (R, G, B) in range [0, 255]
        auto (bool): Automatically calculate padding to make dimensions divisible by 32
        scale_fill (bool): Stretch image to fill new_shape (disables aspect ratio preservation)
        scaleup (bool): Allow scaling up (small images will be scaled up)
        
    Returns:
        img_tensor (torch.Tensor): Letterboxed image tensor [C, H, W]
        ratio (tuple): Width and height ratios
        (dw, dh) (tuple): Padding widths
    """
    import torch
    import torch.nn.functional as F
    
    # Convert tensor to numpy for processing
    if img_tensor.is_cuda:
        img_np = img_tensor.cpu().numpy()
    else:
        img_np = img_tensor.numpy()
    
    # Convert from [C, H, W] to [H, W, C] and from [0, 1] to [0, 255]
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
    img_np = (img_np * 255).astype(np.uint8)
    
    # Apply letterbox
    img_lb, ratio, pad = letterbox(img_np, new_shape, color, auto, scale_fill, scaleup)
    
    # Convert back to tensor format [C, H, W] and [0, 255] to [0, 1]
    img_lb = img_lb.astype(np.float32) / 255.0
    img_tensor_result = torch.from_numpy(np.transpose(img_lb, (2, 0, 1)))
    
    return img_tensor_result, ratio, pad


if __name__ == "__main__":
    # Test the letterbox function
    import torch
    from PIL import Image
    
    # Create a dummy image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Apply letterbox
    img_lb, ratio, pad = letterbox(img, new_shape=640)
    
    print(f"Original shape: {img.shape}")
    print(f"Letterboxed shape: {img_lb.shape}")
    print(f"Ratio: {ratio}")
    print(f"Padding: {pad}")
    
    # Test with tensor
    img_tensor = torch.rand(3, 480, 640)
    img_tensor_lb, ratio, pad = letterbox_tensor(img_tensor, new_shape=640)
    
    print(f"Original tensor shape: {img_tensor.shape}")
    print(f"Letterboxed tensor shape: {img_tensor_lb.shape}")
    print(f"Ratio: {ratio}")
    print(f"Padding: {pad}")