"""
UP-Retinex: Unsupervised Physics-Guided Retinex Network
Training Script
"""

import os
import sys
import argparse
import time
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import UP_Retinex
from losses.loss import TotalLoss
from datasets.dataset import get_train_dataloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """
    Train for one epoch
    
    Args:
        model: UP_Retinex model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        avg_loss_dict: Dictionary of average losses
    """
    model.train()
    
    # Initialize loss accumulators
    total_losses = {
        'total': 0.0,
        'exposure': 0.0,
        'smoothness': 0.0,
        'color': 0.0,
        'spatial': 0.0
    }
    
    num_batches = len(dataloader)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, img_low in enumerate(pbar):
        # Move to device
        img_low = img_low.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        img_enhanced, illu_map = model(img_low)
        
        # Compute loss
        loss, loss_dict = criterion(img_low, img_enhanced, illu_map)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'exp': f"{loss_dict['exposure']:.4f}",
            'smooth': f"{loss_dict['smoothness']:.4f}"
        })
        
        # Log to TensorBoard (every 100 batches)
        if batch_idx % 100 == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'Loss/{key}', value, global_step)
    
    # Calculate average losses
    avg_loss_dict = {key: value / num_batches for key, value in total_losses.items()}
    
    return avg_loss_dict


def save_checkpoint(model, optimizer, epoch, save_dir, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        save_dir: Directory to save checkpoint
        is_best: If True, also save as best model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save as best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save as latest model (overwrite)
    latest_path = os.path.join(save_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        start_epoch: Epoch to resume from
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch


def train(args):
    """
    Main training function
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = UP_Retinex().to(device)
    print(f"Number of parameters: {model.get_num_params():,}")
    
    # Create loss function
    criterion = TotalLoss(
        weight_exp=args.weight_exp,
        weight_smooth=args.weight_smooth,
        weight_col=args.weight_col,
        weight_spa=args.weight_spa
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_gamma
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Create dataloader
    print("Creating dataloader...")
    train_loader = get_train_dataloader(
        image_dir=args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    print(f"Number of training batches: {len(train_loader)}")
    
    # Create TensorBoard writer
    log_dir = os.path.join(args.save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Training loop
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        avg_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Average Losses:")
        for key, value in avg_loss_dict.items():
            print(f"    {key}: {value:.6f}")
        
        # Log to TensorBoard
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        for key, value in avg_loss_dict.items():
            writer.add_scalar(f'Epoch_Loss/{key}', value, epoch)
        
        # Save checkpoint
        is_best = avg_loss_dict['total'] < best_loss
        if is_best:
            best_loss = avg_loss_dict['total']
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, args.save_dir, is_best)
        
        print("=" * 60)
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train UP-Retinex Model')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=640,
                        help='Image size for training crops')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_decay_step', type=int, default=30,
                        help='Learning rate decay step size')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='Learning rate decay gamma')
    
    # Loss weights
    parser.add_argument('--weight_exp', type=float, default=10.0,
                        help='Weight for exposure loss')
    parser.add_argument('--weight_smooth', type=float, default=1.0,
                        help='Weight for smoothness loss')
    parser.add_argument('--weight_col', type=float, default=0.5,
                        help='Weight for color loss')
    parser.add_argument('--weight_spa', type=float, default=1.0,
                        help='Weight for spatial consistency loss')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("UP-Retinex Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()

