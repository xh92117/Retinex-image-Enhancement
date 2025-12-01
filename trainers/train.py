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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import MultiScaleUP_Retinex
from losses.loss import TotalLoss
from datasets.dataset import get_train_dataloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, scaler=None, use_amp=False):
    """
    Train for one epoch with mixed precision training support
    
    Args:
        model: UP_Retinex model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
        scaler: GradScaler for mixed precision training
        use_amp: Whether to use automatic mixed precision
        
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
        'spatial': 0.0,
        'decouple': 0.0,
        'perceptual': 0.0
    }
    
    num_batches = len(dataloader)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, img_low in enumerate(pbar):
        # Move to device
        img_low = img_low.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast():
                # Forward pass
                img_enhanced, reflectance, illu_map = model(img_low)
                
                # Compute loss
                loss, loss_dict = criterion(img_low, img_enhanced, illu_map, reflectance)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            img_enhanced, reflectance, illu_map = model(img_low)
            
            # Compute loss
            loss, loss_dict = criterion(img_low, img_enhanced, illu_map, reflectance)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (for stability)
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
            'smooth': f"{loss_dict['smoothness']:.4f}",
            'decouple': f"{loss_dict['decouple']:.4f}",
            'perceptual': f"{loss_dict['perceptual']:.4f}"
        })
        
        # Log to TensorBoard (every 100 batches)
        if batch_idx % 100 == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'Loss/{key}', value, global_step)
    
    # Calculate average losses
    if num_batches > 0:
        avg_loss_dict = {key: value / num_batches for key, value in total_losses.items()}
    else:
        # If no batches, return the raw losses (or handle appropriately)
        avg_loss_dict = total_losses
    
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
    
    # Save as best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save as latest model (overwrite)
    latest_path = os.path.join(save_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)
    print(f"Saved latest model: {latest_path}")


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
    
    # Create model with improved architecture
    print("Creating model...")
    use_preact = getattr(args, 'use_preact', False)
    use_aspp = getattr(args, 'use_aspp', False)
    model = MultiScaleUP_Retinex(use_preact=use_preact, use_aspp=use_aspp).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if use_preact:
        print("  Using Pre-activation ResBlocks")
    if use_aspp:
        print("  Using ASPP modules")
    
    # Create loss function with improved features
    # Handle potential missing arguments with defaults
    weight_exp = getattr(args, 'weight_exp', 10.0)
    weight_smooth = getattr(args, 'weight_smooth', 1.0)
    weight_col = getattr(args, 'weight_col', 0.5)
    weight_spa = getattr(args, 'weight_spa', 1.0)
    weight_decouple = getattr(args, 'weight_decouple', 0.1)
    weight_perceptual = getattr(args, 'weight_perceptual', 1.0)
    weight_freq = getattr(args, 'weight_freq', 0.5)
    
    use_freq_loss = getattr(args, 'use_freq_loss', False)
    adaptive_weights = getattr(args, 'adaptive_weights', False)
    
    criterion = TotalLoss(
        weight_exp=weight_exp,
        weight_smooth=weight_smooth,
        weight_col=weight_col,
        weight_spa=weight_spa,
        weight_decouple=weight_decouple,
        weight_perceptual=weight_perceptual,
        weight_freq=weight_freq,
        use_freq_loss=use_freq_loss,
        adaptive_weights=adaptive_weights
    ).to(device)
    
    print(f"Loss function configured:")
    print(f"  - Frequency loss: {'enabled' if use_freq_loss else 'disabled'}")
    print(f"  - Adaptive weights: {'enabled' if adaptive_weights else 'disabled'}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - 使用CosineAnnealingWarmRestarts更好的性能
    use_cosine_scheduler = getattr(args, 'use_cosine_scheduler', False)
    if use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        print("Using CosineAnnealingWarmRestarts scheduler")
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_gamma
        )
    
    # 混合精度训练初始化
    use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (AMP)")
    
    # 早停机制初始化
    patience = getattr(args, 'patience', 20)
    best_loss = float('inf')
    patience_counter = 0
    print(f"Early stopping patience: {patience}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Create dataloader
    print("Creating dataloader...")
    advanced_augment = getattr(args, 'advanced_augment', False)
    train_loader = get_train_dataloader(
        image_dir=args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        shuffle=True,
        advanced_augment=advanced_augment
    )
    print(f"Number of training batches: {len(train_loader)}")
    if advanced_augment:
        print("  Using advanced data augmentation")
    
    # Check if there are any training batches
    if len(train_loader) == 0:
        print("错误：训练数据不足，无法创建批次。请增加训练数据或减小batch_size。")
        print(f"当前设置：样本数={len(train_loader.dataset)}, batch_size={args.batch_size}")
        return
    
    # Create TensorBoard writer
    log_dir = os.path.join(args.save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Initialize lists to store loss history for visualization
    loss_history = {
        'total': [],
        'exposure': [],
        'smoothness': [],
        'color': [],
        'spatial': [],
        'decouple': [],
        'perceptual': []
    }
    
    # Training loop
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        # Train one epoch with mixed precision support
        avg_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            scaler=scaler, use_amp=use_amp
        )
        
        # Save sample visualization every 10 epochs
        if epoch % 10 == 0:
            save_sample_visualizations(model, train_loader, device, epoch, args.save_dir)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store loss history for visualization
        for key, value in avg_loss_dict.items():
            loss_history[key].append(value)
        
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
        
        # 早停机制检查
        current_total_loss = avg_loss_dict['total']
        if current_total_loss < best_loss:
            best_loss = current_total_loss
            patience_counter = 0
            is_best = True
            print(f"  ✓ New best loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            is_best = False
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Always save the latest model and save best model when improved
        save_checkpoint(model, optimizer, epoch, args.save_dir, is_best)
        
        # 检查是否应该早停
        if patience_counter >= patience:
            print("\n" + "=" * 60)
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best loss: {best_loss:.6f}")
            print("=" * 60)
            break
        
        print("=" * 60)
    
    # Close TensorBoard writer
    writer.close()
    
    # Save loss history for visualization
    save_loss_curves(loss_history, args.save_dir)
    
    # Save results to CSV
    save_results_to_csv(loss_history, args.save_dir)
    
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
    parser.add_argument('--weight_decouple', type=float, default=0.1,
                        help='Weight for illumination-reflectance decoupling loss')
    parser.add_argument('--weight_perceptual', type=float, default=1.0,
                        help='Weight for perceptual loss')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Advanced training arguments
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--use_cosine_scheduler', action='store_true',
                        help='Use cosine annealing scheduler instead of step scheduler')
    
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


def save_sample_visualizations(model, train_loader, device, epoch, save_dir):
    """
    Save sample visualizations of low-light and enhanced images
    
    Args:
        model: Trained model
        train_loader: Training data loader
        device: Device to run inference on
        epoch: Current epoch number
        save_dir: Directory to save visualizations
    """
    import torchvision.utils as vutils
    from utils.utils import visualize_results
    
    model.eval()  # Set model to evaluation mode
    
    # Create directory for visualizations
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get a batch of samples from the training loader
    with torch.no_grad():
        for batch_idx, img_low in enumerate(train_loader):
            if batch_idx >= 2:  # Only process first 2 batches
                break
                
            # Move to device
            img_low = img_low.to(device)
            
            # Forward pass
            img_enhanced, reflectance, illu_map = model(img_low)
            
            # Save visualizations for first few samples in the batch
            for i in range(min(2, img_low.size(0))):  # Save max 2 samples per batch
                # Create visualization
                save_path = os.path.join(vis_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png')
                visualize_results(
                    img_low[i:i+1], 
                    img_enhanced[i:i+1], 
                    illu_map[i:i+1], 
                    save_path=save_path
                )
    
    model.train()  # Set model back to training mode


def save_loss_curves(loss_history, save_dir):
    """
    Save loss curves as images
    
    Args:
        loss_history: Dictionary containing loss history
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot individual loss curves
    for key, values in loss_history.items():
        if values:  # Only plot if there are values
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'{key.capitalize()} Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(plot_dir, f'{key}_curve.png')
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Saved {key} curve: {plot_path}")
    
    # Plot combined loss curve
    plt.figure(figsize=(12, 8))
    for key, values in loss_history.items():
        if values and key != 'total':  # Skip total for clarity
            plt.plot(values, label=key.capitalize())
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(plot_dir, 'combined_loss_curves.png')
    plt.savefig(combined_plot_path)
    plt.close()
    
    print(f"Saved combined loss curves: {combined_plot_path}")


def save_results_to_csv(loss_history, save_dir):
    """
    Save loss history to CSV file
    
    Args:
        loss_history: Dictionary containing loss history
        save_dir: Directory to save the CSV
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')
    
    # Get the number of epochs
    num_epochs = len(next(iter(loss_history.values())))
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch'] + list(loss_history.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for epoch in range(num_epochs):
            row = {'epoch': epoch}
            for key, values in loss_history.items():
                row[key] = values[epoch] if epoch < len(values) else ''
            writer.writerow(row)
    
    print(f"Saved results to CSV: {csv_path}")