"""
Phase 3: Complete CIFAR-10 Transfer Learning with NCT
======================================================
完整的 CIFAR-10 迁移学习训练脚本

功能：
1. 从 MNIST 预训练权重迁移
2. 数据增强（RandomCrop, HorizontalFlip, ColorJitter）
3. OneCycleLR 学习率调度
4. 早停机制
5. 完整训练历史保存和可视化

使用方法：
    python experiments/run_cifar10_full.py --epochs 100 --batch_size 128 --lr 0.001

Author: WENG YONGGANG
Affiliation: NeuroConscious Lab, Universiti Teknologi Malaysia
Date: February 24, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import NCT modules - 修复路径问题
import sys
import os
# 添加项目根目录到 Python 路径（无论从哪里执行）
project_root = Path(__file__).parent.parent  # experiments/.. = project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_cifar10 import NCTForCIFAR10


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCT CIFAR-10 Transfer Learning')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (default: 512)')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers (default: 4)')
    parser.add_argument('--dim_ff', type=int, default=1024, help='Feedforward dimension (default: 1024)')
    parser.add_argument('--n_candidates', type=int, default=20, help='Number of candidates (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs (default: 10)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    
    # 数据增强
    parser.add_argument('--augmentation', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_false', dest='augmentation', help='Disable data augmentation')
    
    # 迁移学习
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained MNIST model')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results/cifar10', help='Output directory')
    parser.add_argument('--name', type=str, default='', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    return parser.parse_args()


def create_model(args):
    """创建模型并加载预训练权重"""
    print("\n" + "=" * 60)
    print("Creating NCTForCIFAR10 Model")
    print("=" * 60)
    
    model = NCTForCIFAR10(
        input_shape=(3, 32, 32),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        num_classes=10,
        dropout_rate=args.dropout,
        n_candidates=args.n_candidates
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights if provided
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\nLoading pretrained weights from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        # Filter state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle architecture mismatch
        filtered_state = {}
        for key, value in state_dict.items():
            # Skip classifier if different
            if 'classifier' in key and args.freeze_encoder:
                continue
            # Adapt first conv layer if needed
            if 'encoder.0' in key and value.shape[1] == 1:
                # Expand from 1 channel to 3 channels
                if 'encoder.0.weight' in key:
                    filtered_state[key] = value.repeat(1, 3, 1, 1) / 3.0  # Average
                else:
                    filtered_state[key] = value
            else:
                filtered_state[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        
        print(f"  Loaded weights:")
        print(f"    Missing keys: {len(missing_keys)}")
        print(f"    Unexpected keys: {len(unexpected_keys)}")
        
        if args.freeze_encoder:
            print(f"  Freezing encoder layers...")
            for name, param in model.named_parameters():
                if 'encoder' in name or 'transformer' in name or 'candidate_generator' in name:
                    param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters after freezing: {trainable_params:,}")
    
    return model


def get_data_loaders(args):
    """创建数据加载器"""
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 Dataset")
    print("=" * 60)
    
    # Data augmentation
    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        print("  ✓ Data augmentation enabled")
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        print("  ✗ Data augmentation disabled")
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Classes: {train_dataset.classes}")
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output_dict = model(data)
        output = output_dict['output']
        
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Collect metrics
        all_phis.extend(output_dict['phi'].detach().cpu().numpy())
        all_pes.extend(output_dict['prediction_error'].detach().cpu().numpy())
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.mean(all_phis), np.mean(all_pes)


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_phis = []
    all_pes = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        output_dict = model(data)
        output = output_dict['output']
        
        loss = criterion(output, target)
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Collect metrics
        all_phis.extend(output_dict['phi'].cpu().numpy())
        all_pes.extend(output_dict['prediction_error'].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.mean(all_phis), np.mean(all_pes)


def visualize_training(history, save_path):
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2, color='blue')
    ax.plot(history['val_acc'], label='Val Acc', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax = axes[0, 2]
    ax.plot(history['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (OneCycleLR)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Φ value trajectory
    ax = axes[1, 0]
    ax.plot(history['train_phi'], label='Train Φ', linewidth=2, color='purple')
    ax.plot(history['val_phi'], label='Val Φ', linewidth=2, color='orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Φ Value (Information Integration)', fontsize=12)
    ax.set_title('Information Integration During Training', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Prediction Error trajectory
    ax = axes[1, 1]
    ax.plot(history['train_pe'], label='Train PE', linewidth=2, color='brown')
    ax.plot(history['val_pe'], label='Val PE', linewidth=2, color='pink')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Prediction Error (Free Energy)', fontsize=12)
    ax.set_title('Free Energy (Prediction Error) During Training', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Validation accuracy distribution
    ax = axes[1, 2]
    ax.hist(history['val_acc'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Validation Accuracy Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_acc = np.mean(history['val_acc'])
    std_acc = np.std(history['val_acc'])
    max_acc = np.max(history['val_acc'])
    
    stats_text = f'Mean: {mean_acc:.3f}\nStd: {std_acc:.3f}\nMax: {max_acc:.3f}'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def main():
    """主训练流程"""
    args = get_args()
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"cifar10_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 3: CIFAR-10 Transfer Learning with NCT")
    print("=" * 80)
    print(f"\nExperiment: {exp_name}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {args.device}")
    
    # Save configuration
    config = vars(args)
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    model = create_model(args)
    model = model.to(args.device)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Separate learning rates for encoder and classifier
    if args.freeze_encoder:
        optimizer = optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy='cos',
        cycle_momentum=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_phi': [],
        'val_phi': [],
        'train_pe': [],
        'val_pe': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc, train_phi, train_pe = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, args.device, epoch
        )
        
        # Evaluate
        val_loss, val_acc, val_phi, val_pe = evaluate(
            model, test_loader, criterion, args.device
        )
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_phi'].append(train_phi)
        history['val_phi'].append(val_phi)
        history['train_pe'].append(train_pe)
        history['val_pe'].append(val_pe)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Train Φ: {train_phi:.4f}, Val Φ: {val_phi:.4f}")
            print(f"  Train PE: {train_pe:.4f}, Val PE: {val_pe:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, results_dir / 'best_model.pt')
            
            print(f"  ⭐ New best model! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }, results_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  Checkpoint saved at epoch {epoch+1}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final epoch: {len(history['train_loss'])}")
    
    # Load best model for final test
    best_model_path = results_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Save training history
    results_summary = {
        'config': config,
        'training_history': history,
        'best_val_acc': best_val_acc,
        'final_epoch': len(history['train_loss']),
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1]
    }
    
    with open(results_dir / 'cifar10_training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_training(history, results_dir / 'cifar10_training_curves.png')
    
    # Compare with MNIST
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"MNIST V3 Accuracy:   99.2%")
    print(f"CIFAR-10 Accuracy:   {best_val_acc:.1%}")
    print(f"Performance gap:     {(0.992 - best_val_acc)*100:.1f}%")
    print(f"Total epochs:        {len(history['train_loss'])}")
    print(f"Best epoch:          {checkpoint.get('epoch', 0) + 1 if 'checkpoint' in locals() else 'N/A'}")
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("Phase 3 Completed Successfully!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    results = main()
