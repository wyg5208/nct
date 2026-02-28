"""
Phase 3: CIFAR-10 Transfer Learning with NCT V3
================================================
迁移学习：从 MNIST（灰度）到 CIFAR-10（彩色）

关键调整：
1. 输入层：1 通道 → 3 通道（RGB）
2. 图像尺寸：28x28 → 32x32
3. Patch size 调整以适应彩色图像
4. 数据增强：RandomCrop, RandomHorizontalFlip

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
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import NCT modules
import sys
sys.path.append('..')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


@dataclass
class CIFAR10Config:
    """CIFAR-10 训练配置"""
    # 架构参数（与 V3 一致）
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 3
    dim_ff: int = 768
    dropout_rate: float = 0.4
    
    # 训练参数
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # 数据增强
    use_augmentation: bool = True
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
    
    # 优化器
    warmup_epochs: int = 10
    patience: int = 20  # Early stopping
    
    # 输出
    version: str = "v1"
    results_dir: str = None
    
    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.results_dir is None:
            self.results_dir = Path(f'results/cifar10_{timestamp}')
        else:
            self.results_dir = Path(self.results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)


def create_model(config, pretrained_mnist_path=None):
    """创建 CIFAR-10 模型，可选择加载 MNIST 预训练权重"""
    
    # 直接在这里定义模型类（从 V3 复制）
    from experiments.run_optimized_training_v3 import NCTClassifierV3
    
    # 创建模型（适配 CIFAR-10：3 通道输入）
    model = NCTClassifierV3(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dim_ff=config.dim_ff,
        num_classes=10,
        dropout_rate=config.dropout_rate,
        input_channels=3  # RGB
    )
    
    # 加载 MNIST 预训练权重（迁移学习）
    if pretrained_mnist_path and os.path.exists(pretrained_mnist_path):
        print(f"\nLoading MNIST pretrained weights from {pretrained_mnist_path}...")
        checkpoint = torch.load(pretrained_mnist_path, map_location='cpu')
        
        # 过滤 classifier 层
        state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith('classifier'):
                state_dict[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"  Loaded weights:")
        print(f"    Missing keys: {len(missing_keys)} (expected: classifier layers)")
        print(f"    Unexpected keys: {len(unexpected_keys)}")
        print(f"  ✓ Transfer learning initialized!")
    
    return model


def get_cifar10_dataloaders(config):
    """创建 CIFAR-10 数据加载器"""
    
    # 数据增强
    if config.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=config.random_crop_padding),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载数据集
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += data.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    """主训练流程"""
    print("=" * 80)
    print("Phase 3: CIFAR-10 Transfer Learning with NCT")
    print("=" * 80)
    
    # 配置
    config = CIFAR10Config(
        d_model=384,
        n_heads=6,
        n_layers=3,
        batch_size=128,
        n_epochs=100,
        learning_rate=0.001,
        use_augmentation=True
    )
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Data augmentation: {config.use_augmentation}")
    print(f"  Results dir: {config.results_dir}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 创建模型（带 MNIST 预训练权重）
    mnist_model_path = Path('../results/training_v3/best_model_v3.pt')
    model = create_model(config, pretrained_mnist_path=str(mnist_model_path) if mnist_model_path.exists() else None)
    model = model.to(device)
    
    # 数据加载器
    train_loader, test_loader = get_cifar10_dataloaders(config)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_epochs / config.n_epochs,
        anneal_strategy='cos'
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(config.n_epochs):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{config.n_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(config)
            }, config.results_dir / 'best_model.pt')
            
            print(f"  ⭐ New best model! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config.patience} epochs)")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for final test
    best_model_path = config.results_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Save results
    results_summary = {
        'config': vars(config),
        'training_history': history,
        'best_val_acc': best_val_acc,
        'final_epoch': len(history['train_loss'])
    }
    
    with open(config.results_dir / 'cifar10_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_training(history, config.results_dir)
    
    # Compare with MNIST
    print("\nComparison with MNIST V3:")
    print(f"  MNIST V3 Accuracy:   99.2%")
    print(f"  CIFAR-10 Accuracy:   {best_val_acc:.1%}")
    print(f"  Performance gap:     {(0.992 - best_val_acc)*100:.1f}%")
    
    print(f"\nResults saved to: {config.results_dir}")
    print("\n" + "=" * 80)
    
    return results_summary


def visualize_training(history, save_dir):
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax = axes[1, 0]
    ax.plot(history['learning_rates'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (OneCycleLR)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Val accuracy distribution
    ax = axes[1, 1]
    ax.hist(history['val_acc'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequency')
    ax.set_title('Validation Accuracy Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cifar10_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_dir / 'cifar10_training_curves.png'}")


if __name__ == '__main__':
    results = main()
