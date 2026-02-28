"""
NCT Optimized Training V2 - 改进版训练脚本

核心优化:
1. 数据量: 每类 10 -> 100 样本 (共500个训练样本)
2. 模型简化: d_model 504->256, n_layers 4->2 (避免过拟合)
3. 更强正则化: Dropout 0.5, 更强数据增强
4. 训练策略: 100 epochs + 早停机制

版本: V2
目标: 验证准确率 70%+
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from nct_modules.nct_batched import BatchedNCTManager
from nct_modules.nct_core import NCTConfig

# 版本号
VERSION = "v2"

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class OptimizedConfigV2:
    """优化版配置 V2"""
    # 版本
    version: str = VERSION
    
    # 模型配置 - 简化版 (避免过拟合)
    n_heads: int = 4           # 减少 head 数量 (7->4)
    n_layers: int = 2          # 减少层数 (4->2)
    d_model: int = 256         # 减小维度 (504->256)
    
    # 数据配置 - 增加数据量
    n_classes: int = 5         # MNIST 0-4
    n_samples_per_class: int = 100  # 增加数据量 (10->100)
    batch_size: int = 64       # 更大的批次
    
    # 训练配置
    n_epochs: int = 100        # 更多轮次
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3  # 更强的L2正则化
    warmup_epochs: int = 10
    patience: int = 20         # 早停耐心值
    
    # 正则化
    dropout_rate: float = 0.5   # 更高的Dropout
    
    # 其他
    seed: int = 42
    results_dir: str = f'results/training_{VERSION}'


# ============================================================================
# 数据增强
# ============================================================================

class StrongAugmentation:
    """强数据增强"""
    
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
        ])
    
    def __call__(self, img):
        img_pil = transforms.ToPILImage()(img)
        img_aug = self.transforms(img_pil)
        img_tensor = transforms.ToTensor()(img_aug)
        
        # 随机噪声
        if np.random.random() < 0.3:
            noise = torch.randn_like(img_tensor) * 0.15
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        return img_tensor


# ============================================================================
# 数据集
# ============================================================================

class NCTMNISTDatasetV2(Dataset):
    """改进版 MNIST 数据集"""
    
    def __init__(self, mnist_dataset, indices=None, augment=False, max_samples=None):
        self.dataset = mnist_dataset
        self.indices = indices if indices is not None else list(range(len(mnist_dataset)))
        self.augment = augment
        self.augmenter = StrongAugmentation() if augment else None
        
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.dataset[actual_idx]
        
        if self.augment and self.augmenter is not None:
            img = self.augmenter(img)
        
        visual = img.squeeze(0).numpy()
        return {'visual': visual}, label


def create_balanced_dataset(root, n_classes, n_samples_per_class, seed):
    """创建平衡数据集"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.ToTensor()
    
    full_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    selected_indices = []
    for class_id in range(n_classes):
        class_indices = [i for i in range(len(full_dataset)) 
                        if full_dataset.targets[i] == class_id]
        
        n_select = min(n_samples_per_class, len(class_indices))
        selected = np.random.choice(class_indices, n_select, replace=False)
        selected_indices.extend(selected)
    
    np.random.shuffle(selected_indices)
    
    return full_dataset, selected_indices


# ============================================================================
# 简化版分类器
# ============================================================================

class SimpleClassifierV2(nn.Module):
    """简化版分类头 V2"""
    
    def __init__(self, input_dim, n_classes, dropout_rate=0.5):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# 训练器
# ============================================================================

class OptimizedTrainerV2:
    """优化版训练器 V2"""
    
    def __init__(self, manager, config: OptimizedConfigV2):
        self.manager = manager
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.manager.to(self.device)
        
        # 分类头
        self.classifier = SimpleClassifierV2(
            config.d_model, 
            config.n_classes,
            config.dropout_rate
        ).to(self.device)
        
        # 优化器
        params = list(self.manager.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs - config.warmup_epochs,
            eta_min=1e-6,
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [],
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 参数量统计
        total_params = sum(p.numel() for p in params)
        logger.info(f"Total parameters: {total_params:,}")
    
    def _warmup_lr(self, epoch):
        """Warmup 学习率"""
        if epoch < self.config.warmup_epochs:
            lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return None
    
    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.manager.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.n_epochs}')
        
        for batch_data, labels in pbar:
            batch_tensors = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    batch_tensors[key] = torch.from_numpy(value).float().to(self.device)
                elif isinstance(value, torch.Tensor):
                    batch_tensors[key] = value.float().to(self.device)
            
            labels = labels.to(self.device)
            
            # 前向
            batch_state = self.manager.process_batch(batch_tensors)
            representations = batch_state['representations']
            predictions = self.classifier(representations)
            
            # 损失
            loss = self.criterion(predictions, labels)
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.manager.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * labels.shape[0]
            _, predicted = predictions.max(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.shape[0]
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{total_correct/total_samples:.2%}"})
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """评估"""
        self.manager.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_data, labels in val_loader:
            batch_tensors = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    batch_tensors[key] = torch.from_numpy(value).float().to(self.device)
                elif isinstance(value, torch.Tensor):
                    batch_tensors[key] = value.float().to(self.device)
            
            labels = labels.to(self.device)
            
            batch_state = self.manager.process_batch(batch_tensors)
            representations = batch_state['representations']
            predictions = self.classifier(representations)
            
            loss = self.criterion(predictions, labels)
            
            total_loss += loss.item() * labels.shape[0]
            _, predicted = predictions.max(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.shape[0]
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}
    
    def fit(self, train_loader, val_loader):
        """完整训练 (带早停)"""
        logger.info(f"\nStarting training for {self.config.n_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.n_epochs):
            # Warmup
            self._warmup_lr(epoch)
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # 最佳模型检查
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                logger.info(f"★ New Best: {self.best_val_acc:.2%} at epoch {self.best_epoch}")
            else:
                self.patience_counter += 1
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.config.n_epochs} - "
                f"Train: {train_metrics['accuracy']:.2%}, Val: {val_metrics['accuracy']:.2%}, "
                f"LR: {current_lr:.2e}, Patience: {self.patience_counter}/{self.config.patience}"
            )
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # 达到目标检查
            if self.best_val_acc >= 0.70:
                logger.info(f"Target 70% achieved! Best: {self.best_val_acc:.2%}")
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time/60:.1f} min, Best: {self.best_val_acc:.2%}")
        
        return self.history


# ============================================================================
# 可视化
# ============================================================================

def plot_results_v2(history, save_path, config):
    """绘制训练结果 V2"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', lw=2, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curve ({VERSION})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], 'b-', lw=2, label='Train')
    ax.plot(epochs, [a*100 for a in history['val_acc']], 'r-', lw=2, label='Val')
    ax.axhline(y=70, color='g', ls='--', alpha=0.7, label='70% Target')
    ax.axhline(y=20, color='gray', ls=':', alpha=0.5, label='Random (20%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Curve ({VERSION})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], 'g-', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'LR Schedule ({VERSION})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Gap
    ax = axes[1, 1]
    gap = [(t - v)*100 for t, v in zip(history['train_acc'], history['val_acc'])]
    ax.plot(epochs, gap, 'purple', lw=2)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.fill_between(epochs, 0, gap, alpha=0.3, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Val Acc (%)')
    ax.set_title(f'Overfitting Gap ({VERSION})')
    ax.grid(True, alpha=0.3)
    
    # 添加配置信息
    info_text = (
        f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
        f"samples/class={config.n_samples_per_class}"
    )
    fig.suptitle(info_text, fontsize=10, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    config = OptimizedConfigV2()
    
    # 创建结果目录 (带版本号)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"NCT Optimized Training {VERSION.upper()}")
    logger.info("=" * 70)
    logger.info(f"Config: {asdict(config)}")
    
    # 随机种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # =========================================================================
    # 数据
    # =========================================================================
    logger.info("\n[1/4] Preparing data...")
    
    full_dataset, train_indices = create_balanced_dataset(
        root='data',
        n_classes=config.n_classes,
        n_samples_per_class=config.n_samples_per_class,
        seed=config.seed,
    )
    
    train_dataset = NCTMNISTDatasetV2(
        full_dataset,
        indices=train_indices,
        augment=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    
    logger.info(f"  Train samples: {len(train_dataset)} ({config.n_samples_per_class}/class)")
    
    # 测试集
    transform = transforms.ToTensor()
    test_full = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    test_indices = [i for i in range(len(test_full)) if test_full.targets[i] < config.n_classes]
    test_dataset = NCTMNISTDatasetV2(test_full, indices=test_indices, max_samples=1000)
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # =========================================================================
    # 模型
    # =========================================================================
    logger.info("\n[2/4] Creating NCT model...")
    
    nct_config = NCTConfig(
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_model=config.d_model,
    )
    
    manager = BatchedNCTManager(nct_config)
    
    logger.info(f"  n_heads: {config.n_heads}, n_layers: {config.n_layers}, d_model: {config.d_model}")
    
    # =========================================================================
    # 训练
    # =========================================================================
    logger.info("\n[3/4] Training...")
    
    trainer = OptimizedTrainerV2(manager, config)
    
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader)
    total_time = time.time() - start_time
    
    # =========================================================================
    # 保存结果 (带版本号)
    # =========================================================================
    logger.info("\n[4/4] Saving results...")
    
    # 训练历史
    history_path = results_dir / f'history_{VERSION}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  History: {history_path}")
    
    # 图表
    plot_path = results_dir / f'results_{VERSION}.png'
    plot_results_v2(history, plot_path, config)
    
    # 配置
    config_path = results_dir / f'config_{VERSION}.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"  Config: {config_path}")
    
    # 总结
    summary = {
        'version': VERSION,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'total_epochs': len(history['train_acc']),
        'total_time_min': total_time / 60,
        'target_achieved': trainer.best_val_acc >= 0.70,
    }
    summary_path = results_dir / f'summary_{VERSION}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary: {summary_path}")
    
    # =========================================================================
    # 打印总结
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - {VERSION.upper()}")
    print("=" * 70)
    print(f"  Total Time:        {total_time/60:.1f} min")
    print(f"  Total Epochs:      {len(history['train_acc'])}")
    print(f"  Best Val Accuracy: {trainer.best_val_acc:.2%} (epoch {trainer.best_epoch})")
    print(f"  Final Train Acc:   {history['train_acc'][-1]:.2%}")
    print(f"  Final Val Acc:     {history['val_acc'][-1]:.2%}")
    print(f"  Target (70%):      {'✓ ACHIEVED' if trainer.best_val_acc >= 0.7 else '✗ NOT YET'}")
    print("=" * 70)
    print(f"Results saved to: {results_dir}/")
    print(f"  - history_{VERSION}.json")
    print(f"  - results_{VERSION}.png")
    print(f"  - config_{VERSION}.json")
    print(f"  - summary_{VERSION}.json")
    
    return trainer.best_val_acc


if __name__ == "__main__":
    main()
