"""
NCT Full MNIST Training - Phase 1 Optimization
完整 50 epochs 训练脚本

优化措施:
1. 完整 50 epochs 训练
2. 学习率调度 (Cosine Annealing)
3. 数据增强
4. 更好的超参数
5. 详细日志记录

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入批量化组件
from nct_modules.nct_batched import BatchedNCTManager
from nct_modules.nct_core import NCTConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    n_heads: int = 7
    n_layers: int = 4
    d_model: int = 504
    
    # 数据配置
    n_classes: int = 5  # MNIST 0-4
    n_samples_per_class: int = 10  # Few-shot
    batch_size: int = 32
    
    # 训练配置
    n_epochs: int = 50
    learning_rate: float = 5e-4  # 调低学习率以稳定训练
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # 其他
    seed: int = 42
    results_dir: str = 'results/full_mnist_training'


# ============================================================================
# 数据集
# ============================================================================

class NCTMNISTDataset(Dataset):
    """NCT 格式 MNIST 数据集"""
    
    def __init__(self, mnist_dataset, indices=None, max_samples=None, augment=False):
        self.dataset = mnist_dataset
        self.indices = indices
        self.max_samples = max_samples
        self.augment = augment
        
        if indices is not None:
            self.length = len(indices)
        else:
            self.length = len(mnist_dataset)
        
        if max_samples is not None:
            self.length = min(self.length, max_samples)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx
        
        img, label = self.dataset[actual_idx]
        
        # 数据增强
        if self.augment:
            img = self._augment(img)
        
        # 转为 NCT 格式 [H, W]
        visual = img.squeeze(0).numpy()
        
        return {'visual': visual}, label
    
    def _augment(self, img):
        """简单的数据增强"""
        # 随机噪声
        if np.random.random() < 0.3:
            noise = torch.randn_like(img) * 0.1
            img = img + noise
            img = torch.clamp(img, 0, 1)
        
        return img


def create_few_shot_dataset(root, n_classes, n_samples_per_class, seed):
    """创建 Few-shot 数据集"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载完整 MNIST
    full_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    # 按类别选择样本
    selected_indices = []
    for class_id in range(n_classes):
        class_indices = [i for i in range(len(full_dataset)) 
                        if full_dataset.targets[i] == class_id]
        
        # 随机选择
        selected = np.random.choice(class_indices, n_samples_per_class, replace=False)
        selected_indices.extend(selected)
    
    return full_dataset, selected_indices


# ============================================================================
# 训练器
# ============================================================================

class OptimizedNCTTrainer:
    """优化版 NCT 训练器"""
    
    def __init__(self, manager, config: TrainingConfig):
        self.manager = manager
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备：{self.device}")
        
        self.manager.to(self.device)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.n_classes),
        ).to(self.device)
        
        # 优化器
        params = list(self.manager.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度 (Cosine Annealing with Warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs - config.warmup_epochs,
            eta_min=1e-6,
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
        }
        
        self.best_val_acc = 0.0
        
        # 统计参数量
        total_params = sum(p.numel() for p in params)
        logger.info(f"模型参数量：{total_params:,}")
    
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
            # 数据移动
            batch_tensors = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    batch_tensors[key] = torch.from_numpy(value).float().to(self.device)
                elif isinstance(value, torch.Tensor):
                    batch_tensors[key] = value.float().to(self.device)
            
            labels = labels.to(self.device)
            
            # 前向传播
            batch_state = self.manager.process_batch(batch_tensors)
            representations = batch_state['representations']
            
            predictions = self.classifier(representations)
            
            # 损失
            loss = self.criterion(predictions, labels)
            
            # 反向传播
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
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct / total_samples:.2%}",
            })
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """评估"""
        self.manager.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_data, labels in tqdm(val_loader, desc='Evaluating'):
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
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }
    
    def fit(self, train_loader, val_loader):
        """完整训练"""
        logger.info(f"\n开始训练 {self.config.n_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.n_epochs):
            # Warmup
            warmup_lr = self._warmup_lr(epoch)
            
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
            
            # 最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                logger.info(f"★ New Best Val Acc: {self.best_val_acc:.2%}")
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.config.n_epochs} - "
                f"Train: {train_metrics['accuracy']:.2%}, "
                f"Val: {val_metrics['accuracy']:.2%}, "
                f"LR: {current_lr:.2e}"
            )
        
        total_time = time.time() - start_time
        
        logger.info(f"\n训练完成！")
        logger.info(f"  - 总耗时：{total_time/60:.1f} 分钟")
        logger.info(f"  - 最佳验证准确率：{self.best_val_acc:.2%}")
        
        return self.history


# ============================================================================
# 可视化
# ============================================================================

def plot_results(history, save_path):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], 'b-', linewidth=2, label='Train Acc')
    ax.plot(epochs, [a*100 for a in history['val_acc']], 'r-', linewidth=2, label='Val Acc')
    ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='70% Target')
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Random (20%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 学习率曲线
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Warmup + Cosine)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. 训练/验证差距（过拟合指标）
    ax = axes[1, 1]
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    ax.plot(epochs, [g*100 for g in gap], 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, 0, [g*100 for g in gap], alpha=0.3, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Val Acc (%)')
    ax.set_title('Overfitting Gap')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"结果图表已保存到 {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 配置
    config = TrainingConfig()
    
    # 创建结果目录
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("NCT Full MNIST Training - Phase 1 Optimization")
    logger.info("="*70)
    logger.info(f"配置：{asdict(config)}")
    
    # 设置随机种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # =========================================================================
    # 数据准备
    # =========================================================================
    logger.info("\n[1/4] 准备数据集...")
    
    # 训练集：Few-shot
    full_dataset, train_indices = create_few_shot_dataset(
        root='data',
        n_classes=config.n_classes,
        n_samples_per_class=config.n_samples_per_class,
        seed=config.seed,
    )
    
    train_dataset = NCTMNISTDataset(
        full_dataset,
        indices=train_indices,
        augment=True,  # 启用数据增强
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    logger.info(f"  - 训练样本：{len(train_dataset)} (每类 {config.n_samples_per_class} 个)")
    
    # 测试集
    transform = transforms.ToTensor()
    test_full = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )
    
    test_indices = [i for i in range(len(test_full)) if test_full.targets[i] < config.n_classes]
    test_dataset = NCTMNISTDataset(
        test_full,
        indices=test_indices,
        max_samples=1000,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    logger.info(f"  - 测试样本：{len(test_dataset)}")
    
    # =========================================================================
    # 模型创建
    # =========================================================================
    logger.info("\n[2/4] 创建 NCT 模型...")
    
    nct_config = NCTConfig(
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_model=config.d_model,
    )
    
    manager = BatchedNCTManager(nct_config)
    
    logger.info(f"  - n_heads: {config.n_heads}")
    logger.info(f"  - n_layers: {config.n_layers}")
    logger.info(f"  - d_model: {config.d_model}")
    
    # =========================================================================
    # 训练
    # =========================================================================
    logger.info("\n[3/4] 开始训练...")
    
    trainer = OptimizedNCTTrainer(manager, config)
    
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader)
    total_time = time.time() - start_time
    
    # =========================================================================
    # 保存结果
    # =========================================================================
    logger.info("\n[4/4] 保存结果...")
    
    # 保存训练历史
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"  - 训练历史已保存到 {history_path}")
    
    # 绘制结果
    plot_path = results_dir / 'training_results.png'
    plot_results(history, plot_path)
    
    # 保存配置
    config_path = results_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Total Time:        {total_time/60:.1f} minutes")
    print(f"  Best Val Accuracy: {trainer.best_val_acc:.2%}")
    print(f"  Final Train Acc:   {history['train_acc'][-1]:.2%}")
    print(f"  Final Val Acc:     {history['val_acc'][-1]:.2%}")
    print(f"  Target (70%):      {'ACHIEVED' if trainer.best_val_acc >= 0.7 else 'NOT YET'}")
    print("="*70)
    print(f"Results saved to: {results_dir}")
    
    return trainer.best_val_acc


if __name__ == "__main__":
    main()
