"""
NCT 批量化架构测试
Batched NCT Architecture Test

快速验证批量化架构是否工作

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入批量化组件
from nct_modules.nct_batched import BatchedNCTManager, BatchedAttentionWorkspace
from nct_modules.nct_core import NCTConfig
from experiments.batched_trainer import BatchedNCTTrainer
from experiments.datasets import load_mnist

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_batched_workspace():
    """测试批量化工作空间"""
    logger.info("\n" + "="*70)
    logger.info("测试 1: BatchedAttentionWorkspace")
    logger.info("="*70)
    
    # 创建工作空间
    workspace = BatchedAttentionWorkspace(
        d_model=504,
        n_heads=7,
        gamma_freq=40.0,
    )
    
    # 模拟 batch 输入
    B = 10  # batch size
    D = 504  # 表征维度
    
    # 创建候选（每个候选都是 [B, D]）
    candidates = [
        torch.randn(B, D),  # 整合表征
        torch.randn(B, D),  # 视觉表征
        torch.randn(B, D),  # 听觉表征
    ]
    
    # 前向传播
    winners, info = workspace(candidates)
    
    logger.info(f"✓ 输入：{len(candidates)} 个候选，每个 shape = {candidates[0].shape}")
    logger.info(f"✓ 输出：winners shape = {winners.shape}")
    logger.info(f"✓ 注意力权重 shape = {info['attention_weights'].shape}")
    logger.info(f"✓ 获胜者索引 = {info['winner_indices']}")
    logger.info(f"✓ 显著性 = {info['salience']}")
    
    assert winners.shape == (B, D), f"winners shape 错误：{winners.shape}"
    assert info['winner_indices'].shape == (B,), f"winner_indices shape 错误"
    
    logger.info("✓ 测试通过！")
    
    return True


def test_batched_manager():
    """测试批量化管理器"""
    logger.info("\n" + "="*70)
    logger.info("测试 2: BatchedNCTManager")
    logger.info("="*70)
    
    # 创建配置和管理器
    config = NCTConfig(
        n_heads=7,
        n_layers=4,
        d_model=504,
    )
    
    manager = BatchedNCTManager(config)
    manager.start()
    
    # 模拟 batch 输入
    B = 5
    H, W = 28, 28
    
    batch_sensory_data = {
        'visual': torch.randn(B, H, W),
    }
    
    # 前向传播
    start_time = time.time()
    batch_state = manager.process_batch(batch_sensory_data)
    elapsed = time.time() - start_time
    
    logger.info(f"✓ 输入：batch_size = {B}")
    logger.info(f"✓ 输出：representations shape = {batch_state['representations'].shape}")
    logger.info(f"✓ 显著性 = {batch_state['salience']}")
    logger.info(f"✓ 处理时间：{elapsed*1000:.2f} ms")
    
    assert batch_state['representations'].shape == (B, 504), "representations shape 错误"
    
    logger.info("✓ 测试通过！")
    
    return True


def test_batched_training():
    """测试批量化训练"""
    logger.info("\n" + "="*70)
    logger.info("测试 3: BatchedNCTTrainer - MNIST 小样本")
    logger.info("="*70)
    
    # 配置
    config = NCTConfig(
        n_heads=7,
        n_layers=4,
        d_model=504,
    )
    
    # 创建管理器
    manager = BatchedNCTManager(config)
    
    # 加载数据
    logger.info("加载 MNIST 数据集...")
    
    # 训练集：few-shot
    train_loader, _ = load_mnist(
        root='data',
        batch_size=32,
        n_samples_per_class=10,
        n_classes=5,
        seed=42,
    )
    
    # 测试集：同样只使用前 5 类
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )
    
    # 只保留前 5 类
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] < 5]
    test_dataset_subset = Subset(test_dataset, test_indices)
    
    # 创建数据加载器（不使用 FewShotDataset，手动转换）
    from torch.utils.data import Dataset
    
    class SimpleTestDataset(Dataset):
        def __init__(self, subset):
            self.subset = subset
        
        def __len__(self):
            return min(len(self.subset), 1000)  # 只用前 1000 个样本
        
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            # 转为 NCT 格式
            visual = img.squeeze(0).numpy()  # [H, W]
            return {'visual': visual}, label
    
    test_dataset_wrapped = SimpleTestDataset(test_dataset_subset)
    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=32,
        shuffle=False,
    )
    
    # 创建训练器
    trainer = BatchedNCTTrainer(
        manager=manager,
        learning_rate=1e-3,
        n_epochs=10,  # 快速测试，只训练 10 个 epoch
    )
    
    # 训练
    logger.info("开始训练...")
    start_time = time.time()
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
    )
    
    elapsed = time.time() - start_time
    
    # 结果
    logger.info(f"\n训练完成！")
    logger.info(f"  - 耗时：{elapsed:.1f} 秒")
    logger.info(f"  - 最终训练准确率：{history['train_acc'][-1]:.2%}")
    logger.info(f"  - 最终验证准确率：{history['val_acc'][-1]:.2%}")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], 'b-', label='Train Acc')
    if history['val_acc']:
        axes[1].plot(history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/batched_training_results.png', dpi=150)
    plt.close()
    
    logger.info(f"✓ 训练曲线已保存到 results/batched_training_results.png")
    
    return True


def compare_single_vs_batched():
    """对比单样本 vs 批量化处理"""
    logger.info("\n" + "="*70)
    logger.info("测试 4: 性能对比 - 单样本 vs 批量化")
    logger.info("="*70)
    
    config = NCTConfig(
        n_heads=7,
        n_layers=4,
        d_model=504,
    )
    
    # 测试不同 batch size
    batch_sizes = [1, 4, 8, 16, 32]
    times = []
    
    manager = BatchedNCTManager(config)
    manager.start()
    
    for B in batch_sizes:
        batch_sensory_data = {
            'visual': torch.randn(B, 28, 28),
        }
        
        # 预热
        _ = manager.process_batch(batch_sensory_data)
        
        # 计时
        start_time = time.time()
        for _ in range(10):  # 运行 10 次取平均
            _ = manager.process_batch(batch_sensory_data)
        elapsed = (time.time() - start_time) / 10
        
        times.append(elapsed)
        samples_per_sec = B / elapsed
        
        logger.info(f"Batch size = {B:2d}: {elapsed*1000:6.2f} ms, {samples_per_sec:8.1f} samples/sec")
    
    # 计算加速比
    speedup = times[0] / times[-1]
    logger.info(f"\n✓ 最大加速比：{speedup:.1f}x (batch_size=32 vs 1)")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    samples_per_sec = [b / t for b, t in zip(batch_sizes, times)]
    
    ax.bar(range(len(batch_sizes)), samples_per_sec, color='steelblue', alpha=0.7)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Samples/Second')
    ax.set_title('Batched Processing Performance Improvement')
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (b, sps) in enumerate(zip(batch_sizes, samples_per_sec)):
        ax.text(i, sps + 5, f'{sps:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/batched_speedup.png', dpi=150)
    plt.close()
    
    logger.info(f"✓ 性能对比图已保存到 results/batched_speedup.png")
    
    return True


def main():
    """运行所有测试"""
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    logger.info("="*70)
    logger.info("NCT 批量化架构测试套件")
    logger.info("="*70)
    
    try:
        # 测试 1: 工作空间
        test_batched_workspace()
        
        # 测试 2: 管理器
        test_batched_manager()
        
        # 测试 3: 训练
        test_batched_training()
        
        # 测试 4: 性能对比
        compare_single_vs_batched()
        
        logger.info("\n" + "="*70)
        logger.info("所有测试通过！")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"测试失败：{str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
