"""
NeuroConscious Transformer - MNIST 小样本学习实验
NCT MNIST Few-Shot Learning Experiment

实验目标:
1. 验证 NCT 能否通过少量样本（每类 10 个）学会识别数字
2. 对比传统 CNN 的样本效率
3. 测量 Φ 值、自由能等意识指标的变化

实验设计:
- 任务：MNIST 数字 0-4 识别（5 分类）
- 训练集：每类仅 10 个样本（共 50 个）
- 测试集：标准 1000 个样本
- 基线：简单 CNN（相同训练集）

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入 NCT 组件
from nct_modules import NCTManager, NCTConfig
from experiments.nct_trainer import NCTTrainer, TrainingConfig
from experiments.datasets import load_mnist

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集
    dataset_name: str = 'mnist'
    n_classes: int = 5  # 数字 0-4
    n_samples_per_class: int = 10  # 每类 10 个样本
    img_size: int = 28
    
    # NCT 配置
    n_heads: int = 7  # Miller 定律 7±2
    n_layers: int = 4  # 皮层 4 层
    d_model: int = 504  # 能被 7 整除
    gamma_freq: float = 40.0
    
    # 训练配置
    n_epochs: int = 50
    batch_size: int = 1  # 使用 batch_size=1 以适配 NCT 的单样本处理模式
    learning_rate: float = 1e-3
    
    # 损失权重
    lambda_classification: float = 1.0
    lambda_prediction_error: float = 0.5
    lambda_phi_regularization: float = 0.1
    lambda_sparsity: float = 0.01
    
    # STDP
    use_stdp: bool = True
    stdp_learning_rate: float = 0.01
    
    # 其他
    seed: int = 42
    checkpoint_dir: str = 'checkpoints/mnist_few_shot'
    results_dir: str = 'results/mnist_few_shot'


# ============================================================================
# 实验器
# ============================================================================

class MNISTFewShotExperiment:
    """MNIST 小样本学习实验器"""
    
    def __init__(self, config: ExperimentConfig):
        """初始化实验器"""
        self.config = config
        
        # 设置随机种子
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # 创建结果目录
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("NCT MNIST 小样本学习实验")
        logger.info("=" * 70)
        
        # 加载数据
        self._load_data()
        
        # 创建 NCT 模型
        self._create_model()
        
        # 创建训练器
        self._create_trainer()
    
    def _load_data(self):
        """加载数据集"""
        logger.info(f"\n加载数据集：{self.config.dataset_name}")
        logger.info(f"  - 类别数：{self.config.n_classes}")
        logger.info(f"  - 每类样本数：{self.config.n_samples_per_class}")
        
        self.train_loader, self.test_loader = load_mnist(
            root='data',
            batch_size=self.config.batch_size,
            n_samples_per_class=self.config.n_samples_per_class,
            n_classes=self.config.n_classes,
            img_size=self.config.img_size,
            seed=self.config.seed,
        )
        
        logger.info(f"  - 训练样本总数：{len(self.train_loader.dataset)}")
        logger.info(f"  - 测试样本总数：{len(self.test_loader.dataset)}")
    
    def _create_model(self):
        """创建 NCT 模型"""
        logger.info(f"\n创建 NCT 模型:")
        logger.info(f"  - n_heads: {self.config.n_heads}")
        logger.info(f"  - n_layers: {self.config.n_layers}")
        logger.info(f"  - d_model: {self.config.d_model}")
        
        config = NCTConfig(
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            gamma_freq=self.config.gamma_freq,
        )
        
        self.manager = NCTManager(config)
        logger.info(f"  - [OK] NCT 管理器已创建")
    
    def _create_trainer(self):
        """创建训练器"""
        logger.info(f"\n创建训练器:")
        
        training_config = TrainingConfig(
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            lambda_classification=self.config.lambda_classification,
            lambda_prediction_error=self.config.lambda_prediction_error,
            lambda_phi_regularization=self.config.lambda_phi_regularization,
            lambda_sparsity=self.config.lambda_sparsity,
            use_stdp=self.config.use_stdp,
            stdp_learning_rate=self.config.stdp_learning_rate,
            lr_scheduler_type='cosine',
            checkpoint_dir=self.config.checkpoint_dir,
        )
        
        self.trainer = NCTTrainer(
            manager=self.manager,
            config=training_config,
        )
        
        logger.info(f"  - [OK] 训练器已创建")
    
    def run(self):
        """运行完整实验"""
        logger.info("\n" + "=" * 70)
        logger.info("开始实验")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # 训练
        logger.info("\n开始训练...")
        self.trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
        )
        
        # 计算总时间
        total_time = time.time() - start_time
        
        logger.info(f"\n实验完成!")
        logger.info(f"  - 总耗时：{total_time/60:.1f} 分钟")
        logger.info(f"  - 最佳准确率：{self.trainer.state.best_accuracy:.2%}")
        
        # 保存结果
        self._save_results(total_time)
        
        # 可视化
        self._visualize_results()
        
        return self.trainer.state.best_accuracy
    
    def _save_results(self, total_time: float):
        """保存实验结果"""
        results = {
            'config': vars(self.config),
            'best_accuracy': self.trainer.state.best_accuracy,
            'total_time_minutes': total_time,
            'training_metrics': self.trainer.state.training_metrics,
            'validation_metrics': self.trainer.state.validation_metrics,
        }
        
        # 保存为 numpy 格式
        save_path = Path(self.config.results_dir) / 'results.npz'
        np.savez(save_path, **results)
        
        logger.info(f"[OK] 结果已保存到 {save_path}")
    
    def _visualize_results(self):
        """可视化训练结果"""
        logger.info("\n生成可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 训练损失曲线
        ax = axes[0, 0]
        epochs = range(len(self.trainer.state.training_metrics))
        
        total_losses = [m['total_loss'] for m in self.trainer.state.training_metrics]
        ax.plot(epochs, total_losses, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 验证准确率曲线
        ax = axes[0, 1]
        val_accs = [m.get('val_accuracy', 0) for m in self.trainer.state.validation_metrics]
        ax.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
        ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='80% 基线')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        ax = axes[1, 0]
        lrs = [m.get('learning_rate', 1e-3) for m in self.trainer.state.training_metrics]
        ax.plot(epochs, lrs, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (Cosine Annealing)')
        ax.grid(True, alpha=0.3)
        
        # 4. Φ值变化（如果有记录）
        ax = axes[1, 1]
        phi_values = []
        for metrics in self.trainer.state.training_metrics:
            # 从训练指标中提取（需要在 trainer 中添加记录）
            phi = metrics.get('phi_value', 0)
            phi_values.append(phi)
        
        if any(phi_values):
            ax.plot(epochs, phi_values, 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Φ Value')
            ax.set_title('Information Integration (Phi) Evolution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Φ值数据未记录\n需在 trainer 中添加', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Phi Value (No Data)')
        
        plt.tight_layout()
        save_path = Path(self.config.results_dir) / 'training_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] 可视化图表已保存到 {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    from dataclasses import dataclass
    
    # 创建实验配置
    config = ExperimentConfig()
    
    # 创建并运行实验
    experiment = MNISTFewShotExperiment(config)
    
    try:
        best_accuracy = experiment.run()
        
        # 打印总结
        print("\n" + "=" * 70)
        print("实验总结")
        print("=" * 70)
        print(f"✓ 训练完成!")
        print(f"✓ 最佳准确率：{best_accuracy:.2%}")
        print(f"✓ 结果保存在：{experiment.config.results_dir}/")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"实验失败：{str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
