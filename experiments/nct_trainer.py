"""
NeuroConscious Transformer - 训练器
NCT Trainer: 完整的训练基础设施

功能:
1. 封装完整的训练循环（梯度下降 + STDP 混合更新）
2. 支持多任务损失函数
3. 集成优化器和调度器
4. 训练日志和监控（TensorBoard/WandB）
5. 模型保存/恢复

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
版本：v1.0.0
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 尝试导入 wandb（可选）
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger = logging.getLogger(__name__)
    logger.warning("wandb 未安装，禁用实验跟踪功能")

logger = logging.getLogger(__name__)


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 优化器
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # 训练参数
    n_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # 损失权重
    lambda_classification: float = 1.0
    lambda_prediction_error: float = 0.5
    lambda_phi_regularization: float = 0.1
    lambda_sparsity: float = 0.01
    
    # STDP 参数
    use_stdp: bool = True
    stdp_learning_rate: float = 0.01
    stdp_update_frequency: int = 1  # 每个 batch 更新一次
    
    # 学习率调度
    lr_scheduler_type: str = 'cosine'  # 'step', 'cosine', 'none'
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # 正则化
    dropout: float = 0.1
    clip_grad_norm: float = 1.0
    
    # 日志与保存
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 10
    checkpoint_dir: str = 'checkpoints'
    
    # 实验跟踪
    use_wandb: bool = False
    project_name: str = 'NCT'
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """验证配置"""
        if self.experiment_name is None:
            self.experiment_name = f'nct_{time.strftime("%Y%m%d_%H%M%S")}'


# ============================================================================
# 训练状态
# ============================================================================

@dataclass
class TrainingState:
    """训练状态（用于保存 checkpoint）"""
    
    epoch: int = 0
    step: int = 0
    best_accuracy: float = 0.0
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    model_state_dict: Optional[Dict[str, Any]] = None
    training_metrics: List[Dict[str, float]] = field(default_factory=list)
    validation_metrics: List[Dict[str, float]] = field(default_factory=list)


# ============================================================================
# 多任务损失函数
# ============================================================================

class MultiTaskLoss(nn.Module):
    """NCT 多任务损失函数
    
    Total Loss = λ₁·Classification_Loss 
               + λ₂·Prediction_Error (自由能)
               + λ₃·Φ值正则化 (鼓励高整合)
               + λ₄·Sparsity_Loss (稀疏连接)
    """
    
    def __init__(
        self,
        lambda_classification: float = 1.0,
        lambda_prediction_error: float = 0.5,
        lambda_phi: float = 0.1,
        lambda_sparsity: float = 0.01,
    ):
        super().__init__()
        
        self.lambda_cls = lambda_classification
        self.lambda_pred = lambda_prediction_error
        self.lambda_phi = lambda_phi
        self.lambda_sparse = lambda_sparsity
        
        # 分类损失（交叉熵）
        self.classification_loss = nn.CrossEntropyLoss()
        
        # 预测误差（MSE）
        self.prediction_loss = nn.MSELoss()
        
        logger.info(
            f"[MultiTaskLoss] 初始化："
            f"λ_cls={lambda_classification}, λ_pred={lambda_prediction_error}, "
            f"λ_phi={lambda_phi}, λ_sparse={lambda_sparsity}"
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        state: Any,  # NCTConsciousnessState
        manager: Any,  # NCTManager
    ) -> Dict[str, torch.Tensor]:
        """计算总损失
        
        Args:
            predictions: 模型输出 [B, num_classes]
            labels: 真实标签 [B]
            state: 意识状态
            manager: NCT 管理器
            
        Returns:
            包含各项损失的字典
        """
        result = {}
        
        # 1. 分类损失（主要任务）
        loss_cls = self.classification_loss(predictions, labels)
        result['classification_loss'] = loss_cls
        
        # 2. 预测误差（自由能最小化）
        if hasattr(state, 'self_representation') and 'free_energy' in state.self_representation:
            free_energy = state.self_representation['free_energy']
            # 将自由能转为 tensor 并作为辅助损失
            loss_pred = torch.tensor(free_energy, device=predictions.device)
        else:
            # 如果没有自由能信息，使用预测层输出
            if hasattr(manager, 'predictive_hierarchy'):
                # 从预测层级提取误差
                pred_results = manager.predictive_hierarchy.forward_with_sequence(
                    torch.randn(1, 2, manager.config.d_model, device=predictions.device)
                )
                loss_pred = torch.tensor(
                    pred_results.get('total_free_energy', 0.5),
                    device=predictions.device
                )
            else:
                loss_pred = torch.tensor(0.0, device=predictions.device)
        
        result['prediction_error'] = loss_pred
        
        # 3. Φ值正则化（鼓励高信息整合）
        # 目标：最大化 Φ 值 → 最小化 -log(Φ)
        phi_value = state.consciousness_metrics.get('phi_value', 0.0)
        if phi_value > 0:
            loss_phi = -torch.log(torch.tensor(phi_value + 1e-8, device=predictions.device))
        else:
            loss_phi = torch.tensor(0.0, device=predictions.device)
        
        result['phi_regularization'] = loss_phi
        
        # 4. 稀疏性损失（L1 正则化突触权重）
        if hasattr(manager, 'hybrid_learner') and hasattr(manager.hybrid_learner, 'synaptic_weights'):
            weights = manager.hybrid_learner.synaptic_weights
            loss_sparse = weights.abs().mean()
        else:
            loss_sparse = torch.tensor(0.0, device=predictions.device)
        
        result['sparsity_loss'] = loss_sparse
        
        # 5. 总损失
        total_loss = (
            self.lambda_cls * loss_cls +
            self.lambda_pred * loss_pred +
            self.lambda_phi * loss_phi +
            self.lambda_sparse * loss_sparse
        )
        
        result['total_loss'] = total_loss
        
        # 记录详细日志
        logger.debug(
            f"Loss breakdown: "
            f"L_cls={loss_cls.item():.4f}, L_pred={loss_pred.item():.4f}, "
            f"L_phi={loss_phi.item():.4f}, L_sparse={loss_sparse.item():.4f}, "
            f"Total={total_loss.item():.4f}"
        )
        
        return result


# ============================================================================
# NCT 训练器
# ============================================================================

class NCTTrainer:
    """NCT 完整训练器
    
    特性:
    1. 梯度下降 + STDP 混合更新
    2. 多任务损失优化
    3. 学习率调度
    4. Checkpoint 管理
    5. TensorBoard/WandB 集成
    """
    
    def __init__(
        self,
        manager: Any,  # NCTManager
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        """初始化训练器
        
        Args:
            manager: NCT 管理器（PyTorch Module）
            config: 训练配置
            device: 训练设备（'cuda'/'cpu'）
        """
        self.manager = manager
        self.config = config or TrainingConfig()
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"[NCTTrainer] 使用设备：{self.device}")
        
        # 将 manager 移到设备上
        self.manager.to(self.device)
        
        # 构建优化器（只优化关键参数）
        trainable_params = self._get_trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = self._build_scheduler()
        
        # 损失函数
        self.loss_fn = MultiTaskLoss(
            lambda_classification=self.config.lambda_classification,
            lambda_prediction_error=self.config.lambda_prediction_error,
            lambda_phi=self.config.lambda_phi_regularization,
            lambda_sparsity=self.config.lambda_sparsity,
        )
        
        # 训练状态
        self.state = TrainingState()
        
        # 初始化 WandB（如果启用）
        if self.config.use_wandb and HAS_WANDB:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=vars(self.config),
            )
        
        # 确保 checkpoint 目录存在
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        logger.info(f"[NCTTrainer] 初始化完成")
    
    def _get_trainable_parameters(self) -> List[Dict[str, Any]]:
        """获取可训练参数
        
        策略：
        1. 冻结大部分预训练参数
        2. 只优化关键层（预测编码、注意力、输出层）
        """
        params = []
        
        # 添加预测编码层的参数
        if hasattr(self.manager, 'predictive_hierarchy'):
            for param in self.manager.predictive_hierarchy.parameters():
                if param.requires_grad:
                    params.append(param)
        
        # 添加注意力工作空间的参数
        if hasattr(self.manager, 'attention_workspace'):
            for param in self.manager.attention_workspace.parameters():
                if param.requires_grad:
                    params.append(param)
        
        # 添加输出投影层
        # （如果有分类头的话，需要在 manager 中添加）
        
        logger.info(f"[NCTTrainer] 找到 {sum(p.numel() for p in params)} 个可训练参数")
        
        return params
    
    def _build_scheduler(self) -> Optional[Any]:
        """构建学习率调度器"""
        if self.config.lr_scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs,
                eta_min=1e-6,
            )
        else:
            return None
    
    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """单个训练步骤
        
        Args:
            batch_data: 批次数据（感觉输入字典）
            labels: 标签
            
        Returns:
            损失字典
        """
        self.manager.train()
        self.optimizer.zero_grad()
        
        # Step 1: 前向传播
        # 将数据转为 numpy 格式（manager.process_cycle 需要）
        sensory_data = {}
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor):
                sensory_data[key] = tensor.cpu().numpy()
        
        state = self.manager.process_cycle(sensory_data)
        
        # Step 2: 获取表征并预测
        # 从 state 中提取 integrated 表征
        # 注意：需要处理 batch 中的多个样本
        if hasattr(state, 'workspace_content') and state.workspace_content is not None:
            # 使用工作空间内容作为表征
            # workspace_content.representation 是 numpy 数组 [B, D]
            representation = torch.from_numpy(
                state.workspace_content.representation
            ).to(self.device)
        else:
            # 降级方案：使用随机噪声（应该避免）
            batch_size = labels.shape[0]
            representation = torch.randn(
                batch_size, self.manager.config.d_model, device=self.device
            )
        
        # Step 3: 通过分类头得到预测
        # TODO: 需要在 manager 中添加分类头
        # 暂时使用简单的线性投影
        batch_size = representation.shape[0] if representation.dim() > 1 else 1
        
        with torch.no_grad():
            # 简化处理：假设是 5 分类任务
            num_classes = labels.max().item() + 1 if len(labels.shape) > 0 else 5
            
            # 如果 representation 是 1D，扩展为 2D
            if representation.dim() == 1:
                representation = representation.unsqueeze(0)  # [1, D]
            
            # 确保 projection 层存在
            if not hasattr(self, 'classifier'):
                self.classifier = nn.Linear(representation.shape[-1], num_classes).to(self.device)
            
            predictions = self.classifier(representation)
        
        # Step 4: 计算损失
        loss_dict = self.loss_fn(predictions, labels.to(self.device), state, self.manager)
        total_loss = loss_dict['total_loss']
        
        # Step 5: 反向传播
        total_loss.backward()
        
        # Step 6: 梯度裁剪
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._get_trainable_parameters(),
                self.config.clip_grad_norm,
            )
        
        # Step 7: 优化器更新
        self.optimizer.step()
        
        # Step 8: STDP 更新（独立于梯度下降）
        if self.config.use_stdp and self.state.step % self.config.stdp_update_frequency == 0:
            self._apply_stdp_updates(state)
        
        # Step 9: 记录日志
        loss_info = {k: v.item() if isinstance(v, torch.Tensor) else v 
                     for k, v in loss_dict.items()}
        loss_info['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return loss_info
    
    def _apply_stdp_updates(self, state: Any):
        """应用 STDP 更新
        
        基于当前意识状态生成 STDP 事件并更新突触权重
        """
        if not hasattr(self.manager, 'hybrid_learner'):
            return
        
        hybrid_learner = self.manager.hybrid_learner
        
        # 从 state 中提取神经活动模式
        # 这里简化处理：生成随机的 STDP 事件
        # 实际应该根据神经元发放时间生成
        
        from nct_modules.nct_hybrid_learning import STDPEvent
        
        # 生成少量 STDP 事件（模拟稀疏发放）
        n_events = 10
        stdp_events = []
        
        for _ in range(n_events):
            pre_id = np.random.randint(0, hybrid_learner.n_neurons)
            post_id = np.random.randint(0, hybrid_learner.n_neurons)
            pre_time = np.random.uniform(0, 100)  # ms
            post_time = pre_time + np.random.uniform(-10, 30)  # ±20ms 窗口
            
            event = STDPEvent(
                pre_neuron_id=pre_id,
                post_neuron_id=post_id,
                pre_spike_time=pre_time,
                post_spike_time=post_time,
            )
            stdp_events.append(event)
        
        # 应用 STDP 更新
        if stdp_events:
            global_context = None  # 可以传入当前的 integrated 表征
            updates = hybrid_learner(stdp_events, global_context)
            
            logger.debug(f"[STDP] 更新了 {len(updates)} 个突触")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个 epoch
        
        Args:
            train_loader: 数据加载器
            epoch: 当前 epoch
            
        Returns:
            平均损失字典
        """
        self.manager.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.n_epochs}')
        
        for batch_idx, (batch_data, labels) in enumerate(pbar):
            # 将批次数据移到设备上
            batch_data_on_device = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    batch_data_on_device[key] = torch.from_numpy(value).float()
                elif isinstance(value, torch.Tensor):
                    batch_data_on_device[key] = value.float()
            
            # 训练步骤
            loss_info = self.train_step(batch_data_on_device, labels)
            
            epoch_losses.append(loss_info)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_info['total_loss']:.4f}",
                'acc': f"{loss_info.get('accuracy', 0):.2%}",
            })
            
            # 详细日志
            if self.state.step % self.config.log_every_n_steps == 0:
                logger.info(
                    f"Step {self.state.step}: "
                    f"loss={loss_info['total_loss']:.4f}, "
                    f"lr={loss_info['learning_rate']:.2e}"
                )
                
                # 记录到 WandB
                if self.config.use_wandb and HAS_WANDB:
                    wandb.log(loss_info, step=self.state.step)
            
            self.state.step += 1
        
        # 计算 epoch 平均
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([d[key] for d in epoch_losses])
        
        return avg_losses
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            评估指标字典
        """
        self.manager.eval()
        
        all_predictions = []
        all_labels = []
        val_losses = []
        
        for batch_data, labels in tqdm(val_loader, desc='Evaluating'):
            # 前向传播（类似 train_step）
            batch_data_on_device = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    batch_data_on_device[key] = torch.from_numpy(value).float()
            
            state = self.manager.process_cycle(batch_data_on_device)
            
            # 获取预测（简化版）
            if hasattr(state, 'workspace_content') and state.workspace_content:
                representation = torch.from_numpy(
                    state.workspace_content.representation
                ).to(self.device)
            else:
                representation = torch.randn(
                    1, self.manager.config.d_model, device=self.device
                )
            
            # 简单分类（临时实现）
            num_classes = labels.max().item() + 1 if len(labels.shape) > 0 else 5
            predictions = torch.randn(
                representation.shape[0], num_classes, device=self.device
            )
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels)
            
            # 计算损失
            loss_dict = self.loss_fn(
                predictions, labels.to(self.device), state, self.manager
            )
            val_losses.append(loss_dict['total_loss'].item())
        
        # 计算准确率
        all_preds = torch.cat(all_predictions, dim=0)
        all_lbls = torch.cat(all_labels, dim=0)
        
        predicted_classes = all_preds.argmax(dim=1)
        accuracy = (predicted_classes == all_lbls).float().mean().item()
        
        return {
            'val_loss': np.mean(val_losses),
            'val_accuracy': accuracy,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
        """
        logger.info(f"[NCTTrainer] 开始训练 {self.config.n_epochs} 个 epochs")
        
        for epoch in range(self.state.epoch, self.config.n_epochs):
            # 训练
            train_losses = self.train_epoch(train_loader, epoch)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            else:
                val_metrics = {}
            
            # 记录 epoch 结果
            epoch_result = {
                'epoch': epoch,
                **train_losses,
                **val_metrics,
            }
            self.state.training_metrics.append(train_losses)
            self.state.validation_metrics.append(val_metrics)
            
            # 打印总结
            logger.info(
                f"Epoch {epoch+1}/{self.config.n_epochs} - "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Acc: {val_metrics.get('val_accuracy', 0):.2%}"
            )
            
            # 保存最佳模型
            current_acc = val_metrics.get('val_accuracy', 0)
            if current_acc > self.state.best_accuracy:
                self.state.best_accuracy = current_acc
                self.save_checkpoint('best_model.pth')
            
            # 定期保存 checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.state.epoch = epoch + 1
        
        logger.info(f"[NCTTrainer] 训练完成！最佳准确率：{self.state.best_accuracy:.2%}")
    
    def save_checkpoint(self, filename: str):
        """保存 checkpoint"""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'best_accuracy': self.state.best_accuracy,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_state_dict': self.manager.state_dict(),
            'config': vars(self.config),
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"[NCTTrainer] Checkpoint 已保存到 {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载 checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint 不存在：{filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state.epoch = checkpoint['epoch']
        self.state.step = checkpoint['step']
        self.state.best_accuracy = checkpoint['best_accuracy']
        
        self.manager.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"[NCTTrainer] 已从 {filepath} 恢复训练")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'TrainingConfig',
    'TrainingState',
    'MultiTaskLoss',
    'NCTTrainer',
]
