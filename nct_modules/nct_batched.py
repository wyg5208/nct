"""
NeuroConscious Transformer - 批量处理架构升级
NCT Batched Processing Architecture Upgrade

核心改进：
1. 将单样本处理改为批量处理
2. 利用 PyTorch 的向量化操作提升效率
3. 保持意识的单一性（每个样本一个获胜者）

架构变化：
- process_cycle → process_batch
- 单样本候选竞争 → 批量候选竞争
- 标量指标 → 向量化指标

预期提升：
- 训练速度：10-100 倍（GPU 并行）
- 内存效率：更优（batch 处理）
- 可扩展性：支持更大 batch

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 批量化注意力工作空间
# ============================================================================

class BatchedAttentionWorkspace(nn.Module):
    """支持多样本并行处理的注意力工作空间
    
    核心创新：
    1. 每个 batch 样本有独立的工作空间通道
    2. 批量化的 Multi-Head Attention
    3. 向量化的获胜者选择
    
    理论基础：
    - GWT（全局工作空间理论）：意识内容单一
    - 应用到 batch：每个样本有一个独立的意识通道
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        gamma_freq: float = 40.0,
        consciousness_threshold: float = 0.7,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.gamma_freq = gamma_freq
        self.consciousness_threshold = consciousness_threshold
        
        # Multi-Head Attention（支持 batch）
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # 可学习的 Workspace Query（每个样本一个）
        self.workspace_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Head 角色定义
        self.head_roles = {
            0: "视觉显著性检测",
            1: "听觉显著性检测",
            2: "情感价值评估",
            3: "动机调制",
            4: "任务相关性",
            5: "目标匹配度",
            6: "新颖性检测",
            7: "意外性评估",
        }
        
        logger.info(
            f"[BatchedAttentionWorkspace] 初始化："
            f"{n_heads} heads, γ={gamma_freq}Hz, threshold={consciousness_threshold}"
        )
    
    def forward(
        self,
        batch_candidates: List[torch.Tensor],
        neuromodulator_state: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """批量处理候选意识内容
        
        Args:
            batch_candidates: 候选表征列表，每个 shape [B, D]
                例如：[integrated_repr, visual_repr, auditory_repr]
                其中每个 repr 的 shape 是 [B, D]
            neuromodulator_state: 神经递质状态（可选调制）
        
        Returns:
            winners: [B, D] 每个 batch 样本的获胜表征
            info: 诊断信息字典
                - 'attention_weights': [B, n_heads, N_candidates]
                - 'winner_indices': [B]
                - 'salience': [B]
        """
        if not batch_candidates:
            raise ValueError("candidates 不能为空")
        
        # 获取 batch size
        B = batch_candidates[0].shape[0]
        N_candidates = len(batch_candidates)
        
        # Step 1: 堆叠候选为 [B, N_candidates, D]
        candidates_stack = torch.stack(batch_candidates, dim=1)  # [B, N, D]
        
        # Step 2: 扩展 query 到 batch
        q = self.workspace_query.expand(B, -1, -1)  # [B, 1, D]
        
        # Step 3: Multi-head attention（并行处理所有样本）
        # q: [B, 1, D], k=v: [B, N, D]
        attn_output, attn_weights = self.attention(q, candidates_stack, candidates_stack)
        # attn_output: [B, 1, D]
        # attn_weights: [B, n_heads, 1, N]
        
        # Step 4: 计算显著性（每个样本）
        # 平均所有 heads 的注意力权重
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)  # [B, N]
        
        # Step 5: 为每个样本选择获胜候选
        winner_indices = avg_attn_weights.argmax(dim=1)  # [B]
        
        # Step 6: 提取获胜表征
        # 使用 advanced indexing
        batch_indices = torch.arange(B, device=candidates_stack.device)
        winners = candidates_stack[batch_indices, winner_indices]  # [B, D]
        
        # Step 7: 计算显著性（获胜候选的注意力权重）
        salience = avg_attn_weights[batch_indices, winner_indices]  # [B]
        
        # Step 8: 输出投影
        winners = self.output_proj(winners)
        
        # 构建诊断信息
        info = {
            'attention_weights': attn_weights.squeeze(2),  # [B, n_heads, N]
            'avg_attention_weights': avg_attn_weights,  # [B, N]
            'winner_indices': winner_indices,  # [B]
            'salience': salience,  # [B]
            'consciousness_level': (salience > self.consciousness_threshold).float(),  # [B]
        }
        
        logger.debug(
            f"[BatchedAttentionWorkspace] 处理完成："
            f"batch_size={B}, N_candidates={N_candidates}, "
            f"avg_salience={salience.mean().item():.3f}"
        )
        
        return winners, info


# ============================================================================
# 批量化 NCT 管理器
# ============================================================================

class BatchedNCTManager(nn.Module):
    """支持批量处理的 NCT 管理器
    
    关键改进：
    1. process_batch 方法支持 [B, ...] 输入
    2. 所有模块都支持 batch 维度
    3. 返回 batch 级别的意识状态
    """
    
    def __init__(self, config: Any):
        super().__init__()
        
        from .nct_core import NCTConfig, MultiModalEncoder
        from .nct_cross_modal import CrossModalIntegration
        from .nct_predictive_coding import PredictiveHierarchy
        from .nct_metrics import ConsciousnessMetrics
        from .nct_gamma_sync import GammaSynchronizer
        from .nct_hybrid_learning import TransformerSTDP
        
        self.config = config if isinstance(config, NCTConfig) else NCTConfig()
        
        # 1. 多模态编码器（已支持 batch）
        self.multimodal_encoder = MultiModalEncoder(self.config)
        
        # 2. Cross-Modal 整合（需要修改为支持 batch）
        self.cross_modal_integration = CrossModalIntegration(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
        )
        
        # 3. 批量化工作空间（新！）
        self.attention_workspace = BatchedAttentionWorkspace(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            gamma_freq=self.config.gamma_freq,
        )
        
        # 4. 预测编码层次
        self.predictive_hierarchy = PredictiveHierarchy(
            config={
                'layer0_dim': self.config.d_model,
                'layer1_dim': self.config.d_model,
                'layer2_dim': self.config.d_model,
                'layer3_dim': self.config.d_model,
                'n_heads': self.config.n_heads,
            }
        )
        
        # 5. 意识度量
        self.consciousness_metrics = ConsciousnessMetrics()
        
        # 6. γ同步器
        self.gamma_synchronizer = GammaSynchronizer(
            frequency=self.config.gamma_freq
        )
        
        # 7. STDP 学习器
        self.hybrid_learner = TransformerSTDP(
            n_neurons=self.config.d_model,
            d_model=self.config.d_model,
        )
        
        # 运行状态
        self.is_running = False
        self.total_batches = 0
        
        logger.info(f"[BatchedNCTManager] 初始化完成")
    
    def start(self):
        """启动系统"""
        self.is_running = True
        self.total_batches = 0
        logger.info("[BatchedNCTManager] 系统启动")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        logger.info("[BatchedNCTManager] 系统停止")
    
    def process_batch(
        self,
        batch_sensory_data: Dict[str, torch.Tensor],
        neurotransmitter_state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """批量处理多个样本
        
        Args:
            batch_sensory_data: 感觉输入字典
                - 'visual': [B, H, W] 或 [B, C, H, W]
                - 'auditory': [B, T, F] (可选)
                - 'interoceptive': [B, 10] (可选)
            neurotransmitter_state: 神经递质状态
        
        Returns:
            batch_state: 批量意识状态
                - 'representations': [B, D]
                - 'consciousness_metrics': Dict
                - 'workspace_info': Dict
        """
        if not self.is_running:
            logger.warning("[BatchedNCTManager] 系统未运行，自动启动")
            self.start()
        
        self.total_batches += 1
        
        # 获取 batch size
        B = batch_sensory_data['visual'].shape[0]
        
        logger.debug(f"[BatchedNCTManager] 处理 batch: size={B}")
        
        # Step 1: 多模态编码（已支持 batch）
        embeddings = self.multimodal_encoder(batch_sensory_data)
        # embeddings['visual_emb']: [B, N_v, D]
        
        # Step 2: Cross-Modal 整合
        integrated, cross_modal_info = self.cross_modal_integration(embeddings)
        # integrated: [B, D]
        
        # Step 3: 构建批量候选
        batch_candidates = [integrated]  # [B, D]
        
        if 'visual_emb' in embeddings:
            visual_repr = embeddings['visual_emb'].mean(dim=1)  # [B, D]
            batch_candidates.append(visual_repr)
        
        if 'audio_emb' in embeddings:
            audio_repr = embeddings['audio_emb'].mean(dim=1)  # [B, D]
            batch_candidates.append(audio_repr)
        
        # Step 4: 工作空间竞争（批量）
        winners, workspace_info = self.attention_workspace(
            batch_candidates,
            neuromodulator_state=neurotransmitter_state,
        )
        # winners: [B, D]
        
        # Step 5: 预测编码
        # 需要序列输入 [B, T, D]
        if integrated.dim() == 2:
            integrated_seq = integrated.unsqueeze(1)  # [B, 1, D]
        else:
            integrated_seq = integrated
        
        prediction_results = self.predictive_hierarchy.forward_with_sequence(
            integrated_seq.expand(-1, 2, -1)  # [B, 2, D] 假设有 2 个时间步
        )
        
        # Step 6: 计算意识度量
        metrics = self.consciousness_metrics(
            attention_maps=workspace_info['attention_weights'].unsqueeze(2),  # [B, H, 1, N]
            neural_activity=integrated.unsqueeze(1),  # [B, 1, D]
            prediction_error=prediction_results.get('total_free_energy', 0.5),
        )
        
        # 构建批量状态
        batch_state = {
            'representations': winners,  # [B, D]
            'salience': workspace_info['salience'],  # [B]
            'consciousness_metrics': metrics,
            'workspace_info': workspace_info,
            'cross_modal_info': cross_modal_info,
            'prediction_results': prediction_results,
        }
        
        return batch_state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'total_batches': self.total_batches,
        }


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'BatchedAttentionWorkspace',
    'BatchedNCTManager',
]
