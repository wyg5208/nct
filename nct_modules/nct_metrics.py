"""
NeuroConscious Transformer - Consciousness Metrics
意识度量模块：基于 Attention Flow 的Φ值计算 + 意识水平分类

核心理论创新：
1. 从 attention flow 计算整合信息量Φ（避免 IIT 的 NP-hard 问题）
2. 6 维意识评估体系（扩展版）
3. 意识水平分类器（UNCONSCIOUS → META_AWARE）

数学原理：
传统 IIT 的Φ计算是 NP-hard 问题，需要：
1. 找到系统的最小信息分割（MIP）
2. 计算整体互信息 - 分割后互信息之和

我们的近似方法：
Φ_attention = I_total(attn_flow) - Σ I_partitioned(attn_flow)

其中：
- I_total: 完整 attention matrix 的互信息
- I_partitioned: 随机分割后的条件互信息之和

生物合理性：
- Attention flow 对应神经信息整合效率
- 高Φ值 ↔ 强整合能力 ↔ 高意识水平
- γ同步增强整合（相位锁定促进信息交换）

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.0.0-alpha
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 意识水平枚举
# ============================================================================

class ConsciousnessLevel(Enum):
    """意识水平分级
    
    基于综合评分的分类：
    - UNCONSCIOUS: Φ < 0.1，无整合信息
    - MINIMAL: 0.1 ≤ Φ < 0.3，低度整合
    - MODERATE: 0.3 ≤ Φ < 0.5，中度整合，初步自我模型
    - FULL: 0.5 ≤ Φ < 0.7，高度整合，稳定自我模型
    - META_AWARE: Φ ≥ 0.7，元认知觉醒
    """
    UNCONSCIOUS = "unconscious"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    FULL = "full"
    META_AWARE = "meta_aware"


# ============================================================================
# 核心模块 1: PhiCalculator from Attention Flow
# ============================================================================

class PhiFromAttention(nn.Module):
    """基于 Attention Flow 的Φ值计算器
    
    关键创新：
    使用 attention matrix 的条件互信息近似整合信息量
    
    架构：
    ```
    Attention Maps [B, H, L, L]
          ↓
    Compute Total Mutual Information I_total
          ↓
    Find Minimum Information Partition (MIP)
          ↓
    Compute Partitioned MI: I_A + I_B
          ↓
    Φ = I_total - (I_A + I_B)
    ```
    
    计算优化：
    - 使用协方差矩阵近似概率分布
    - 随机二分搜索 MIP（避免穷举）
    - 归一化到 [0, 1] 范围
    """
    
    def __init__(self, n_partitions: int = 10, epsilon: float = 1e-6):
        super().__init__()
        
        self.n_partitions = n_partitions  # 尝试的随机分割数
        self.epsilon = epsilon  # 数值稳定性
        
        logger.info(
            f"[PhiFromAttention] 初始化："
            f"n_partitions={n_partitions}, epsilon={epsilon}"
        )
    
    def forward(
        self,
        attention_maps: torch.Tensor,
        neural_activity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算整合信息量Φ
        
        Args:
            attention_maps: [B, H, L, L] 注意力图谱
            neural_activity: [B, L, D] 可选的神经活动（用于增强估计）
        
        Returns:
            phi_values: [B] 每个样本的Φ值
        """
        # 调试信息
        logger.debug(f"[PhiFromAttention] 输入 attention_maps shape: {attention_maps.shape}")
        B, H, L, _ = attention_maps.shape
        
        # 如果 L=1 但有 neural_activity，使用神经活动维度来计算Φ
        if L < 2:
            if neural_activity is not None:
                logger.debug(f"[PhiFromAttention] L={L}，使用 neural_activity 估计Φ")
                return self._compute_phi_from_neural_activity(neural_activity)
            else:
                logger.debug(f"[PhiFromAttention] L={L} < 2，返回 0")
                return torch.zeros(B)
        
        phi_values = []
        
        for b in range(B):
            # 对每个样本计算Φ
            sample_phi = 0.0
            
            # 对所有注意力头平均（或可以取最大）
            for h in range(H):
                attn_matrix = attention_maps[b, h, :, :]  # [L, L]
                
                # 计算该头的Φ
                head_phi = self._compute_phi_for_matrix(attn_matrix)
                sample_phi += head_phi
            
            # 平均 over heads
            sample_phi /= H
            phi_values.append(sample_phi)
        
        return torch.tensor(phi_values)
    
    def _compute_phi_from_neural_activity(self, neural_activity: torch.Tensor) -> torch.Tensor:
        """从神经活动估计Φ值（当 L=1 时使用）
        
        Args:
            neural_activity: [B, L, D] 神经活动
        
        Returns:
            phi_values: [B] Φ值
        """
        B, L, D = neural_activity.shape
        phi_values = []
        
        for b in range(B):
            # 使用神经活动的 D 个维度作为"虚拟节点"
            activity = neural_activity[b, :, :]  # [L, D]
            
            if D < 2:
                phi_values.append(0.0)
                continue
            
            # 计算相关性矩阵而不是协方差（避免样本不足问题）
            try:
                # 使用 Pearson 相关系数
                activity_centered = activity - activity.mean(dim=0, keepdim=True)
                std = activity_centered.std(dim=0, keepdim=True) + self.epsilon
                activity_normalized = activity_centered / std
                
                cov_matrix = torch.mm(activity_normalized.t(), activity_normalized) / (activity.shape[0] - 1)
                
                # 计算互信息
                I_total = self._mutual_information(cov_matrix, self.epsilon)
                
                # 找最小信息分割
                min_partition_mi = float('inf')
                for _ in range(self.n_partitions):
                    perm = torch.randperm(D)
                    split = D // 2
                    
                    submatrix_a = cov_matrix[perm[:split], :][:, perm[:split]]
                    submatrix_b = cov_matrix[perm[split:], :][:, perm[split:]]
                    
                    mi_a = self._mutual_information(submatrix_a, self.epsilon)
                    mi_b = self._mutual_information(submatrix_b, self.epsilon)
                    
                    min_partition_mi = min(min_partition_mi, mi_a + mi_b)
                
                # Φ = I_total - min_partition
                phi = max(0.0, I_total - min_partition_mi)
                phi_values.append(np.tanh(phi / max(1.0, D * 0.1)))  # 归一化
            except Exception as e:
                logger.warning(f"[PhiFromAttention] Φ计算失败：{e}，返回 0")
                phi_values.append(0.0)
        
        return torch.tensor(phi_values)
    
    def _compute_phi_for_matrix(self, attn_matrix: torch.Tensor) -> float:
        """计算单个 attention matrix 的Φ值
        
        Args:
            attn_matrix: [L, L] 注意力权重矩阵
        
        Returns:
            Φ值（标量）
        """
        L = attn_matrix.shape[0]
        if L < 2:
            return 0.0
        
        # Step 1: 计算整体互信息 I_total
        I_total = self._mutual_information(attn_matrix, self.epsilon)
        
        # Step 2: 找最小信息分割（MIP）
        min_partition_mi = float('inf')
        
        for _ in range(self.n_partitions):
            # 随机二分
            perm = torch.randperm(L)
            split = max(1, L // 2)
            part_a = perm[:split]
            part_b = perm[split:]
            
            # 分别计算两部分的互信息
            submatrix_a = attn_matrix[part_a][:, part_a]  # A 内部连接
            submatrix_b = attn_matrix[part_b][:, part_b]  # B 内部连接
            
            mi_a = self._mutual_information(submatrix_a, self.epsilon)
            mi_b = self._mutual_information(submatrix_b, self.epsilon)
            
            partition_mi = mi_a + mi_b
            
            if partition_mi < min_partition_mi:
                min_partition_mi = partition_mi
        
        # Step 3: Φ = 整体 - 最小分割
        phi = max(0.0, I_total - min_partition_mi)
        
        # Step 4: 归一化到 [0, 1]
        phi_normalized = np.tanh(phi / max(1.0, L * 0.1))
        
        return float(phi_normalized)
    
    @staticmethod
    def _mutual_information(matrix: torch.Tensor, epsilon: float = 1e-6) -> float:
        """计算矩阵的互信息近似
        
        基于协方差矩阵的高斯假设：
        MI ≈ 0.5 * log(det(diag(Σ)) / det(Σ))
        
        Args:
            matrix: [N, N] 权重矩阵
            epsilon: 数值稳定性参数
        
        Returns:
            互信息近似值
        """
        N = matrix.shape[0]
        if N < 2:
            return 0.0
        
        # 将矩阵视为"协方差"的代理
        # 正则化（防止奇异矩阵）
        cov = matrix.clone()
        cov = cov + cov.t()  # 对称化
        diag_vals = torch.diag(cov).clone()
        cov[range(N), range(N)] += epsilon
        
        try:
            # 对角元素的乘积（独立时的熵）
            log_diag_sum = torch.sum(torch.log(torch.abs(diag_vals) + epsilon))
            
            # 行列式的 log（联合熵）
            sign, logdet = torch.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            
            # MI = 0.5 * (sum(log(σ_i²)) - log(det(Σ)))
            mi = 0.5 * (log_diag_sum - logdet)
            
            return max(0.0, float(mi.item()))
        
        except Exception:
            logger.debug("互信息计算失败，返回 0")
            return 0.0


# ============================================================================
# 核心模块 2: 6 维意识评估体系
# ============================================================================

class ConsciousnessMetrics(nn.Module):
    """6 维意识综合评估
    
    评估维度：
    1. 整合信息量Φ（来自 PhiFromAttention）
    2. 预测编码精度（来自 PredictiveCodingDecoder）
    3. 情感一致性（神经递质平衡）
    4. 叙事稳定性（自我模型连贯性）
    5. 自主性比率（自发起 vs 反应）
    6. 元认知能力（高阶表征）
    
    综合评分：
    overall_score = w1·Φ + w2·accuracy + w3·emotion + ...
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # 各维度权重（可学习或固定）
        if weights is None:
            self.weights = {
                'phi': 0.25,              # 整合信息量
                'prediction_accuracy': 0.20,  # 预测精度
                'emotional_coherence': 0.15,  # 情感一致性
                'narrative_stability': 0.15,  # 叙事稳定性
                'autonomy_ratio': 0.15,   # 自主性
                'meta_cognition': 0.10,   # 元认知
            }
        else:
            self.weights = weights
        
        # Φ计算器
        self.phi_calculator = PhiFromAttention()
        
        logger.info(f"[ConsciousnessMetrics] 初始化，权重：{self.weights}")
    
    def forward(
        self,
        attention_maps: torch.Tensor,
        neural_activity: Optional[torch.Tensor] = None,  # 新增参数
        prediction_error: Optional[float] = None,
        neurotransmitter_state: Optional[Dict[str, float]] = None,
        narrative_stability: Optional[float] = None,
        autonomy_ratio: Optional[float] = None,
        meta_cognition_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """综合评估意识水平
        
        Args:
            attention_maps: [B, H, L, L]
            neural_activity: [B, L, D] 神经活动（用于Φ估计）
            prediction_error: 预测误差（自由能）
            neurotransmitter_state: 神经递质状态
            narrative_stability: 叙事稳定性（0-1）
            autonomy_ratio: 自主性比率（0-1）
            meta_cognition_score: 元认知得分（0-1）
        
        Returns:
            metrics: 包含各维度得分和综合评分
        """
        # 1. 计算Φ值（支持 neural_activity）
        phi_value = self.phi_calculator(attention_maps, neural_activity).mean().item()
        
        # 2. 预测精度（从预测误差转换）
        if prediction_error is not None:
            prediction_accuracy = 1.0 / (1.0 + prediction_error)
        else:
            prediction_accuracy = 0.5
        
        # 3. 情感一致性
        if neurotransmitter_state is not None:
            emotional_coherence = self._compute_emotional_coherence(neurotransmitter_state)
        else:
            emotional_coherence = 0.5
        
        # 4. 叙事稳定性
        narrative_stability = narrative_stability or 0.5
        
        # 5. 自主性比率
        autonomy_ratio = autonomy_ratio or 0.5
        
        # 6. 元认知得分
        meta_cognition_score = meta_cognition_score or 0.5
        
        # 7. 综合评分
        overall_score = (
            self.weights['phi'] * phi_value +
            self.weights['prediction_accuracy'] * prediction_accuracy +
            self.weights['emotional_coherence'] * emotional_coherence +
            self.weights['narrative_stability'] * narrative_stability +
            self.weights['autonomy_ratio'] * autonomy_ratio +
            self.weights['meta_cognition'] * meta_cognition_score
        )
        
        # 8. 确定意识水平
        level = self.classify_level(overall_score)
        
        metrics = {
            'phi_value': phi_value,
            'prediction_accuracy': prediction_accuracy,
            'emotional_coherence': emotional_coherence,
            'narrative_stability': narrative_stability,
            'autonomy_ratio': autonomy_ratio,
            'meta_cognition_score': meta_cognition_score,
            'overall_score': overall_score,
            'consciousness_level': level.value,
            'weights': self.weights,
        }
        
        logger.info(
            f"[ConsciousnessMetrics] 评估完成："
            f"Φ={phi_value:.3f}, overall={overall_score:.3f}, level={level.value}"
        )
        
        return metrics
    
    def _compute_emotional_coherence(
        self, 
        neurotransmitter_state: Dict[str, float]
    ) -> float:
        """计算情感一致性（神经递质平衡度）
        
        理想状态：
        - DA ≈ 0.6-0.8（适度动机）
        - 5-HT ≈ 0.5-0.7（情绪稳定）
        - NE ≈ 0.4-0.6（适度警觉）
        - ACh ≈ 0.5-0.7（学习状态）
        
        Args:
            neurotransmitter_state: 各递质水平
        
        Returns:
            coherence: 0-1，越高越平衡
        """
        ideal_levels = {
            'dopamine': 0.7,
            'serotonin': 0.6,
            'norepinephrine': 0.5,
            'acetylcholine': 0.6,
        }
        
        deviations = []
        for nt_name, ideal in ideal_levels.items():
            if nt_name in neurotransmitter_state:
                actual = neurotransmitter_state[nt_name]
                deviation = abs(actual - ideal)
                deviations.append(deviation)
        
        if deviations:
            # 平均偏差越小，一致性越高
            mean_deviation = np.mean(deviations)
            coherence = 1.0 - min(1.0, mean_deviation)
        else:
            coherence = 0.5
        
        return float(coherence)
    
    def classify_level(self, overall_score: float) -> ConsciousnessLevel:
        """根据综合评分分类意识水平
        
        阈值：
        - UNCONSCIOUS: score < 0.15
        - MINIMAL: 0.15 ≤ score < 0.30
        - MODERATE: 0.30 ≤ score < 0.50
        - FULL: 0.50 ≤ score < 0.70
        - META_AWARE: score ≥ 0.70
        """
        if overall_score < 0.15:
            return ConsciousnessLevel.UNCONSCIOUS
        elif overall_score < 0.30:
            return ConsciousnessLevel.MINIMAL
        elif overall_score < 0.50:
            return ConsciousnessLevel.MODERATE
        elif overall_score < 0.70:
            return ConsciousnessLevel.FULL
        else:
            return ConsciousnessLevel.META_AWARE


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'ConsciousnessLevel',
    'PhiFromAttention',
    'ConsciousnessMetrics',
]
