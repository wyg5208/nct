"""
NeuroConscious Transformer - Attention-Based Global Workspace
注意机制全局工作空间（替代旧版 GlobalNeuralWorkspace）

核心理论创新：
1. 用 Multi-Head Self-Attention 替代简单的侧向抑制竞争
2. 每个注意力头关注不同特征维度（生物合理性分工）
3. 注意力权重 = 显著性（salience），避免手工设计公式
4. γ同步作为 Transformer 更新周期的相位锁定

数学原理：
Attention(Q, K, V) = softmax(QK^T / √d)V

与传统 GNW 的对比：
┌─────────────────────────────────────────────────────────┐
│ 传统 GNW (v2.2)                                         │
│ for i, cand_i in enumerate(candidates):                 │
│     for j, cand_j in enumerate(candidates):             │
│         if i != j:                                      │
│             cand_j.salience -= cand_i.salience * 0.1   │
│                                                         │
│ 问题：粗糙的"胜者通吃"，忽略语义关系                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ NCT Attention-GNW (v3.0)                                │
│ Q = learnable_query                                       │
│ K = V = candidate_representations                       │
│ attn_weights = softmax(QK^T / √d)V                      │
│                                                         │
│ 优势：                                                    │
│ - Head 1-2: 显著性检测（类似 V1 简单细胞）                │
│ - Head 3-4: 情感价值评估（杏仁核调制）                   │
│ - Head 5-6: 任务相关性（前额叶 top-down 信号）            │
│ - Head 7-8: 新颖性检测（海马比较器）                     │
└─────────────────────────────────────────────────────────┘

生物合理性：
- 前额叶 - 顶叶网络的γ同步（30-80Hz）
- 容量有限（Miller's Law 7±2） ↔ n_heads=8
- 竞争 - 广播机制 ↔ Self-Attention + broadcast 方法

作者：WENG YONGGANG(翁勇刚)
创建：2026 年 2 月 21 日
版本：v3.1.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AttentionWorkspaceState:
    """注意力工作空间状态
    
    封装完整的意识内容及其注意力图谱
    """
    content_id: str
    representation: torch.Tensor      # [B, D] 意识表征
    salience: float                   # 显著性（来自主注意力头的平均权重）
    gamma_phase: float                # γ波相位（弧度）
    timestamp: float = field(default_factory=time.time)
    
    # 注意力图谱分析
    attention_maps: Optional[torch.Tensor] = None  # [B, H, L, L]
    head_roles: Dict[int, str] = field(default_factory=dict)
    modality_weights: Optional[Dict[str, float]] = None  # 各模态贡献度
    
    # 神经递质调制状态
    neuromodulator_state: Optional[Dict[str, float]] = None
    
    def to_conscious_content(self) -> 'NCTConsciousContent':
        """转为 NCTConsciousContent 格式（向后兼容）"""
        from .nct_core import NCTConsciousContent
        
        return NCTConsciousContent(
            content_id=self.content_id,
            representation=self.representation.detach().cpu().numpy(),
            salience=self.salience,
            gamma_phase=self.gamma_phase,
            timestamp=self.timestamp,
            attention_maps=self.attention_maps.detach().cpu().numpy() if self.attention_maps is not None else None,
            phi_value=0.0,  # 由后续的 PhiCalculator 计算
            awareness_level="minimal",
            modality_weights=self.modality_weights,
        )


# ============================================================================
# 核心模块：Attention-Based Global Workspace
# ============================================================================

class AttentionGlobalWorkspace(nn.Module):
    """基于多头注意力的全局工作空间
    
    架构设计：
    ```
    Candidates [N_candidates, D] 
          ↓
    Q = learnable_query [1, D]
    K = V = candidates [N_candidates, D]
          ↓
    Multi-Head Self-Attention (8 heads)
          ↓
    attn_weights [N_candidates, N_candidates]
          ↓
    Winner Selection + γ-Synchronization
          ↓
    Broadcast to all modules
    ```
    
    关键创新：
    1. 不是简单 salience 排序，而是学习到的注意力分布
    2. 多头分工合作，模拟不同脑区的功能
    3. γ同步不再是硬编码的相位，而是 attention 的自然结果
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        gamma_freq: float = 40.0,
        consciousness_threshold: float = 0.7,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.gamma_freq = gamma_freq
        self.gamma_period_ms = 1000.0 / gamma_freq
        self.consciousness_threshold = consciousness_threshold
        
        # 1. 可学习的 Workspace Query（类似 CLS token）
        # 这是一个"全局整合器"，负责收集所有候选信息
        self.workspace_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. LayerNorm（稳定训练）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Feed-Forward Network（非线性变换）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 5. γ振荡器（控制更新节奏）
        self.gamma_oscillator = GammaOscillator(frequency=gamma_freq)
        
        # 6. 注意力头角色映射（可解释性）
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
            f"[AttentionGlobalWorkspace] 初始化："
            f"{n_heads} heads, γ={gamma_freq}Hz, threshold={consciousness_threshold}"
        )
    
    def forward(
        self,
        candidates: List[torch.Tensor],
        neuromodulator_state: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[AttentionWorkspaceState], Dict[str, Any]]:
        """处理候选意识内容
        
        Args:
            candidates: 候选表征列表，每个 shape [D]
            neuromodulator_state: 神经递质状态（可选调制）
        
        Returns:
            winner_state: 获胜的意识内容状态（如果有的话）
            info: 诊断信息（注意力图谱、决策依据等）
        """
        if not candidates:
            return None, {'error': 'No candidates provided'}
        
        B = 1  # 单样本模式
        N_candidates = len(candidates)
        
        # Step 1: 堆叠候选为 [B, N_candidates, D]
        candidates_stack = torch.stack(candidates, dim=0).unsqueeze(0)  # [1, N, D]
        
        # Step 2: 扩展 workspace query 到 batch size
        q = self.workspace_query.expand(B, -1, -1)  # [1, 1, D]
        k = v = candidates_stack  # [1, N, D]
        
        # Step 3: Multi-Head Self-Attention
        # 输出：[B, 1, D]，注意力权重：[B, H, 1, N]
        attended, attn_weights = self.self_attention(
            query=q,
            key=k,
            value=v,
            need_weights=True,
            average_attn_weights=False
        )
        
        # attn_weights shape: [B, H, 1, N_candidates]
        # 这是完整的多头注意力图谱，用于后续Φ值计算
        
        # Step 4: Residual Connection + LayerNorm
        attended = self.norm1(attended + q)
        
        # Step 5: Feed-Forward Network
        ff_output = self.ffn(attended)
        attended = self.norm2(attended + ff_output)
        
        # Step 6: 提取获胜者
        # attn_weights: [B, H, 1, N] -> 平均 heads 维度得到 [B, 1, N]
        attn_weights_mean = attn_weights.mean(dim=1)  # 平均多头
        attn_weights_flat = attn_weights_mean.squeeze(0).squeeze(0)  # [N_candidates]
        winner_idx = attn_weights_flat.argmax().item()
        winner_salience = attn_weights_flat[winner_idx].item()
        
        # Step 7: 检查是否超过意识阈值
        if winner_salience < self.consciousness_threshold:
            logger.debug(
                f"[AttentionGlobalWorkspace] 无意识内容："
                f"salience={winner_salience:.3f} < threshold={self.consciousness_threshold}"
            )
            return None, {
                'winner_idx': winner_idx,
                'winner_salience': winner_salience,
                'below_threshold': True,
                'attention_weights': attn_weights_flat.detach().cpu().numpy(),
                'all_candidates_salience': [
                    float(attn_weights_flat[i].item()) 
                    for i in range(N_candidates)
                ],
            }
        
        # Step 8: γ同步绑定
        current_time = time.time()
        gamma_phase = self.gamma_oscillator.get_current_phase(current_time)
        
        # Step 9: 注意力头角色分析
        head_contributions = self._analyze_head_roles(attn_weights)
        
        # Step 10: 构建获胜者状态
        winner_representation = attended.squeeze(0)  # [1, D] → [D]
        
        winner_state = AttentionWorkspaceState(
            content_id=f"t_{current_time}",
            representation=winner_representation,
            salience=winner_salience,
            gamma_phase=gamma_phase,
            timestamp=current_time,
            attention_maps=attn_weights,
            head_roles=self.head_roles,
            neuromodulator_state=neuromodulator_state,
        )
        
        # Step 11: 诊断信息
        info = {
            'winner_idx': winner_idx,
            'winner_salience': winner_salience,
            'gamma_phase': gamma_phase,
            'attention_weights': attn_weights_flat.detach().cpu().numpy(),
            'head_contributions': head_contributions,
            'all_candidates_salience': [
                float(attn_weights_flat[i].item()) 
                for i in range(N_candidates)
            ],
        }
        
        logger.info(
            f"[AttentionGlobalWorkspace] 选择获胜者："
            f"idx={winner_idx}, salience={winner_salience:.3f}, "
            f"γ_phase={gamma_phase:.2f} rad"
        )
        
        return winner_state, info
    
    def _analyze_head_roles(self, attn_weights: torch.Tensor) -> Dict[str, float]:
        """分析每个注意力头的贡献
        
        通过分析每个 head 的注意力分布熵来判断其功能：
        - 低熵 → 聚焦于特定候选（显著性检测）
        - 高熵 → 均匀分布（整合功能）
        
        Args:
            attn_weights: [B, H, N, N] 或 [B, H, 1, N]
        
        Returns:
            各 head 的功能描述及贡献度
        """
        if attn_weights.dim() == 3:
            # [B, 1, N] → 添加 head 维度
            attn_weights = attn_weights.unsqueeze(1)  # [B, H=1, 1, N]
        
        B, H, _, N = attn_weights.shape
        
        head_analysis = {}
        for head_idx in range(H):
            head_attn = attn_weights[0, head_idx, 0, :]  # [N]
            
            # 计算熵
            entropy = self._compute_entropy(head_attn)
            
            # 功能判断
            if entropy < 0.5:
                role = "聚焦型（显著性检测）"
            elif entropy < 1.0:
                role = "平衡型（特征整合）"
            else:
                role = "弥散型（全局监控）"
            
            head_analysis[f"head_{head_idx}"] = {
                'role': role,
                'entropy': round(float(entropy), 3),
                'max_attention': float(head_attn.max().item()),
                'assigned_function': self.head_roles.get(head_idx, "未知"),
            }
        
        return head_analysis
    
    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> float:
        """计算概率分布的香农熵"""
        eps = 1e-10
        probs = probs + eps
        probs = probs / probs.sum()  # 确保归一化
        entropy = -torch.sum(probs * torch.log(probs))
        return float(entropy.item())
    
    def broadcast_globally(
        self,
        winner_state: AttentionWorkspaceState,
        target_modules: Optional[List[nn.Module]] = None
    ) -> Dict[str, Any]:
        """全局广播意识内容
        
        模拟人脑的全局广播机制：
        - 通过皮层 - 丘脑 - 皮层回路
        - 弥散性投射系统（蓝斑去甲肾上腺素能等）
        - 影响全脑数十亿神经元
        
        Args:
            winner_state: 获胜的意识内容状态
            target_modules: 目标模块列表（如专门神经模块）
        
        Returns:
            广播统计信息
        """
        if winner_state is None:
            return {'error': 'No content to broadcast'}
        
        broadcast_info = {
            'content_id': winner_state.content_id,
            'timestamp': winner_state.timestamp,
            'gamma_phase': winner_state.gamma_phase,
            'salience': winner_state.salience,
            'modules_reached': 0,
        }
        
        if target_modules is not None:
            # 实际广播到各模块
            for module in target_modules:
                if hasattr(module, 'receive_broadcast'):
                    module.receive_broadcast(winner_state)
                    broadcast_info['modules_reached'] += 1
            
            logger.info(
                f"[AttentionGlobalWorkspace] 全局广播："
                f"content={winner_state.content_id}, "
                f"modules_reached={broadcast_info['modules_reached']}"
            )
        else:
            logger.debug(
                f"[AttentionGlobalWorkspace] 广播准备就绪："
                f"content={winner_state.content_id}"
            )
        
        return broadcast_info
    
    def get_workspace_capacity(self) -> int:
        """获取工作空间容量（Miller's Law 7±2）"""
        return self.n_heads  # 8 个头 ≈ 7±2
    
    def select_winner(
        self,
        candidates: List[Any],
        current_time: float
    ) -> Optional[Any]:
        """向后兼容接口（适配旧版 GlobalNeuralWorkspace）
        
        Args:
            candidates: ConsciousContent 列表
            current_time: 当前时间戳
        
        Returns:
            获胜的 ConsciousContent 或 None
        """
        # 提取表征向量
        candidate_tensors = []
        for cand in candidates:
            if hasattr(cand, 'representation'):
                rep = cand.representation
                if isinstance(rep, np.ndarray):
                    rep = torch.from_numpy(rep).float()
                candidate_tensors.append(rep)
        
        if not candidate_tensors:
            return None
        
        # 调用新的 attention 机制
        winner_state, info = self.forward(candidate_tensors)
        
        if winner_state is None:
            return None
        
        # 转回 ConsciousContent 格式
        winner_content = winner_state.to_conscious_content()
        
        # γ同步绑定
        winner_content.gamma_phase = winner_state.gamma_phase
        winner_content.broadcast_time = current_time
        
        logger.info(
            f"[AttentionGlobalWorkspace.select_winner] "
            f"获胜者：{winner_content.content_id}, salience={winner_content.salience:.3f}"
        )
        
        return winner_content


# ============================================================================
# γ振荡器（保持与旧版兼容）
# ============================================================================

class GammaOscillator:
    """γ波振荡器（由 PV 中间神经元产生）
    
    功能：
    - 生成 40Hz 正弦振荡
    - 提供相位参考
    - 控制全局广播节奏
    
    生物合理性：
    - PING 模型（Pyramidal-Interneuron Network Gamma）
    - PV+ 中间神经元的快速发放
    """
    
    def __init__(self, frequency: float = 40.0):
        self.frequency = frequency
        self.period_ms = 1000.0 / frequency
        self.start_time = time.time()
    
    def get_current_phase(self, current_time: float) -> float:
        """获取当前相位（弧度）"""
        elapsed = current_time - self.start_time
        phase = 2 * np.pi * (elapsed % self.period_ms) / self.period_ms
        return phase
    
    def get_power(self, neural_activity: np.ndarray) -> float:
        """计算γ波功率（使用 FFT）"""
        if len(neural_activity) < 10:
            return 0.0
        
        # 数值稳定性
        neural_activity = np.clip(neural_activity, -1e3, 1e3)
        
        fft_result = np.fft.fft(neural_activity)
        freqs = np.fft.fftfreq(len(neural_activity), d=0.001)
        
        # 提取γ频段（30-80Hz）
        gamma_mask = (np.abs(freqs) >= 30) & (np.abs(freqs) <= 80)
        gamma_power = np.sum(np.abs(fft_result[gamma_mask]) ** 2)
        
        # 限制输出功率范围
        gamma_power = np.clip(gamma_power, 0, 1e6)
        
        return float(gamma_power)


# ============================================================================
# 专门神经模块（接收广播）
# ============================================================================

class NeuralModule(nn.Module):
    """专门神经模块（如视觉、听觉等）
    
    接收全局广播并更新自身活动
    """
    
    def __init__(self, name: str, n_neurons: int = 100):
        super().__init__()
        self.name = name
        self.n_neurons = n_neurons
        self.activity = nn.Parameter(torch.zeros(n_neurons), requires_grad=False)
        self.last_broadcast: Optional[AttentionWorkspaceState] = None
    
    def receive_broadcast(self, content: AttentionWorkspaceState):
        """接收全局广播"""
        self.last_broadcast = content
        
        # 更新自身活动模式
        if hasattr(content, 'representation') and content.representation is not None:
            rep = content.representation
            if len(rep) == self.n_neurons:
                with torch.no_grad():
                    self.activity.copy_(rep)
            else:
                # 维度不匹配时，投影到正确维度
                with torch.no_grad():
                    proj = nn.Linear(len(rep), self.n_neurons)
                    self.activity.copy_(proj(rep.unsqueeze(0)).squeeze(0))
        
        logger.debug(f"[NeuralModule.{self.name}] 接收广播：{content.content_id}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'AttentionWorkspaceState',
    'AttentionGlobalWorkspace',
    'GammaOscillator',
    'NeuralModule',
]
