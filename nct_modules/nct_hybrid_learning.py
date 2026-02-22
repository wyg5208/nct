"""
NeuroConscious Transformer - Transformer-STDP Hybrid Learning
混合学习规则：局部 STDP + 全局 Attention + 神经递质调制

核心理论创新：
1. 经典 STDP（局部时间相关）提供基础可塑性
2. Attention 梯度（全局语义相关）提供上下文调制
3. 神经递质（DA/5-HT/NE/ACh）调节学习速率

数学公式：
Δw = (δ_STDP + λ·δ_attention) · η_neuromodulator

其中：
- δ_STDP = A₊·exp(-Δt/τ₊) if Δt > 0 (LTP)
         = -A₋·exp(Δt/τ₋) if Δt < 0 (LTD)
- δ_attention = ∂Loss/∂W (attention weights 的梯度)
- η_neuromodulator = f(DA, 5-HT, NE, ACh)

生物合理性：
- STDP: 突触前→后神经元的时间依赖可塑性
- Attention: 前额叶 top-down 信号的全球工作空间调制
- Neuromodulators: 蓝斑（NE）、VTA（DA）、中缝核（5-HT）、基底前脑（ACh）

预期效果：
- 比纯 STDP 快 3-5 倍收敛（利用全局信息）
- 比纯 Attention 更节能（稀疏事件驱动更新）
- 抗遗忘能力更强（双重编码：局部 + 全局）

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.0.0-alpha
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
class STDPEvent:
    """STDP 事件记录
    
    封装一对前后神经元的脉冲时间
    """
    pre_neuron_id: int
    post_neuron_id: int
    pre_spike_time: float       # 毫秒
    post_spike_time: float      # 毫秒
    synapse_key: Tuple[int, int] = field(init=False)
    
    def __post_init__(self):
        self.synapse_key = (self.pre_neuron_id, self.post_neuron_id)
    
    @property
    def delta_t(self) -> float:
        """计算时间差 Δt = t_post - t_pre"""
        return self.post_spike_time - self.pre_spike_time


@dataclass
class SynapticUpdate:
    """突触更新记录"""
    synapse_key: Tuple[int, int]
    old_weight: float
    new_weight: float
    delta_w_std: float          # STDP 贡献
    delta_w_attn: float         # Attention 贡献
    modulation_factor: float    # 神经递质调制
    timestamp: float = field(default_factory=time.time)
    
    @property
    def total_delta_w(self) -> float:
        """总更新量"""
        return self.new_weight - self.old_weight


# ============================================================================
# 核心模块：Transformer-STDP Hybrid Learning
# ============================================================================

class TransformerSTDP(nn.Module):
    """Transformer-STDP 混合学习规则
    
    架构设计：
    ```
    Pre-synaptic spike ─┬─→ Classic STDP ─→ δ_std
                        │
    Post-synaptic spike ─┘
                         ↓
    Global context ─→ Attention Gradient ─→ δ_attn
                         ↓
    Neuromodulators ─→ Gate ─→ η
                         ↓
    Δw = (δ_std + λ·δ_attn) · η
    ```
    
    关键组件：
    1. ClassicSTDP: 生物物理 STDP 规则
    2. AttentionLearner: 从 attention 梯度提取全局信息
    3. NeuromodulatorGate: 神经递质门控
    """
    
    def __init__(
        self,
        n_neurons: int = 768,  # 修复：默认与 d_model 一致
        d_model: int = 768,
        stdp_learning_rate: float = 0.01,
        attention_modulation_lambda: float = 0.1,
        sparsity: float = 0.01,
    ):
        super().__init__()
        
        # 确保 n_neurons 与 d_model 对齐
        if n_neurons != d_model:
            logger.warning(
                f"[TransformerSTDP] n_neurons({n_neurons}) != d_model({d_model}), "
                f"建议统一以避免维度不匹配"
            )
        
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.stdp_lr = stdp_learning_rate
        self.attention_lambda = attention_modulation_lambda
        self.sparsity = sparsity
        
        # 1. 突触权重矩阵（稀疏初始化）
        # 使用 d_model 维度以匹配 attention gradients
        self.synaptic_weights = nn.Parameter(
            self._initialize_sparse_weights(),
            requires_grad=True
        )
        
        # 1.1 维度对齐投影层（如果 n_neurons != d_model）
        if n_neurons != d_model:
            self.neuron_to_model = nn.Linear(n_neurons, d_model)
            self.model_to_neuron = nn.Linear(d_model, n_neurons)
        else:
            self.neuron_to_model = None
            self.model_to_neuron = None
        
        # 2. 经典 STDP 规则
        self.stdp_rule = ClassicSTDP(
            A_plus=0.01,
            A_minus=0.012,
            tau_plus=20.0,
            tau_minus=20.0,
        )
        
        # 3. Attention Learner（提取全局上下文）
        self.attention_learner = AttentionGradientLearner(
            d_model=d_model,
            n_heads=8
        )
        
        # 4. 神经递质门控
        self.neuromodulator_gate = NeuromodulatorGate()
        
        # 5. 突触更新历史（用于调试和分析）
        self.update_history: List[SynapticUpdate] = []
        
        logger.info(
            f"[TransformerSTDP] 初始化："
            f"n_neurons={n_neurons}, d_model={d_model}, "
            f"stdp_lr={self.stdp_lr}, attn_lambda={self.attention_lambda}"
        )
    
    def _initialize_sparse_weights(self) -> torch.Tensor:
        """初始化稀疏突触连接（类似生物脑 1% 连接率）"""
        weights = torch.zeros(self.n_neurons, self.n_neurons)
        
        # 随机连接（概率 = sparsity）
        mask = torch.rand(self.n_neurons, self.n_neurons) < self.sparsity
        weights[mask] = torch.rand(mask.sum().item()) * 0.5 + 0.1
        
        # 确保自连接为 0
        torch.diag(weights).zero_()
        
        logger.debug(f"[TransformerSTDP] 初始化稀疏权重，连接率={self.sparsity:.2%}")
        return weights
    
    def forward(
        self,
        stdp_events: List[STDPEvent],
        global_context: Optional[torch.Tensor] = None,
        neurotransmitter_state: Optional[Dict[str, float]] = None,
    ) -> List[SynapticUpdate]:
        """执行突触更新
        
        Args:
            stdp_events: STDP 事件列表
            global_context: 全局上下文（意识内容的 embedding）
            neurotransmitter_state: 神经递质状态
        
        Returns:
            突触更新记录列表
        """
        updates = []
        
        # Step 1: 计算 Attention 梯度（如果有全局上下文）
        if global_context is not None:
            attn_gradients = self.attention_learner.compute_gradient(global_context)
        else:
            attn_gradients = None
        
        # Step 2: 获取神经递质调制因子
        if neurotransmitter_state is not None:
            modulation = self.neuromodulator_gate.get_learning_rate(neurotransmitter_state)
        else:
            modulation = 1.0
        
        # Step 3: 处理每个 STDP 事件
        for event in stdp_events:
            update = self._update_synapse(
                event=event,
                attn_gradients=attn_gradients,
                modulation=modulation
            )
            updates.append(update)
        
        # Step 4: 记录更新历史
        self.update_history.extend(updates)
        
        # Step 5: 权重约束（保持在 [0, 1] 范围）
        with torch.no_grad():
            self.synaptic_weights.data.clamp_(0.0, 1.0)
        
        logger.debug(
            f"[TransformerSTDP] 更新了 {len(updates)} 个突触，"
            f"平均 Δw={np.mean([u.total_delta_w for u in updates]):.4f}"
        )
        
        return updates
    
    def _update_synapse(
        self,
        event: STDPEvent,
        attn_gradients: Optional[torch.Tensor],
        modulation: float
    ) -> SynapticUpdate:
        """更新单个突触
        
        公式：Δw = (δ_STDP + λ·δ_attention) · η
        """
        i, j = event.synapse_key
        old_weight = self.synaptic_weights[i, j].item()
        
        # 1. 经典 STDP 贡献
        delta_w_std = self.stdp_rule.compute(event.delta_t)
        
        # 2. Attention 梯度贡献（如果有）
        delta_w_attn = 0.0
        if attn_gradients is not None and i < attn_gradients.shape[0] and j < attn_gradients.shape[1]:
            delta_w_attn = attn_gradients[i, j].item() * self.attention_lambda
        
        # 3. 神经递质调制
        total_delta_w = (delta_w_std + delta_w_attn) * modulation
        
        # 4. 应用更新
        new_weight = old_weight + total_delta_w * self.stdp_lr
        
        # 5. 记录更新
        update = SynapticUpdate(
            synapse_key=event.synapse_key,
            old_weight=old_weight,
            new_weight=new_weight,
            delta_w_std=delta_w_std,
            delta_w_attn=delta_w_attn,
            modulation_factor=modulation,
        )
        
        # 6. 实际更新权重
        self.synaptic_weights.data[i, j] = new_weight
        
        return update
    
    def get_weight_matrix(self) -> np.ndarray:
        """获取突触权重矩阵（用于可视化或分析）"""
        return self.synaptic_weights.detach().cpu().numpy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取突触统计信息"""
        weights = self.synaptic_weights.detach().cpu().numpy()
        non_zero = weights[weights > 0]
        
        return {
            'total_neurons': self.n_neurons,
            'total_synapses': len(non_zero),
            'sparsity': len(non_zero) / (self.n_neurons ** 2),
            'mean_weight': float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0,
            'std_weight': float(np.std(non_zero)) if len(non_zero) > 0 else 0.0,
            'min_weight': float(np.min(non_zero)) if len(non_zero) > 0 else 0.0,
            'max_weight': float(np.max(non_zero)) if len(non_zero) > 0 else 0.0,
            'total_updates': len(self.update_history),
        }
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度分析"""
        if not self.update_history:
            return {
                'total_updates': 0,
                'avg_delta_w': 0.0,
                'ltp_count': 0,
                'ltd_count': 0,
                'attention_contribution': 0.0,
            }
        
        recent_updates = self.update_history[-100:]  # 最近 100 次更新
        
        delta_ws = [u.total_delta_w for u in recent_updates]
        attn_contribs = [abs(u.delta_w_attn) for u in recent_updates]
        
        return {
            'total_updates': len(self.update_history),
            'avg_delta_w': float(np.mean(delta_ws)),
            'std_delta_w': float(np.std(delta_ws)),
            'ltp_count': sum(1 for d in delta_ws if d > 0),
            'ltd_count': sum(1 for d in delta_ws if d < 0),
            'attention_contribution': float(np.mean(attn_contribs)),
            'recent_trend': 'enhancing' if np.mean(delta_ws[-10:]) > 0 else 'depressing',
        }
    
    def visualize_weight_distribution(self, save_path: str = None) -> Dict[str, Any]:
        """可视化权重分布（返回统计数据，可选保存）"""
        weights = self.synaptic_weights.detach().cpu().numpy()
        non_zero = weights[weights > 0]
        
        # 计算分布统计
        hist, bins = np.histogram(non_zero, bins=50, range=(0, 1))
        
        result = {
            'histogram': hist.tolist(),
            'bins': bins.tolist(),
            'mean': float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0,
            'median': float(np.median(non_zero)) if len(non_zero) > 0 else 0.0,
            'skewness': float((np.mean(non_zero) - np.median(non_zero)) / (np.std(non_zero) + 1e-8)) if len(non_zero) > 0 else 0.0,
        }
        
        if save_path:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 权重分布直方图
            axes[0].hist(non_zero, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Synaptic Weight')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Weight Distribution')
            axes[0].axvline(np.mean(non_zero), color='red', linestyle='--', label=f'Mean: {np.mean(non_zero):.3f}')
            axes[0].legend()
            
            # 权重矩阵热图（采样）
            sample_size = min(100, self.n_neurons)
            sample_weights = weights[:sample_size, :sample_size]
            im = axes[1].imshow(sample_weights, cmap='viridis', aspect='auto')
            axes[1].set_xlabel('Neuron j')
            axes[1].set_ylabel('Neuron i')
            axes[1].set_title(f'Weight Matrix Sample ({sample_size}x{sample_size})')
            plt.colorbar(im, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result['saved_path'] = save_path
            logger.info(f"[TransformerSTDP] 权重分布图已保存到 {save_path}")
        
        return result
    
    def reset_history(self):
        """重置更新历史"""
        self.update_history = []
        logger.info("[TransformerSTDP] 更新历史已重置")
    
    def export_learning_state(self) -> Dict[str, Any]:
        """导出学习状态（用于保存/恢复）"""
        return {
            'synaptic_weights': self.synaptic_weights.detach().cpu().numpy().tolist(),
            'n_neurons': self.n_neurons,
            'd_model': self.d_model,
            'stdp_lr': self.stdp_lr,
            'attention_lambda': self.attention_lambda,
            'sparsity': self.sparsity,
            'update_count': len(self.update_history),
        }
    
    def import_learning_state(self, state: Dict[str, Any]):
        """导入学习状态（用于恢复）"""
        self.synaptic_weights.data = torch.tensor(state['synaptic_weights'])
        logger.info(f"[TransformerSTDP] 已恢复学习状态，更新次数={state.get('update_count', 0)}")


# ============================================================================
# 组件 1: 经典 STDP 规则
# ============================================================================

class ClassicSTDP:
    """经典 STDP 学习规则
    
    生物物理机制：
    - 前神经元先发放 → Ca²⁺大量内流 → LTP（长时程增强）
    - 后神经元先发放 → Ca²⁺少量内流 → LTD（长时程抑制）
    
    数学公式：
    Δw = A₊·exp(-Δt/τ₊)  if Δt > 0（前→后，LTP）
    Δw = -A₋·exp(Δt/τ₋)  if Δt < 0（后→前，LTD）
    """
    
    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
    
    def compute(self, delta_t: float) -> float:
        """计算 STDP 权重变化
        
        Args:
            delta_t: t_post - t_pre（毫秒）
        
        Returns:
            Δw（可正可负）
        """
        if delta_t > 0:
            # LTP（长时程增强）
            delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # LTD（长时程抑制）
            delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)
        
        return float(delta_w)
    
    def get_curve(self, delta_t_range: np.ndarray) -> np.ndarray:
        """获取完整的 STDP 曲线（用于可视化）"""
        return np.array([self.compute(dt) for dt in delta_t_range])


# ============================================================================
# 组件 2: Attention Gradient Learner
# ============================================================================

class AttentionGradientLearner(nn.Module):
    """Attention 梯度学习者
    
    功能：
    从全局上下文提取 attention weights 的梯度，
    作为"自上而下"的调制信号。
    
    理论基础：
    - 前额叶皮层的 top-down 信号
    - 注意力的增益调控机制
    
    v3.1 改进：
    compute_gradient() 现在通过可学习的投影层直接从全局上下文
    生成 N×N 调制矩阵，确保每个突触都能获得结构化的 attention 梯度。
    """
    
    def __init__(self, d_model: int = 768, n_heads: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention 层（用于 forward）
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 可学习的 query（全局工作空间 query）
        self.workspace_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # v3.1: 用于 compute_gradient 的投影层
        # context → pre-synaptic neuron relevance scores
        self.modulation_pre = nn.Linear(d_model, d_model, bias=False)
        # context → post-synaptic neuron relevance scores
        self.modulation_post = nn.Linear(d_model, d_model, bias=False)
        
        # 初始化投影层：gain=1.0 确保输出量级 ~O(1/sqrt(d_model))
        # 经过 tanh 和外积后，gradient[i,j] ~ O(2/d_model)，与 STDP 的 A_plus (~0.01) 同阶
        nn.init.xavier_normal_(self.modulation_pre.weight, gain=1.0)
        nn.init.xavier_normal_(self.modulation_post.weight, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播并保留梯度
        
        Args:
            x: 输入表征 [B, N, D]
        
        Returns:
            output: attention 输出 [B, 1, D]
            attn_weights: 注意力权重 [B, H, 1, N]
        """
        q = self.workspace_query.expand(x.shape[0], -1, -1)
        k = v = x
        
        output, attn_weights = self.attention(
            query=q,
            key=k,
            value=v,
            need_weights=True
        )
        
        return output, attn_weights
    
    def compute_gradient(self, global_context: torch.Tensor, target_size: int = None) -> np.ndarray:
        """计算 attention-based 突触调制矩阵
        
        v3.1 改进：直接从全局上下文计算 N×N 调制矩阵，
        确保每个突触 (i,j) 都获得结构化、上下文相关的梯度信号。
        
        数学原理：
        gradient[i,j] = tanh(W_post · ctx)[i] * tanh(W_pre · ctx)[j] / N
        
        这等价于 attention 机制中 query-key 外积的简化形式，
        其中 W_pre/W_post 是可学习的投影层。
        
        Args:
            global_context: 全局上下文 [D] 或 [B, D]
            target_size: 目标矩阵大小（默认使用 d_model）
        
        Returns:
            gradient: 调制矩阵 [target_size, target_size]
        """
        # 确保是 2D
        if global_context.dim() == 1:
            global_context = global_context.unsqueeze(0)  # [1, D]
        
        B, D = global_context.shape
        
        if target_size is None:
            target_size = D
        
        # 投影为 pre/post 突触相关性分数
        # tanh 保证输出在 [-1, 1]，避免梯度爆炸
        pre_scores = torch.tanh(self.modulation_pre(global_context))   # [B, D]
        post_scores = torch.tanh(self.modulation_post(global_context))  # [B, D]
        
        # 截断或填充到 target_size
        if D < target_size:
            pre_pad = torch.zeros(B, target_size, device=global_context.device)
            post_pad = torch.zeros(B, target_size, device=global_context.device)
            pre_pad[:, :D] = pre_scores
            post_pad[:, :D] = post_scores
            pre_scores, post_scores = pre_pad, post_pad
        elif D > target_size:
            pre_scores = pre_scores[:, :target_size]
            post_scores = post_scores[:, :target_size]
        
        # 外积生成 N×N 调制矩阵
        # gradient[i,j] = post_scores[i] * pre_scores[j]
        gradient = torch.bmm(
            post_scores.unsqueeze(2),  # [B, N, 1]
            pre_scores.unsqueeze(1)    # [B, 1, N]
        )  # [B, N, N]
        
        # 不做额外缩放：xavier_normal(gain=1.0) 的输出经 tanh 后 ~O(1/sqrt(N))
        # 外积元素 ~O(1/N)，与 STDP 的 A_plus (~0.01) 自然同阶
        
        return gradient.squeeze(0).detach().cpu().numpy()


# ============================================================================
# 组件 3: Neuromodulator Gate
# ============================================================================

class NeuromodulatorGate(nn.Module):
    """神经递质门控机制
    
    四种主要神经调质对学习率的调制：
    - 多巴胺（DA）：奖赏预测误差，动机学习
    - 血清素（5-HT）：情绪调节，冲动控制
    - 去甲肾上腺素（NE）：警觉，应激反应
    - 乙酰胆碱（ACh）：注意门控，学习可塑性
    
    调制函数：
    η = 1.0 + w_DA·DA + w_5HT·5HT + w_NE·NE + w_ACh·ACh
    """
    
    def __init__(self):
        super().__init__()
        
        # 各神经调质的权重（可学习）
        self.weights = nn.ParameterDict({
            'dopamine': nn.Parameter(torch.tensor(0.5)),      # 正相关
            'serotonin': nn.Parameter(torch.tensor(0.2)),     # 轻微正相关
            'norepinephrine': nn.Parameter(torch.tensor(0.3)), # 中等正相关（警觉促进学习）
            'acetylcholine': nn.Parameter(torch.tensor(0.6)),  # 强正相关（ACh 直接调控可塑性）
        })
        
        # 基线水平
        self.baselines = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'acetylcholine': 0.5,
        }
    
    def forward(self, neurotransmitter_state: Dict[str, float]) -> torch.Tensor:
        """计算调制因子
        
        v3.1 改进：使用指数门控替代线性门控，放大神经调质偏离基线的效果。
        η = exp(Σ w_k · (n_k - baseline_k))
        
        Args:
            neurotransmitter_state: 各神经递质当前水平（0-1）
        
        Returns:
            modulation_factor: 学习率调制因子
        """
        exponent = 0.0
        
        for nt_name, weight_param in self.weights.items():
            if nt_name in neurotransmitter_state:
                current_level = neurotransmitter_state[nt_name]
                baseline = self.baselines[nt_name]
                
                # 偏离基线的程度
                deviation = current_level - baseline
                
                # 加权
                weight = weight_param.item()
                exponent += weight * deviation
        
        # 指数门控：放大偏离基线的效果
        modulation = float(np.exp(exponent))
        
        # 限制在合理范围 [0.1, 3.0]
        modulation = max(0.1, min(3.0, modulation))
        
        return torch.tensor(modulation)
    
    def get_learning_rate(self, neurotransmitter_state: Dict[str, float]) -> float:
        """获取学习率调制因子（向后兼容）"""
        modulation = self.forward(neurotransmitter_state)
        return float(modulation.item())
    
    def analyze_modulation(self, neurotransmitter_state: Dict[str, float]) -> Dict[str, Any]:
        """分析神经调制的效果
        
        Returns:
            详细分析报告
        """
        modulation = self.forward(neurotransmitter_state)
        
        analysis = {
            'total_modulation': float(modulation.item()),
            'contributions': {},
        }
        
        for nt_name, weight_param in self.weights.items():
            if nt_name in neurotransmitter_state:
                current = neurotransmitter_state[nt_name]
                baseline = self.baselines[nt_name]
                deviation = current - baseline
                contribution = weight_param.item() * deviation
                
                analysis['contributions'][nt_name] = {
                    'current_level': current,
                    'baseline': baseline,
                    'deviation': deviation,
                    'weighted_contribution': contribution,
                    'effect': '促进' if contribution > 0 else '抑制',
                }
        
        return analysis


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'STDPEvent',
    'SynapticUpdate',
    'TransformerSTDP',
    'ClassicSTDP',
    'AttentionGradientLearner',
    'NeuromodulatorGate',
]
