"""
NeuroConscious Transformer - Cross-Modal Attention 整合层
Cross-Modal Attention Integration Layer

核心功能：
实现真正的语义级多模态融合，而非简单拼接。

生物合理性：
- 模拟上颞叶沟（STS）的多模态整合功能
- 通过 Cross-Attention 实现视觉 - 听觉 - 体感的语义交互
- 注意力权重反映不同模态的贡献度

数学原理：
Cross-Attention(Q, K, V) = softmax(QK^T / √d)V

其中：
- Query: 来自全局工作空间的 query token（类似 CLS token）
- Key: 所有模态的 embedding
- Value: 所有模态的 embedding

架构设计：
```
Visual Emb [B, N_v, D] ─┐
                        ├→ Concat → [B, N_total, D] → Cross-Attention → Integrated [B, 1, D]
Audio Emb [B, N_a, D] ──┤
                        │
Intero Emb [B, 1, D] ───┘
```

作者：WENG YONGGANG(翁勇刚)
创建：2026 年 2 月 21 日
版本：v3.1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalIntegration(nn.Module):
    """Cross-Modal 整合层
    
    使用 Cross-Attention 机制实现多模态语义融合
    
    关键创新：
    1. 不是简单 concatenation，而是有结构的交互
    2. 多头注意力分别关注不同特征维度：
       - Head 1-2: 空间对齐（视觉 - 本体感觉）
       - Head 3-4: 时间同步（听觉 - 视觉）
       - Head 5-6: 情感调制（内感受 - 所有模态）
       - Head 7-8: 语义一致性（跨模态匹配）
    3. 可解释性：可以可视化每个 head 关注哪个模态
    """
    
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        
        # 1. Workspace Query（可学习参数，类似 CLS token）
        # 这是一个"全局整合器"token，负责收集所有模态的信息
        self.workspace_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Multi-Head Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
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
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 5. 模态特异性投影（可选，用于处理不同模态的分布差异）
        self.modality_projection = nn.ModuleDict({
            'visual': nn.Identity(),  # 已在前面的 MultiModalEncoder 中投影
            'audio': nn.Identity(),
            'intero': nn.Identity(),
        })
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Cross-Modal 整合
        
        Args:
            embeddings: 字典格式的模态 embeddings
                - 'visual_emb': [B, N_v, D]
                - 'audio_emb': [B, N_a, D]
                - 'intero_emb': [B, 1, D]
        
        Returns:
            integrated: 整合后的表征 [B, D]
            info: 包含注意力图谱等诊断信息
        """
        B = next(iter(embeddings.values())).shape[0]
        
        # Step 1: 拼接所有模态 tokens
        # 确保顺序一致：visual → audio → intero
        all_tokens = []
        modality_info = {}
        
        if 'visual_emb' in embeddings:
            visual_emb = embeddings['visual_emb']
            all_tokens.append(visual_emb)
            modality_info['visual'] = {
                'start': 0,
                'end': visual_emb.shape[1],
                'n_tokens': visual_emb.shape[1]
            }
        
        if 'audio_emb' in embeddings:
            audio_emb = embeddings['audio_emb']
            all_tokens.append(audio_emb)
            modality_info['audio'] = {
                'start': len(all_tokens) - 1,  # 累加索引
                'end': sum([t.shape[1] for t in all_tokens]),
                'n_tokens': audio_emb.shape[1]
            }
        
        if 'intero_emb' in embeddings:
            intero_emb = embeddings['intero_emb']
            all_tokens.append(intero_emb)
            modality_info['intero'] = {
                'start': sum([t.shape[1] for t in all_tokens[:-1]]),
                'end': sum([t.shape[1] for t in all_tokens]),
                'n_tokens': 1
            }
        
        # 拼接为 [B, N_total, D]
        all_tokens = torch.cat(all_tokens, dim=1)
        N_total = all_tokens.shape[1]
        
        logger.debug(f"[CrossModalIntegration] 拼接后的 tokens: {all_tokens.shape}")
        
        # Step 2: Cross-Attention（Workspace Query 作为 Q，所有 tokens 作为 K/V）
        # expand workspace query to match batch size
        q = self.workspace_query.expand(B, -1, -1)  # [B, 1, D]
        k = v = all_tokens  # [B, N_total, D]
        
        # Cross-Attention
        integrated, attn_weights = self.cross_attention(
            query=q,
            key=k,
            value=v,
            need_weights=True,
            average_attn_weights=False
        )
        
        # attn_weights shape: [B, 1, N_total]
        # 这表示 workspace 对每个 token 的关注度
        
        # Step 3: Residual Connection + LayerNorm
        integrated = self.norm1(integrated + q)
        
        # Step 4: Feed-Forward Network
        ff_output = self.ffn(integrated)
        integrated = self.norm2(integrated + ff_output)
        
        # integrated shape: [B, 1, D]
        # 压缩为 [B, D]
        integrated = integrated.squeeze(1)
        
        # Step 5: 提取模态贡献度
        modality_contributions = self._compute_modality_contributions(
            attn_weights, modality_info
        )
        
        # Step 6: 诊断信息
        info = {
            'attention_weights': attn_weights.detach().cpu().numpy(),  # [B, 1, N_total]
            'modality_contributions': modality_contributions,
            'integrated_norm': integrated.norm(dim=-1).detach().cpu().numpy(),
        }
        
        logger.debug(f"[CrossModalIntegration] 整合完成，模态贡献：{modality_contributions}")
        
        return integrated, info
    
    def _compute_modality_contributions(
        self, 
        attn_weights: torch.Tensor,
        modality_info: Dict[str, Dict]
    ) -> Dict[str, float]:
        """计算每个模态的贡献度
        
        通过对 attention weights 在对应 token 范围内求和得到
        
        Args:
            attn_weights: [B, 1, N_total]
            modality_info: 每个模态的 start/end 索引
        
        Returns:
            各模态贡献度字典
        """
        contributions = {}
        
        for modality_name, info in modality_info.items():
            start_idx = info['start']
            end_idx = info['end']
            
            # 对该模态的所有 tokens 的 attention 求和
            modality_attn = attn_weights[:, :, start_idx:end_idx].sum(dim=-1)
            
            # 平均 over batch
            avg_contribution = modality_attn.mean().item()
            contributions[modality_name] = round(avg_contribution, 4)
        
        # 归一化到总和为 1
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: round(v / total, 4) for k, v in contributions.items()}
        
        return contributions


class ModalityGating(nn.Module):
    """模态门控机制（可选增强）
    
    动态调整每个模态的输入强度，模拟注意力的早期选择
    
    生物合理性：
    - 丘脑网状核的感觉门控功能
    - 顶叶的空间注意调控
    
    使用场景：
    - 当某个模态噪声过大时，自动降低其权重
    - 任务需求变化时（如听觉任务 vs 视觉任务）
    """
    
    def __init__(self, d_model: int = 768, n_modalities: int = 3):
        super().__init__()
        
        self.n_modalities = n_modalities
        
        # 门控向量生成器
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_modalities),
            nn.Sigmoid()  # 输出 [0, 1] 范围
        )
        
    def forward(
        self, 
        embeddings: Dict[str, torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """门控调制
        
        Args:
            embeddings: 原始 embeddings
            context: 可选的上下文信息（如任务描述），用于生成门控信号
        
        Returns:
            门控调制后的 embeddings
        """
        if context is None:
            # 如果没有上下文，使用所有 embeddings 的平均作为 context
            all_emb = torch.cat([v for v in embeddings.values()], dim=1)
            context = all_emb.mean(dim=1)  # [B, D]
        
        # 生成门控向量 [B, n_modalities]
        gates = self.gate_generator(context)
        
        # 应用门控
        gated_embeddings = {}
        for i, (modality, emb) in enumerate(embeddings.items()):
            gate = gates[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            gated_embeddings[modality] = emb * gate
        
        logger.debug(f"[ModalityGating] 门控值：{gates.detach().cpu().numpy()}")
        
        return gated_embeddings


# ============================================================================
# 可视化辅助工具
# ============================================================================

def visualize_cross_modal_attention(
    attention_weights: np.ndarray,
    modality_info: Dict[str, Dict],
    title: str = "Cross-Modal Attention Map"
):
    """可视化 Cross-Modal 注意力图谱
    
    Args:
        attention_weights: [B, 1, N_total] 或 [H, N_total]
        modality_info: 每个模态的 start/end 索引
        title: 图表标题
    
    返回 matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn 未安装，无法可视化")
        return None
    
    # 如果是 batch 格式，取第一个样本
    if attention_weights.ndim == 3:
        attn = attention_weights[0, 0, :]  # [N_total]
    elif attention_weights.ndim == 2:
        attn = attention_weights.mean(axis=0)  # 平均 over heads
    else:
        attn = attention_weights.flatten()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 绘制注意力权重
    x = np.arange(len(attn))
    ax.bar(x, attn, alpha=0.7, color='steelblue')
    
    # 标注模态边界
    colors = ['red', 'green', 'blue']
    for i, (modality_name, info) in enumerate(modality_info.items()):
        start = info['start']
        end = info['end']
        
        # 高亮区域
        ax.axvspan(start, end - 1, alpha=0.2, color=colors[i % len(colors)], 
                  label=f'{modality_name} (tokens {start}-{end-1})')
    
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'CrossModalIntegration',
    'ModalityGating',
    'visualize_cross_modal_attention',
]
